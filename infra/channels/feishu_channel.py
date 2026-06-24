# -*- coding: utf-8 -*-
"""飞书 Channel 接入适配层（基于 websockets 直接实现）

不依赖飞书 SDK，自己实现 WebSocket 长连接接入 MessageBus 架构：

入站流程：
    飞书用户私聊 → WebSocket → _on_message()
    → 转换为 InboundMessage → MessageBus.publish_inbound()

出站流程：
    MessageBus → OutboundMessage → _on_response()
    → REST API 发送消息

流式处理：
    EventBus → StreamDeltaReady 事件 → _on_stream_delta()

增强功能（借鉴 hermes-agent 飞书实现）：
    - 去重持久化（重启后不重复）
    - Markdown → 飞书 Post 渲染
    - 消息类型标准化（post/image/file 等）
    - Reaction 反馈（Typing → CrossMark）
    - 重试 + Reply 回退机制
    - Sender 名字缓存
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import websockets

from bus.event_bus import EventBus
from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import StreamDeltaReady, ToolCallCompleted, ToolCallStarted, TurnCommitted, TurnStarted
from bus.queue import MessageBus
from agent.looping.interrupt import InterruptController

# protobuf 帧解析（飞书使用 PBBP2 二进制协议）
try:
    from lark_oapi.ws.pb.pbbp2_pb2 import Frame
    from lark_oapi.ws.enum import FrameType

    _HAS_PBBP2 = True
except ImportError:
    _HAS_PBBP2 = False
    Frame = None
    FrameType = None

logger = logging.getLogger("feishu")
_CHANNEL = "feishu"

# 去重滑动窗口大小
_SEEN_MSG_MAXSIZE = 500
_MESSAGE_TEXT_CACHE_MAXSIZE = 500

# 重试配置
_FEISHU_SEND_ATTEMPTS = 3
_FEISHU_REPLY_FALLBACK_CODES = frozenset({230011, 231003})

# Reaction 配置（使用飞书标准 emoji_type 字符串）
_REACTION_IN_PROGRESS = "SMILE"  # 表示处理中
_REACTION_FAILURE = "SAD"         # 表示失败
_REACTION_CACHE_SIZE = 512

# Sender 名字缓存 TTL（秒）
_SENDER_NAME_TTL_SECONDS = 10 * 60

# 飞书 API 配置
_FEISHU_API_BASE = "https://open.feishu.cn"
_TOKEN_URL = f"{_FEISHU_API_BASE}/open-apis/auth/v3/tenant_access_token/internal"
_WS_GATEWAY_URL = f"{_FEISHU_API_BASE}/callback/ws/endpoint"

# Markdown 检测正则
_MARKDOWN_HINT_RE = re.compile(
    r"(^#{1,6}\s)|(^\s*[-*]\s)|(^\s*\d+\.\s)|(^\s*---+\s*$)|(```)|(`[^`\n]+`)"
    r"|(\*\*[^*\n].+?\*\*)|(~~[^~\n].+?~~)|(\*[^*\n]+\*)|(\[[^\]]+\]\([^)]+\))",
    re.MULTILINE,
)
_MARKDOWN_SPECIAL_CHARS_RE = re.compile(r"([\\`*_{}\[\]()#+\-!|>~])")


@dataclass
class _TokenCache:
    token: str
    expires_at: float


class FeishuSendError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        response: object | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.response = response


def _get_hermes_home() -> Path:
    """获取 hermes home 路径"""
    import os
    from pathlib import Path
    home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
    return Path(home)


def _atomic_json_write(path: Path, data: dict) -> None:
    """原子写入 JSON 文件"""
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _escape_markdown_text(text: str) -> str:
    return _MARKDOWN_SPECIAL_CHARS_RE.sub(r"\\\1", text)


class FeishuChannel:
    """飞书 Channel 适配器（基于 websockets 直接实现）"""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        bus: MessageBus,
        allow_from: list[str] | None = None,
        event_bus: EventBus | None = None,
        interrupt_controller: InterruptController | None = None,
        enable_thinking: bool = False,
        session_manager: Any | None = None,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._bus = bus
        self._allow_from = set(allow_from) if allow_from else set()
        self._interrupt_controller = interrupt_controller
        self._thinking_enabled = enable_thinking
        self._session_manager = session_manager

        # HTTP 客户端
        self._http = httpx.AsyncClient(timeout=30.0)

        # 状态
        self._token: _TokenCache | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        self._ws_url: str | None = None

        # 流式控制器
        self._stream_controllers: dict[str, Any] = {}

        # ── 去重持久化 ────────────────────────────────────────────────────
        self._seen_message_ids: list[str] = []
        self._dedup_state_path = _get_hermes_home() / "feishu_seen_message_ids.json"
        self._load_seen_message_ids()
        self._message_text_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._message_text_cache_path = _get_hermes_home() / "feishu_message_text_cache.json"
        self._load_message_text_cache()

        # ── Reaction 反馈 ────────────────────────────────────────────────
        # message_id → reaction_id（用于完成后清除/替换 reaction）
        self._pending_processing_reactions: "OrderedDict[str, str]" = OrderedDict()

        # ── Sender 名字缓存 ──────────────────────────────────────────────
        self._sender_name_cache: dict[str, tuple[str, float]] = {}  # sender_id → (name, expires_at)

        # ── Session → Message ID 追踪（用于 Reaction） ──────────────────
        # session_key → message_id（追踪每个 turn 的原始消息，用于 reaction 更新）
        self._session_last_message_id: dict[str, str] = {}

        # ── 卡片流式（Card Kit v3） ──────────────────────────────────────
        # session_key → 卡片状态
        self._card_id: dict[str, str] = {}            # session_key → card_id
        self._card_seq: dict[str, int] = {}           # session_key → PUT sequence
        self._card_tool_states: dict[str, list[dict[str, Any]]] = {}  # session_key → [{name, status, result_preview}]
        self._card_reply_buf: dict[str, str] = {}     # session_key → 累积回复
        self._card_thinking_buf: dict[str, str] = {}  # session_key → 累积思考过程
        self._card_last_push: dict[str, float] = {}   # session_key → 上次推送时间
        self._card_last_think_push: dict[str, float] = {}  # session_key → 上次思考推送时间
        self._card_last_think_len: dict[str, int] = {}    # session_key → 上次推送时的思考长度
        self._card_last_reply_len: dict[str, int] = {}    # session_key → 上次推送时的回复长度
        self._card_done: set[str] = set()             # 已用卡片回复过的 session_key
        self._card_message_ids: dict[str, str] = {}   # card_id → message_id（用于回复上下文缓存）

        # 订阅事件总线
        if event_bus is not None:
            event_bus.on(TurnStarted, self._on_turn_started)
            event_bus.on(StreamDeltaReady, self._on_stream_delta)
            event_bus.on(ToolCallStarted, self._on_tool_call_started)
            event_bus.on(ToolCallCompleted, self._on_tool_call_completed)
            event_bus.on(TurnCommitted, self._on_turn_committed)

    # ── 去重持久化 ──────────────────────────────────────────────────────────

    def _load_seen_message_ids(self) -> None:
        """从磁盘加载已见消息 ID 列表"""
        try:
            if self._dedup_state_path.exists():
                data = json.loads(self._dedup_state_path.read_text(encoding="utf-8"))
                self._seen_message_ids = data.get("message_ids", [])[-_SEEN_MSG_MAXSIZE:]
                logger.debug("[feishu] 加载 %d 条去重记录", len(self._seen_message_ids))
        except Exception as e:
            logger.warning("[feishu] 加载去重状态失败: %s", e)
            self._seen_message_ids = []

    def _save_seen_message_ids(self) -> None:
        """持久化已见消息 ID 列表"""
        try:
            _atomic_json_write(self._dedup_state_path, {"message_ids": self._seen_message_ids})
        except Exception as e:
            logger.warning("[feishu] 保存去重状态失败: %s", e)

    def _mark_seen(self, message_id: str) -> bool:
        """标记消息已见。返回 True 表示重复，False 表示新消息"""
        if message_id in self._seen_message_ids:
            return True
        self._seen_message_ids.append(message_id)
        while len(self._seen_message_ids) > _SEEN_MSG_MAXSIZE:
            self._seen_message_ids.pop(0)
        self._save_seen_message_ids()
        return False

    def _load_message_text_cache(self) -> None:
        """加载最近消息文本缓存，用于还原用户回复的历史/卡片消息。"""
        try:
            if not self._message_text_cache_path.exists():
                return
            data = json.loads(self._message_text_cache_path.read_text(encoding="utf-8"))
            items = data.get("items", [])
            if not isinstance(items, list):
                return
            for item in items[-_MESSAGE_TEXT_CACHE_MAXSIZE:]:
                if not isinstance(item, dict):
                    continue
                message_id = str(item.get("message_id") or "")
                if not message_id:
                    continue
                self._message_text_cache[message_id] = {
                    "text": str(item.get("text") or ""),
                    "sender": str(item.get("sender") or ""),
                    "msg_type": str(item.get("msg_type") or ""),
                }
        except Exception as e:
            logger.warning("[feishu] 加载消息文本缓存失败: %s", e)
            self._message_text_cache.clear()

    def _save_message_text_cache(self) -> None:
        try:
            items = [
                {"message_id": message_id, **data}
                for message_id, data in self._message_text_cache.items()
            ]
            _atomic_json_write(self._message_text_cache_path, {"items": items})
        except Exception as e:
            logger.warning("[feishu] 保存消息文本缓存失败: %s", e)

    def _cache_message_text(
        self,
        message_id: str,
        text: str,
        *,
        sender: str = "",
        msg_type: str = "",
        session_key: str = "",
        session_message_id: str = "",
    ) -> None:
        message_id = str(message_id or "")
        text = str(text or "").strip()
        if not message_id or not text:
            return
        self._message_text_cache[message_id] = {
            "text": text,
            "sender": sender,
            "msg_type": msg_type,
        }
        self._message_text_cache.move_to_end(message_id)
        while len(self._message_text_cache) > _MESSAGE_TEXT_CACHE_MAXSIZE:
            self._message_text_cache.popitem(last=False)
        self._save_message_text_cache()
        self._remember_channel_message_ref(
            message_id=message_id,
            text=text,
            sender=sender,
            msg_type=msg_type,
            session_key=session_key,
            session_message_id=session_message_id,
        )

    def _get_cached_message_text(self, message_id: str) -> dict[str, Any] | None:
        stored = self._get_channel_message_ref(message_id)
        if stored:
            return stored
        cached = self._message_text_cache.get(str(message_id or ""))
        if cached:
            self._message_text_cache.move_to_end(str(message_id))
        return cached

    def _remember_channel_message_ref(
        self,
        *,
        message_id: str,
        text: str,
        sender: str = "",
        msg_type: str = "",
        session_key: str = "",
        session_message_id: str = "",
    ) -> None:
        remember = getattr(self._session_manager, "remember_channel_message_ref", None)
        if not callable(remember):
            return
        try:
            remember(
                channel=_CHANNEL,
                channel_message_id=message_id,
                session_key=session_key,
                session_message_id=session_message_id,
                sender=sender,
                msg_type=msg_type,
                text=text,
            )
        except Exception as e:
            logger.warning("[feishu] 保存消息引用索引失败 message_id=%s: %s", message_id, e)

    def _get_channel_message_ref(self, message_id: str) -> dict[str, Any] | None:
        get_ref = getattr(self._session_manager, "get_channel_message_ref", None)
        if not callable(get_ref):
            return None
        try:
            ref = get_ref(channel=_CHANNEL, channel_message_id=str(message_id or ""))
        except Exception as e:
            logger.warning("[feishu] 读取消息引用索引失败 message_id=%s: %s", message_id, e)
            return None
        if not isinstance(ref, dict):
            return None
        text = str(ref.get("text") or "").strip()
        if not text:
            return None
        return {
            "text": text,
            "sender": str(ref.get("sender") or ""),
            "msg_type": str(ref.get("msg_type") or ""),
        }

    # ── Sender 名字缓存 ──────────────────────────────────────────────────────

    def _get_sender_name(self, sender_id: str, default_name: str = "") -> str:
        """从缓存或默认值获取 sender 显示名"""
        now = time.time()
        cached = self._sender_name_cache.get(sender_id)
        if cached and cached[1] > now:
            return cached[0]
        return default_name

    def _cache_sender_name(self, sender_id: str, name: str) -> None:
        """缓存 sender 名字，TTL 10 分钟"""
        self._sender_name_cache[sender_id] = (name, time.time() + _SENDER_NAME_TTL_SECONDS)

    # ── Reaction 反馈 ───────────────────────────────────────────────────────

    def _cache_reaction(self, message_id: str, reaction_id: str) -> None:
        """缓存 message_id → reaction_id 映射（LRU）"""
        self._pending_processing_reactions[message_id] = reaction_id
        while len(self._pending_processing_reactions) > _REACTION_CACHE_SIZE:
            self._pending_processing_reactions.popitem(last=False)

    def _pop_reaction(self, message_id: str) -> str | None:
        """弹出并返回 reaction_id"""
        return self._pending_processing_reactions.pop(message_id, None)

    async def _add_reaction(self, message_id: str, emoji_type: str) -> str | None:
        """添加 reaction，返回 reaction_id"""
        logger.debug(f"[feishu] 尝试添加 reaction: message_id={message_id} emoji={emoji_type}")
        try:
            token = await self._get_access_token()
            # 飞书 API: POST /open-apis/im/v1/messages/{message_id}/reactions
            resp = await self._http.post(
                f"{_FEISHU_API_BASE}/open-apis/im/v1/messages/{message_id}/reactions",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={
                    "reaction_type": {"emoji_type": emoji_type},
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                logger.debug(f"[feishu] 添加 reaction 成功 message_id={message_id}")
                return data.get("data", {}).get("reaction_id")
            logger.warning("[feishu] 添加 reaction 失败 status=%d: %s", resp.status_code, resp.text)
        except Exception as e:
            logger.warning("[feishu] 添加 reaction 异常: %s", e)
        return None

    async def _remove_reaction(self, message_id: str, reaction_id: str) -> bool:
        """删除 reaction"""
        logger.debug(f"[feishu] 尝试移除 reaction: message_id={message_id} reaction_id={reaction_id}")
        try:
            token = await self._get_access_token()
            # 飞书 API: DELETE /open-apis/im/v1/messages/{message_id}/reactions/{reaction_id}
            resp = await self._http.delete(
                f"{_FEISHU_API_BASE}/open-apis/im/v1/messages/{message_id}/reactions/{reaction_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            logger.debug(f"[feishu] 移除 reaction 成功 message_id={message_id}")
            return resp.status_code == 200
        except Exception as e:
            logger.warning("[feishu] 删除 reaction 失败: %s", e)
            return False

    # ── 媒体资源下载 ──────────────────────────────────────────────────────────

    def _media_cache_dir(self) -> Path:
        p = Path.home() / ".akashic" / "workspace" / "media_cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    async def _download_image_resource(
        self, message_id: str, image_key: str
    ) -> str | None:
        """下载飞书消息中的图片到本地缓存，返回本地路径"""
        try:
            token = await self._get_access_token()
            resp = await self._http.get(
                f"{_FEISHU_API_BASE}/open-apis/im/v1/messages/{message_id}/resources/{image_key}",
                params={"type": "image"},
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code != 200:
                logger.warning(
                    "[feishu] 下载图片失败 status=%d image_key=%s: %s",
                    resp.status_code, image_key, resp.text[:200],
                )
                return None

            content_type = resp.headers.get("content-type", "")
            ext = self._guess_image_ext(content_type)
            local_path = self._media_cache_dir() / f"{image_key}{ext}"
            local_path.write_bytes(resp.content)
            logger.debug(
                "[feishu] 图片已缓存 image_key=%s -> %s (%d bytes)",
                image_key, local_path.name, len(resp.content),
            )
            return str(local_path)
        except Exception as e:
            logger.warning("[feishu] 下载图片异常 image_key=%s: %s", image_key, e)
            return None

    @staticmethod
    def _guess_image_ext(content_type: str) -> str:
        ct = (content_type or "").split(";")[0].strip().lower()
        mapping = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
        }
        return mapping.get(ct, ".jpg")

    # ── 文件上传（发送用） ────────────────────────────────────────────────────

    async def _upload_image(self, image_path: str) -> str | None:
        """上传图片到飞书，返回 image_key"""
        import io

        p = Path(image_path)
        if not p.is_file():
            logger.warning("[feishu] 图片文件不存在: %s", image_path)
            return None

        token = await self._get_access_token()
        content = p.read_bytes()

        # httpx multipart: 直接用 files 参数
        resp = await self._http.post(
            f"{_FEISHU_API_BASE}/open-apis/im/v1/images",
            headers={"Authorization": f"Bearer {token}"},
            data={"image_type": "message"},
            files={"image": (p.name, io.BytesIO(content), self._guess_mime(p))},
        )
        if resp.status_code != 200:
            logger.warning("[feishu] 上传图片失败 status=%d: %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        code = data.get("code", -1)
        if code != 0:
            logger.warning("[feishu] 上传图片 API 错误 code=%s msg=%s", code, data.get("msg", ""))
            return None
        image_key = (data.get("data") or {}).get("image_key")
        if image_key:
            logger.debug("[feishu] 图片上传成功 image_key=%s", image_key)
        return image_key

    async def _upload_file(self, file_path: str, file_name: str | None = None) -> str | None:
        """上传文件到飞书，返回 file_key"""
        import io

        p = Path(file_path)
        if not p.is_file():
            logger.warning("[feishu] 文件不存在: %s", file_path)
            return None

        name = file_name or p.name
        ext = p.suffix.lower()
        # 飞书 file_type: pdf/doc/xls/ppt/stream（通用）
        doc_types = {
            ".pdf": "pdf", ".doc": "doc", ".docx": "doc",
            ".xls": "xls", ".xlsx": "xls",
            ".ppt": "ppt", ".pptx": "ppt",
        }
        file_type = doc_types.get(ext, "stream")

        token = await self._get_access_token()
        content = p.read_bytes()

        resp = await self._http.post(
            f"{_FEISHU_API_BASE}/open-apis/im/v1/files",
            headers={"Authorization": f"Bearer {token}"},
            data={"file_type": file_type, "file_name": name},
            files={"file": (name, io.BytesIO(content), self._guess_mime(p))},
        )
        if resp.status_code != 200:
            logger.warning("[feishu] 上传文件失败 status=%d: %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        code = data.get("code", -1)
        if code != 0:
            logger.warning("[feishu] 上传文件 API 错误 code=%s msg=%s", code, data.get("msg", ""))
            return None
        file_key = (data.get("data") or {}).get("file_key")
        if file_key:
            logger.debug("[feishu] 文件上传成功 file_key=%s", file_key)
        return file_key

    @staticmethod
    def _guess_mime(p: Path) -> str:
        ext = p.suffix.lower()
        return {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp", ".bmp": "image/bmp",
            ".pdf": "application/pdf",
            ".mp4": "video/mp4", ".mov": "video/quicktime",
            ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
        }.get(ext, "application/octet-stream")

    # ── Markdown → 飞书 Post ────────────────────────────────────────────────

    def _build_markdown_post_payload(self, content: str) -> str:
        """将 Markdown 内容转换为飞书 post 格式"""
        rows = self._build_markdown_post_rows(content)
        return json.dumps({"zh_cn": {"content": rows}}, ensure_ascii=False)

    def _build_markdown_post_rows(self, content: str) -> list[list[dict]]:
        """构建飞书 post rows，隔离代码块避免渲染问题"""
        if not content:
            return [[{"tag": "md", "text": ""}]]
        if "```" not in content:
            return [[{"tag": "md", "text": content}]]

        rows: list[list[dict]] = []
        current: list[str] = []
        in_code_block = False

        def _flush_current() -> None:
            nonlocal current
            if not current:
                return
            segment = "\n".join(current)
            if segment.strip():
                rows.append([{"tag": "md", "text": segment}])
            current = []

        for raw_line in content.splitlines():
            stripped_line = raw_line.strip()
            # 检测代码块开关
            if stripped_line.startswith("```"):
                if not in_code_block:
                    _flush_current()
                current.append(raw_line)
                in_code_block = not in_code_block
                if not in_code_block:
                    _flush_current()
                continue
            current.append(raw_line)
        _flush_current()
        return rows or [[{"tag": "md", "text": content}]]

    def _detect_msg_type(self, content: str) -> tuple[str, str]:
        """检测内容类型，返回 (msg_type, content_payload)"""
        if _MARKDOWN_HINT_RE.search(content):
            return "post", self._build_markdown_post_payload(content)
        return "text", json.dumps({"text": content}, ensure_ascii=False)

    async def start(self) -> None:
        """启动 WebSocket 连接"""
        _ = self._stopped.clear()
        self._task = asyncio.create_task(self._gateway_loop())
        self._bus.subscribe_outbound(_CHANNEL, self._on_response)
        logger.info("[feishu] FeishuChannel 已启动")

    async def stop(self) -> None:
        """停止连接"""
        self._stopped.set()
        if self._task:
            _ = self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._http.aclose()
        logger.info("[feishu] FeishuChannel 已停止")

    # ── Token 管理 ──────────────────────────────────────────────────────────

    async def _get_access_token(self) -> str:
        """获取 tenant_access_token"""
        cached = self._token
        if cached and time.time() < cached.expires_at - 60:
            return cached.token

        resp = await self._http.post(
            _TOKEN_URL,
            json={
                "app_id": self._app_id,
                "app_secret": self._app_secret,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        token = str(data.get("tenant_access_token") or "")
        expires_in = int(data.get("expires_in") or 0)
        if not token:
            raise RuntimeError("Failed to get tenant_access_token")
        self._token = _TokenCache(token=token, expires_at=time.time() + expires_in)
        return token

    # ── 网关连接循环 ─────────────────────────────────────────────────────────

    async def _gateway_loop(self) -> None:
        """主循环：获取 WS 地址 → 连接 → 断线重连"""
        while not self._stopped.is_set():
            try:
                ws_info = await self._get_ws_gateway()
                # 响应结构: {"code": 0, "data": {"URL": "wss://..."}}
                data = ws_info.get("data") or {}
                self._ws_url = data.get("URL") or ""
                if not self._ws_url:
                    raise RuntimeError("No URL in response data")
                await self._run_ws(self._ws_url)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[feishu] gateway 连接失败: %s", e)
                await asyncio.sleep(5)

    async def _get_ws_gateway(self) -> dict[str, Any]:
        """获取 WebSocket Gateway URL"""
        resp = await self._http.post(
            _WS_GATEWAY_URL,
            json={
                "AppID": self._app_id,
                "AppSecret": self._app_secret,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        code = data.get("code", 0)
        if code != 0:
            raise RuntimeError(f"Failed to get WS gateway: code={code}")
        return data

    async def _run_ws(self, url: str) -> None:
        """运行 WebSocket 连接"""
        async with websockets.connect(url) as ws:
            # 认证握手
            await ws.send(json.dumps({
                "schema": "v1",
                "type": "websocket",
            }))
            logger.debug("[feishu] WebSocket 连接已建立")

            async for raw in ws:
                logger.debug(f"[feishu] 收到原始数据: type={type(raw).__name__}, len={len(raw)}, hex={raw[:50].hex() if isinstance(raw, bytes) else raw[:50]!r}")
                try:
                    payload = self._parse_frame(raw)
                    logger.debug(f"[feishu] 解析后 payload: {payload}")
                    if payload is not None:
                        await self._handle_ws_message(payload)
                except Exception as e:
                    logger.warning("[feishu] 处理消息失败: %s", e)

    def _parse_frame(self, raw: bytes | str) -> dict[str, Any] | None:
        """解析 WebSocket 帧（支持 protobuf 二进制和 JSON 文本）"""
        if isinstance(raw, str):
            # JSON 文本帧
            try:
                return json.loads(raw)
            except Exception:
                return None

        # 二进制 protobuf 帧
        if not _HAS_PBBP2 or Frame is None:
            logger.warning("[feishu] 收到二进制帧但无可用 protobuf 解析器")
            return None

        frame = Frame()
        try:
            frame.ParseFromString(raw)
        except Exception as e:
            logger.warning("[feishu] protobuf 解析失败: %s", e)
            return None

        ft = FrameType(frame.method)
        if ft == FrameType.CONTROL:
            # CONTROL 帧：处理 ping/pong 等控制消息
            self._handle_control_frame(frame)
            return None
        elif ft == FrameType.DATA:
            # DATA 帧：提取业务数据
            return self._handle_data_frame(frame)
        else:
            logger.debug(f"[feishu] 未知帧类型: {ft}")
            return None

    def _handle_control_frame(self, frame: Frame) -> None:
        """处理 CONTROL 帧（ping/pong）"""
        # CONTROL 帧的 payload 是简单字符串如 "ping"
        logger.debug(f"[feishu] 控制帧: {frame.payload!r}")

    def _handle_data_frame(self, frame: Frame) -> dict[str, Any]:
        """处理 DATA 帧，提取业务数据"""
        # DATA 帧的 payload 是 JSON 编码的事件数据（如 {"schema": "v1", "type": "im.message.receive_v1", "data": {...}}）
        try:
            return json.loads(frame.payload.decode("utf-8"))
        except Exception:
            return {"raw": frame.payload.hex()}

    async def _handle_ws_message(self, payload: dict[str, Any]) -> None:
        """处理 WebSocket 消息（DATA 帧解析后的 JSON）"""
        schema = payload.get("schema")
        # 飞书使用 "v1" 或 "2.0" schema
        if schema not in ("v1", "2.0"):
            logger.debug(f"[feishu] 未知 schema: {schema}")
            return

        msg_type = payload.get("header", {}).get("event_type") or payload.get("type", "")
        data = payload.get("event", {})

        # 飞书 2.0 schema: data 就是 event 对象
        # 飞书 v1 schema: data 是 {"event": {...}} 结构
        if "event" in data and "message" not in data:
            data = data["event"]

        if msg_type == "im.message.receive_v1":
            await self._handle_message(data)

    # ── 消息处理 ─────────────────────────────────────────────────────────────

    async def _handle_message(self, data: dict[str, Any]) -> None:
        """处理接收到的消息（data 已是 event 对象）"""
        message = data.get("message", {})
        sender = data.get("sender", {})

        chat_id = str(message.get("chat_id") or "")
        sender_id = str(sender.get("sender_id", {}).get("open_id") or "")
        message_id = str(message.get("message_id") or "")
        content_raw = message.get("content", "{}")
        msg_type = message.get("message_type", "text")

        # 解析 content（JSON 字符串）
        try:
            content = json.loads(content_raw)
        except Exception:
            content = {}

        # ── 跳过延迟重推的老消息（飞书 WebSocket 断连期间的消息可能被延迟推送）──
        create_time_ms = message.get("create_time", "")
        if create_time_ms:
            try:
                age_s = time.time() - int(create_time_ms) / 1000
                _STALE_MESSAGE_MAX_AGE_S = 300  # 超过 5 分钟的消息视为过期
                if age_s > _STALE_MESSAGE_MAX_AGE_S:
                    logger.info(
                        "[feishu] 跳过过期消息 message_id=%s age=%.0fs msg_type=%s",
                        message_id, age_s, msg_type,
                    )
                    # 标记为已见，避免后续重推
                    if message_id:
                        self._mark_seen(message_id)
                    return
            except (ValueError, TypeError):
                pass

        # 获取文本内容（支持 post/interactive 等富文本消息）
        text = self._extract_text_content(msg_type, content)
        if not text and msg_type == "text":
            return

        # 去重持久化
        if message_id and self._mark_seen(message_id):
            logger.debug(f"[feishu] 重复消息已忽略  message_id={message_id}")
            return

        # 私聊 only
        chat_type = message.get("chat_type", "")
        if chat_type != "p2p":
            logger.debug(f"[feishu] 忽略非私聊消息 chat_type={chat_type}")
            return

        # 白名单检查
        if self._allow_from and sender_id not in self._allow_from:
            logger.warning(f"[feishu] 拒绝未授权用户 open_id={sender_id}")
            return

        # 获取 sender 显示名（优先用缓存）
        sender_nickname = sender.get("sender_nickname") or ""
        if sender_nickname:
            self._cache_sender_name(sender_id, sender_nickname)
        else:
            sender_nickname = self._get_sender_name(sender_id, sender_nickname)

        logger.info(
            f"[feishu] 收到消息 chat_id={chat_id} "
            f"sender_id={sender_id} msg_type={msg_type} content={text[:50]!r}..."
        )

        self._cache_message_text(
            message_id,
            text,
            sender=sender_nickname or sender_id,
            msg_type=msg_type,
            session_key=f"{_CHANNEL}:{chat_id}",
        )

        # 如果是图片类型，下载到本地缓存（Core 管线会自动 base64 编码发送给 VL 模型）
        media: list[str] = []
        if msg_type == "image":
            image_key = content.get("image_key") or ""
            if image_key:
                logger.info("[feishu] 收到图片，开始下载 image_key=%s", image_key)
                local_path = await self._download_image_resource(message_id, image_key)
                if local_path:
                    media.append(local_path)
                    logger.info("[feishu] 图片已下载 -> %s", local_path)
                else:
                    logger.warning("[feishu] 图片下载失败 image_key=%s", image_key)

        # 追踪 session → message_id（用于 reaction 反馈）
        session_key = f"{_CHANNEL}:{chat_id}"
        # 新消息到达时，清理同一 session 的旧 pending reaction（如果有的话）
        old_msg_id = self._session_last_message_id.get(session_key)
        if old_msg_id and old_msg_id != message_id:
            old_reaction_id = self._pending_processing_reactions.pop(old_msg_id, None)
            if old_reaction_id:
                logger.debug(f"[feishu] 清理旧 reaction: message_id={old_msg_id} reaction_id={old_reaction_id}")
                await self._remove_reaction(old_msg_id, old_reaction_id)
        self._session_last_message_id[session_key] = message_id
        logger.debug(f"[feishu] 收到消息 session_key={session_key} message_id={message_id}")

        # 添加 "SMILE" reaction 表示正在处理
        if message_id:
            reaction_id = await self._add_reaction(message_id, _REACTION_IN_PROGRESS)
            if reaction_id:
                self._cache_reaction(message_id, reaction_id)
                logger.debug(f"[feishu] 添加 SMILE reaction 成功 message_id={message_id}")
            else:
                logger.warning(f"[feishu] 添加 SMILE reaction 失败 message_id={message_id}")

        reply_meta = await self._build_reply_context_metadata(message)

        await self._bus.publish_inbound(
            InboundMessage(
                channel=_CHANNEL,
                sender=sender_id,
                chat_id=chat_id,
                content=text,
                media=media,
                metadata={
                    "sender_name": sender_nickname,
                    "message_id": message_id,
                    "msg_type": msg_type,
                    **reply_meta,
                },
            )
        )

    def _extract_text_content(self, msg_type: str, content: dict[str, Any]) -> str:
        """根据消息类型提取文本内容"""
        if msg_type == "text":
            return str(content.get("text") or "")
        if msg_type == "post":
            return self._parse_post_content(content)
        if msg_type == "interactive":
            return self._parse_interactive_content(content)
        if msg_type == "image":
            return "[图片]"
        if msg_type in ("file", "audio", "media"):
            file_name = content.get("file_name") or content.get("title") or ""
            return f"[附件: {file_name}]" if file_name else "[附件]"
        if msg_type == "share_chat":
            chat_name = content.get("chat_name") or content.get("name") or ""
            return f"[分享的聊天: {chat_name}]" if chat_name else "[分享的聊天]"
        if msg_type == "merge_forward":
            title = content.get("title") or ""
            return f"[转发消息: {title}]" if title else "[转发消息]"
        # 回退：尝试提取 text 或 content 字段
        return content.get("text") or content.get("content") or ""

    def _parse_interactive_content(self, content: dict[str, Any]) -> str:
        """从飞书卡片消息中提取可读文本。"""
        direct_text = content.get("text") or content.get("content")
        if direct_text:
            return str(direct_text)

        card = content
        data = content.get("data")
        if isinstance(data, dict):
            card_json = data.get("card_json") or data.get("card")
            if isinstance(card_json, str):
                try:
                    card = json.loads(card_json)
                except Exception:
                    card = data
            elif isinstance(card_json, dict):
                card = card_json
            else:
                card = data

        parts: list[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                tag = str(value.get("tag") or "")
                if tag in {"plain_text", "markdown", "text", "lark_md"}:
                    text = value.get("content") or value.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                for key, child in value.items():
                    if key in {"content", "text"} and isinstance(child, str):
                        continue
                    walk(child)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(card)
        text = "\n".join(dict.fromkeys(parts)).strip()
        return text or "[卡片消息]"

    def _reply_target_message_id(self, message: dict[str, Any]) -> str:
        """提取飞书回复目标。通常 parent_id 是直接被回复消息，root_id 是线程根。"""
        for key in ("parent_id", "root_id", "reply_to", "thread_id"):
            value = message.get(key)
            if value:
                return str(value)
        mentions = message.get("mentions")
        if isinstance(mentions, list):
            for mention in mentions:
                if not isinstance(mention, dict):
                    continue
                for key in ("message_id", "id"):
                    value = mention.get(key)
                    if value:
                        return str(value)
        return ""

    async def _fetch_message_text(self, message_id: str) -> dict[str, Any] | None:
        cached = self._get_cached_message_text(message_id)
        if cached:
            return cached
        if not message_id:
            return None
        try:
            token = await self._get_access_token()
            resp = await self._http.get(
                f"{_FEISHU_API_BASE}/open-apis/im/v1/messages/{message_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code != 200:
                logger.warning(
                    "[feishu] 拉取被回复消息失败 status=%d message_id=%s body=%s",
                    resp.status_code,
                    message_id,
                    resp.text[:300],
                )
                return None
            data = resp.json()
            if data.get("code", 0) != 0:
                logger.warning(
                    "[feishu] 拉取被回复消息 API 错误 code=%s msg=%s message_id=%s",
                    data.get("code"),
                    data.get("msg", ""),
                    message_id,
                )
                return None
            item = data.get("data", {}).get("item") or data.get("data", {}).get("message") or data.get("data", {})
            if not isinstance(item, dict):
                return None
            content_raw = item.get("content", "{}")
            try:
                content = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
            except Exception:
                content = {}
            if not isinstance(content, dict):
                content = {}
            msg_type = str(item.get("message_type") or "text")
            text = self._extract_text_content(msg_type, content).strip()
            sender = self._sender_label_from_message(item)
            if text:
                self._cache_message_text(message_id, text, sender=sender, msg_type=msg_type)
                return {"text": text, "sender": sender, "msg_type": msg_type}
        except Exception as e:
            logger.warning("[feishu] 拉取被回复消息异常 message_id=%s: %s", message_id, e)
        return None

    def _sender_label_from_message(self, message: dict[str, Any]) -> str:
        sender = message.get("sender") or {}
        if not isinstance(sender, dict):
            return ""
        for key in ("sender_nickname", "name"):
            value = sender.get(key)
            if value:
                return str(value)
        sender_id = sender.get("sender_id")
        if isinstance(sender_id, dict):
            return str(sender_id.get("open_id") or sender_id.get("user_id") or "")
        return ""

    async def _build_reply_context_metadata(
        self,
        message: dict[str, Any],
    ) -> dict[str, str]:
        reply_message_id = self._reply_target_message_id(message)
        if not reply_message_id:
            return {}

        reply_data = await self._fetch_message_text(reply_message_id)
        reply_meta = {"reply_to_message_id": reply_message_id}
        if not reply_data:
            return reply_meta

        reply_text = str(reply_data.get("text") or "").strip()
        if not reply_text:
            return reply_meta
        sender_label = str(reply_data.get("sender") or "").strip() or "未知发送者"
        author_role = self._reply_author_role(sender_label)
        reply_meta["reply_to_sender"] = sender_label
        reply_meta["reply_to_msg_type"] = str(reply_data.get("msg_type") or "")
        reply_meta["reply_to_role"] = author_role
        reply_meta["reply_context_text"] = reply_text
        reply_meta["reply_context_hint"] = (
            "<quoted_message>\n"
            f"author_role: {author_role}\n"
            f"author_name: {sender_label}\n"
            f"message_id: {reply_message_id}\n"
            f"msg_type: {reply_meta['reply_to_msg_type']}\n"
            "content:\n"
            f"{reply_text}\n"
            "</quoted_message>"
        ).strip()
        return reply_meta

    @staticmethod
    def _reply_author_role(sender_label: str) -> str:
        normalized = str(sender_label or "").strip().lower()
        if normalized in {"akashic", "bot", "assistant"}:
            return "assistant"
        if not normalized or normalized == "未知发送者":
            return "unknown"
        return "user"

    def _parse_post_content(self, content: dict[str, Any]) -> str:
        """解析飞书 post 类型的富文本内容"""
        # 飞书 post content 格式: {"zh_cn": {"content": [[{tag: "md", text: "..."}], ...]}}
        parts: list[str] = []
        # 尝试多个可能的 key
        for locale_key in ("zh_cn", "en_us", "ja_jp"):
            locale_data = content.get(locale_key) or content.get(locale_key.replace("_", "-"))
            if not locale_data:
                continue
            if isinstance(locale_data, dict):
                content_list = locale_data.get("content", [])
            elif isinstance(locale_data, list):
                content_list = locale_data
            else:
                continue

            for row in content_list:
                if not isinstance(row, list):
                    continue
                for element in row:
                    if not isinstance(element, dict):
                        continue
                    tag = element.get("tag", "")
                    if tag == "text":
                        text = element.get("text", "")
                        if text:
                            parts.append(text)
                    elif tag == "at":
                        user_name = element.get("user_name", "@user")
                        parts.append(f"@{user_name}")
                    elif tag == "md":
                        text = element.get("text", "")
                        if text:
                            parts.append(text)
        return "\n".join(parts).strip() or "[富文本消息]"

    # ── 出站处理 ─────────────────────────────────────────────────────────────

    async def _on_response(self, msg: OutboundMessage) -> None:
        """处理出站消息（AgentLoop → 飞书）"""
        session_key = f"{_CHANNEL}:{msg.chat_id}"

        # 卡片流式活跃时，卡片本身就是输出，跳过单独的消息发送
        if session_key in self._card_id:
            return
        if session_key in self._card_done and self._is_passive_turn_outbound(msg):
            return

        # 检查是否有正在进行的流式输出
        if session_key in self._stream_controllers:
            controller = self._stream_controllers.pop(session_key)
            if msg.content.strip():
                await controller.set_content(msg.content)
            return

        # 非流式发送
        if not msg.content.strip():
            return

        # 从 metadata 提取回复目标和建议的消息类型
        metadata = msg.metadata or {}
        reply_to = msg.reply_to or metadata.get("reply_to")
        force_msg_type = metadata.get("msg_type")

        # 自动检测消息类型（Markdown → post）
        if force_msg_type:
            msg_type, content_payload = force_msg_type, self._build_markdown_post_payload(msg.content) if force_msg_type == "post" else json.dumps({"text": msg.content}, ensure_ascii=False)
        else:
            msg_type, content_payload = self._detect_msg_type(msg.content)

        sent_response = await self._send_with_retry(
            chat_id=msg.chat_id,
            msg_type=msg_type,
            content_payload=content_payload,
            reply_to=reply_to,
            fallback_msg_type="text" if msg_type == "post" else None,
            fallback_content_payload=(
                json.dumps({"text": msg.content}, ensure_ascii=False)
                if msg_type == "post"
                else None
            ),
        )
        sent_message_id = self._message_id_from_response(sent_response)
        if sent_message_id:
            self._cache_message_text(
                sent_message_id,
                msg.content,
                sender="Akashic",
                msg_type=msg_type,
                session_key=f"{_CHANNEL}:{msg.chat_id}",
            )

    def _is_passive_turn_outbound(self, msg: OutboundMessage) -> bool:
        metadata = msg.metadata or {}
        if metadata.get("source") == "proactive":
            return False
        return any(
            key in metadata
            for key in (
                "streamed_reply",
                "tools_used",
                "tool_chain",
                "context_retry",
            )
        )

    async def _send_with_retry(
        self,
        *,
        chat_id: str,
        msg_type: str,
        content_payload: str,
        reply_to: str | None = None,
        sent_message_id: str | None = None,
        fallback_msg_type: str | None = None,
        fallback_content_payload: str | None = None,
    ) -> Any:
        """发送消息，带重试和 reply 回退机制"""
        try:
            return await self._send_with_retry_once(
                chat_id=chat_id,
                msg_type=msg_type,
                content_payload=content_payload,
                reply_to=reply_to,
            )
        except Exception as primary_error:
            if (
                fallback_msg_type
                and fallback_content_payload
                and fallback_msg_type != msg_type
            ):
                logger.warning(
                    "[feishu] %s 消息发送失败，降级为 %s 重试: %s",
                    msg_type,
                    fallback_msg_type,
                    primary_error,
                )
                try:
                    return await self._send_with_retry_once(
                        chat_id=chat_id,
                        msg_type=fallback_msg_type,
                        content_payload=fallback_content_payload,
                        reply_to=None,
                    )
                except Exception as fallback_error:
                    raise FeishuSendError(
                        "飞书消息发送失败，降级纯文本后仍失败",
                        response={
                            "primary_error": str(primary_error),
                            "fallback_error": str(fallback_error),
                        },
                    ) from fallback_error
            raise

    async def _send_with_retry_once(
        self,
        *,
        chat_id: str,
        msg_type: str,
        content_payload: str,
        reply_to: str | None = None,
    ) -> Any:
        last_error: Exception | None = None
        effective_reply_to = reply_to

        for attempt in range(_FEISHU_SEND_ATTEMPTS):
            try:
                response = await self._send_raw_message(
                    chat_id=chat_id,
                    msg_type=msg_type,
                    content=content_payload,
                    reply_to=effective_reply_to,
                )
                # 检查 reply 失败回退
                if effective_reply_to and not self._response_succeeded(response):
                    code = getattr(response, "code", None)
                    if code in _FEISHU_REPLY_FALLBACK_CODES:
                        logger.warning(
                            "[feishu] 回复 %s 失败 (code %s)，回退到普通消息",
                            effective_reply_to, code,
                        )
                        effective_reply_to = None
                        response = await self._send_raw_message(
                            chat_id=chat_id,
                            msg_type=msg_type,
                            content=content_payload,
                            reply_to=None,
                        )
                if self._response_succeeded(response):
                    return response
                last_error = self._response_error(response)
            except Exception as exc:
                last_error = exc
            if attempt < _FEISHU_SEND_ATTEMPTS - 1:
                wait_seconds = 2 ** attempt
                logger.warning(
                    "[feishu] 发送失败 (attempt %d/%d): %s，%ds 后重试",
                    attempt + 1, _FEISHU_SEND_ATTEMPTS, last_error, wait_seconds,
                )
                await asyncio.sleep(wait_seconds)
        logger.error("[feishu] 发送消息失败: %s", last_error)
        if last_error is not None:
            raise last_error
        raise FeishuSendError("飞书消息发送失败：无响应")

    def _message_id_from_response(self, response: Any) -> str:
        data = getattr(response, "data", None)
        if not isinstance(data, dict):
            return ""
        return self._message_id_from_response_data(data)

    def _message_id_from_response_data(self, data: dict[str, Any]) -> str:
        payload = data.get("data")
        if isinstance(payload, dict):
            for key in ("message_id", "messageId", "id"):
                value = payload.get(key)
                if value:
                    return str(value)
            message = payload.get("message")
            if isinstance(message, dict):
                value = message.get("message_id") or message.get("id")
                if value:
                    return str(value)
        return str(data.get("message_id") or data.get("id") or "")

    def _response_succeeded(self, response: Any) -> bool:
        """检查飞书 API 响应是否成功"""
        if not response:
            return False
        code = getattr(response, "code", None)
        return code == 0

    def _response_error(self, response: Any) -> FeishuSendError:
        code = getattr(response, "code", None)
        data = getattr(response, "data", None)
        message = ""
        if isinstance(data, dict):
            message = str(data.get("msg") or data.get("message") or "")
        if not message:
            message = f"飞书 API 返回失败 code={code}"
        return FeishuSendError(message, code=code, response=data)

    async def _send_raw_message(
        self,
        *,
        chat_id: str,
        msg_type: str,
        content: str,
        reply_to: str | None = None,
    ) -> Any:
        """发送原始消息（不走重试）。有 reply_to 时走 Reply 端点，否则走 Create 端点。"""
        token = await self._get_access_token()

        if reply_to:
            # Reply 端点: POST /im/v1/messages/{message_id}/reply
            payload: dict[str, Any] = {
                "content": content,
                "msg_type": msg_type,
            }
            url = f"{_FEISHU_API_BASE}/open-apis/im/v1/messages/{reply_to}/reply"
            resp = await self._http.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        else:
            # Create 端点: POST /im/v1/messages
            payload = {
                "receive_id": chat_id,
                "msg_type": msg_type,
                "content": content,
            }
            resp = await self._http.post(
                f"{_FEISHU_API_BASE}/open-apis/im/v1/messages",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                params={"receive_id_type": "chat_id"},
                json=payload,
            )

        resp.raise_for_status()
        data = resp.json()
        logger.debug("[feishu] 发送消息成功 chat_id=%s msg_type=%s", chat_id, msg_type)
        # 返回响应数据供调用方判断 code
        class Response:
            def __init__(self, data):
                self.data = data
                self.code = data.get("code", 0)
        return Response(data)

    # ── 卡片流式（Card Kit v3） ────────────────────────────────────────────

    # 节流参数
    _CARD_PUSH_MIN_INTERVAL = 0.15     # 最小推送间隔 (秒)
    _CARD_PUSH_MIN_CHARS = 20          # 最小增量字符数
    _CARD_PUSH_FORCE_INTERVAL = 0.5    # 强制推送间隔 (秒)
    _CARD_THINK_PUSH_INTERVAL = 0.2   # 思考面板独立推送间隔（积累后批量推送）

    # 思考面板元素 ID（需与 _build_card_json 中的 element_id 一致）
    _THINK_ELEMENT_ID = "think_md"

    def _build_card_json(self, content: str) -> str:
        """构建 Card JSON 2.0（带 streaming_mode，选配思考折叠面板）"""
        elements: list[dict[str, Any]] = []

        # 思考折叠面板：默认折叠，流式更新内嵌 markdown
        if self._thinking_enabled:
            elements.append({
                "tag": "collapsible_panel",
                "expanded": False,
                "header": {
                    "title": {"tag": "plain_text", "content": "💭 思考过程"},
                    "vertical_align": "center",
                },
                "vertical_spacing": "4px",
                "padding": "4px 4px 4px 4px",
                "border": {"color": "grey", "corner_radius": "4px"},
                "elements": [{
                    "tag": "markdown",
                    "element_id": self._THINK_ELEMENT_ID,
                    "content": "—",
                }],
            })

        # 主回复
        elements.append({
            "tag": "markdown",
            "element_id": "markdown_1",
            "content": content,
        })

        return json.dumps({
            "schema": "2.0",
            "config": {
                "streaming_mode": True,
                "update_multi": True,
                "summary": {"content": ""},
                "streaming_config": {
                    "print_frequency_ms": {"default": 30},
                    "print_step": {"default": 1},
                    "print_strategy": "fast",
                },
            },
            "body": {
                "elements": elements,
            },
        }, ensure_ascii=False)

    async def _create_card_entity(self, content: str) -> str | None:
        """POST /cardkit/v1/cards — 创建卡片实体，返回 card_id"""
        try:
            token = await self._get_access_token()
            card_json = json.loads(self._build_card_json(content))
            resp = await self._http.post(
                f"{_FEISHU_API_BASE}/open-apis/cardkit/v1/cards",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "type": "card_json",
                    "data": json.dumps(card_json, ensure_ascii=False),
                },
            )
            if resp.status_code != 200:
                logger.warning(f"[feishu] 创建卡片失败 status={resp.status_code}: {resp.text}")
                return None
            data = resp.json()
            card_id = (data.get("data") or {}).get("card_id")
            if card_id:
                logger.debug(f"[feishu] 卡片实体创建成功 card_id={card_id}")
            else:
                logger.warning(f"[feishu] 卡片创建返回200但无card_id: {resp.text[:500]}")
            return card_id
        except Exception as e:
            logger.warning(f"[feishu] 创建卡片异常: {e}")
            return None

    async def _send_card_message(
        self,
        chat_id: str,
        card_id: str,
        reply_to: str | None = None,
        *,
        session_key: str = "",
    ) -> Any:
        """发送卡片消息。有 reply_to 时走 Reply 端点（显示回复横条），否则走 Create 端点。"""
        token = await self._get_access_token()
        content = json.dumps({"type": "card", "data": {"card_id": card_id}}, ensure_ascii=False)

        if reply_to:
            # Reply 端点: POST /im/v1/messages/{message_id}/reply
            # 不传 receive_id/receive_id_type，消息自动挂在被回复消息的会话下
            payload: dict[str, Any] = {
                "content": content,
                "msg_type": "interactive",
            }
            url = f"{_FEISHU_API_BASE}/open-apis/im/v1/messages/{reply_to}/reply"
            resp = await self._http.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        else:
            # Create 端点: POST /im/v1/messages
            payload = {
                "receive_id": chat_id,
                "msg_type": "interactive",
                "content": content,
            }
            resp = await self._http.post(
                f"{_FEISHU_API_BASE}/open-apis/im/v1/messages",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                params={"receive_id_type": "chat_id"},
                json=payload,
            )

        if resp.status_code != 200:
            logger.warning(
                "[feishu] 发送卡片消息失败 status=%d reply_to=%s body=%s",
                resp.status_code, reply_to, resp.text[:500],
            )
        else:
            data = resp.json()
            code = data.get("code", 0)
            if code != 0:
                logger.warning(
                    "[feishu] 发送卡片消息 API 错误 code=%s msg=%s reply_to=%s",
                    code, data.get("msg", ""), reply_to,
                )
            else:
                message_id = self._message_id_from_response_data(data)
                if message_id:
                    self._card_message_ids[card_id] = message_id
                    self._cache_message_text(
                        message_id,
                        self._render_card_content(session_key) if session_key else "[卡片消息]",
                        sender="Akashic",
                        msg_type="interactive",
                        session_key=session_key,
                    )
        return resp

    async def _update_element_content(self, card_id: str, element_id: str, content: str, seq: int) -> bool:
        """PUT /cardkit/v1/cards/{id}/elements/{eid}/content — 流式更新元素"""
        try:
            token = await self._get_access_token()
            resp = await self._http.put(
                f"{_FEISHU_API_BASE}/open-apis/cardkit/v1/cards/{card_id}/elements/{element_id}/content",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={"content": content, "sequence": seq},
            )
            if resp.status_code != 200:
                logger.warning(f"[feishu] 更新 {element_id} 失败 status={resp.status_code}: {resp.text}")
                return False
            return True
        except Exception as e:
            logger.warning(f"[feishu] 更新 {element_id} 异常: {e}")
            return False

    async def _update_card_content(self, card_id: str, content: str, seq: int) -> bool:
        """流式更新主回复 markdown_1"""
        return await self._update_element_content(card_id, "markdown_1", content, seq)

    async def _update_thinking_content(self, card_id: str, content: str, seq: int) -> bool:
        """流式更新思考面板 think_md"""
        return await self._update_element_content(card_id, self._THINK_ELEMENT_ID, content, seq)

    async def _disable_streaming_mode(self, card_id: str, seq: int) -> bool:
        """PATCH /cardkit/v1/cards/{id}/settings — 关闭 streaming_mode"""
        try:
            token = await self._get_access_token()
            resp = await self._http.patch(
                f"{_FEISHU_API_BASE}/open-apis/cardkit/v1/cards/{card_id}/settings",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "settings": json.dumps({"config": {"streaming_mode": False}}),
                    "sequence": seq,
                },
            )
            if resp.status_code != 200:
                logger.warning(f"[feishu] 关闭 streaming_mode 失败 status={resp.status_code}: {resp.text}")
                return False
            return True
        except Exception as e:
            logger.warning(f"[feishu] 关闭 streaming_mode 异常: {e}")
            return False

    # ── 卡片内容渲染 ─────────────────────────────────────────────────────────

    def _render_card_content(self, session_key: str, *, finalize: bool = False) -> str:
        """根据当前状态渲染卡片主 markdown 内容（工具链 + 回复）"""
        tools = self._card_tool_states.get(session_key, [])
        reply = self._card_reply_buf.get(session_key, "")

        # 工具链
        tool_lines: list[str] = []
        for t in tools:
            name = t.get("name", "unknown")
            desc = t.get("description", "")
            status = t.get("status", "running")
            label = f"{desc} - `{name}`" if desc else f"`{name}`"
            if status == "done":
                tool_lines.append(f"✅ {label} 完成")
            else:
                tool_lines.append(f"🔧 {label}...")

        tool_block = "\n".join(tool_lines)

        # finalize 模式：工具链用引用块弱化视觉，回复单独展示
        if finalize:
            if tool_block and reply:
                quoted_tools = "\n".join("> " + line for line in tool_block.split("\n"))
                return quoted_tools + "\n\n" + reply
            return reply or tool_block or "—"

        # 流式模式（完全不动）
        if tool_block and reply:
            return tool_block + "\n\n───\n\n" + reply
        if tool_block:
            return tool_block
        if reply:
            return reply
        # 未开启思考时展示占位；开启时面板已有 "💭 思考过程"，主区域用最小占位
        return "💭 思考中..." if not self._thinking_enabled else "—"

    # ── 卡片推送（带节流） ───────────────────────────────────────────────────

    # 卡片创建并发锁（防止 TurnStarted 和 StreamDeltaReady 同时创建）
    _PENDING = "__pending__"

    async def _push_card_update(
        self, session_key: str, *, finalize: bool = False, force: bool = False
    ) -> None:
        """推送卡片内容更新（带节流控制）"""
        card_id = self._card_id.get(session_key)

        # 卡片正在创建中，跳过（内容会在创建完成后通过后续推送更新）
        if card_id == self._PENDING:
            return

        if not card_id:
            # 还没有卡片 — 设置锁，防止并发重复创建
            self._card_id[session_key] = self._PENDING
            try:
                content = self._render_card_content(session_key, finalize=finalize)
                card_id = await self._create_card_entity(content)
                if not card_id:
                    self._card_id.pop(session_key, None)
                    logger.warning(f"[feishu] 卡片创建失败，流式降级 session_key={session_key}")
                    return
                self._card_id[session_key] = card_id
                self._card_seq[card_id] = 1

                # 从 _session_last_message_id 获取 reply_to
                reply_to = self._session_last_message_id.get(session_key)
                resp = await self._send_card_message(
                    session_key.split(":", 1)[1] if ":" in session_key else session_key,
                    card_id,
                    reply_to=reply_to,
                    session_key=session_key,
                )
                logger.debug(f"[feishu] 卡片消息已发送 card_id={card_id}")
                return
            except Exception:
                self._card_id.pop(session_key, None)
                raise

        # 节流检查
        now = time.time()
        last = self._card_last_push.get(session_key, 0)
        seq = self._card_seq.get(card_id, 0)

        # 思考面板：静默累积，工具开始时一次性展示（不流式推送）
        if self._thinking_enabled:
            thinking = self._card_thinking_buf.get(session_key, "")
            if thinking and force and not finalize:
                seq += 1
                await self._update_thinking_content(card_id, thinking, seq)
            elif finalize and thinking:
                seq += 1
                await self._update_thinking_content(card_id, thinking, seq)

        # 主回复：思考阶段跳过，force/finalize 时正常推送
        if not force and not finalize:
            reply = self._card_reply_buf.get(session_key, "")
            thinking_active = bool(self._card_thinking_buf.get(session_key, "")) and not reply
            if thinking_active:
                return  # 思考还在进行中，不推送主回复
            if now - last < self._CARD_PUSH_MIN_INTERVAL:
                return
            # 增量不够不推，除非超过强制间隔
            if now - last < self._CARD_PUSH_FORCE_INTERVAL:
                reply_grew = len(self._card_reply_buf.get(session_key, ""))
                last_reply_len = self._card_last_reply_len.get(session_key, 0)
                if reply_grew - last_reply_len < self._CARD_PUSH_MIN_CHARS:
                    return

        content = self._render_card_content(session_key, finalize=finalize)
        seq += 1
        success = await self._update_card_content(card_id, content, seq)
        if success:
            self._card_seq[card_id] = seq
            self._card_last_push[session_key] = now
            self._card_last_reply_len[session_key] = len(self._card_reply_buf.get(session_key, ""))

        if finalize:
            seq += 1
            await self._disable_streaming_mode(card_id, seq)

    # ── 事件处理 ─────────────────────────────────────────────────────────────

    async def _on_turn_started(self, event: TurnStarted) -> None:
        """初始化新 turn 的卡片流式状态，并立即展示思考指示器"""
        if event.channel != _CHANNEL:
            return
        session_key = event.session_key
        self._card_id.pop(session_key, None)
        self._card_seq.pop(session_key, None)
        self._card_tool_states.pop(session_key, None)
        self._card_reply_buf.pop(session_key, None)
        self._card_thinking_buf.pop(session_key, None)
        self._card_last_push.pop(session_key, None)
        self._card_last_think_push.pop(session_key, None)
        self._card_last_reply_len.pop(session_key, None)
        self._card_done.discard(session_key)

        # 立即创建卡片展示思考指示器（不依赖 reasoning 模型）
        await self._push_card_update(session_key, force=True)
        logger.debug(f"[feishu] 卡片流式状态已初始化 session_key={session_key}")

    async def _on_tool_call_started(self, event: ToolCallStarted) -> None:
        """工具调用开始 → 追加到工具链 → 推卡片"""
        if event.channel != _CHANNEL:
            return
        session_key = event.session_key
        tools = self._card_tool_states.setdefault(session_key, [])
        tools.append({
            "call_id": event.call_id,
            "name": event.tool_name,
            "description": (event.arguments or {}).get("description", ""),
            "status": "running",
        })
        logger.debug(f"[feishu] 工具开始: {event.tool_name} session_key={session_key}")
        await self._push_card_update(session_key, force=True)

    async def _on_tool_call_completed(self, event: ToolCallCompleted) -> None:
        """工具调用完成 → 更新状态 → 推卡片"""
        if event.channel != _CHANNEL:
            return
        session_key = event.session_key
        tools = self._card_tool_states.get(session_key, [])
        for t in tools:
            if t.get("call_id") == event.call_id:
                t["status"] = "done"
                t["result_preview"] = event.result_preview[:200]
                break
        logger.debug(f"[feishu] 工具完成: {event.tool_name} status={event.status} session_key={session_key}")
        await self._push_card_update(session_key, force=True)

    async def _on_stream_delta(self, event: StreamDeltaReady) -> None:
        """流式文本到达 → 累积回复/思考 → 推卡片"""
        if event.channel != _CHANNEL:
            return
        if not event.content_delta and not event.thinking_delta:
            return
        session_key = event.session_key

        # 累积回复文本
        if event.content_delta:
            current = self._card_reply_buf.get(session_key, "")
            self._card_reply_buf[session_key] = current + event.content_delta

        # 累积思考过程
        if event.thinking_delta:
            current = self._card_thinking_buf.get(session_key, "")
            self._card_thinking_buf[session_key] = current + event.thinking_delta
            logger.debug(f"[feishu] 思考到达 len={len(event.thinking_delta)} total={len(self._card_thinking_buf[session_key])}")

        # 检查增量是否够推送
        card_id = self._card_id.get(session_key)
        if card_id == self._PENDING:
            # 卡片正在创建中，只累积不推送（创建完成后由后续事件更新）
            return
        if card_id is None:
            # 还没有卡片 — 等 ToolCallStarted 或直接创建
            tools = self._card_tool_states.get(session_key, [])
            if not tools:
                # 没有工具调用，纯文本回复，直接创建卡片流式
                await self._push_card_update(session_key, force=True)
                return
            # 有工具调用在等，不急着推
            return

        # 已创建卡片，节流推送
        await self._push_card_update(session_key)

    # ── 公共发送接口 ─────────────────────────────────────────────────────────

    async def send(self, chat_id: str, content: str) -> None:
        """发送文本消息（公共接口，供外部调用）"""
        msg_type, content_payload = self._detect_msg_type(content)
        await self._send_with_retry(
            chat_id=chat_id,
            msg_type=msg_type,
            content_payload=content_payload,
            fallback_msg_type="text" if msg_type == "post" else None,
            fallback_content_payload=(
                json.dumps({"text": content}, ensure_ascii=False)
                if msg_type == "post"
                else None
            ),
        )

    async def send_image(self, chat_id: str, image_path: str) -> None:
        """发送图片（公共接口）。支持本地路径或 URL"""
        import io
        import tempfile

        # URL 图片：先下载到临时文件
        if image_path.startswith(("http://", "https://")):
            try:
                resp = await self._http.get(image_path)
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "")
                ext = self._guess_image_ext(ct)
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(resp.content)
                    local_path = tmp.name
            except Exception as e:
                logger.warning("[feishu] 下载远程图片失败: %s", e)
                raise
        else:
            local_path = image_path

        image_key = await self._upload_image(local_path)
        if not image_key:
            raise RuntimeError("图片上传失败")

        content_payload = json.dumps({"image_key": image_key}, ensure_ascii=False)
        await self._send_with_retry(
            chat_id=chat_id,
            msg_type="image",
            content_payload=content_payload,
        )

    async def send_file(
        self, chat_id: str, file_path: str, file_name: str | None = None
    ) -> None:
        """发送文件（公共接口）。支持 PDF、Word、Excel、PPT 等"""
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_key = await self._upload_file(file_path, file_name)
        if not file_key:
            raise RuntimeError("文件上传失败")

        content_payload = json.dumps({"file_key": file_key}, ensure_ascii=False)
        await self._send_with_retry(
            chat_id=chat_id,
            msg_type="file",
            content_payload=content_payload,
        )

    # ── TurnCommitted 事件处理（Reaction 反馈 + 卡片 Finalize） ──────────────

    async def _on_turn_committed(self, event: TurnCommitted) -> None:
        """处理 TurnCommitted：finalize 卡片 + 清除 reaction"""
        if event.channel != _CHANNEL:
            return

        session_key = event.session_key
        logger.debug(
            f"[feishu] _on_turn_committed session_key={session_key} "
            f"card_active={session_key in self._card_id}"
        )

        # ── 卡片 finalize ──
        card_id = self._card_id.get(session_key)
        if card_id == self._PENDING:
            # 卡片还在创建中，无法 finalize；清理锁让后续流程处理
            self._card_id.pop(session_key, None)
        elif card_id is not None:
            # 确保 reply_buffer 有最终回复
            if event.assistant_response and not self._card_reply_buf.get(session_key):
                self._card_reply_buf[session_key] = event.assistant_response
            await self._push_card_update(session_key, finalize=True)
            card_message_id = self._card_message_ids.get(card_id)
            if card_message_id:
                self._cache_message_text(
                    card_message_id,
                    self._render_card_content(session_key, finalize=True),
                    sender="Akashic",
                    msg_type="interactive",
                    session_key=session_key,
                )
            # 标记：已用卡片回复，阻止后续 _on_response 重复发送
            self._card_done.add(session_key)
            # 清理卡片状态
            self._card_id.pop(session_key, None)
            self._card_seq.pop(session_key, None)
            self._card_message_ids.pop(card_id, None)
            self._card_tool_states.pop(session_key, None)
            self._card_reply_buf.pop(session_key, None)
            self._card_thinking_buf.pop(session_key, None)
            self._card_last_push.pop(session_key, None)
            self._card_last_think_push.pop(session_key, None)
            self._card_last_reply_len.pop(session_key, None)
            logger.debug(f"[feishu] 卡片已 finalize session_key={session_key}")

        # ── Reaction 清除 ──
        message_id: str | None = None

        extra = getattr(event, "extra", None) or {}
        message_id = extra.get("feishu_reaction_message_id")

        if not message_id:
            message_id = self._session_last_message_id.get(session_key)

        if not message_id:
            return

        reaction_id = self._pending_processing_reactions.get(message_id)
        logger.debug(f"[feishu] 移除 reaction message_id={message_id} reaction_id={reaction_id}")
        if not reaction_id:
            self._session_last_message_id.pop(session_key, None)
            return

        logger.debug(f"[feishu] 处理完成，移除 reaction message_id={message_id}")
        await self._remove_reaction(message_id, reaction_id)

        self._pending_processing_reactions.pop(message_id, None)
        self._session_last_message_id.pop(session_key, None)
