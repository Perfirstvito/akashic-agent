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
from bus.events_lifecycle import StreamDeltaReady, TurnCommitted
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
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._bus = bus
        self._allow_from = set(allow_from) if allow_from else set()
        self._interrupt_controller = interrupt_controller

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

        # ── Reaction 反馈 ────────────────────────────────────────────────
        # message_id → reaction_id（用于完成后清除/替换 reaction）
        self._pending_processing_reactions: "OrderedDict[str, str]" = OrderedDict()

        # ── Sender 名字缓存 ──────────────────────────────────────────────
        self._sender_name_cache: dict[str, tuple[str, float]] = {}  # sender_id → (name, expires_at)

        # ── Session → Message ID 追踪（用于 Reaction） ──────────────────
        # session_key → message_id（追踪每个 turn 的原始消息，用于 reaction 更新）
        self._session_last_message_id: dict[str, str] = {}

        # 订阅事件总线
        if event_bus is not None:
            event_bus.on(StreamDeltaReady, self._on_stream_delta)
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
                logger.info(f"[feishu] 添加 reaction 成功，返回体: {data}")
                # reaction_id 直接在 data 下，不是 data.reaction.reaction_id
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
            logger.debug(f"[feishu] 移除 reaction 响应: status={resp.status_code} body={resp.text}")
            logger.info(f"[feishu] 移除 reaction 响应体: {resp.text}")
            return resp.status_code == 200
        except Exception as e:
            logger.warning("[feishu] 删除 reaction 失败: %s", e)
            return False

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
        msg_type = message.get("msg_type", "text")

        # 解析 content（JSON 字符串）
        try:
            content = json.loads(content_raw)
        except Exception:
            content = {}

        # 获取文本内容（支持 post 等富文本消息）
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

        # 如果是图片/文件/音频类型，收集 media 信息
        media: list[str] = []
        if msg_type == "image":
            image_key = content.get("image_key") or ""
            if image_key:
                media.append(f"image:{image_key}")

        # 追踪 session → message_id（用于 reaction 反馈）
        session_key = f"{_CHANNEL}:{chat_id}"
        self._session_last_message_id[session_key] = message_id
        logger.warning(f"[feishu] ====== 收到消息 ====== session_key={session_key} message_id={message_id}")

        # 添加 "SMILE" reaction 表示正在处理
        if message_id:
            reaction_id = await self._add_reaction(message_id, _REACTION_IN_PROGRESS)
            if reaction_id:
                self._cache_reaction(message_id, reaction_id)
                logger.warning(f"[feishu] ====== 添加 SMILE reaction 成功 ====== message_id={message_id} reaction_id={reaction_id}")
            else:
                logger.warning(f"[feishu] ====== 添加 SMILE reaction 失败 ====== message_id={message_id}")

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
                },
            )
        )

    def _extract_text_content(self, msg_type: str, content: dict[str, Any]) -> str:
        """根据消息类型提取文本内容"""
        if msg_type == "text":
            return str(content.get("text") or "")
        if msg_type == "post":
            return self._parse_post_content(content)
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

        await self._send_with_retry(
            chat_id=msg.chat_id,
            msg_type=msg_type,
            content_payload=content_payload,
            reply_to=reply_to,
        )

    async def _send_with_retry(
        self,
        *,
        chat_id: str,
        msg_type: str,
        content_payload: str,
        reply_to: str | None = None,
        sent_message_id: str | None = None,
    ) -> None:
        """发送消息，带重试和 reply 回退机制"""
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
                return
            except Exception as exc:
                last_error = exc
                if attempt < _FEISHU_SEND_ATTEMPTS - 1:
                    wait_seconds = 2 ** attempt
                    logger.warning(
                        "[feishu] 发送失败 (attempt %d/%d): %s，%ds 后重试",
                        attempt + 1, _FEISHU_SEND_ATTEMPTS, exc, wait_seconds,
                    )
                    await asyncio.sleep(wait_seconds)
        logger.error("[feishu] 发送消息失败: %s", last_error)

    def _response_succeeded(self, response: Any) -> bool:
        """检查飞书 API 响应是否成功"""
        if not response:
            return False
        code = getattr(response, "code", None)
        return code == 0

    async def _send_raw_message(
        self,
        *,
        chat_id: str,
        msg_type: str,
        content: str,
        reply_to: str | None = None,
    ) -> Any:
        """发送原始消息（不走重试）"""
        token = await self._get_access_token()
        payload: dict[str, Any] = {
            "receive_id": chat_id,
            "msg_type": msg_type,
            "content": content,
        }
        params: dict[str, Any] = {"receive_id_type": "chat_id"}

        if reply_to:
            payload["root_id"] = reply_to
            params["reply_to_message_id"] = reply_to

        resp = await self._http.post(
            f"{_FEISHU_API_BASE}/open-apis/im/v1/messages",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info("[feishu] 发送消息成功 chat_id=%s msg_type=%s", chat_id, msg_type)
        # 返回响应数据供调用方判断 code
        class Response:
            def __init__(self, data):
                self.data = data
                self.code = data.get("code", 0)
        return Response(data)

    # ── 流式处理 ──────────────────────────────────────────────────────────────

    async def _on_stream_delta(self, event: StreamDeltaReady) -> None:
        """处理流式输出事件"""
        if event.channel != _CHANNEL:
            return
        # 流式功能暂未实现
        pass

    # ── 公共发送接口 ─────────────────────────────────────────────────────────

    async def send(self, chat_id: str, content: str) -> None:
        """发送文本消息（公共接口，供外部调用）"""
        msg_type, content_payload = self._detect_msg_type(content)
        await self._send_with_retry(
            chat_id=chat_id,
            msg_type=msg_type,
            content_payload=content_payload,
        )

    # ── TurnCommitted 事件处理（Reaction 反馈） ───────────────────────────────

    async def _on_turn_committed(self, event: TurnCommitted) -> None:
        """处理 TurnCommitted 事件，更新 reaction 状态"""
        logger.warning(
            f"[feishu] ====== _on_turn_committed ====== channel={event.channel} "
            f"session_key={event.session_key} "
            f"_session_last_message_id={dict(self._session_last_message_id)}"
        )
        if event.channel != _CHANNEL:
            return

        session_key = event.session_key
        message_id: str | None = None

        # 优先从 extra 中获取飞书 message_id（由 handle_message 通过 extra slot 传入）
        extra = getattr(event, "extra", None) or {}
        message_id = extra.get("feishu_reaction_message_id")

        # 回退：从 _session_last_message_id 查找
        if not message_id and session_key in self._session_last_message_id:
            message_id = self._session_last_message_id.pop(session_key, None)

        if not message_id:
            return

        reaction_id = self._pop_reaction(message_id)
        logger.warning(f"[feishu] ====== _on_turn_committed 尝试移除 ====== message_id={message_id} reaction_id={reaction_id}")
        if not reaction_id:
            return

        # 检查成功/失败：通过 tool_call_groups 中的 call status 判断
        has_failure = any(
            call.status == "error"
            for tg in (event.tool_call_groups or [])
            for call in (tg.calls or [])
        )

        if has_failure:
            # 失败：移除 SMILE，添加 SAD
            logger.debug(f"[feishu] 处理失败，替换 reaction 为 SAD message_id={message_id}")
            await self._remove_reaction(message_id, reaction_id)
            await self._add_reaction(message_id, _REACTION_FAILURE)
        else:
            # 成功：移除 reaction
            logger.debug(f"[feishu] 处理完成，移除 reaction message_id={message_id}")
            await self._remove_reaction(message_id, reaction_id)