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
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, cast

import httpx
import websockets

from bus.event_bus import EventBus
from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import StreamDeltaReady
from bus.queue import MessageBus
from infra.channels.base import MessageDeduper
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

logger = logging.getLogger(__name__)
_CHANNEL = "feishu"
_SEEN_MSG_MAXSIZE = 500  # 滑动窗口大小，防止内存无限增长

# 飞书 API 配置
_FEISHU_API_BASE = "https://open.feishu.cn"
_TOKEN_URL = f"{_FEISHU_API_BASE}/open-apis/auth/v3/tenant_access_token/internal"
_WS_GATEWAY_URL = f"{_FEISHU_API_BASE}/callback/ws/endpoint"


@dataclass
class _TokenCache:
    token: str
    expires_at: float


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

        # 消息去重
        self._message_deduper = MessageDeduper(_SEEN_MSG_MAXSIZE)

        # 订阅事件总线
        if event_bus is not None:
            event_bus.on(StreamDeltaReady, self._on_stream_delta)

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

        # 获取文本内容
        if msg_type == "text":
            text = str(content.get("text") or "")
        elif msg_type == "image":
            text = "[图片]"
        else:
            text = content.get("text") or content.get("content") or ""

        if not text and msg_type == "text":
            return

        # 去重
        if message_id and self._message_deduper.seen(message_id):
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

        logger.info(
            f"[feishu] 收到消息 chat_id={chat_id} "
            f"sender_id={sender_id} content={text[:50]!r}..."
        )

        await self._bus.publish_inbound(
            InboundMessage(
                channel=_CHANNEL,
                sender=sender_id,
                chat_id=chat_id,
                content=text,
                media=[],
                metadata={
                    "sender_name": sender.get("sender_nickname") or "",
                    "message_id": message_id,
                    "msg_type": msg_type,
                },
            )
        )

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
        if msg.content.strip():
            await self.send(msg.chat_id, msg.content)

    async def send(self, chat_id: str, content: str) -> None:
        """发送文本消息"""
        try:
            token = await self._get_access_token()
            resp = await self._http.post(
                f"{_FEISHU_API_BASE}/open-apis/im/v1/messages",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "receive_id": chat_id,
                    "msg_type": "text",
                    "content": json.dumps({"text": content}),
                },
                params={"receive_id_type": "chat_id"},
            )
            resp.raise_for_status()
            logger.info(f"[feishu] 发送消息 chat_id={chat_id}")
        except Exception as e:
            logger.error(f"[feishu] 发送失败: {e}")

    # ── 流式处理 ──────────────────────────────────────────────────────────────

    async def _on_stream_delta(self, event: StreamDeltaReady) -> None:
        """处理流式输出事件"""
        if event.channel != _CHANNEL:
            return
        # 流式功能暂未实现
        pass