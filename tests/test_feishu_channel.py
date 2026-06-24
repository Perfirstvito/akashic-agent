from __future__ import annotations

import json
import warnings
from types import SimpleNamespace

import pytest

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"lark_oapi\..*",
)

import infra.channels.feishu_channel as feishu_module
from infra.channels.feishu_channel import FeishuChannel
from bus.events import OutboundMessage
from bus.queue import MessageBus


@pytest.fixture
async def feishu_channel(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    channel = FeishuChannel(
        app_id="app",
        app_secret="secret",
        bus=MessageBus(),
    )
    try:
        yield channel
    finally:
        await channel._http.aclose()


@pytest.mark.asyncio
async def test_feishu_send_raises_after_retries(feishu_channel, monkeypatch):
    attempts = 0

    async def fake_sleep(_seconds: float) -> None:
        return None

    async def fail_send(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("network down")

    monkeypatch.setattr(feishu_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(feishu_channel, "_send_raw_message", fail_send)

    with pytest.raises(RuntimeError, match="network down"):
        await feishu_channel.send("chat_id", "plain text")

    assert attempts == feishu_module._FEISHU_SEND_ATTEMPTS


@pytest.mark.asyncio
async def test_feishu_post_send_falls_back_to_plain_text(feishu_channel, monkeypatch):
    calls: list[dict[str, str]] = []

    async def fake_sleep(_seconds: float) -> None:
        return None

    async def fake_send(**kwargs):
        calls.append(dict(kwargs))
        if kwargs["msg_type"] == "post":
            raise RuntimeError("post rejected")
        return SimpleNamespace(code=0, data={"code": 0})

    monkeypatch.setattr(feishu_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(feishu_channel, "_send_raw_message", fake_send)

    await feishu_channel.send("chat_id", "# Title\n\nbody")

    assert [call["msg_type"] for call in calls] == ["post", "post", "post", "text"]
    assert json.loads(calls[-1]["content"]) == {"text": "# Title\n\nbody"}


@pytest.mark.asyncio
async def test_feishu_card_done_does_not_block_proactive_outbound(
    feishu_channel,
    monkeypatch,
):
    calls: list[dict[str, str]] = []

    async def fake_send_with_retry(**kwargs):
        calls.append(dict(kwargs))

    monkeypatch.setattr(feishu_channel, "_send_with_retry", fake_send_with_retry)
    feishu_channel._card_done.add("feishu:chat_id")

    await feishu_channel._on_response(
        OutboundMessage(
            channel="feishu",
            chat_id="chat_id",
            content="proactive arxiv digest",
        )
    )

    assert len(calls) == 1
    assert calls[0]["chat_id"] == "chat_id"


@pytest.mark.asyncio
async def test_feishu_card_done_still_blocks_passive_duplicate(
    feishu_channel,
    monkeypatch,
):
    calls: list[dict[str, str]] = []

    async def fake_send_with_retry(**kwargs):
        calls.append(dict(kwargs))

    monkeypatch.setattr(feishu_channel, "_send_with_retry", fake_send_with_retry)
    feishu_channel._card_done.add("feishu:chat_id")

    await feishu_channel._on_response(
        OutboundMessage(
            channel="feishu",
            chat_id="chat_id",
            content="passive duplicate",
            metadata={"streamed_reply": True},
        )
    )

    assert calls == []


@pytest.mark.asyncio
async def test_feishu_reply_stores_cached_card_message_as_metadata(feishu_channel, monkeypatch):
    async def fake_add_reaction(*_args, **_kwargs):
        return None

    monkeypatch.setattr(feishu_channel, "_add_reaction", fake_add_reaction)
    feishu_channel._cache_message_text(
        "card-msg-1",
        "这是上一条卡片里的最终回复",
        sender="Akashic",
        msg_type="interactive",
    )

    await feishu_channel._handle_message(
        {
            "sender": {
                "sender_id": {"open_id": "user-1"},
                "sender_nickname": "User",
            },
            "message": {
                "chat_id": "chat-1",
                "chat_type": "p2p",
                "message_id": "msg-2",
                "message_type": "text",
                "parent_id": "card-msg-1",
                "content": json.dumps({"text": "这里再解释一下"}),
            },
        }
    )

    inbound = await feishu_channel._bus.consume_inbound()
    assert inbound.content == "这里再解释一下"
    assert inbound.metadata["reply_to_message_id"] == "card-msg-1"
    assert inbound.metadata["reply_to_msg_type"] == "interactive"
    assert inbound.metadata["reply_context_text"] == "这是上一条卡片里的最终回复"
    assert "被回复消息（来自 Akashic）" in inbound.metadata["reply_context_hint"]


@pytest.mark.asyncio
async def test_feishu_reply_uses_session_message_ref_without_short_cache(
    feishu_channel,
    monkeypatch,
):
    async def fake_add_reaction(*_args, **_kwargs):
        return None

    class RefStore:
        def get_channel_message_ref(self, **kwargs):
            assert kwargs == {
                "channel": "feishu",
                "channel_message_id": "old-card-msg",
            }
            return {
                "text": "很久以前的卡片正文",
                "sender": "Akashic",
                "msg_type": "interactive",
            }

    monkeypatch.setattr(feishu_channel, "_add_reaction", fake_add_reaction)
    feishu_channel._session_manager = RefStore()
    feishu_channel._message_text_cache.clear()

    await feishu_channel._handle_message(
        {
            "sender": {"sender_id": {"open_id": "user-1"}},
            "message": {
                "chat_id": "chat-1",
                "chat_type": "p2p",
                "message_id": "msg-3",
                "message_type": "text",
                "parent_id": "old-card-msg",
                "content": json.dumps({"text": "继续这个"}),
            },
        }
    )

    inbound = await feishu_channel._bus.consume_inbound()
    assert inbound.content == "继续这个"
    assert inbound.metadata["reply_context_text"] == "很久以前的卡片正文"
    assert inbound.metadata["reply_to_msg_type"] == "interactive"


def test_feishu_extracts_interactive_card_text(feishu_channel):
    text = feishu_channel._extract_text_content(
        "interactive",
        {
            "data": {
                "card_json": json.dumps(
                    {
                        "schema": "2.0",
                        "body": {
                            "elements": [
                                {"tag": "markdown", "content": "卡片 Markdown 正文"},
                                {
                                    "tag": "collapsible_panel",
                                    "header": {
                                        "title": {
                                            "tag": "plain_text",
                                            "content": "思考过程",
                                        }
                                    },
                                },
                            ]
                        },
                    },
                    ensure_ascii=False,
                )
            }
        },
    )

    assert "卡片 Markdown 正文" in text
    assert "思考过程" in text
