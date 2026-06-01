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
