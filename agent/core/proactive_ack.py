"""Proactive ACK 副作用。

从 proactive_turn.py 抽出的纯函数组：在 resolve/deliver 各分支对 cited /
discarded / 未命中的候选条目回写 ACK TTL，控制 MCP 源端不再重复推送同一条内容。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from proactive_v2.contracts import build_compound_key

if TYPE_CHECKING:
    from proactive_v2.context import AgentTickContext

# ── ACK TTL 常量（小时） ───────────────────────────────────────────────────
_CITED_ACK_TTL = 168
_UNCITED_ACK_TTL = 24
_POST_GUARD_ACK_TTL = 24
_DISCARDED_ACK_TTL = 720


async def ack_discarded(ctx: "AgentTickContext", ack_fn) -> None:
    if ack_fn is None:
        return
    for key in ctx.discarded_item_ids:
        await ack_fn(key, _DISCARDED_ACK_TTL)


async def ack_post_guard_fail(ctx: "AgentTickContext", ack_fn, *, alert_ack_fn=None) -> None:
    if ack_fn is None:
        return
    fetched_alert_keys = {
        build_compound_key(e["ack_server"], e.get("event_id") or e.get("id", ""))
        for e in ctx.fetched_alerts
    }
    cited_set = set(ctx.cited_item_ids)

    async def _ack_alert(key: str) -> None:
        if alert_ack_fn is not None:
            await alert_ack_fn(key)
        else:
            await ack_fn(key, _POST_GUARD_ACK_TTL)

    for key in cited_set - fetched_alert_keys:
        await ack_fn(key, _POST_GUARD_ACK_TTL)
    for key in cited_set & fetched_alert_keys:
        await _ack_alert(key)
    for key in fetched_alert_keys - cited_set:
        await _ack_alert(key)
    for key in (ctx.interesting_item_ids - cited_set) - fetched_alert_keys:
        await ack_fn(key, _POST_GUARD_ACK_TTL)
    for key in ctx.discarded_item_ids:
        await ack_fn(key, _DISCARDED_ACK_TTL)


async def ack_on_success(ctx: "AgentTickContext", ack_fn, *, alert_ack_fn=None) -> None:
    if ack_fn is None:
        return
    fetched_alert_keys = {
        build_compound_key(e["ack_server"], e.get("event_id") or e.get("id", ""))
        for e in ctx.fetched_alerts
    }
    fetched_content_keys = {
        build_compound_key(e["ack_server"], e.get("event_id") or e.get("id", ""))
        for e in ctx.fetched_contents
    }
    cited_set = set(ctx.cited_item_ids)
    for key in cited_set & fetched_content_keys:
        await ack_fn(key, _CITED_ACK_TTL)
    for key in cited_set & fetched_alert_keys:
        if alert_ack_fn is not None:
            await alert_ack_fn(key)
        else:
            await ack_fn(key, _CITED_ACK_TTL)
    for key in (ctx.interesting_item_ids - cited_set) - fetched_alert_keys:
        await ack_fn(key, _UNCITED_ACK_TTL)
    for key in ctx.discarded_item_ids:
        await ack_fn(key, _DISCARDED_ACK_TTL)


async def _mark_delivery(*, state_store: Any, session_key: str, delivery_key: str) -> None:
    state_store.mark_delivery(session_key, delivery_key)


async def _mark_context_only_send(
    *,
    state_store: Any,
    session_key: str,
    context_as_fallback_open: bool,
    has_cited: bool,
) -> None:
    if context_as_fallback_open and not has_cited:
        state_store.mark_context_only_send(session_key)
