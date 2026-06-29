"""Proactive tick 日志记录。

从 proactive_turn.py 抽出的协作者：把每轮 tick 的 gate 退出、终局动作、工具步
逐条落盘到 proactive.db 的 tick log，供事后回看主动链路决策路径。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from agent.turns.result import TurnResult

if TYPE_CHECKING:
    from proactive_v2.context import AgentTickContext
    from proactive_v2.gateway import GatewayResult

logger = logging.getLogger(__name__)


def log_content_candidates(gw: "GatewayResult") -> None:
    if not gw.content_meta:
        logger.info("[proactive_v2] content candidates: 0")
        return
    lines: list[str] = []
    for index, item in enumerate(gw.content_meta, 1):
        title = str(item.get("title") or "").strip() or "(no title)"
        source = str(item.get("source") or "").strip()
        line = f"[{index}] {title}"
        if source:
            line += f" | source={source}"
        lines.append(line)
    logger.info(
        "[proactive_v2] content candidates: %d\n%s",
        len(gw.content_meta),
        "\n".join(lines),
    )


class TickLogger:
    """每轮 tick 的落盘日志记录器，与 pipeline 编排解耦。"""

    def __init__(self, *, state_store: Any, session_key: str) -> None:
        self._state_store = state_store
        self._session_key = session_key

    def record_tick_log_start(self, ctx: "AgentTickContext") -> None:
        self._state_store.record_tick_log_start(
            tick_id=ctx.tick_id,
            session_key=self._session_key,
            started_at=ctx.now_utc.isoformat(),
            gate_exit=None,
        )

    def record_tick_log_finish(
        self,
        ctx: "AgentTickContext",
        *,
        gate_exit: str | None = None,
        result: TurnResult | None = None,
        dispatch_sent: bool | None = None,
    ) -> None:
        decision = result.decision if result is not None else ctx.terminal_action
        if result is not None and result.decision == "reply" and dispatch_sent is False:
            decision = "send_failed"
        if ctx.drift_entered and result is None and decision is None:
            decision = "reply" if ctx.drift_message_sent else "skip"
        trace_extra = result.trace.extra if result is not None and result.trace is not None else {}
        skip_reason = str(trace_extra.get("skip_reason") or ctx.skip_reason or "")
        if decision == "send_failed" and not skip_reason:
            skip_reason = "send_failed"
        final_message = ""
        if result is not None and result.outbound is not None:
            final_message = str(result.outbound.content or "")
        elif ctx.final_message:
            final_message = ctx.final_message
        self._state_store.record_tick_log_finish(
            tick_id=ctx.tick_id,
            session_key=self._session_key,
            started_at=ctx.now_utc.isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
            gate_exit=gate_exit,
            terminal_action=decision,
            skip_reason=skip_reason,
            steps_taken=ctx.steps_taken,
            alert_count=len(ctx.fetched_alerts),
            content_count=len(ctx.fetched_contents),
            context_count=len(ctx.fetched_context),
            interesting_ids=sorted(ctx.interesting_item_ids),
            discarded_ids=sorted(ctx.discarded_item_ids),
            cited_ids=list(ctx.cited_item_ids),
            drift_entered=ctx.drift_entered,
            final_message=final_message,
        )

    def record_tick_step(
        self,
        ctx: "AgentTickContext",
        *,
        phase: str,
        tool_name: str,
        tool_call_id: str,
        tool_args: dict[str, Any],
        tool_result_text: str,
    ) -> None:
        self._state_store.record_tick_step_log(
            tick_id=ctx.tick_id,
            step_index=ctx.steps_taken,
            phase=phase,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_args=tool_args,
            tool_result_text=tool_result_text,
            terminal_action_after=ctx.terminal_action,
            skip_reason_after=ctx.skip_reason,
            interesting_ids_after=sorted(ctx.interesting_item_ids),
            discarded_ids_after=sorted(ctx.discarded_item_ids),
            cited_ids_after=list(ctx.cited_item_ids),
            final_message_after=ctx.final_message,
        )
