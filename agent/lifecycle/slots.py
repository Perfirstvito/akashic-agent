from __future__ import annotations


class PhaseAnchor:
    """Stable lifecycle anchors exposed to plugin phase modules."""

    BEFORE_TURN_SESSION_READY = "before_turn.acquire_session"
    BEFORE_TURN_CTX_EMITTED = "before_turn.emit"
    PROMPT_RENDER_CTX_EMITTED = "prompt_render.emit"
    AFTER_STEP_INPUT_READY = "after_step.copy_input"
    AFTER_REASONING_REPLY_READY = "after_reasoning.build_ctx"
    AFTER_REASONING_CTX_EMITTED = "after_reasoning.emit"


class FrameSlot:
    """Stable keys for PhaseFrame.slots data shared with plugin modules."""

    SESSION = "session:session"
    BEFORE_TURN_CTX = "session:ctx"
    PROMPT_CTX = "prompt:ctx"
    REASONING_CTX = "reasoning:ctx"
    STEP_CTX = "step:ctx"
    STEP_EARLY_STOP_REASON = "step:early_stop_reason"
    PERSIST_ASSISTANT_CITED_MEMORY_IDS = "persist:assistant:cited_memory_ids"


class FrameSlotPrefix:
    """Stable prefixes for plugin-exported PhaseFrame.slots values."""

    STEP_TELEMETRY = "step:telemetry:"


__all__ = ["FrameSlot", "FrameSlotPrefix", "PhaseAnchor"]
