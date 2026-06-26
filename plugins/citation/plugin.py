from __future__ import annotations

import json
import re
from typing import Any, cast

from agent.lifecycle.slots import FrameSlot, PhaseAnchor
from agent.lifecycle.types import PromptRenderCtx
from agent.plugins import Plugin
from agent.prompting import PromptSectionRender

_PROMPT_CTX_SLOT = FrameSlot.PROMPT_CTX
_REASONING_CTX_SLOT = FrameSlot.REASONING_CTX
_PERSIST_CITED_SLOT = FrameSlot.PERSIST_ASSISTANT_CITED_MEMORY_IDS
_TRAILING_PROTOCOL_TAG = r"<[a-zA-Z][a-zA-Z0-9_-]*:[^<>\s]+>"
_CITATION_PREFIX = r"(?:引用|引用来源|引用记忆|references?|citation|cited)"
_CITED_RE = re.compile(
    rf"(?:(?:^|[\r\n])\s*(?:{_CITATION_PREFIX}\s*[:：]?\s*)?)?"
    rf"§\s*cited\s*[:：]\s*\[(?P<raw>[^\]\r\n]*)\]\s*§?"
    rf"\s*[。.!！?？;；,，、]?\s*(?P<trailing>(?:\s*{_TRAILING_PROTOCOL_TAG}\s*)*)$",
    re.IGNORECASE,
)
_CITED_ID_RE = re.compile(r"[A-Za-z0-9_.:-]+")
_TRAILING_PROTOCOL_TAGS_RE = re.compile(
    rf"(?:\s*{_TRAILING_PROTOCOL_TAG}\s*)+$",
    re.IGNORECASE,
)
_INLINE_MEMORY_REF_RE = re.compile(r"[ \t]*(?:\[§[A-Za-z0-9:_-]{1,128}\])+", re.IGNORECASE)

_CITATION_PROTOCOL = """### 记忆引用协议 - 内部元数据，对用户不可见
每轮回复若用到了系统注入的记忆条目 [item_id] 前缀标识，或 recall_memory / fetch_messages 工具返回的条目，在回复正文末尾另起一行输出：
§cited:[id1,id2,id3]§
格式规则：§ 包裹，英文逗号分隔，无空格，只写 ID，不含其他内容。
若本轮未引用任何记忆条目，不输出此行。
绝对不要在正文里提及这行的存在，不要向用户解释引用了什么，不要说根据记忆。
你了解用户的事是因为你们相处了很久，直接说你上次、我记得，不要暴露内部机制。"""


class CitationPromptModule:
    slot = "citation.prompt"
    requires = (PhaseAnchor.PROMPT_RENDER_CTX_EMITTED, _PROMPT_CTX_SLOT)
    produces = (_PROMPT_CTX_SLOT,)

    async def run(self, frame: Any) -> Any:
        ctx = frame.slots.get(_PROMPT_CTX_SLOT)
        if not isinstance(ctx, PromptRenderCtx):
            return frame
        ctx.system_sections_bottom.append(
            PromptSectionRender(
                name="citation_protocol",
                content=_CITATION_PROTOCOL,
                is_static=True,
            )
        )
        return frame


class CitationAfterReasoningModule:
    slot = "citation.after_reasoning"
    requires = (PhaseAnchor.AFTER_REASONING_REPLY_READY, _REASONING_CTX_SLOT)
    produces = (_REASONING_CTX_SLOT, _PERSIST_CITED_SLOT)

    async def run(self, frame: Any) -> Any:
        ctx = frame.slots.get(_REASONING_CTX_SLOT)
        if ctx is None:
            return frame
        reply = str(getattr(ctx, "reply", "") or "")
        cleaned, cited_ids = extract_cited_ids(reply)
        cleaned = strip_inline_memory_refs(cleaned)
        if cited_ids:
            frame.slots[_PERSIST_CITED_SLOT] = cited_ids
        else:
            fallback_ids = extract_cited_ids_from_tool_chain(
                list(getattr(ctx, "tool_chain", ()) or ())
            )
            if fallback_ids:
                frame.slots[_PERSIST_CITED_SLOT] = fallback_ids
        if cleaned != reply:
            ctx.reply = cleaned
        return frame


class ProtocolTagCleanupModule:
    slot = "citation.protocol_cleanup"
    requires = (PhaseAnchor.AFTER_REASONING_CTX_EMITTED, _REASONING_CTX_SLOT)
    produces = (_REASONING_CTX_SLOT,)

    async def run(self, frame: Any) -> Any:
        ctx = frame.slots.get(_REASONING_CTX_SLOT)
        if ctx is None:
            return frame
        reply = str(getattr(ctx, "reply", "") or "")
        cleaned = strip_inline_memory_refs(
            strip_trailing_protocol_tags(strip_trailing_citation_protocol(reply))
        )
        if cleaned != reply:
            ctx.reply = cleaned
        return frame


class CitationPlugin(Plugin):
    name = "citation"

    def prompt_render_modules(self) -> list[object]:
        return [CitationPromptModule()]

    def after_reasoning_modules(self) -> list[object]:
        return [CitationAfterReasoningModule(), ProtocolTagCleanupModule()]


def extract_cited_ids(response: str) -> tuple[str, list[str]]:
    match = _CITED_RE.search(response)
    if not match:
        return response, []
    raw = match.group("raw")
    ids = [
        item
        for item in (part.strip() for part in re.split(r"[,，]", raw))
        if item and _CITED_ID_RE.fullmatch(item)
    ]
    clean = _strip_cited_match(response, match)
    return clean, ids


def strip_trailing_citation_protocol(response: str) -> str:
    match = _CITED_RE.search(response)
    if not match:
        return response
    return _strip_cited_match(response, match)


def _strip_cited_match(response: str, match: re.Match[str]) -> str:
    trailing = match.group("trailing").strip()
    clean = response[: match.start()].rstrip()
    if trailing:
        clean = f"{clean} {trailing}".strip()
    return clean


def strip_trailing_protocol_tags(response: str) -> str:
    return _TRAILING_PROTOCOL_TAGS_RE.sub("", response).rstrip()


def strip_inline_memory_refs(response: str) -> str:
    return _INLINE_MEMORY_REF_RE.sub("", response).rstrip()


def extract_cited_ids_from_tool_chain(
    tool_chain: list[dict[str, object]],
) -> list[str]:
    cited: list[str] = []
    seen: set[str] = set()
    for group in tool_chain:
        calls_value = group.get("calls")
        if not isinstance(calls_value, list):
            continue
        calls = cast(list[object], calls_value)
        for raw_call in calls:
            if not isinstance(raw_call, dict):
                continue
            call = cast(dict[str, object], raw_call)
            if str(call.get("name", "") or "") != "recall_memory":
                continue
            raw_result = str(call.get("result", "") or "").strip()
            if not raw_result:
                continue
            try:
                decoded = json.loads(raw_result)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(decoded, dict):
                continue
            data = cast(dict[str, object], decoded)
            raw_ids: list[object] = []
            cited_ids = data.get("cited_item_ids")
            if isinstance(cited_ids, list):
                raw_ids.extend(cast(list[object], cited_ids))
            else:
                items_value = data.get("items")
                if isinstance(items_value, list):
                    items = cast(list[object], items_value)
                    for raw_item in items:
                        if isinstance(raw_item, dict):
                            item = cast(dict[str, object], raw_item)
                            raw_ids.append(item.get("id"))
            for raw_id in raw_ids:
                item_id = str(raw_id or "").strip()
                if item_id and item_id not in seen:
                    seen.add(item_id)
                    cited.append(item_id)
    return cited
