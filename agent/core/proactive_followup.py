from __future__ import annotations

import re
from typing import Any


_FOLLOWUP_TRIGGERS = (
    "刚才",
    "上一条",
    "上条",
    "你推",
    "推送",
    "主动",
    "那篇",
    "这篇",
    "那条",
    "这条",
    "下面",
    "下边",
    "上面",
    "上边",
    "第二",
    "第一",
    "第2",
    "第1",
    "next",
    "previous",
    "paper",
    "arxiv",
)
_BATCH_TERMS = ("这些", "这批", "这几", "几篇", "全部", "都", "all")
_ORDINAL_PATTERNS = (
    (("第三", "第3", "3篇", "third"), 2),
    (("第二", "第2", "2篇", "下面", "下边", "后一", "下一", "second", "next"), 1),
    (("第一", "第1", "1篇", "上面", "上边", "前一", "上一", "first", "previous"), 0),
)


def resolve_proactive_followup(
    content: str,
    messages: list[dict[str, Any]],
    *,
    lookback: int = 20,
) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if not text:
        return None
    source_message = _find_recent_proactive_message(messages, lookback=lookback)
    if source_message is None:
        return None
    source_refs = _clean_source_refs(source_message.get("source_refs"))
    if not source_refs:
        return None

    matched_refs = _match_source_refs(text, source_refs)
    status = "resolved"
    if not matched_refs:
        index = _ordinal_index(text)
        if index is not None and 0 <= index < len(source_refs):
            matched_refs = [source_refs[index]]
        elif len(source_refs) == 1 and _has_followup_trigger(text):
            matched_refs = [source_refs[0]]
        elif _has_batch_reference(text):
            matched_refs = source_refs
        elif _has_followup_trigger(text):
            matched_refs = source_refs
            status = "ambiguous"

    if not matched_refs:
        return None

    return {
        "proactive_followup": True,
        "proactive_followup_status": status,
        "resolved_from_proactive_message_id": str(source_message.get("id") or ""),
        "resolved_proactive_refs": matched_refs,
        "resolved_proactive_hint": render_proactive_followup_hint(
            source_message=source_message,
            refs=matched_refs,
            status=status,
        ),
    }


def render_proactive_followup_hint(
    *,
    source_message: dict[str, Any],
    refs: list[dict[str, Any]],
    status: str,
) -> str:
    lines = [
        "## resolved_proactive_followup",
        "用户当前消息引用了之前的主动推送。以下来源只用于解析用户指代，不是用户新陈述。",
        f"status={status}",
    ]
    source_id = str(source_message.get("id") or "").strip()
    if source_id:
        lines.append(f"source_message_id={source_id}")
    for index, ref in enumerate(refs, start=1):
        parts = [
            f"[{index}]",
            _field(ref, "id"),
            _field(ref, "source_name") or _field(ref, "source"),
            _field(ref, "title"),
            _field(ref, "url"),
        ]
        lines.append("- " + " | ".join(part for part in parts if part))
    return "\n".join(lines)


def format_proactive_followup_for_conversation(message: dict[str, Any]) -> list[str]:
    if not message.get("proactive_followup"):
        return []
    refs = _clean_source_refs(message.get("resolved_proactive_refs"))
    if not refs:
        return []
    status = str(message.get("proactive_followup_status") or "resolved").strip()
    lines = [f"[proactive_followup status={status}]"]
    source_id = str(message.get("resolved_from_proactive_message_id") or "").strip()
    if source_id:
        lines.append(f"source_message_id={source_id}")
    for index, ref in enumerate(refs, start=1):
        parts = [
            f"[{index}]",
            _field(ref, "id"),
            _field(ref, "source_name") or _field(ref, "source"),
            _field(ref, "title"),
            _field(ref, "url"),
        ]
        lines.append("- " + " | ".join(part for part in parts if part))
    return lines


def _find_recent_proactive_message(
    messages: list[dict[str, Any]],
    *,
    lookback: int,
) -> dict[str, Any] | None:
    tail = messages[-max(1, int(lookback)) :]
    for message in reversed(tail):
        if (
            message.get("role") == "assistant"
            and message.get("proactive")
            and isinstance(message.get("source_refs"), list)
        ):
            return message
    return None


def _clean_source_refs(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        ref = {
            key: str(raw).strip()
            for key, raw in item.items()
            if raw is not None and str(raw).strip()
        }
        if not ref:
            continue
        key = str(ref.get("id") or ref.get("url") or ref.get("title") or len(refs))
        if key in seen:
            continue
        seen.add(key)
        refs.append(ref)
    return refs


def _match_source_refs(text: str, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_text = _normalize(text)
    matched: list[dict[str, Any]] = []
    for ref in refs:
        candidates = [
            _field(ref, "id"),
            _field(ref, "event_id"),
            _field(ref, "title"),
            _field(ref, "url"),
        ]
        arxiv_id = _extract_arxiv_id(_field(ref, "url"))
        if arxiv_id:
            candidates.append(arxiv_id)
        for candidate in candidates:
            normalized_candidate = _normalize(candidate)
            if not normalized_candidate:
                continue
            if len(normalized_candidate) >= 4 and normalized_candidate in normalized_text:
                matched.append(ref)
                break
    return matched


def _has_followup_trigger(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in _FOLLOWUP_TRIGGERS)


def _has_batch_reference(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in _BATCH_TERMS)


def _ordinal_index(text: str) -> int | None:
    lowered = text.lower()
    match = re.search(r"第\s*([1-9])\s*(?:篇|个|条|则|项)?", lowered)
    if match:
        return int(match.group(1)) - 1
    for tokens, index in _ORDINAL_PATTERNS:
        if any(token in lowered for token in tokens):
            return index
    return None


def _extract_arxiv_id(url: str) -> str:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#/]+)", url, flags=re.I)
    return match.group(1).removesuffix(".pdf") if match else ""


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


def _field(ref: dict[str, Any], key: str) -> str:
    return str(ref.get(key) or "").strip()
