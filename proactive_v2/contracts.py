from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from zoneinfo import ZoneInfo
from typing import Any

MAX_METRICS_KEYS = 8
MAX_METRICS_VALUE_STR_LEN = 60
_TIME_KEY_SUFFIXES = ("_at", "_time", "_ts")
_TIME_KEYS = {"last_seen", "updated_at", "published_at", "timestamp", "ts"}

# ── source key 契约 ────────────────────────────────────────────────────────
#
# proactive 链路用两种命名空间 key 标识一条候选内容，二者形状不同、不可混用：
#
# 1. compound_key = "{ack_server}:{event_id}"
#    贯穿 gateway → tools 校验 → delivery 去重 → ack 回执，并硬编码在 system
#    prompt（evidence 格式）与工具 schema description 中供 LLM 产出。格式不可变。
#
# 2. source_key = "mcp:{ack_server}"
#    落盘于 proactive.db 的 seen_items / rejection_cooldown / semantic_items 三表
#    source_key 列，是持久化契约。格式不可变，否则现有去重状态失效。
MCP_SOURCE_PREFIX = "mcp:"


def build_compound_key(ack_server: str, event_id: str) -> str:
    """构造 compound_key "{ack_server}:{event_id}"，幂等。

    若 event_id 已含 ':' 且其前缀等于 ack_server，则不再重复加前缀，
    避免 "feed:feed:x" 这类双前缀。event_id 为空时返回空串。
    """
    raw_id = str(event_id or "").strip()
    ack_server = str(ack_server or "").strip()
    if not raw_id:
        return ""
    if ":" in raw_id:
        prefix = raw_id.partition(":")[0]
        if not ack_server or prefix == ack_server:
            return raw_id
    if ack_server:
        return f"{ack_server}:{raw_id}"
    return raw_id


def build_source_key(ack_server: str) -> str:
    """构造去重库 source_key "mcp:{ack_server}"。"""
    return f"{MCP_SOURCE_PREFIX}{str(ack_server or '').strip()}"


def normalize_source_key(source_key: str) -> str:
    """归一 source_key：剥离 item_id 部分，只保留 "mcp:{ack_server}"。

    "mcp:feed-mcp:c1" → "mcp:feed-mcp"；"mcp:feed-mcp" → "mcp:feed-mcp"；
    非 "mcp:" 前缀的值原样返回（向后兼容历史数据）。
    """
    raw = str(source_key or "").strip()
    if not raw.startswith(MCP_SOURCE_PREFIX):
        return raw
    parts = raw.split(":", 2)
    if len(parts) < 2:
        return raw
    return ":".join(parts[:2])


def _trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _normalize_metrics(metrics: Any) -> dict[str, Any] | None:
    if not isinstance(metrics, dict) or not metrics:
        return None

    normalized: dict[str, Any] = {}
    items = list(metrics.items())
    for key, value in items[:MAX_METRICS_KEYS]:
        key_text = str(key).strip()
        if not key_text:
            continue
        if isinstance(value, str):
            normalized[key_text] = _trim_text(value, MAX_METRICS_VALUE_STR_LEN)
            continue
        if isinstance(value, (int, float, bool)) or value is None:
            normalized[key_text] = value
            continue

        text = json.dumps(value, ensure_ascii=False)
        normalized[key_text] = _trim_text(text, MAX_METRICS_VALUE_STR_LEN)

    truncated = len(items) - MAX_METRICS_KEYS
    if truncated > 0:
        normalized["_truncated_keys"] = truncated

    return normalized or None


def _looks_like_time_key(key: str) -> bool:
    return key in _TIME_KEYS or key.endswith(_TIME_KEY_SUFFIXES)


def _resolve_tz(value: str | tzinfo | None) -> tzinfo | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return ZoneInfo(text)
        except Exception:
            return None
    return value


def _format_local_time(raw: str, local_tz: str | tzinfo | None = None) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return None
    tz = _resolve_tz(local_tz)
    local_dt = dt.astimezone(tz) if tz is not None else dt.astimezone()
    return local_dt.strftime("%Y-%m-%d %H:%M:%S %z")


def _annotate_local_times(value: Any, local_tz: str | tzinfo | None = None) -> Any:
    if isinstance(value, dict):
        annotated: dict[str, Any] = {}
        for key, item in value.items():
            annotated[key] = _annotate_local_times(item, local_tz)
            if isinstance(item, str) and _looks_like_time_key(str(key)):
                local_text = _format_local_time(item, local_tz)
                if local_text:
                    annotated[f"{key}_local"] = local_text
        return annotated
    if isinstance(value, list):
        return [_annotate_local_times(item, local_tz) for item in value]
    return value


@dataclass(slots=True)
class AlertContract:
    item_id: str
    title: str
    content: str
    severity: str
    suggested_tone: str
    metrics: dict[str, Any] | None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_prompt_line(self, index: int) -> str:
        severity_part = f"  severity={self.severity}" if self.severity else ""
        line = f"  [{index}] id={self.item_id}{severity_part}\n       title={self.title}"
        if self.content:
            line += f"\n       内容：{self.content}"
        if self.metrics:
            line += f"\n       metrics：{json.dumps(self.metrics, ensure_ascii=False)}"
        if self.suggested_tone:
            line += f"\n       建议语气：{self.suggested_tone}"
        return line


def normalize_alert(event: dict[str, Any]) -> AlertContract:
    ack_server = str(event.get("ack_server") or "?").strip() or "?"
    event_id = str(event.get("event_id") or event.get("id") or "?").strip() or "?"
    title = str(event.get("title") or "").strip()
    content = str(event.get("content") or event.get("body") or "").strip()
    severity = str(event.get("severity") or "").strip()
    tone = str(event.get("suggested_tone") or "").strip()
    return AlertContract(
        item_id=build_compound_key(ack_server, event_id),
        title=title,
        content=content,
        severity=severity,
        suggested_tone=tone,
        metrics=_normalize_metrics(event.get("metrics")),
        raw=event,
    )


@dataclass(slots=True)
class ContentContract:
    item_id: str
    title: str
    source: str
    url: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_valid_url(self) -> bool:
        return bool(self.url)

    def to_prompt_line(self, index: int, has_content: bool) -> str:
        status = "✓" if has_content else "✗(预取失败)"
        url_part = f"\n       url={self.url}" if self.has_valid_url else ""
        return (
            f"  [{index}] id={self.item_id}\n"
            f"       title={self.title}\n"
            f"       source={self.source}  正文:{status}"
            f"{url_part}"
        )


def normalize_content(item: dict[str, Any]) -> ContentContract:
    return ContentContract(
        item_id=str(item.get("id") or "").strip(),
        title=str(item.get("title") or "").strip(),
        source=str(item.get("source") or item.get("source_name") or "").strip(),
        url=str(item.get("url") or "").strip(),
        raw=item,
    )


@dataclass(slots=True)
class ContextContract:
    available: bool | None
    source: str
    raw: dict[str, Any] = field(default_factory=dict)
    local_tz: str | tzinfo | None = None

    def to_prompt_item(self) -> dict[str, Any]:
        payload = _annotate_local_times(dict(self.raw), self.local_tz)
        if self.available is not None:
            payload["available"] = self.available
        if self.source:
            payload["_source"] = self.source

        if "sleep_prob" in payload and payload["sleep_prob"] is not None:
            payload["awake_prob"] = round(1.0 - float(payload["sleep_prob"]), 3)
        return payload


def normalize_context(
    item: dict[str, Any],
    *,
    local_tz: str | tzinfo | None = None,
) -> ContextContract:
    source = str(item.get("_source") or "").strip()
    available_raw = item.get("available")
    available = None if available_raw is None else bool(available_raw)
    return ContextContract(
        available=available,
        source=source,
        raw=item,
        local_tz=local_tz,
    )
