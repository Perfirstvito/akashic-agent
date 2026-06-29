"""Proactive delivery key / source ref 构建。

从 proactive_turn.py 抽出的纯函数组：把本轮 cited 内容归一为 delivery 去重
hash 与结构化 source_refs（供 followup 解析“第二篇”类指代）。

compound_key 格式 "{ack_server}:{event_id}" 是 LLM 契约（硬编码在 system prompt
与工具 schema description 中），由 proactive_v2.contracts.build_compound_key 统一构造。
"""

from __future__ import annotations

import json
from hashlib import sha1
from typing import TYPE_CHECKING, Any

from proactive_v2.contracts import build_compound_key

if TYPE_CHECKING:
    from proactive_v2.context import AgentTickContext


def _normalize_delivery_url(raw: str) -> str:
    from urllib.parse import urlsplit, urlunsplit
    text = str(raw or "").strip()
    if not text:
        return ""
    parts = urlsplit(text)
    path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _build_delivery_refs(ctx: "AgentTickContext") -> list[str]:
    if not ctx.cited_item_ids:
        return []
    content_map = {
        build_compound_key(e.get("ack_server", ""), e.get("event_id") or e.get("id", "")): e
        for e in ctx.fetched_contents
        if e.get("ack_server") and (e.get("event_id") or e.get("id"))
    }
    refs: list[str] = []
    for key in sorted(set(ctx.cited_item_ids)):
        meta = content_map.get(key)
        if meta is None:
            refs.append(f"id:{key}")
            continue
        url = _normalize_delivery_url(str(meta.get("url") or ""))
        if url:
            refs.append(f"url:{url}")
            continue
        source = str(meta.get("source") or meta.get("source_name") or "").strip().lower()
        title = str(meta.get("title") or "").strip().lower()
        if title:
            refs.append(f"title:{source}|{title}")
            continue
        refs.append(f"id:{key}")
    return sorted(set(refs))


def _compound_source_key(item: dict[str, Any]) -> str:
    return build_compound_key(
        str(item.get("ack_server") or ""),
        str(item.get("event_id") or item.get("id") or ""),
    )


def _source_ref_event_id(item: dict[str, Any], key: str) -> str:
    explicit = str(item.get("event_id") or "").strip()
    if explicit:
        return explicit
    ack_server = str(item.get("ack_server") or "").strip()
    raw_id = str(item.get("id") or "").strip()
    if ack_server and raw_id.startswith(f"{ack_server}:"):
        return raw_id.split(":", 1)[1]
    if key.startswith(f"{ack_server}:"):
        return key.split(":", 1)[1]
    return raw_id or key


def _source_ref_from_item(
    key: str,
    item: dict[str, Any],
    *,
    kind: str,
) -> dict[str, str]:
    ref: dict[str, str] = {"id": key, "kind": kind}
    event_id = _source_ref_event_id(item, key)
    if event_id:
        ref["event_id"] = event_id
    field_map = {
        "ack_server": ("ack_server",),
        "source_name": ("source_name", "source"),
        "source": ("source", "source_name"),
        "title": ("title",),
        "url": ("url",),
        "published_at": ("published_at",),
    }
    for output_key, candidates in field_map.items():
        for candidate in candidates:
            value = str(item.get(candidate) or "").strip()
            if value:
                ref[output_key] = value
                break
    return ref


def _build_proactive_source_refs(ctx: "AgentTickContext") -> list[dict[str, str]]:
    if not ctx.cited_item_ids:
        return []

    content_map = {
        key: item
        for item in ctx.fetched_contents
        if (key := _compound_source_key(item))
    }
    alert_map = {
        key: item
        for item in ctx.fetched_alerts
        if (key := _compound_source_key(item))
    }

    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for key in ctx.cited_item_ids:
        if key in seen:
            continue
        seen.add(key)
        if key in content_map:
            refs.append(_source_ref_from_item(key, content_map[key], kind="content"))
        elif key in alert_map:
            refs.append(_source_ref_from_item(key, alert_map[key], kind="alert"))
        else:
            refs.append({"id": key, "kind": "unknown"})
    return refs


def build_delivery_key(ctx: "AgentTickContext") -> str:
    refs = _build_delivery_refs(ctx)
    if refs and any(not ref.startswith("id:") for ref in refs):
        key_src = json.dumps(refs)
    elif ctx.cited_item_ids:
        key_src = json.dumps(sorted(ctx.cited_item_ids))
    else:
        key_src = ctx.final_message[:500]
    return sha1(key_src.encode()).hexdigest()[:16]
