from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_RULES_DIR = "proactive_rules"
DEFAULT_MAX_RULE_CHARS = 3000
DEFAULT_MAX_RULE_FILES = 12


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.endswith(".md"):
        text = text[:-3]
    chars: list[str] = []
    last_dash = False
    for ch in text:
        if ch.isalnum() or ch == "-":
            chars.append(ch)
            last_dash = False
            continue
        if not last_dash:
            chars.append("-")
            last_dash = True
    return "".join(chars).strip("-_")


def _normalize_rule_key(value: Any) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return ""
    parts = [_slug(part) for part in raw.split("/") if part.strip()]
    if not parts:
        return ""
    return "/".join(part for part in parts if part)


def _iter_raw_rule_keys(value: Any) -> Iterable[Any]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Iterable):
        for item in value:
            yield item


def _add_key(keys: list[str], seen: set[str], key: str) -> None:
    normalized = _normalize_rule_key(key)
    if normalized and normalized not in seen:
        keys.append(normalized)
        seen.add(normalized)


def _infer_content_type(item: dict[str, Any]) -> str:
    for key in ("source_type", "subscription_type"):
        slug = _slug(item.get(key))
        if slug:
            return slug

    source = str(item.get("source") or item.get("source_name") or "").lower()
    url = str(item.get("url") or "")
    host = urlparse(url).netloc.lower()
    haystack = f"{source} {host}"

    if "arxiv" in haystack:
        return "arxiv"
    if "bilibili" in haystack or "b23.tv" in haystack or "b站" in source:
        return "bilibili"
    if "youtube" in haystack or "youtu.be" in haystack:
        return "youtube"
    if "twitter" in haystack or "x.com" in haystack:
        return "x"
    return ""


def collect_content_rule_keys(content_meta: Iterable[dict[str, Any]]) -> list[str]:
    items = [item for item in content_meta if isinstance(item, dict)]
    if not items:
        return []

    keys: list[str] = []
    seen: set[str] = set()
    _add_key(keys, seen, "content")

    for item in items:
        content_type = _infer_content_type(item)
        if content_type:
            _add_key(keys, seen, f"content/{content_type}")

        subscription_type = _slug(item.get("subscription_type"))
        if subscription_type and subscription_type != content_type:
            _add_key(keys, seen, f"content/{subscription_type}")

        subscription_id = _slug(item.get("subscription_id"))
        if subscription_id:
            _add_key(keys, seen, f"subscriptions/{subscription_id}")

        for raw_key in _iter_raw_rule_keys(
            item.get("_rule_keys")
            or item.get("rule_keys")
            or item.get("proactive_rule_keys")
        ):
            _add_key(keys, seen, raw_key)

    return keys


def read_dynamic_rule_context(
    workspace: Path | str | None,
    content_meta: Iterable[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_MAX_RULE_CHARS,
    max_files: int = DEFAULT_MAX_RULE_FILES,
) -> str:
    if workspace is None:
        return ""
    root = Path(workspace) / DEFAULT_RULES_DIR
    if not root.exists():
        return ""

    chunks: list[str] = []
    remaining = max(0, int(max_chars))
    if remaining <= 0:
        return ""

    for key in collect_content_rule_keys(content_meta)[:max_files]:
        path = root / f"{key}.md"
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not text:
            continue
        label = f"[{DEFAULT_RULES_DIR}/{key}.md]\n"
        budget = remaining - len(label)
        if budget <= 0:
            break
        clipped = text[:budget]
        chunks.append(label + clipped)
        remaining -= len(label) + len(clipped)
        if remaining <= 0:
            break

    return "\n\n".join(chunks)
