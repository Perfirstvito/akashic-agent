from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SHORT_QUERY_MAX_CHARS = 10
DEFAULT_SHORT_QUERY_MIN_HITS = 4
DEFAULT_STICKY_MIN_CONTEXTS = 20
DEFAULT_MAX_MEMORY_PROBES_PER_TYPE = 25


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path

    @property
    def sessions_db(self) -> Path:
        return self.root / "sessions.db"

    @property
    def memory_db(self) -> Path:
        return self.root / "memory" / "memory2.db"

    @property
    def recall_log(self) -> Path:
        return self.root / "observe" / "recall_inspector.jsonl"


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def _read_only_connect(path: Path, *, immutable: bool = False) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(path)
    mode = "immutable=1" if immutable else "mode=ro"
    conn = sqlite3.connect(f"file:{path.as_posix()}?{mode}", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                records.append(
                    {
                        "kind": "parse_error",
                        "line_no": line_no,
                        "raw_preview": text[:200],
                    }
                )
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _case_id(prefix: str, *parts: object) -> str:
    raw = "_".join(str(part) for part in parts if str(part))
    clean = re.sub(r"[^a-zA-Z0-9_\-]+", "_", raw).strip("_").lower()
    if len(clean) > 80:
        clean = clean[:80].rstrip("_")
    return f"{prefix}_{clean}" if clean else prefix


def _safe_filename(case_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", case_id).strip("_") + ".json"


def _split_source_ids(source_ref: str) -> list[str]:
    raw = str(source_ref or "").strip()
    if not raw:
        return []
    base = raw.split("#", 1)[0].strip()
    if not base:
        return []
    try:
        value = json.loads(base)
    except json.JSONDecodeError:
        return [base]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _fetch_messages_by_ids(conn: sqlite3.Connection, ids: list[str]) -> list[dict[str, Any]]:
    clean_ids = [str(item).strip() for item in ids if str(item).strip()]
    if not clean_ids:
        return []
    placeholders = ",".join("?" for _ in clean_ids)
    rows = conn.execute(
        f"""
        SELECT id, session_key, seq, role, content, ts
        FROM messages
        WHERE id IN ({placeholders})
        ORDER BY session_key, seq
        """,
        tuple(clean_ids),
    ).fetchall()
    return [
        {
            "id": str(row["id"]),
            "session_key": str(row["session_key"]),
            "seq": int(row["seq"]),
            "role": str(row["role"]),
            "content": str(row["content"] or ""),
            "timestamp": str(row["ts"] or ""),
        }
        for row in rows
    ]


def _load_active_memory_items(memory_conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = memory_conn.execute(
        """
        SELECT id, memory_type, summary, source_ref, happened_at,
               created_at, updated_at, reinforcement, emotional_weight
        FROM memory_items
        WHERE status = 'active'
        ORDER BY updated_at DESC
        """
    ).fetchall()
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        result[str(row["id"])] = {
            "id": str(row["id"]),
            "memory_type": str(row["memory_type"]),
            "summary": str(row["summary"] or ""),
            "source_ref": str(row["source_ref"] or ""),
            "happened_at": str(row["happened_at"] or ""),
            "created_at": str(row["created_at"] or ""),
            "updated_at": str(row["updated_at"] or ""),
            "reinforcement": int(row["reinforcement"] or 0),
            "emotional_weight": int(row["emotional_weight"] or 0),
        }
    return result


def _group_recall_log(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    turns: dict[str, dict[str, Any]] = {}
    for record in records:
        kind = record.get("kind")
        if kind not in {"context_prepare", "recall_memory"}:
            continue
        turn_id = str(record.get("turn_id") or "")
        if not turn_id:
            continue
        turn = turns.setdefault(
            turn_id,
            {
                "turn_id": turn_id,
                "session_key": str(record.get("session_key") or ""),
                "channel": str(record.get("channel") or ""),
                "chat_id": str(record.get("chat_id") or ""),
                "timestamp": str(record.get("timestamp") or ""),
                "user_text": str(record.get("user_text") or ""),
                "context_prepare": None,
                "recall_memory_calls": [],
            },
        )
        if kind == "context_prepare":
            turn["session_key"] = str(record.get("session_key") or turn["session_key"])
            turn["channel"] = str(record.get("channel") or turn["channel"])
            turn["chat_id"] = str(record.get("chat_id") or turn["chat_id"])
            turn["timestamp"] = str(record.get("timestamp") or turn["timestamp"])
            turn["user_text"] = str(record.get("user_text") or "")
            turn["context_prepare"] = record.get("context_prepare") or {}
        elif kind == "recall_memory":
            turn["recall_memory_calls"].append(record.get("recall_memory") or {})
    return turns


def _compact_memory_item(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not item:
        return None
    return {
        "id": item["id"],
        "memory_type": item["memory_type"],
        "summary": item["summary"],
        "source_ref": item["source_ref"],
        "happened_at": item["happened_at"],
        "reinforcement": item["reinforcement"],
        "emotional_weight": item["emotional_weight"],
    }


def _make_base_case(
    *,
    case_id: str,
    source: str,
    failure_type: str,
    task: str,
    workspace: Path,
    turn: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "case_id": case_id,
        "source": source,
        "failure_type": failure_type,
        "task": task,
        "source_workspace": str(workspace),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if turn:
        payload["turn"] = {
            "turn_id": turn.get("turn_id"),
            "session_key": turn.get("session_key"),
            "channel": turn.get("channel"),
            "chat_id": turn.get("chat_id"),
            "timestamp": turn.get("timestamp"),
            "user_text": turn.get("user_text"),
        }
    return payload


def _write_case(
    *,
    badcase_dir: Path,
    report_path: Path,
    case_payload: dict[str, Any],
    report_payload: dict[str, Any],
) -> Path:
    path = badcase_dir / _safe_filename(str(case_payload["case_id"]))
    _json_dump(path, case_payload)
    report = dict(report_payload)
    report.update(
        {
            "kind": "case",
            "case_id": case_payload["case_id"],
            "failure_type": case_payload.get("failure_type"),
            "task": case_payload.get("task"),
            "case_path": str(path),
        }
    )
    _append_jsonl(report_path, report)
    return path


def _emit_explicit_recall_cases(
    *,
    turns: dict[str, dict[str, Any]],
    memory_items: dict[str, dict[str, Any]],
    workspace: Path,
    badcase_dir: Path,
    report_path: Path,
) -> int:
    emitted = 0
    for turn in turns.values():
        calls = list(turn.get("recall_memory_calls") or [])
        if not calls:
            continue
        for index, call in enumerate(calls, 1):
            args = call.get("arguments") or {}
            items = [
                item
                for item in call.get("items") or []
                if isinstance(item, dict)
            ]
            hit_ids = [str(item.get("id") or "") for item in items if item.get("id")]
            count = int(call.get("count") or 0)
            failure_type = "explicit_empty_recall" if count == 0 else "explicit_recall_review"
            case_id = _case_id(
                "daily",
                failure_type,
                turn.get("turn_id"),
                index,
            )
            case_payload = _make_base_case(
                case_id=case_id,
                source="daily",
                failure_type=failure_type,
                task="retrieve_explicit",
                workspace=workspace,
                turn=turn,
            )
            case_payload.update(
                {
                    "probe": {
                        "query": str(args.get("query") or ""),
                        "memory_type": str(args.get("memory_type") or ""),
                        "search_mode": str(args.get("search_mode") or "semantic"),
                        "limit": int(args.get("limit") or 8),
                        "description": str(args.get("description") or ""),
                    },
                    "expected": {
                        "manual_review": True,
                        "notes": (
                            "Explicit recall was observed in daily usage. "
                            "Gold relevance should be labeled manually unless expected_ids are added."
                        ),
                    },
                    "observed": {
                        "count": count,
                        "hit_ids": hit_ids,
                        "hits": [
                            {
                                "id": str(item.get("id") or ""),
                                "memory_type": str(item.get("memory_type") or ""),
                                "score": item.get("score"),
                                "summary": str(item.get("summary") or ""),
                            }
                            for item in items
                        ],
                    },
                    "observed_memory_items": [
                        _compact_memory_item(memory_items.get(item_id))
                        for item_id in hit_ids
                        if memory_items.get(item_id)
                    ],
                }
            )
            _write_case(
                badcase_dir=badcase_dir,
                report_path=report_path,
                case_payload=case_payload,
                report_payload={
                    "query": case_payload["probe"]["query"],
                    "hit_ids": hit_ids,
                    "observed_count": count,
                },
            )
            emitted += 1
    return emitted


def _emit_short_query_over_recall_cases(
    *,
    turns: dict[str, dict[str, Any]],
    memory_items: dict[str, dict[str, Any]],
    workspace: Path,
    badcase_dir: Path,
    report_path: Path,
    max_chars: int,
    min_hits: int,
    max_cases: int,
) -> int:
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for turn_id, turn in turns.items():
        text = str(turn.get("user_text") or "").strip()
        cp = turn.get("context_prepare") or {}
        items = cp.get("items") or []
        count = int(cp.get("count") or 0)
        if 0 < len(text) <= max_chars and count >= min_hits:
            candidates.append((count, -len(text), turn_id, turn))
    candidates.sort(reverse=True)

    emitted = 0
    for count, _neg_len, turn_id, turn in candidates[:max_cases]:
        cp = turn.get("context_prepare") or {}
        items = [item for item in cp.get("items") or [] if isinstance(item, dict)]
        injected = [
            item
            for item in cp.get("injected_items") or []
            if isinstance(item, dict)
        ]
        hit_ids = [str(item.get("id") or "") for item in items if item.get("id")]
        injected_ids = [
            str(item.get("id") or "") for item in injected if item.get("id")
        ]
        case_id = _case_id("daily", "short_query_over_recall", turn_id)
        case_payload = _make_base_case(
            case_id=case_id,
            source="daily",
            failure_type="short_query_over_recall",
            task="retrieve_context",
            workspace=workspace,
            turn=turn,
        )
        case_payload.update(
            {
                "probe": {
                    "query": str(turn.get("user_text") or ""),
                    "top_k": 8,
                    "mode": "passive_context_prepare",
                },
                "expected": {
                    "max_injected_count": 0,
                    "manual_review": True,
                    "notes": "Short/low-information query should usually avoid memory injection.",
                },
                "observed": {
                    "context_count": count,
                    "hit_ids": hit_ids,
                    "injected_ids": injected_ids,
                    "hits": [
                        {
                            "id": str(item.get("id") or ""),
                            "memory_type": str(item.get("memory_type") or ""),
                            "score": item.get("score"),
                            "injected": bool(item.get("injected")),
                            "summary": str(item.get("summary") or ""),
                        }
                        for item in items
                    ],
                },
                "observed_memory_items": [
                    _compact_memory_item(memory_items.get(item_id))
                    for item_id in hit_ids
                    if memory_items.get(item_id)
                ],
            }
        )
        _write_case(
            badcase_dir=badcase_dir,
            report_path=report_path,
            case_payload=case_payload,
            report_payload={
                "query": case_payload["probe"]["query"],
                "hit_ids": hit_ids,
                "injected_ids": injected_ids,
                "observed_count": count,
            },
        )
        emitted += 1
    return emitted


def _emit_sticky_memory_cases(
    *,
    turns: dict[str, dict[str, Any]],
    memory_items: dict[str, dict[str, Any]],
    workspace: Path,
    badcase_dir: Path,
    report_path: Path,
    min_contexts: int,
    max_cases: int,
) -> int:
    item_counts: Counter[str] = Counter()
    item_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in turns.values():
        cp = turn.get("context_prepare") or {}
        for raw_item in cp.get("injected_items") or cp.get("items") or []:
            if not isinstance(raw_item, dict):
                continue
            item_id = str(raw_item.get("id") or "")
            if not item_id:
                continue
            item_counts[item_id] += 1
            if len(item_examples[item_id]) < 5:
                item_examples[item_id].append(
                    {
                        "turn_id": turn.get("turn_id"),
                        "session_key": turn.get("session_key"),
                        "timestamp": turn.get("timestamp"),
                        "user_text": turn.get("user_text"),
                    }
                )

    emitted = 0
    for item_id, frequency in item_counts.most_common(max_cases):
        if frequency < min_contexts:
            break
        item = memory_items.get(item_id)
        case_id = _case_id("daily", "sticky_memory", item_id)
        case_payload = _make_base_case(
            case_id=case_id,
            source="daily",
            failure_type="sticky_memory",
            task="injection_frequency_review",
            workspace=workspace,
        )
        case_payload.update(
            {
                "probe": {
                    "memory_id": item_id,
                },
                "expected": {
                    "manual_review": True,
                    "notes": "Review whether this item is over-injected across unrelated daily turns.",
                },
                "observed": {
                    "injected_context_count": frequency,
                    "sample_turns": item_examples[item_id],
                },
                "memory_item": _compact_memory_item(item),
            }
        )
        _write_case(
            badcase_dir=badcase_dir,
            report_path=report_path,
            case_payload=case_payload,
            report_payload={
                "memory_id": item_id,
                "injected_context_count": frequency,
                "memory_type": item.get("memory_type") if item else "",
            },
        )
        emitted += 1
    return emitted


def _strip_event_timestamp(summary: str) -> str:
    return re.sub(r"^\[[^\]]+\]\s*", "", summary).strip()


def _emit_memory_item_probe_cases(
    *,
    sessions_conn: sqlite3.Connection,
    memory_items: dict[str, dict[str, Any]],
    workspace: Path,
    badcase_dir: Path,
    report_path: Path,
    max_per_type: int,
    include_source_messages: bool,
) -> int:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in memory_items.values():
        by_type[str(item["memory_type"])].append(item)

    emitted = 0
    for memory_type, items in sorted(by_type.items()):
        # Recent and reinforced items are the best positive controls.
        items.sort(
            key=lambda item: (
                int(item.get("reinforcement") or 0),
                str(item.get("updated_at") or ""),
            ),
            reverse=True,
        )
        for item in items[:max_per_type]:
            source_ids = _split_source_ids(str(item.get("source_ref") or ""))
            messages = (
                _fetch_messages_by_ids(sessions_conn, source_ids)
                if include_source_messages
                else []
            )
            query = _strip_event_timestamp(str(item.get("summary") or ""))
            case_id = _case_id("daily", "memory_item_probe", memory_type, item["id"])
            case_payload = _make_base_case(
                case_id=case_id,
                source="daily",
                failure_type="positive_control",
                task="retrieve_explicit",
                workspace=workspace,
            )
            case_payload.update(
                {
                    "seed": {
                        "source_ref": item.get("source_ref"),
                        "source_ids": source_ids,
                        "messages": messages,
                    },
                    "probe": {
                        "query": query,
                        "memory_type": memory_type,
                        "search_mode": "semantic",
                        "limit": 8,
                    },
                    "expected": {
                        "memory_ids": [item["id"]],
                        "source_ref": item.get("source_ref"),
                        "must_recall_at_k": 8,
                    },
                    "memory_item": _compact_memory_item(item),
                }
            )
            _write_case(
                badcase_dir=badcase_dir,
                report_path=report_path,
                case_payload=case_payload,
                report_payload={
                    "memory_id": item["id"],
                    "memory_type": memory_type,
                    "query": query,
                    "source_message_count": len(messages),
                },
            )
            emitted += 1
    return emitted


def _write_summary_report(
    *,
    report_path: Path,
    workspace: Path,
    turns: dict[str, dict[str, Any]],
    memory_items: dict[str, dict[str, Any]],
    emitted_counts: dict[str, int],
) -> None:
    recall_calls = [
        call
        for turn in turns.values()
        for call in turn.get("recall_memory_calls") or []
    ]
    context_counts = [
        int((turn.get("context_prepare") or {}).get("count") or 0)
        for turn in turns.values()
        if turn.get("context_prepare") is not None
    ]
    memory_type_counts = Counter(
        str(item.get("memory_type") or "") for item in memory_items.values()
    )
    _append_jsonl(
        report_path,
        {
            "kind": "summary",
            "workspace": str(workspace),
            "turn_count": len(turns),
            "context_prepare_count": len(context_counts),
            "context_nonzero_count": sum(1 for value in context_counts if value > 0),
            "recall_call_count": len(recall_calls),
            "recall_empty_count": sum(
                1 for call in recall_calls if int(call.get("count") or 0) == 0
            ),
            "active_memory_count": len(memory_items),
            "memory_type_counts": dict(sorted(memory_type_counts.items())),
            "emitted_counts": emitted_counts,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract daily memory-engine badcases from a workspace."
    )
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument(
        "--badcase-dir",
        type=Path,
        default=Path("eval/memory_engine/badcases/daily"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSONL report path. Defaults to eval/memory_engine/reports/daily_extract_<timestamp>.jsonl",
    )
    parser.add_argument("--short-query-max-chars", type=int, default=DEFAULT_SHORT_QUERY_MAX_CHARS)
    parser.add_argument("--short-query-min-hits", type=int, default=DEFAULT_SHORT_QUERY_MIN_HITS)
    parser.add_argument("--short-query-max-cases", type=int, default=50)
    parser.add_argument("--sticky-min-contexts", type=int, default=DEFAULT_STICKY_MIN_CONTEXTS)
    parser.add_argument("--sticky-max-cases", type=int, default=30)
    parser.add_argument(
        "--max-memory-probes-per-type",
        type=int,
        default=DEFAULT_MAX_MEMORY_PROBES_PER_TYPE,
    )
    parser.add_argument(
        "--no-source-messages",
        action="store_true",
        help="Do not embed source message windows in memory_item_probe cases.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = WorkspacePaths(args.workspace.expanduser().resolve())
    report_path = args.report or Path(
        f"eval/memory_engine/reports/daily_extract_{_now_stamp()}.jsonl"
    )
    badcase_dir = args.badcase_dir

    records = _iter_jsonl(paths.recall_log)
    turns = _group_recall_log(records)

    with _read_only_connect(paths.memory_db) as memory_conn, _read_only_connect(
        paths.sessions_db
    ) as sessions_conn:
        memory_items = _load_active_memory_items(memory_conn)
        emitted_counts = {
            "explicit_recall": _emit_explicit_recall_cases(
                turns=turns,
                memory_items=memory_items,
                workspace=paths.root,
                badcase_dir=badcase_dir,
                report_path=report_path,
            ),
            "short_query_over_recall": _emit_short_query_over_recall_cases(
                turns=turns,
                memory_items=memory_items,
                workspace=paths.root,
                badcase_dir=badcase_dir,
                report_path=report_path,
                max_chars=max(1, args.short_query_max_chars),
                min_hits=max(1, args.short_query_min_hits),
                max_cases=max(0, args.short_query_max_cases),
            ),
            "sticky_memory": _emit_sticky_memory_cases(
                turns=turns,
                memory_items=memory_items,
                workspace=paths.root,
                badcase_dir=badcase_dir,
                report_path=report_path,
                min_contexts=max(1, args.sticky_min_contexts),
                max_cases=max(0, args.sticky_max_cases),
            ),
            "memory_item_probe": _emit_memory_item_probe_cases(
                sessions_conn=sessions_conn,
                memory_items=memory_items,
                workspace=paths.root,
                badcase_dir=badcase_dir,
                report_path=report_path,
                max_per_type=max(0, args.max_memory_probes_per_type),
                include_source_messages=not args.no_source_messages,
            ),
        }

    _write_summary_report(
        report_path=report_path,
        workspace=paths.root,
        turns=turns,
        memory_items=memory_items,
        emitted_counts=emitted_counts,
    )

    print(f"badcase_dir: {badcase_dir}")
    print(f"report: {report_path}")
    print(json.dumps(emitted_counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
