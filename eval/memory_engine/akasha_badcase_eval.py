from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, cast

from agent.config_models import Config
from core.memory.engine import MemoryQuery, MemoryQueryFilters, MemoryScope
from core.net.http import SharedHttpResources
from eval.memory_engine.badcase_lane_eval import (
    DEFAULT_BADCASE_DIR,
    DEFAULT_REPORT_DIR,
    _as_dict,
    _case_query,
    _safe_int,
    _unique_ids,
)
from plugins.akasha.config import load_akasha_config
from plugins.akasha.core import turn_key
from plugins.akasha.engine import AkashaMemoryEngine


DEFAULT_WORKSPACE = Path.home() / ".akashic" / "workspace"
DEFAULT_SANDBOX_ROOT = Path("eval/memory_engine/sandbox")


@dataclass(frozen=True)
class ExpectedTurns:
    turn_keys: list[str]
    missing_message_ids: list[str]


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _copy_sqlite(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        src = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
        dst = sqlite3.connect(str(destination))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
    except Exception:
        if destination.exists():
            destination.unlink()
        shutil.copy2(source, destination)


def _prepare_sandbox(source_workspace: Path, sandbox_root: Path) -> Path:
    workspace = sandbox_root / "workspace"
    _copy_sqlite(source_workspace / "sessions.db", workspace / "sessions.db")
    _copy_sqlite(source_workspace / "memory" / "akasha.db", workspace / "memory" / "akasha.db")
    return workspace


def _source_ref_message_ids(source_ref: object) -> list[str]:
    text = str(source_ref or "").strip()
    if not text:
        return []
    json_part = text.split("#", 1)[0]
    try:
        value = cast(object, json.loads(json_part))
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return _unique_ids([str(item) for item in cast(list[object], value)])


def _message_index(sessions_db: Path) -> dict[str, tuple[str, int, str]]:
    conn = sqlite3.connect(f"file:{sessions_db.as_posix()}?mode=ro", uri=True)
    try:
        return {
            str(row[0]): (str(row[1]), int(row[2]), str(row[3] or ""))
            for row in conn.execute("SELECT id, session_key, seq, role FROM messages")
        }
    finally:
        conn.close()


def _expected_turns(case: dict[str, Any], *, index: dict[str, tuple[str, int, str]]) -> ExpectedTurns:
    expected = _as_dict(case.get("expected"))
    memory_items = [
        item for item in expected.get("memory_items") or [] if isinstance(item, dict)
    ]
    message_ids: list[str] = []
    for item in memory_items:
        message_ids.extend(_source_ref_message_ids(item.get("source_ref")))
    missing: list[str] = []
    turn_keys: list[str] = []
    for message_id in _unique_ids(message_ids):
        row = index.get(message_id)
        if row is None:
            missing.append(message_id)
            continue
        session_key, seq, role = row
        _, _, key = turn_key(session_key, seq, role)
        turn_keys.append(key)
    return ExpectedTurns(
        turn_keys=_unique_ids(turn_keys),
        missing_message_ids=missing,
    )


def _rank_metrics(hit_ids: list[str], expected_ids: list[str]) -> dict[str, Any]:
    hits = _unique_ids(hit_ids)
    expected = set(_unique_ids(expected_ids))
    matched = [item_id for item_id in hits if item_id in expected]
    ranks = [hits.index(item_id) + 1 for item_id in matched]
    first_rank = min(ranks) if ranks else None
    precision = len(matched) / len(hits) if hits else 0.0
    recall = len(set(matched)) / len(expected) if expected else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if isinstance(recall, float) and precision + recall > 0
        else None
    )
    return {
        "top_n": len(hits),
        "expected_count": len(expected),
        "matched_count": len(set(matched)),
        "matched_ids": matched,
        "unmatched_expected_ids": sorted(expected - set(matched)),
        "precision@topn": round(precision, 6) if expected else None,
        "recall@topn": round(recall, 6) if isinstance(recall, float) else None,
        "f1@topn": round(f1, 6) if isinstance(f1, float) else None,
        "first_rank": first_rank,
        "mrr": round(1.0 / first_rank, 6) if first_rank else 0.0,
        "hit@1": bool(hits[:1] and set(hits[:1]) & expected),
        "hit@3": bool(set(hits[:3]) & expected),
        "hit@5": bool(set(hits[:5]) & expected),
        "hit@8": bool(set(hits[:8]) & expected),
    }


def _records_by_lane(records: list[Any]) -> dict[str, list[str]]:
    lanes: dict[str, list[str]] = {"dense": [], "ripple": [], "all": []}
    for record in records:
        record_id = str(getattr(record, "id", "") or "")
        if not record_id:
            continue
        signals = getattr(record, "signals", {}) or {}
        source = str(signals.get("source") or "").lower()
        if source == "dense":
            lanes["dense"].append(record_id)
        elif source:
            lanes["ripple"].append(record_id)
        lanes["all"].append(record_id)
    return {key: _unique_ids(value) for key, value in lanes.items()}


def _case_timestamp(case: dict[str, Any]) -> datetime:
    raw = str(_as_dict(case.get("turn")).get("timestamp") or "").strip()
    if raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _compact_records(records: list[Any], *, max_items: int = 12) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for record in records[:max_items]:
        compact.append(
            {
                "id": str(getattr(record, "id", "") or ""),
                "score": getattr(record, "score", None),
                "source": (getattr(record, "signals", {}) or {}).get("source"),
                "summary": str(getattr(record, "summary", "") or "")[:360],
            }
        )
    return compact


async def _eval_case(
    engine: AkashaMemoryEngine,
    case_path: Path,
    *,
    message_index: dict[str, tuple[str, int, str]],
) -> dict[str, Any]:
    case = _read_json(case_path)
    case_id = str(case.get("case_id") or case_path.stem)
    task = str(case.get("task") or "")
    query = _case_query(case)
    expected = _expected_turns(case, index=message_index)
    probe = _as_dict(case.get("probe"))
    turn = _as_dict(case.get("turn"))
    limit = max(1, _safe_int(probe.get("limit") or probe.get("top_k"), 8))
    intent = "context" if task == "retrieve_context" else "answer"
    if not query:
        return {
            "kind": "akasha_badcase_eval_case",
            "case_id": case_id,
            "case_key": case_path.stem,
            "case_path": str(case_path),
            "status": "skipped",
            "reason": "case has no retrieval query",
        }
    result = await engine.query(
        MemoryQuery(
            text=query,
            intent=intent,
            scope=MemoryScope(
                session_key=str(turn.get("session_key") or ""),
                channel=str(turn.get("channel") or ""),
                chat_id=str(turn.get("chat_id") or ""),
            ),
            filters=MemoryQueryFilters(),
            limit=limit,
            timestamp=_case_timestamp(case),
        )
    )
    lanes = _records_by_lane(result.records)
    metrics = {
        lane: _rank_metrics(ids, expected.turn_keys)
        for lane, ids in lanes.items()
    }
    return {
        "kind": "akasha_badcase_eval_case",
        "case_id": case_id,
        "case_key": case_path.stem,
        "case_path": str(case_path),
        "failure_type": str(case.get("failure_type") or ""),
        "task": task,
        "intent": intent,
        "query": query,
        "status": "ok",
        "expected_turn_keys": expected.turn_keys,
        "expected_missing_message_ids": expected.missing_message_ids,
        "include_in_pr_summary": bool(expected.turn_keys),
        "metrics": metrics,
        "hit_ids": lanes,
        "records": _compact_records(result.records),
        "trace": result.trace,
    }


async def _eval_sticky_case(
    engine: AkashaMemoryEngine,
    case_path: Path,
    *,
    message_index: dict[str, tuple[str, int, str]],
) -> dict[str, Any]:
    case = _read_json(case_path)
    case_id = str(case.get("case_id") or case_path.stem)
    expected = _expected_turns(case, index=message_index)
    sample_turns = [
        item
        for item in _as_dict(case.get("observed")).get("sample_turns") or []
        if isinstance(item, dict)
    ]
    if not sample_turns:
        return {
            "kind": "akasha_badcase_eval_case",
            "case_id": case_id,
            "case_key": case_path.stem,
            "case_path": str(case_path),
            "failure_type": str(case.get("failure_type") or ""),
            "task": str(case.get("task") or ""),
            "status": "skipped",
            "reason": "sticky case has no sample_turns",
        }

    per_turn: list[dict[str, Any]] = []
    for index, turn in enumerate(sample_turns, 1):
        query = str(turn.get("user_text") or "").strip()
        if not query:
            continue
        synthetic_case = {
            "turn": {
                "timestamp": turn.get("timestamp") or "",
            }
        }
        result = await engine.query(
            MemoryQuery(
                text=query,
                intent="context",
                scope=MemoryScope(
                    session_key=str(turn.get("session_key") or ""),
                    channel=str(turn.get("channel") or ""),
                    chat_id=str(turn.get("chat_id") or ""),
                ),
                filters=MemoryQueryFilters(),
                limit=8,
                timestamp=_case_timestamp(synthetic_case),
            )
        )
        lanes = _records_by_lane(result.records)
        metrics = {
            lane: _rank_metrics(ids, expected.turn_keys)
            for lane, ids in lanes.items()
        }
        per_turn.append(
            {
                "index": index,
                "turn_id": turn.get("turn_id"),
                "query": query,
                "metrics": metrics,
                "hit_ids": lanes,
                "trace": result.trace,
            }
        )

    aggregate: dict[str, dict[str, Any]] = {}
    for lane in ("dense", "ripple", "all"):
        hits = [
            item
            for item in per_turn
            if int(_as_dict(_as_dict(item.get("metrics")).get(lane)).get("matched_count") or 0) > 0
        ]
        aggregate[lane] = {
            "sample_turn_count": len(per_turn),
            "target_presence_count": len(hits),
            "target_presence_rate": round(len(hits) / len(per_turn), 6) if per_turn else 0.0,
        }

    return {
        "kind": "akasha_badcase_eval_case",
        "case_id": case_id,
        "case_key": case_path.stem,
        "case_path": str(case_path),
        "failure_type": str(case.get("failure_type") or ""),
        "task": str(case.get("task") or ""),
        "query": "",
        "status": "ok",
        "expected_turn_keys": expected.turn_keys,
        "expected_missing_message_ids": expected.missing_message_ids,
        "include_in_pr_summary": False,
        "metric_semantics": "sticky target presence, not gold relevance",
        "metrics": aggregate,
        "sample_turns": per_turn,
    }


def _avg(results: list[dict[str, Any]], lane: str, key: str) -> float | None:
    values: list[float] = []
    for result in results:
        value = _as_dict(_as_dict(result.get("metrics")).get(lane)).get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return round(mean(values), 6) if values else None


def _summarize(results: list[dict[str, Any]], *, sandbox_workspace: Path) -> dict[str, Any]:
    ok = [result for result in results if result.get("status") == "ok"]
    gold = [result for result in ok if result.get("include_in_pr_summary")]
    sticky = [result for result in ok if result.get("task") == "injection_frequency_review"]
    return {
        "kind": "akasha_badcase_eval_summary",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(results),
        "status_counts": dict(Counter(str(result.get("status") or "") for result in results)),
        "failure_type_counts": dict(Counter(str(result.get("failure_type") or "") for result in ok)),
        "gold_pr_case_count": len(gold),
        "sticky_case_count": len(sticky),
        "expected_missing_message_case_count": len(
            [result for result in ok if result.get("expected_missing_message_ids")]
        ),
        "gold_pr_metrics": {
            lane: {
                "avg_returned_n": _avg(gold, lane, "top_n"),
                "macro_precision@topn": _avg(gold, lane, "precision@topn"),
                "macro_recall@topn": _avg(gold, lane, "recall@topn"),
                "macro_f1@topn": _avg(gold, lane, "f1@topn"),
                "hit@1_rate": _avg(gold, lane, "hit@1"),
                "hit@3_rate": _avg(gold, lane, "hit@3"),
                "hit@5_rate": _avg(gold, lane, "hit@5"),
                "hit@8_rate": _avg(gold, lane, "hit@8"),
                "macro_mrr": _avg(gold, lane, "mrr"),
            }
            for lane in ("dense", "ripple", "all")
        },
        "sandbox_workspace": sandbox_workspace.as_posix(),
    }


def _format_metric(value: Any) -> str:
    return "N/A" if value is None else str(value)


def _write_markdown(path: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    rows = [
        "# Akasha Badcase Eval",
        "",
        f"Cases: `{summary['total_cases']}`. Gold cases: `{summary['gold_pr_case_count']}`.",
        f"Sandbox: `{summary['sandbox_workspace']}`.",
        "",
        "Metrics map expected `memory_items.source_ref` message ids to Akasha turn keys.",
        "",
        "## Aggregate",
        "",
        "| Lane | Macro P@N | Macro R@N | Macro F1@N | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR | Avg N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lane, metrics in _as_dict(summary.get("gold_pr_metrics")).items():
        metric = _as_dict(metrics)
        rows.append(
            "| "
            + " | ".join(
                [
                    lane,
                    _format_metric(metric.get("macro_precision@topn")),
                    _format_metric(metric.get("macro_recall@topn")),
                    _format_metric(metric.get("macro_f1@topn")),
                    _format_metric(metric.get("hit@1_rate")),
                    _format_metric(metric.get("hit@3_rate")),
                    _format_metric(metric.get("hit@5_rate")),
                    _format_metric(metric.get("hit@8_rate")),
                    _format_metric(metric.get("macro_mrr")),
                    _format_metric(metric.get("avg_returned_n")),
                ]
            )
            + " |"
        )
    rows.extend(
        [
            "",
            "## Cases",
            "",
            "| # | Case | Type | Query | Expected | All Hit/R/F1/MRR | Dense Hit/R/F1/MRR | Ripple Hit/R/F1/MRR |",
            "|---:|---|---|---|---:|---|---|---|",
        ]
    )
    for index, result in enumerate([r for r in results if r.get("status") == "ok"], 1):
        metrics = _as_dict(result.get("metrics"))
        def cells(lane: str) -> str:
            item = _as_dict(metrics.get(lane))
            return "/".join(
                [
                    str(item.get("matched_count")),
                    _format_metric(item.get("recall@topn")),
                    _format_metric(item.get("f1@topn")),
                    _format_metric(item.get("mrr")),
                ]
            )
        query = str(result.get("query") or "").replace("\n", " ").replace("|", "\\|")
        if len(query) > 72:
            query = query[:72] + "..."
        rows.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{result.get('case_key')}`",
                    str(result.get("failure_type") or ""),
                    query,
                    str(len(result.get("expected_turn_keys") or [])),
                    cells("all"),
                    cells("dense"),
                    cells("ripple"),
                ]
            )
            + " |"
        )
    sticky = [
        result
        for result in results
        if result.get("status") == "ok" and result.get("task") == "injection_frequency_review"
    ]
    if sticky:
        rows.extend(
            [
                "",
                "## Sticky Target Presence",
                "",
                "| # | Case | Dense | Ripple | All |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for index, result in enumerate(sticky, 1):
            metrics = _as_dict(result.get("metrics"))
            rows.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        f"`{result.get('case_key')}`",
                        _format_metric(_as_dict(metrics.get("dense")).get("target_presence_rate")),
                        _format_metric(_as_dict(metrics.get("ripple")).get("target_presence_rate")),
                        _format_metric(_as_dict(metrics.get("all")).get("target_presence_rate")),
                    ]
                )
                + " |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Akasha retrieval on curated memory-engine badcases in a sandbox workspace."
    )
    parser.add_argument("--badcase-dir", type=Path, default=DEFAULT_BADCASE_DIR)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--sandbox-root", type=Path, default=DEFAULT_SANDBOX_ROOT)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--markdown-report", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser


async def _amain() -> None:
    args = _build_parser().parse_args()
    stamp = _now_stamp()
    report_path = args.report or DEFAULT_REPORT_DIR / f"akasha_badcase_eval_{stamp}.jsonl"
    markdown_path = args.markdown_report or report_path.with_suffix(".md")
    sandbox_workspace = _prepare_sandbox(
        args.workspace.expanduser().resolve(),
        args.sandbox_root / f"akasha_badcase_{stamp}",
    )
    case_paths = sorted(
        path
        for path in args.badcase_dir.glob("*.json")
        if not path.name.startswith("_")
    )
    if args.limit:
        case_paths = case_paths[: max(0, int(args.limit))]
    if not case_paths:
        raise FileNotFoundError(f"no case json files under {args.badcase_dir}")

    config = Config.load(args.config)
    http_resources = SharedHttpResources()
    engine = AkashaMemoryEngine(
        config=config,
        akasha_config=load_akasha_config(),
        workspace=sandbox_workspace,
        http_resources=http_resources,
    )
    message_index = _message_index(sandbox_workspace / "sessions.db")
    results: list[dict[str, Any]] = []
    try:
        for case_path in case_paths:
            case = _read_json(case_path)
            if str(case.get("task") or "") == "injection_frequency_review":
                result = await _eval_sticky_case(
                    engine,
                    case_path,
                    message_index=message_index,
                )
            else:
                result = await _eval_case(engine, case_path, message_index=message_index)
            result["index"] = len(results) + 1
            result["sandbox_workspace"] = sandbox_workspace.as_posix()
            results.append(result)
    finally:
        for closeable in reversed(getattr(engine, "closeables", [])):
            close = getattr(closeable, "close", None)
            if callable(close):
                close()
        await http_resources.aclose()

    summary = _summarize(results, sandbox_workspace=sandbox_workspace)
    _write_jsonl(report_path, [*results, summary])
    _write_markdown(markdown_path, results, summary)
    print(f"report: {report_path}")
    print(f"markdown_report: {markdown_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
