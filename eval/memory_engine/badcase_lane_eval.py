from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from eval.memory_engine.daily_badcase_eval import (
    DEFAULT_REPORT_DIR,
    DEFAULT_SANDBOX_ROOT,
    DEFAULT_WORKSPACE,
    EXPLICIT_VECTOR_SCORE_THRESHOLD,
    EXPLICIT_VECTOR_TOP_K,
    EvalEmbedder,
    _as_dict,
    _bm25_summary_search,
    _build_retriever,
    _case_query,
    _copy_memory_db_to_sandbox,
    _load_embedding_settings,
    _load_retrieval_settings,
    _memory_types,
    _read_json,
    _scope,
)


DEFAULT_BADCASE_DIR = Path("eval/memory_engine/badcases/badcase_with_expectedid")
SPARSE_LIMIT_FLOOR = 30
SPARSE_LIMIT_MULTIPLIER = 2


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _unique_ids(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item_id) for item_id in ids if str(item_id).strip()))


def _hit_ids(hits: list[dict[str, Any]]) -> list[str]:
    return _unique_ids([str(hit.get("id") or "") for hit in hits])


def _compact_hits(hits: list[dict[str, Any]], *, max_hits: int = 12) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for hit in hits[:max_hits]:
        compact.append(
            {
                "id": str(hit.get("id") or ""),
                "memory_type": str(hit.get("memory_type") or ""),
                "score": hit.get("score"),
                "rrf_score": hit.get("rrf_score"),
                "semantic_score": _as_dict(hit.get("_score_debug")).get("semantic"),
                "keyword_score": hit.get("keyword_score"),
                "bm25_score": hit.get("bm25_score"),
                "summary": str(hit.get("summary") or ""),
            }
        )
    return compact


def _expected_ids(case: dict[str, Any]) -> list[str]:
    return _unique_ids(
        [
            str(item_id)
            for item_id in _as_dict(case.get("expected")).get("memory_ids") or []
        ]
    )


def _active_memory_ids(db_path: Path) -> set[str]:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return {
            str(row[0])
            for row in conn.execute("SELECT id FROM memory_items WHERE status='active'")
        }
    finally:
        conn.close()


def _lane_metrics(ids: list[str], expected_ids: list[str]) -> dict[str, Any]:
    unique_hits = _unique_ids(ids)
    expected = set(_unique_ids(expected_ids))
    matched = [item_id for item_id in unique_hits if item_id in expected]
    matched_unique = set(matched)
    result: dict[str, Any] = {
        "top_n": len(unique_hits),
        "returned_count": len(unique_hits),
        "expected_count": len(expected),
        "matched_count": len(matched_unique),
        "matched_ids": matched,
        "unmatched_expected_ids": sorted(expected - matched_unique),
        "unjudged_returned_count": len(unique_hits) - len(matched),
    }
    if not expected:
        result.update(
            {
                "precision@topn": None,
                "recall@topn": None,
                "f1@topn": None,
            }
        )
        return result

    precision = len(matched) / len(unique_hits) if unique_hits else 0.0
    recall = len(matched_unique) / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result.update(
        {
            "precision@topn": round(precision, 6),
            "recall@topn": round(recall, 6),
            "f1@topn": round(f1, 6),
        }
    )
    return result


def _hit(metrics: dict[str, Any]) -> bool:
    return int(metrics.get("matched_count") or 0) > 0


def _classify_chain(
    *,
    dense_ids: list[str],
    keyword_ids: list[str],
    metrics: dict[str, dict[str, Any]],
    expected_ids: list[str],
    task: str,
) -> str:
    if task == "injection_frequency_review":
        return "sticky_presence_diagnostic"
    if not expected_ids:
        return "no_expected_ids"

    dense_hit = _hit(metrics["dense"])
    keyword_hit = _hit(metrics["keyword"])
    fusion_hit = _hit(metrics["fusion"])

    if dense_hit and keyword_hit and fusion_hit:
        return "dense_keyword_fusion_all_hit"
    if dense_hit and keyword_hit and not fusion_hit:
        return "dense_keyword_hit_but_rrf_missed"
    if not dense_hit and keyword_hit and fusion_hit:
        return "dense_miss_keyword_hit_fusion_hit"
    if not dense_hit and keyword_hit and not fusion_hit:
        return "dense_miss_keyword_hit_but_rrf_missed"
    if dense_hit and not keyword_hit and fusion_hit:
        return "dense_hit_keyword_miss_fusion_hit"
    if dense_hit and not keyword_hit and not fusion_hit:
        return "dense_hit_keyword_miss_but_rrf_missed"

    dense_set = set(dense_ids)
    keyword_set = set(keyword_ids)
    if not dense_set and not keyword_set:
        return "dense_keyword_both_empty"
    if dense_set != keyword_set:
        return "dense_keyword_both_miss_different"
    return "dense_keyword_both_miss_same"


def _classify_bm25(metrics: dict[str, dict[str, Any]], expected_ids: list[str], task: str) -> str:
    if task == "injection_frequency_review":
        return "sticky_presence_diagnostic"
    if not expected_ids:
        return "no_expected_ids"
    dense_hit = _hit(metrics["dense"])
    bm25_hit = _hit(metrics["bm25_summary"])
    if dense_hit and bm25_hit:
        return "dense_bm25_both_hit"
    if dense_hit and not bm25_hit:
        return "dense_hit_bm25_miss"
    if not dense_hit and bm25_hit:
        return "dense_miss_bm25_hit"
    return "dense_bm25_both_miss"


def _contribution(
    *,
    fusion_ids: list[str],
    dense_ids: list[str],
    keyword_ids: list[str],
    bm25_ids: list[str],
    expected_ids: list[str],
) -> dict[str, Any]:
    fusion = set(fusion_ids)
    dense = set(dense_ids)
    keyword = set(keyword_ids)
    bm25 = set(bm25_ids)
    expected = set(expected_ids)
    payload: dict[str, Any] = {
        "fusion_count": len(fusion_ids),
        "dense_count": len(dense_ids),
        "keyword_count": len(keyword_ids),
        "bm25_count": len(bm25_ids),
        "dense_in_fusion_count": len(fusion & dense),
        "keyword_in_fusion_count": len(fusion & keyword),
        "bm25_overlap_fusion_count": len(fusion & bm25),
        "dense_keyword_overlap_count": len(dense & keyword),
        "dense_bm25_overlap_count": len(dense & bm25),
        "keyword_bm25_overlap_count": len(keyword & bm25),
    }
    if expected:
        payload.update(
            {
                "expected_count": len(expected),
                "dense_expected_count": len(expected & dense),
                "keyword_expected_count": len(expected & keyword),
                "bm25_expected_count": len(expected & bm25),
                "fusion_expected_count": len(expected & fusion),
                "dense_expected_lost_by_fusion": sorted(expected & dense - fusion),
                "keyword_expected_lost_by_fusion": sorted(expected & keyword - fusion),
                "bm25_expected_not_in_fusion": sorted(expected & bm25 - fusion),
            }
        )
    return payload


def _retrieval_plan(case: dict[str, Any]) -> dict[str, Any]:
    task = str(case.get("task") or "")
    probe = _as_dict(case.get("probe"))
    if task == "retrieve_explicit":
        final_limit = max(1, _safe_int(probe.get("limit"), 8))
        candidate_top_k = max(final_limit, EXPLICIT_VECTOR_TOP_K)
        return {
            "task": task,
            "final_limit": final_limit,
            "candidate_top_k": candidate_top_k,
            "memory_types": _memory_types(case),
            "scope_channel": None,
            "scope_chat_id": None,
            "require_scope_match": False,
            "score_threshold": EXPLICIT_VECTOR_SCORE_THRESHOLD,
        }
    final_limit = max(1, _safe_int(probe.get("top_k"), 8))
    scope_channel, scope_chat_id = _scope(case)
    return {
        "task": task,
        "final_limit": final_limit,
        "candidate_top_k": final_limit,
        "memory_types": None,
        "scope_channel": scope_channel,
        "scope_chat_id": scope_chat_id,
        "require_scope_match": False,
        "score_threshold": None,
    }


async def _run_lanes(
    retriever: Any,
    case: dict[str, Any],
    *,
    query: str,
) -> dict[str, Any]:
    plan = _retrieval_plan(case)
    task = plan["task"]
    if task not in {"retrieve_explicit", "retrieve_context"}:
        raise ValueError(f"unsupported direct retrieval task: {task}")

    candidate_top_k = int(plan["candidate_top_k"])
    final_limit = int(plan["final_limit"])
    score_threshold = (
        retriever._score_threshold  # type: ignore[attr-defined]
        if plan["score_threshold"] is None
        else float(plan["score_threshold"])
    )
    dense_hits = await retriever._retrieve_vector_lanes(  # type: ignore[attr-defined]
        [query],
        actual_top_k=candidate_top_k,
        memory_types=plan["memory_types"],
        score_threshold=score_threshold,
        scope_channel=plan["scope_channel"],
        scope_chat_id=plan["scope_chat_id"],
        require_scope_match=bool(plan["require_scope_match"]),
        time_start=None,
        time_end=None,
    )
    keyword_hits = retriever._retrieve_keyword_lane(  # type: ignore[attr-defined]
        query,
        actual_top_k=candidate_top_k,
        memory_types=plan["memory_types"],
        scope_channel=plan["scope_channel"],
        scope_chat_id=plan["scope_chat_id"],
        require_scope_match=bool(plan["require_scope_match"]),
        time_start=None,
        time_end=None,
    )
    sparse_limit = max(SPARSE_LIMIT_FLOOR, candidate_top_k * SPARSE_LIMIT_MULTIPLIER)
    bm25_hits, bm25_terms = _bm25_summary_search(
        retriever._store,  # type: ignore[attr-defined]
        query,
        top_k=sparse_limit,
        memory_types=plan["memory_types"],
        scope_channel=plan["scope_channel"],
        scope_chat_id=plan["scope_chat_id"],
        require_scope_match=bool(plan["require_scope_match"]),
    )
    fusion_hits = await retriever.retrieve(
        query,
        memory_types=plan["memory_types"],
        top_k=candidate_top_k if task == "retrieve_explicit" else final_limit,
        scope_channel=plan["scope_channel"],
        scope_chat_id=plan["scope_chat_id"],
        require_scope_match=bool(plan["require_scope_match"]),
        score_threshold=score_threshold,
        keyword_enabled=True,
    )
    output_hits = list(fusion_hits)[:final_limit]
    injected_ids: list[str] = []
    if task == "retrieve_context":
        _text_block, injected_ids = retriever.build_injection_block(output_hits)

    return {
        "plan": {
            "final_limit": final_limit,
            "dense_candidate_top_k": candidate_top_k,
            "keyword_candidate_limit": max(
                SPARSE_LIMIT_FLOOR,
                candidate_top_k * SPARSE_LIMIT_MULTIPLIER,
            ),
            "bm25_candidate_limit": sparse_limit,
            "score_threshold": score_threshold,
        },
        "dense": {
            "algorithm": "memory2.vector_cosine_hotness",
            "ids": _hit_ids(dense_hits),
            "hits": _compact_hits(dense_hits),
        },
        "keyword": {
            "algorithm": "memory2.summary_like_terms",
            "ids": _hit_ids(keyword_hits),
            "hits": _compact_hits(keyword_hits),
        },
        "bm25_summary": {
            "algorithm": "eval.memory_items_summary_bm25",
            "terms": bm25_terms,
            "ids": _hit_ids(bm25_hits),
            "hits": _compact_hits(bm25_hits),
        },
        "fusion": {
            "algorithm": "memory2.rrf_dense_keyword",
            "ids": _hit_ids(output_hits),
            "hits": _compact_hits(output_hits),
        },
        "injected_ids": injected_ids,
    }


async def _eval_direct_case(
    retriever: Any,
    case_path: Path,
    *,
    active_ids: set[str],
) -> dict[str, Any]:
    case = _read_json(case_path)
    case_id = str(case.get("case_id") or case_path.stem)
    task = str(case.get("task") or "")
    query = _case_query(case)
    expected_ids = _expected_ids(case)
    result: dict[str, Any] = {
        "kind": "badcase_lane_eval_case",
        "case_id": case_id,
        "case_key": case_path.stem,
        "case_path": str(case_path),
        "failure_type": str(case.get("failure_type") or ""),
        "task": task,
        "query": query,
        "probe_search_mode": str(_as_dict(case.get("probe")).get("search_mode") or ""),
        "expected_memory_ids": expected_ids,
        "expected_missing_in_db": sorted(set(expected_ids) - active_ids),
        "include_in_pr_summary": bool(expected_ids),
    }
    if not query:
        result.update({"status": "skipped", "reason": "case has no retrieval query"})
        return result

    try:
        lanes = await _run_lanes(retriever, case, query=query)
    except Exception as exc:
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return result

    ids_by_lane = {
        lane: lanes[lane]["ids"]
        for lane in ("dense", "keyword", "bm25_summary", "fusion")
    }
    metrics = {
        lane: _lane_metrics(ids, expected_ids)
        for lane, ids in ids_by_lane.items()
    }
    result.update(
        {
            "status": "ok",
            "plan": lanes["plan"],
            "metrics": metrics,
            "category": _classify_chain(
                dense_ids=ids_by_lane["dense"],
                keyword_ids=ids_by_lane["keyword"],
                metrics=metrics,
                expected_ids=expected_ids,
                task=task,
            ),
            "bm25_category": _classify_bm25(metrics, expected_ids, task),
            "contribution": _contribution(
                fusion_ids=ids_by_lane["fusion"],
                dense_ids=ids_by_lane["dense"],
                keyword_ids=ids_by_lane["keyword"],
                bm25_ids=ids_by_lane["bm25_summary"],
                expected_ids=expected_ids,
            ),
            "dense": lanes["dense"],
            "keyword": lanes["keyword"],
            "bm25_summary": lanes["bm25_summary"],
            "fusion": lanes["fusion"],
            "injected_ids": lanes["injected_ids"],
        }
    )
    return result


async def _eval_sticky_case(
    retriever: Any,
    case_path: Path,
    *,
    active_ids: set[str],
) -> dict[str, Any]:
    case = _read_json(case_path)
    case_id = str(case.get("case_id") or case_path.stem)
    target_ids = _expected_ids(case) or _unique_ids([str(_as_dict(case.get("probe")).get("memory_id") or "")])
    turns = [
        turn
        for turn in _as_dict(case.get("observed")).get("sample_turns") or []
        if isinstance(turn, dict)
    ]
    result: dict[str, Any] = {
        "kind": "badcase_lane_eval_case",
        "case_id": case_id,
        "case_key": case_path.stem,
        "case_path": str(case_path),
        "failure_type": str(case.get("failure_type") or ""),
        "task": str(case.get("task") or ""),
        "query": "",
        "expected_memory_ids": target_ids,
        "expected_missing_in_db": sorted(set(target_ids) - active_ids),
        "include_in_pr_summary": False,
        "metric_semantics": "sticky target presence, not gold relevance",
    }
    per_turn: list[dict[str, Any]] = []
    for index, turn in enumerate(turns, 1):
        query = str(turn.get("user_text") or "").strip()
        if not query:
            continue
        synthetic_case = {
            "task": "retrieve_context",
            "probe": {"query": query, "top_k": 8},
            "turn": {
                "channel": turn.get("channel") or "",
                "chat_id": turn.get("chat_id") or "",
            },
        }
        try:
            lanes = await _run_lanes(retriever, synthetic_case, query=query)
        except Exception as exc:
            per_turn.append(
                {
                    "index": index,
                    "turn_id": turn.get("turn_id"),
                    "query": query,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        ids_by_lane = {
            lane: lanes[lane]["ids"]
            for lane in ("dense", "keyword", "bm25_summary", "fusion")
        }
        metrics = {
            lane: _lane_metrics(ids, target_ids)
            for lane, ids in ids_by_lane.items()
        }
        per_turn.append(
            {
                "index": index,
                "turn_id": turn.get("turn_id"),
                "query": query,
                "status": "ok",
                "metrics": metrics,
                "category": _classify_chain(
                    dense_ids=ids_by_lane["dense"],
                    keyword_ids=ids_by_lane["keyword"],
                    metrics=metrics,
                    expected_ids=target_ids,
                    task="retrieve_context",
                ),
                "bm25_category": _classify_bm25(metrics, target_ids, "retrieve_context"),
                "contribution": _contribution(
                    fusion_ids=ids_by_lane["fusion"],
                    dense_ids=ids_by_lane["dense"],
                    keyword_ids=ids_by_lane["keyword"],
                    bm25_ids=ids_by_lane["bm25_summary"],
                    expected_ids=target_ids,
                ),
                "dense_ids": ids_by_lane["dense"],
                "keyword_ids": ids_by_lane["keyword"],
                "bm25_summary_ids": ids_by_lane["bm25_summary"],
                "fusion_ids": ids_by_lane["fusion"],
            }
        )

    ok_turns = [turn for turn in per_turn if turn.get("status") == "ok"]
    aggregate_metrics: dict[str, dict[str, Any]] = {}
    for lane in ("dense", "keyword", "bm25_summary", "fusion"):
        presence = [
            1
            for turn in ok_turns
            if int(_as_dict(_as_dict(turn.get("metrics")).get(lane)).get("matched_count") or 0) > 0
        ]
        aggregate_metrics[lane] = {
            "sample_turn_count": len(ok_turns),
            "target_presence_count": len(presence),
            "target_presence_rate": round(len(presence) / len(ok_turns), 6) if ok_turns else 0.0,
        }
    result.update(
        {
            "status": "ok",
            "sample_turns": per_turn,
            "metrics": aggregate_metrics,
            "category": "sticky_presence_diagnostic",
            "bm25_category": "sticky_presence_diagnostic",
        }
    )
    return result


def _source_workspace_for_case(case: dict[str, Any], fallback: Path) -> Path:
    source_workspace = str(case.get("source_workspace") or "").strip()
    if source_workspace:
        return Path(source_workspace).expanduser().resolve()
    memory_db = str(_as_dict(case.get("review_set")).get("memory_db") or "").strip()
    if memory_db:
        db_path = Path(memory_db).expanduser()
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path
        return db_path.resolve().parent.parent
    return fallback.expanduser().resolve()


def _workspace_slug(path: Path) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.as_posix()).strip("_")
    return text[-80:] if len(text) > 80 else text


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _format_metric(value: Any) -> str:
    return "N/A" if value is None else str(value)


def _markdown_text(value: Any, *, max_length: int = 64) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    return f"{text[:max_length]}..." if len(text) > max_length else text


def _score_for_sort(result: dict[str, Any]) -> tuple[float, float, float, str]:
    metrics = _as_dict(result.get("metrics"))
    fusion = _as_dict(metrics.get("fusion"))
    f1 = fusion.get("f1@topn")
    recall = fusion.get("recall@topn")
    precision = fusion.get("precision@topn")
    return (
        float(f1) if isinstance(f1, int | float) else 2.0,
        float(recall) if isinstance(recall, int | float) else 2.0,
        float(precision) if isinstance(precision, int | float) else 2.0,
        str(result.get("case_id") or ""),
    )


def _lane_cells(metrics: dict[str, Any]) -> list[str]:
    return [
        str(metrics.get("top_n") if metrics.get("top_n") is not None else "N/A"),
        str(metrics.get("matched_count") if metrics.get("matched_count") is not None else "N/A"),
        _format_metric(metrics.get("precision@topn")),
        _format_metric(metrics.get("recall@topn")),
        _format_metric(metrics.get("f1@topn")),
    ]


def _write_markdown(path: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gold_results = [
        result
        for result in results
        if result.get("status") == "ok" and result.get("include_in_pr_summary")
    ]
    worst = sorted(gold_results, key=_score_for_sort)
    rows = [
        "# Badcase Lane Precision/Recall",
        "",
        f"Cases: `{summary['total_cases']}`. Gold PR cases: `{summary['gold_pr_case_count']}`.",
        "",
        "Metrics use each lane's actual returned `N`: `precision@topn`, `recall@topn`, `f1@topn`.",
        "Sticky-memory cases are reported as target-presence diagnostics and are excluded from PR aggregates.",
        "",
        "## Aggregate",
        "",
        "| Lane | Macro P@N | Macro R@N | Macro F1@N | Avg N |",
        "|---|---:|---:|---:|---:|",
    ]
    for lane, metrics in _as_dict(summary.get("gold_pr_metrics")).items():
        rows.append(
            "| "
            + " | ".join(
                [
                    lane,
                    _format_metric(_as_dict(metrics).get("macro_precision@topn")),
                    _format_metric(_as_dict(metrics).get("macro_recall@topn")),
                    _format_metric(_as_dict(metrics).get("macro_f1@topn")),
                    _format_metric(_as_dict(metrics).get("avg_returned_n")),
                ]
            )
            + " |"
        )

    rows.extend(
        [
            "",
            "## Worst Gold Cases",
            "",
            "| # | Category | BM25 Category | Type | Case Key | Query | Gold | Missing DB | Dense N/H/P/R/F1 | Keyword N/H/P/R/F1 | BM25 N/H/P/R/F1 | Fusion N/H/P/R/F1 |",
            "|---:|---|---|---|---|---|---:|---:|---|---|---|---|",
        ]
    )
    for index, result in enumerate(worst, 1):
        metrics = _as_dict(result.get("metrics"))
        rows.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    _markdown_text(result.get("category"), max_length=48),
                    _markdown_text(result.get("bm25_category"), max_length=48),
                    _markdown_text(result.get("failure_type"), max_length=36),
                    f"`{_markdown_text(result.get('case_key') or result.get('case_id'), max_length=140)}`",
                    _markdown_text(result.get("query")),
                    str(len(result.get("expected_memory_ids") or [])),
                    str(len(result.get("expected_missing_in_db") or [])),
                    "/".join(_lane_cells(_as_dict(metrics.get("dense")))),
                    "/".join(_lane_cells(_as_dict(metrics.get("keyword")))),
                    "/".join(_lane_cells(_as_dict(metrics.get("bm25_summary")))),
                    "/".join(_lane_cells(_as_dict(metrics.get("fusion")))),
                ]
            )
            + " |"
        )

    no_gold = [
        result
        for result in results
        if result.get("status") == "ok"
        and not result.get("include_in_pr_summary")
        and result.get("task") != "injection_frequency_review"
    ]
    if no_gold:
        rows.extend(
            [
                "",
                "## No Gold Cases",
                "",
                "| # | Type | Case Key | Query | Dense N | Keyword N | BM25 N | Fusion N |",
                "|---:|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for index, result in enumerate(no_gold, 1):
            metrics = _as_dict(result.get("metrics"))
            rows.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        _markdown_text(result.get("failure_type"), max_length=36),
                        f"`{_markdown_text(result.get('case_key') or result.get('case_id'), max_length=140)}`",
                        _markdown_text(result.get("query")),
                        str(_as_dict(metrics.get("dense")).get("top_n")),
                        str(_as_dict(metrics.get("keyword")).get("top_n")),
                        str(_as_dict(metrics.get("bm25_summary")).get("top_n")),
                        str(_as_dict(metrics.get("fusion")).get("top_n")),
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
                "| # | Case Key | Target | Dense | Keyword | BM25 | Fusion |",
                "|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for index, result in enumerate(sticky, 1):
            metrics = _as_dict(result.get("metrics"))
            rows.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        f"`{_markdown_text(result.get('case_key') or result.get('case_id'), max_length=140)}`",
                        ",".join(result.get("expected_memory_ids") or []),
                        _format_metric(_as_dict(metrics.get("dense")).get("target_presence_rate")),
                        _format_metric(_as_dict(metrics.get("keyword")).get("target_presence_rate")),
                        _format_metric(_as_dict(metrics.get("bm25_summary")).get("target_presence_rate")),
                        _format_metric(_as_dict(metrics.get("fusion")).get("target_presence_rate")),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _avg(results: list[dict[str, Any]], lane: str, key: str) -> float | None:
    values: list[float] = []
    for result in results:
        value = _as_dict(_as_dict(result.get("metrics")).get(lane)).get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return round(mean(values), 6) if values else None


def _summarize(
    results: list[dict[str, Any]],
    *,
    embedder: EvalEmbedder,
    sandbox_workspaces: dict[str, str],
) -> dict[str, Any]:
    ok = [result for result in results if result.get("status") == "ok"]
    gold = [result for result in ok if result.get("include_in_pr_summary")]
    category_counts = Counter(str(result.get("category") or "") for result in ok)
    bm25_category_counts = Counter(str(result.get("bm25_category") or "") for result in ok)
    return {
        "kind": "badcase_lane_eval_summary",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(results),
        "status_counts": dict(Counter(str(result.get("status") or "") for result in results)),
        "failure_type_counts": dict(Counter(str(result.get("failure_type") or "") for result in results)),
        "gold_pr_case_count": len(gold),
        "no_expected_case_count": len([result for result in ok if not result.get("expected_memory_ids")]),
        "sticky_case_count": len([result for result in ok if result.get("task") == "injection_frequency_review"]),
        "expected_missing_in_db_case_count": len([result for result in ok if result.get("expected_missing_in_db")]),
        "category_counts": dict(sorted(category_counts.items())),
        "bm25_category_counts": dict(sorted(bm25_category_counts.items())),
        "gold_pr_metrics": {
            lane: {
                "avg_returned_n": _avg(gold, lane, "top_n"),
                "macro_precision@topn": _avg(gold, lane, "precision@topn"),
                "macro_recall@topn": _avg(gold, lane, "recall@topn"),
                "macro_f1@topn": _avg(gold, lane, "f1@topn"),
            }
            for lane in ("dense", "keyword", "bm25_summary", "fusion")
        },
        "sandbox_workspaces": sandbox_workspaces,
        "embedding": {
            "requested_mode": embedder.requested_mode,
            "effective_mode": embedder.effective_mode,
            "model": embedder.settings.model,
            "base_url_present": bool(embedder.settings.base_url),
            "api_key_present": bool(embedder.settings.api_key),
            "calls": embedder.calls,
            "cache_hits": embedder.cache_hits,
            "disabled_reason": embedder.disabled_reason,
        },
    }


def _copy_cases(source_dir: Path, target_dir: Path, *, overwrite: bool) -> dict[str, Any]:
    source_files = sorted(source_dir.glob("*.json"))
    if not source_files:
        raise FileNotFoundError(f"no case json files under {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for source_path in source_files:
        target_path = target_dir / source_path.name
        if target_path.exists() and not overwrite:
            skipped += 1
            continue
        data = json.loads(source_path.read_text(encoding="utf-8"))
        target_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        copied += 1
    manifest = {
        "schema_version": 1,
        "kind": "badcase_with_expectedid_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "source_count": len(source_files),
        "copied_count": copied,
        "skipped_existing_count": skipped,
        "notes": (
            "Private badcase directory for cases with expected.memory_ids. "
            "Cases without expected ids are kept for no-gold/over-recall diagnostics."
        ),
    }
    (target_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate badcases with expected ids across dense, keyword, BM25, and RRF lanes in sandbox DB copies."
    )
    parser.add_argument("--badcase-dir", type=Path, default=DEFAULT_BADCASE_DIR)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--overwrite-init", action="store_true")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--workspace-mode",
        choices=("case-source", "single"),
        default="case-source",
        help="case-source uses each case.source_workspace; single uses --workspace for all cases.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--markdown-report", type=Path, default=None)
    parser.add_argument("--sandbox-root", type=Path, default=DEFAULT_SANDBOX_ROOT)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--memory-config",
        type=Path,
        default=Path("plugins/default_memory/config.local.toml"),
    )
    parser.add_argument(
        "--embedding-mode",
        choices=("auto", "live", "keyword-only"),
        default="live",
        help="live requires configured embedding; keyword-only skips dense retrieval.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser


async def _amain() -> None:
    args = _build_parser().parse_args()
    if not args.verbose:
        logging.getLogger("memory2.retriever").setLevel(logging.ERROR)
        logging.getLogger("memory2.store").setLevel(logging.ERROR)

    if args.init_from:
        manifest = _copy_cases(args.init_from, args.badcase_dir, overwrite=args.overwrite_init)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))

    stamp = _now_stamp()
    report_path = args.report or DEFAULT_REPORT_DIR / f"badcase_lane_eval_{stamp}.jsonl"
    markdown_report_path = args.markdown_report or report_path.with_suffix(".md")
    sandbox_base = args.sandbox_root / f"badcase_lane_{stamp}"

    case_paths = sorted(args.badcase_dir.glob("*.json"))
    case_paths = [path for path in case_paths if not path.name.startswith("_")]
    if args.limit:
        case_paths = case_paths[: max(0, args.limit)]
    if not case_paths:
        raise FileNotFoundError(f"no case json files under {args.badcase_dir}")

    fallback_workspace = args.workspace.expanduser().resolve()
    workspace_groups: dict[Path, list[Path]] = defaultdict(list)
    for case_path in case_paths:
        case = _read_json(case_path)
        workspace = (
            _source_workspace_for_case(case, fallback_workspace)
            if args.workspace_mode == "case-source"
            else fallback_workspace
        )
        workspace_groups[workspace].append(case_path)

    embedding_settings = _load_embedding_settings(args.config)
    if args.embedding_mode == "live" and (not embedding_settings.base_url or not embedding_settings.api_key):
        raise RuntimeError("live embedding requested, but embedding base_url/api_key is missing")
    retrieval_settings = _load_retrieval_settings(args.memory_config)
    embedder = EvalEmbedder(mode=args.embedding_mode, settings=embedding_settings)

    results: list[dict[str, Any]] = []
    sandbox_workspaces: dict[str, str] = {}
    try:
        for workspace, paths in sorted(workspace_groups.items(), key=lambda item: item[0].as_posix()):
            sandbox_workspace = sandbox_base / _workspace_slug(workspace)
            db_path = _copy_memory_db_to_sandbox(workspace, sandbox_workspace)
            active_ids = _active_memory_ids(db_path)
            sandbox_workspaces[workspace.as_posix()] = sandbox_workspace.as_posix()
            store, retriever = _build_retriever(db_path, embedder, retrieval_settings)
            try:
                for case_path in paths:
                    case = _read_json(case_path)
                    task = str(case.get("task") or "")
                    if task == "injection_frequency_review":
                        result = await _eval_sticky_case(retriever, case_path, active_ids=active_ids)
                    else:
                        result = await _eval_direct_case(retriever, case_path, active_ids=active_ids)
                    result["workspace"] = workspace.as_posix()
                    result["sandbox_workspace"] = sandbox_workspace.as_posix()
                    result["index"] = len(results) + 1
                    result["embedding_effective_mode"] = embedder.effective_mode
                    results.append(result)
            finally:
                store.close()
    finally:
        await embedder.aclose()

    summary = _summarize(results, embedder=embedder, sandbox_workspaces=sandbox_workspaces)
    _write_jsonl(report_path, [*results, summary])
    _write_markdown(markdown_report_path, results, summary)
    print(f"report: {report_path}")
    print(f"markdown_report: {markdown_report_path}")
    print(f"sandbox_base: {sandbox_base}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.embedding_mode == "live" and embedder.effective_mode != "live":
        raise RuntimeError(
            "live embedding was requested but became unavailable: "
            f"{embedder.disabled_reason or 'unknown embedding failure'}"
        )


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
