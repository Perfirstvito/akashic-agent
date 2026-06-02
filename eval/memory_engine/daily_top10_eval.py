from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from eval.memory_engine.daily_badcase_eval import (
    DEFAULT_REPORT_DIR,
    DEFAULT_SANDBOX_ROOT,
    DEFAULT_WORKSPACE,
    EXPLICIT_VECTOR_SCORE_THRESHOLD,
    EvalEmbedder,
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


DEFAULT_BADCASE_DIR = Path("eval/memory_engine/badcases/daily_deduped")
TOP_K = 10


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _format_metric(value: Any) -> str:
    return "N/A" if value is None else str(value)


def _markdown_text(value: Any, *, max_length: int = 56) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    return f"{text[:max_length]}..." if len(text) > max_length else text


def _write_markdown(
    path: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Daily Dense And BM25 Top-N Retrieval Metrics",
        "",
        f"Retrieval limit: `{summary['retrieval_limit']}`. Each lane's `N` is its actual returned count.",
        "",
        "`N/A` means the case has empty expected memory ids, so recall and F1 are undefined.",
        "For empty-gold cases, `N = 0` means the lane correctly rejected long-term memory.",
        "",
        "| # | Type | Case ID | Query | Gold | Dense N | Dense Hits | Dense P@N | Dense R@N | Dense F1@N | BM25 N | BM25 Hits | BM25 P@N | BM25 R@N | BM25 F1@N |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ok_results = [result for result in results if result.get("status") == "ok"]
    for index, result in enumerate(ok_results, 1):
        dense = result["metrics"]["dense"]
        bm25 = result["metrics"]["bm25_summary"]
        rows.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    _markdown_text(result.get("failure_type")),
                    f"`{_markdown_text(result.get('case_id'), max_length=120)}`",
                    _markdown_text(result.get("query")),
                    str(len(result.get("expected_memory_ids") or [])),
                    str(dense["top_n"]),
                    str(dense["matched_count"]),
                    _format_metric(dense["precision@topn"]),
                    _format_metric(dense["recall@topn"]),
                    _format_metric(dense["f1@topn"]),
                    str(bm25["top_n"]),
                    str(bm25["matched_count"]),
                    _format_metric(bm25["precision@topn"]),
                    _format_metric(bm25["recall@topn"]),
                    _format_metric(bm25["f1@topn"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _expected_ids(case: dict[str, Any]) -> list[str]:
    expected = case.get("expected")
    if not isinstance(expected, dict):
        return []
    return [
        str(item_id)
        for item_id in expected.get("memory_ids") or []
        if str(item_id).strip()
    ]


def _compact_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(hit.get("id") or ""),
            "memory_type": str(hit.get("memory_type") or ""),
            "score": hit.get("score"),
            "semantic_score": (hit.get("_score_debug") or {}).get("semantic"),
            "bm25_score": hit.get("bm25_score"),
            "summary": str(hit.get("summary") or ""),
        }
        for hit in hits
    ]


def _metrics(hit_ids: list[str], expected_ids: list[str]) -> dict[str, Any]:
    unique_hits = list(dict.fromkeys(hit_ids[:TOP_K]))
    expected = set(expected_ids)
    matched = [item_id for item_id in unique_hits if item_id in expected]
    if not expected:
        return {
            "top_n": len(unique_hits),
            "expected_count": 0,
            "returned_count": len(unique_hits),
            "matched_count": 0,
            "matched_ids": [],
            "false_positive_count": len(unique_hits),
            "empty_gold_correct": not unique_hits,
            "precision@topn": 0.0 if unique_hits else None,
            "recall@topn": None,
            "f1@topn": None,
        }
    precision = len(matched) / len(unique_hits) if unique_hits else 0.0
    recall = len(set(matched)) / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "top_n": len(unique_hits),
        "expected_count": len(expected),
        "returned_count": len(unique_hits),
        "matched_count": len(matched),
        "matched_ids": matched,
        "false_positive_count": len(unique_hits) - len(matched),
        "empty_gold_correct": None,
        "precision@topn": round(precision, 6),
        "recall@topn": round(recall, 6),
        "f1@topn": round(f1, 6),
    }


def _avg(results: list[dict[str, Any]], lane: str, key: str) -> float | None:
    values = [
        result["metrics"][lane][key]
        for result in results
        if result["metrics"][lane][key] is not None
    ]
    return round(mean(values), 6) if values else None


def _aggregate(results: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    return {
        lane: {
            "avg_returned_n": _avg(results, lane, "top_n"),
            "macro_precision@topn": _avg(results, lane, "precision@topn"),
            "macro_recall@topn": _avg(results, lane, "recall@topn"),
            "macro_f1@topn": _avg(results, lane, "f1@topn"),
        }
        for lane in ("dense", "bm25_summary")
    }


async def _eval_case(
    retriever: Any,
    case_path: Path,
) -> dict[str, Any]:
    case = _read_json(case_path)
    case_id = str(case.get("case_id") or case_path.stem)
    failure_type = str(case.get("failure_type") or "")
    task = str(case.get("task") or "")
    query = _case_query(case)
    expected_ids = _expected_ids(case)
    result: dict[str, Any] = {
        "kind": "top10_eval_case",
        "case_id": case_id,
        "failure_type": failure_type,
        "task": task,
        "case_path": str(case_path),
        "query": query,
        "expected_memory_ids": expected_ids,
    }
    if not query:
        result.update({"status": "skipped", "reason": "case has no retrieval query"})
        return result

    is_explicit = task == "retrieve_explicit"
    memory_types = _memory_types(case) if is_explicit else None
    scope_channel, scope_chat_id = (None, None) if is_explicit else _scope(case)
    threshold = (
        EXPLICIT_VECTOR_SCORE_THRESHOLD
        if is_explicit
        else retriever._score_threshold  # type: ignore[attr-defined]
    )
    try:
        query_vec = await retriever.embed(query)
        store = retriever._store  # type: ignore[attr-defined]
        dense_hits = store.vector_search(
            query_vec=query_vec,
            top_k=TOP_K,
            memory_types=memory_types,
            score_threshold=threshold,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            require_scope_match=False,
            hotness_alpha=retriever._hotness_alpha,  # type: ignore[attr-defined]
            hotness_half_life_days=retriever._hotness_half_life_days,  # type: ignore[attr-defined]
        )
        bm25_hits, bm25_terms = _bm25_summary_search(
            store,
            query,
            top_k=TOP_K,
            memory_types=memory_types,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            require_scope_match=False,
        )
        dense_ids = [str(hit.get("id") or "") for hit in dense_hits if hit.get("id")]
        bm25_ids = [str(hit.get("id") or "") for hit in bm25_hits if hit.get("id")]
        result.update(
            {
                "status": "ok",
                "dense": {
                    "algorithm": "memory2.vector_cosine_hotness",
                    "score_threshold": threshold,
                    "ids": dense_ids,
                    "hits": _compact_hits(dense_hits),
                },
                "bm25_summary": {
                    "algorithm": "eval.memory_items_summary_bm25",
                    "terms": bm25_terms,
                    "ids": bm25_ids,
                    "hits": _compact_hits(bm25_hits),
                },
                "metrics": {
                    "dense": _metrics(dense_ids, expected_ids),
                    "bm25_summary": _metrics(bm25_ids, expected_ids),
                },
            }
        )
        return result
    except Exception as exc:
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return result


def _summarize(
    results: list[dict[str, Any]],
    embedder: EvalEmbedder,
    sandbox_workspace: Path,
) -> dict[str, Any]:
    ok = [result for result in results if result.get("status") == "ok"]
    all_gold = [result for result in ok if result.get("expected_memory_ids")]
    diagnostic = [
        result
        for result in all_gold
        if result.get("failure_type") != "positive_control"
    ]
    empty_gold = [
        result
        for result in ok
        if not result.get("expected_memory_ids")
        and result.get("failure_type") == "short_query_over_recall"
    ]
    return {
        "kind": "top10_eval_summary",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sandbox_workspace": str(sandbox_workspace),
        "retrieval_limit": TOP_K,
        "metrics_scope": "per-lane actual returned top_n",
        "total_cases": len(results),
        "status_counts": dict(Counter(str(result.get("status") or "") for result in results)),
        "all_gold_case_count": len(all_gold),
        "all_gold_metrics": _aggregate(all_gold),
        "diagnostic_strict_case_count": len(diagnostic),
        "diagnostic_strict_metrics": _aggregate(diagnostic),
        "empty_gold_over_recall_case_count": len(empty_gold),
        "empty_gold_false_positive_avg": {
            lane: round(mean(result["metrics"][lane]["false_positive_count"] for result in empty_gold), 6)
            if empty_gold
            else 0.0
            for lane in ("dense", "bm25_summary")
        },
        "embedding": {
            "requested_mode": embedder.requested_mode,
            "effective_mode": embedder.effective_mode,
            "model": embedder.settings.model,
            "calls": embedder.calls,
            "cache_hits": embedder.cache_hits,
            "disabled_reason": embedder.disabled_reason,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure dense-only and BM25-only Top-N retrieval, capped at 10, in an isolated sandbox."
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--badcase-dir", type=Path, default=DEFAULT_BADCASE_DIR)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--markdown-report", type=Path, default=None)
    parser.add_argument("--sandbox-root", type=Path, default=DEFAULT_SANDBOX_ROOT)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--memory-config",
        type=Path,
        default=Path("plugins/default_memory/config.local.toml"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser


async def _amain() -> None:
    args = _build_parser().parse_args()
    if not args.verbose:
        logging.getLogger("memory2.retriever").setLevel(logging.ERROR)
        logging.getLogger("memory2.store").setLevel(logging.ERROR)
    stamp = _now_stamp()
    report_path = args.report or DEFAULT_REPORT_DIR / f"daily_top10_eval_{stamp}.jsonl"
    markdown_report_path = args.markdown_report or report_path.with_suffix(".md")
    sandbox_workspace = args.sandbox_root / f"daily_top10_{stamp}"
    cases = sorted(args.badcase_dir.glob("*.json"))
    if args.limit:
        cases = cases[: max(0, args.limit)]
    if not cases:
        raise FileNotFoundError(f"no case json files under {args.badcase_dir}")

    db_path = _copy_memory_db_to_sandbox(args.workspace.expanduser().resolve(), sandbox_workspace)
    embedding_settings = _load_embedding_settings(args.config)
    retrieval_settings = _load_retrieval_settings(args.memory_config)
    embedder = EvalEmbedder(mode="live", settings=embedding_settings)
    store, retriever = _build_retriever(db_path, embedder, retrieval_settings)
    results: list[dict[str, Any]] = []
    try:
        for index, case_path in enumerate(cases, 1):
            result = await _eval_case(retriever, case_path)
            result["index"] = index
            results.append(result)
    finally:
        store.close()
        await embedder.aclose()

    if embedder.effective_mode != "live":
        raise RuntimeError(
            "live embedding is required for dense top-10 evaluation: "
            f"{embedder.disabled_reason or 'embedding is unavailable'}"
        )
    summary = _summarize(results, embedder, sandbox_workspace)
    _write_jsonl(report_path, [*results, summary])
    _write_markdown(markdown_report_path, results, summary)
    print(f"report: {report_path}")
    print(f"markdown_report: {markdown_report_path}")
    print(f"sandbox_workspace: {sandbox_workspace}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
