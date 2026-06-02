from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 local fallback
    import tomli as tomllib  # type: ignore[no-redef]

from memory2.embedder import Embedder
from memory2.retriever import Retriever
from memory2.store import MemoryStore2


DEFAULT_WORKSPACE = Path.home() / ".akashic" / "workspace"
DEFAULT_BADCASE_DIR = Path("eval/memory_engine/badcases/daily")
DEFAULT_REPORT_DIR = Path("eval/memory_engine/reports")
DEFAULT_SANDBOX_ROOT = Path("eval/memory_engine/sandbox")

EXPLICIT_VECTOR_TOP_K = 15
EXPLICIT_VECTOR_SCORE_THRESHOLD = 0.35
LANE_TRACE_MAX_HITS = 20
BM25_K1 = 1.5
BM25_B = 0.75

_CJK_STOPWORDS = {
    "用户", "助手", "我们", "他们", "这个", "那个", "什么", "如何", "是否",
    "有没", "没有", "有过", "做过", "进行", "完成", "包括", "通过", "实现",
    "行为", "内容", "相关", "情况", "问题", "方式", "时候", "时间", "目前",
    "当前", "最近", "之前", "以前", "后来", "然后", "因为", "所以", "但是",
    "用户在", "用户对", "的行为吗", "进行了",
}


@dataclass(frozen=True)
class RetrievalSettings:
    top_k_history: int = 8
    score_threshold: float = 0.45
    relative_delta: float = 0.2
    thresholds: dict[str, float] | None = None
    inject_max_chars: int = 6000
    inject_max_forced: int = 3
    inject_max_procedure_preference: int = 4
    inject_max_event_profile: int = 4
    inject_line_max: int = 600
    procedure_guard_enabled: bool = True


@dataclass(frozen=True)
class EmbeddingSettings:
    base_url: str = ""
    api_key: str = ""
    model: str = "text-embedding-v3"


class EvalEmbedder:
    def __init__(
        self,
        *,
        mode: str,
        settings: EmbeddingSettings,
    ) -> None:
        self.requested_mode = mode
        self.settings = settings
        self.calls = 0
        self.cache_hits = 0
        self.disabled_reason = ""
        self._cache: dict[str, list[float]] = {}
        self._live: Embedder | None = None
        self._requester: SimpleHttpRequester | None = None

        if mode == "keyword-only":
            self.disabled_reason = "embedding mode is keyword-only"
            return
        if not settings.base_url or not settings.api_key:
            self.disabled_reason = "embedding base_url/api_key is missing"
            return
        self._requester = SimpleHttpRequester()
        self._live = Embedder(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=settings.model,
            requester=self._requester,  # type: ignore[arg-type]
        )

    @property
    def effective_mode(self) -> str:
        return "live" if self._live is not None and not self.disabled_reason else "keyword-only"

    async def embed(self, text: str) -> list[float]:
        key = str(text or "")
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        if self._live is None or self.disabled_reason:
            raise RuntimeError(self.disabled_reason or "embedding unavailable")
        self.calls += 1
        try:
            vector = await self._live.embed(key)
        except Exception as exc:  # Retriever catches this and keeps keyword lane.
            self.disabled_reason = f"{type(exc).__name__}: {str(exc)[:240]}"
            self._live = None
            raise
        self._cache[key] = vector
        return vector

    async def aclose(self) -> None:
        if self._requester is not None:
            await self._requester.aclose()


class SimpleHttpRequester:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient()

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        **_kwargs: Any,
    ) -> httpx.Response:
        return await self._client.post(
            url,
            headers=headers,
            json=json,
            timeout=timeout_s,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def _resolve_config_value(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    resolved = re.sub(
        r"\$\{(\w+)\}",
        lambda match: os.environ.get(match.group(1), match.group(0)),
        text,
    )
    unresolved = re.fullmatch(r"\$\{(\w+)\}", resolved)
    if unresolved:
        key_file = DEFAULT_WORKSPACE / "memory" / unresolved.group(1)
        if key_file.exists():
            return key_file.read_text(encoding="utf-8").strip()
    return resolved


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _load_embedding_settings(config_path: Path) -> EmbeddingSettings:
    data = _load_toml(config_path)
    llm = _as_dict(data.get("llm"))
    main = _as_dict(llm.get("main"))
    fast = _as_dict(llm.get("fast"))
    memory = _as_dict(data.get("memory"))
    embedding = _as_dict(memory.get("embedding"))
    provider = str(llm.get("provider") or data.get("provider") or "").strip()
    presets = {
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "openai": "https://api.openai.com/v1",
    }
    return EmbeddingSettings(
        base_url=(
            str(embedding.get("base_url") or "")
            or str(fast.get("base_url") or "")
            or str(main.get("base_url") or "")
            or str(data.get("light_base_url") or "")
            or str(data.get("base_url") or "")
            or presets.get(provider, "")
        ),
        api_key=_resolve_config_value(
            embedding.get("api_key")
            or fast.get("api_key")
            or main.get("api_key")
            or data.get("light_api_key")
            or data.get("api_key")
            or ""
        ),
        model=str(embedding.get("model") or "text-embedding-v3"),
    )


def _load_retrieval_settings(config_path: Path) -> RetrievalSettings:
    data = _load_toml(config_path)
    retrieval = _as_dict(data.get("retrieval"))
    thresholds = _as_dict(retrieval.get("thresholds"))
    inject = _as_dict(retrieval.get("inject"))
    return RetrievalSettings(
        top_k_history=int(retrieval.get("top_k_history", 8)),
        score_threshold=float(retrieval.get("score_threshold", 0.45)),
        relative_delta=float(retrieval.get("relative_delta", 0.2)),
        thresholds={
            "procedure": float(thresholds.get("procedure", 0.66)),
            "preference": float(thresholds.get("preference", 0.5)),
            "event": float(thresholds.get("event", 0.5)),
            "profile": float(thresholds.get("profile", 0.5)),
        },
        inject_max_chars=int(inject.get("max_chars", 6000)),
        inject_max_forced=int(inject.get("forced", 3)),
        inject_max_procedure_preference=int(inject.get("procedure_preference", 4)),
        inject_max_event_profile=int(inject.get("event_profile", 4)),
        inject_line_max=int(inject.get("line_max", 600)),
        procedure_guard_enabled=bool(retrieval.get("procedure_guard_enabled", True)),
    )


def _copy_memory_db_to_sandbox(source_workspace: Path, sandbox_workspace: Path) -> Path:
    source_db = source_workspace / "memory" / "memory2.db"
    if not source_db.exists():
        raise FileNotFoundError(source_db)
    dest_db = sandbox_workspace / "memory" / "memory2.db"
    dest_db.parent.mkdir(parents=True, exist_ok=True)
    if dest_db.exists():
        dest_db.unlink()

    try:
        src = sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True)
        dst = sqlite3.connect(str(dest_db))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
    except Exception:
        if dest_db.exists():
            dest_db.unlink()
        shutil.copy2(source_db, dest_db)
    return dest_db


def _build_retriever(db_path: Path, embedder: EvalEmbedder, settings: RetrievalSettings) -> tuple[MemoryStore2, Retriever]:
    store = MemoryStore2(db_path)
    retriever = Retriever(
        store,
        embedder,  # type: ignore[arg-type]
        top_k=settings.top_k_history,
        score_threshold=settings.score_threshold,
        score_thresholds=settings.thresholds or {},
        relative_delta=settings.relative_delta,
        inject_max_chars=settings.inject_max_chars,
        inject_max_forced=settings.inject_max_forced,
        inject_max_procedure_preference=settings.inject_max_procedure_preference,
        inject_max_event_profile=settings.inject_max_event_profile,
        inject_line_max=settings.inject_line_max,
        procedure_guard_enabled=settings.procedure_guard_enabled,
        hotness_alpha=0.20,
    )
    return store, retriever


def _case_query(case: dict[str, Any]) -> str:
    probe = _as_dict(case.get("probe"))
    query = str(probe.get("query") or "").strip()
    if query:
        return query
    turn = _as_dict(case.get("turn"))
    return str(turn.get("user_text") or "").strip()


def _memory_types(case: dict[str, Any]) -> list[str] | None:
    memory_type = str(_as_dict(case.get("probe")).get("memory_type") or "").strip()
    return [memory_type] if memory_type else None


def _scope(case: dict[str, Any]) -> tuple[str | None, str | None]:
    turn = _as_dict(case.get("turn"))
    channel = str(turn.get("channel") or "").strip()
    chat_id = str(turn.get("chat_id") or "").strip()
    return channel or None, chat_id or None


def _hit_ids(hits: list[dict[str, Any]]) -> list[str]:
    return [str(hit.get("id") or "") for hit in hits if str(hit.get("id") or "")]


def _compact_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for hit in hits:
        compact.append(
            {
                "id": str(hit.get("id") or ""),
                "memory_type": str(hit.get("memory_type") or ""),
                "score": hit.get("score"),
                "rrf_score": hit.get("rrf_score"),
                "keyword_score": hit.get("keyword_score"),
                "summary": str(hit.get("summary") or ""),
            }
        )
    return compact


def _compact_lane_hits(hits: list[dict[str, Any]], *, max_hits: int = LANE_TRACE_MAX_HITS) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for hit in hits[:max_hits]:
        compact.append(
            {
                "id": str(hit.get("id") or ""),
                "memory_type": str(hit.get("memory_type") or ""),
                "score": hit.get("score"),
                "semantic_score": _as_dict(hit.get("_score_debug")).get("semantic"),
                "keyword_score": hit.get("keyword_score"),
                "bm25_score": hit.get("bm25_score"),
            }
        )
    return compact


def _tokenize_for_bm25(text: str, *, max_terms: int | None = None) -> list[str]:
    terms: list[str] = []
    ascii_tokens = re.findall(r"[a-zA-Z0-9_\-\.]{2,}", text)
    terms.extend(token.lower() for token in ascii_tokens)

    cjk_chunks = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]{2,}", text)
    for chunk in cjk_chunks:
        if len(chunk) <= 4:
            if chunk not in _CJK_STOPWORDS:
                terms.append(chunk)
            continue
        for i in range(len(chunk) - 1):
            bigram = chunk[i:i + 2]
            if bigram not in _CJK_STOPWORDS:
                terms.append(bigram)

    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
        if max_terms is not None and len(deduped) >= max_terms:
            break
    return deduped


def _bm25_summary_search(
    store: MemoryStore2,
    query: str,
    *,
    top_k: int,
    memory_types: list[str] | None,
    scope_channel: str | None,
    scope_chat_id: str | None,
    require_scope_match: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    query_terms = _tokenize_for_bm25(query, max_terms=20)
    if not query_terms:
        return [], []

    sql = """
        SELECT id, memory_type, summary, source_ref, happened_at, extra_json
        FROM memory_items
        WHERE status = 'active'
    """
    params: list[object] = []
    if memory_types:
        placeholders = ",".join("?" for _ in memory_types)
        sql += f" AND memory_type IN ({placeholders})"
        params.extend(memory_types)
    rows = store._db.execute(sql, tuple(params)).fetchall()  # type: ignore[attr-defined]

    docs: list[dict[str, Any]] = []
    scope_channel_value = (scope_channel or "").strip()
    scope_chat_value = (scope_chat_id or "").strip()
    for row in rows:
        item_id, memory_type, summary, source_ref, happened_at, extra_json = row
        extra = _safe_json_object(extra_json)
        if require_scope_match:
            if str(extra.get("scope_channel") or "").strip() != scope_channel_value:
                continue
            if str(extra.get("scope_chat_id") or "").strip() != scope_chat_value:
                continue
        tokens = _tokenize_for_bm25(str(summary or ""))
        if not tokens:
            continue
        docs.append(
            {
                "id": str(item_id),
                "memory_type": str(memory_type),
                "summary": str(summary or ""),
                "source_ref": str(source_ref or ""),
                "happened_at": str(happened_at or ""),
                "tokens": tokens,
            }
        )
    if not docs:
        return [], query_terms

    doc_count = len(docs)
    avg_len = sum(len(doc["tokens"]) for doc in docs) / doc_count
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc["tokens"]))

    scored: list[dict[str, Any]] = []
    for doc in docs:
        term_counts = Counter(doc["tokens"])
        doc_len = len(doc["tokens"])
        score = 0.0
        for term in query_terms:
            tf = term_counts.get(term, 0)
            if tf <= 0:
                continue
            idf = math.log(1.0 + (doc_count - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * doc_len / avg_len)
            score += idf * (tf * (BM25_K1 + 1.0)) / denom
        if score <= 0:
            continue
        scored.append(
            {
                "id": doc["id"],
                "memory_type": doc["memory_type"],
                "summary": doc["summary"],
                "source_ref": doc["source_ref"],
                "happened_at": doc["happened_at"],
                "score": round(score, 6),
                "bm25_score": round(score, 6),
            }
        )
    scored.sort(key=lambda item: float(item.get("bm25_score") or 0.0), reverse=True)
    return scored[:top_k], query_terms


def _safe_json_object(raw: object) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _lane_expected_metrics(lane_ids: list[str], expected_ids: list[str]) -> dict[str, Any]:
    expected = [item_id for item_id in expected_ids if item_id]
    if not expected:
        return {}
    ranks = [
        lane_ids.index(item_id) + 1
        for item_id in expected
        if item_id in lane_ids
    ]
    first_rank = min(ranks) if ranks else None
    metrics: dict[str, Any] = {
        "matched_expected_count": len(ranks),
        "first_rank": first_rank,
        "mrr": round(1.0 / first_rank, 6) if first_rank else 0.0,
    }
    for k in (1, 3, 5, 8):
        top = set(lane_ids[:k])
        matched = sum(1 for item_id in expected if item_id in top)
        metrics[f"hit@{k}"] = matched > 0
        metrics[f"recall@{k}"] = round(matched / len(expected), 6)
        metrics[f"expected_hit_fraction@{k}"] = round(matched / k, 6)
    return metrics


def _lane_contribution(
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
    contribution: dict[str, Any] = {
        "fusion_count": len(fusion_ids),
        "dense_in_fusion_count": len(fusion & dense),
        "keyword_in_fusion_count": len(fusion & keyword),
        "bm25_in_fusion_count": len(fusion & bm25),
        "dense_only_in_fusion_count": len(fusion & dense - keyword),
        "keyword_only_in_fusion_count": len(fusion & keyword - dense),
        "dense_and_keyword_in_fusion_count": len(fusion & dense & keyword),
        "bm25_only_vs_dense_in_fusion_count": len(fusion & bm25 - dense),
        "dense_only_vs_bm25_in_fusion_count": len(fusion & dense - bm25),
    }
    if expected:
        contribution.update(
            {
                "expected_count": len(expected),
                "dense_expected_count": len(expected & dense),
                "keyword_expected_count": len(expected & keyword),
                "bm25_expected_count": len(expected & bm25),
                "fusion_expected_count": len(expected & fusion),
                "dense_only_expected_count": len(expected & dense - keyword),
                "keyword_only_expected_count": len(expected & keyword - dense),
                "dense_and_keyword_expected_count": len(expected & dense & keyword),
                "bm25_only_vs_dense_expected_count": len(expected & bm25 - dense),
                "dense_only_vs_bm25_expected_count": len(expected & dense - bm25),
            }
        )
    return contribution


async def _trace_lanes(
    retriever: Retriever,
    *,
    query: str,
    fusion_hits: list[dict[str, Any]],
    actual_top_k: int,
    memory_types: list[str] | None,
    scope_channel: str | None,
    scope_chat_id: str | None,
    require_scope_match: bool,
    score_threshold: float | None,
    keyword_enabled: bool,
    expected_ids: list[str],
) -> dict[str, Any]:
    store = retriever._store  # type: ignore[attr-defined]
    threshold = (
        retriever._score_threshold  # type: ignore[attr-defined]
        if score_threshold is None
        else float(score_threshold)
    )
    lane_top_k = max(actual_top_k, LANE_TRACE_MAX_HITS)

    dense_hits: list[dict[str, Any]] = []
    dense_error = ""
    try:
        query_vec = await retriever.embed(query)
        dense_hits = store.vector_search(
            query_vec=query_vec,
            top_k=lane_top_k,
            memory_types=memory_types,
            score_threshold=threshold,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            require_scope_match=require_scope_match,
            hotness_alpha=retriever._hotness_alpha,  # type: ignore[attr-defined]
            hotness_half_life_days=retriever._hotness_half_life_days,  # type: ignore[attr-defined]
        )
    except Exception as exc:
        dense_error = f"{type(exc).__name__}: {str(exc)[:240]}"

    keyword_hits: list[dict[str, Any]] = []
    keyword_terms = _tokenize_for_bm25(query, max_terms=20)
    if keyword_enabled:
        try:
            keyword_hits = retriever._retrieve_keyword_lane(  # type: ignore[attr-defined]
                query,
                actual_top_k=actual_top_k,
                memory_types=memory_types,
                scope_channel=scope_channel,
                scope_chat_id=scope_chat_id,
                require_scope_match=require_scope_match,
                time_start=None,
                time_end=None,
            )
        except Exception:
            keyword_hits = []

    bm25_hits, bm25_terms = _bm25_summary_search(
        store,
        query,
        top_k=lane_top_k,
        memory_types=memory_types,
        scope_channel=scope_channel,
        scope_chat_id=scope_chat_id,
        require_scope_match=require_scope_match,
    )

    fusion_ids = _hit_ids(fusion_hits)
    dense_ids = _hit_ids(dense_hits)
    keyword_ids = _hit_ids(keyword_hits)
    bm25_ids = _hit_ids(bm25_hits)
    lane_metrics = {
        "fusion": _lane_expected_metrics(fusion_ids, expected_ids),
        "dense": _lane_expected_metrics(dense_ids, expected_ids),
        "keyword": _lane_expected_metrics(keyword_ids, expected_ids),
        "bm25_summary": _lane_expected_metrics(bm25_ids, expected_ids),
    }

    return {
        "query": query,
        "top_k": actual_top_k,
        "dense": {
            "algorithm": "memory2.vector_cosine_hotness",
            "score_threshold": threshold,
            "error": dense_error,
            "ids": dense_ids[:LANE_TRACE_MAX_HITS],
            "hits": _compact_lane_hits(dense_hits),
        },
        "keyword": {
            "algorithm": "memory2.summary_like_terms",
            "terms": keyword_terms,
            "ids": keyword_ids[:LANE_TRACE_MAX_HITS],
            "hits": _compact_lane_hits(keyword_hits),
        },
        "bm25_summary": {
            "algorithm": "eval.memory_items_summary_bm25",
            "terms": bm25_terms,
            "ids": bm25_ids[:LANE_TRACE_MAX_HITS],
            "hits": _compact_lane_hits(bm25_hits),
        },
        "fusion": {
            "algorithm": "memory2.rrf_dense_keyword",
            "ids": fusion_ids,
            "hits": _compact_lane_hits(fusion_hits),
        },
        "contribution": _lane_contribution(
            fusion_ids=fusion_ids,
            dense_ids=dense_ids,
            keyword_ids=keyword_ids,
            bm25_ids=bm25_ids,
            expected_ids=expected_ids,
        ),
        "metrics": lane_metrics,
    }


def _rank_metrics(hit_ids: list[str], expected_ids: list[str]) -> dict[str, Any]:
    expected = [item_id for item_id in expected_ids if item_id]
    if not expected:
        return {}
    ranks = [
        hit_ids.index(item_id) + 1
        for item_id in expected
        if item_id in hit_ids
    ]
    first_rank = min(ranks) if ranks else None
    metrics: dict[str, Any] = {
        "expected_count": len(expected),
        "matched_expected_count": len(ranks),
        "first_rank": first_rank,
        "mrr": round(1.0 / first_rank, 6) if first_rank else 0.0,
    }
    for k in (1, 3, 5, 8):
        top = set(hit_ids[:k])
        matched = sum(1 for item_id in expected if item_id in top)
        metrics[f"hit@{k}"] = matched > 0
        metrics[f"recall@{k}"] = round(matched / len(expected), 6)
    return metrics


def _observed_overlap_metrics(hit_ids: list[str], case: dict[str, Any]) -> dict[str, Any]:
    observed = _as_dict(case.get("observed"))
    old_hit_ids = [
        str(item_id)
        for item_id in observed.get("hit_ids") or []
        if str(item_id).strip()
    ]
    if not old_hit_ids:
        return {}
    old_set = set(old_hit_ids)
    new_set = set(hit_ids)
    union = old_set | new_set
    return {
        "observed_overlap_count": len(old_set & new_set),
        "observed_jaccard": round(len(old_set & new_set) / len(union), 6) if union else 1.0,
        "observed_top1_same": bool(old_hit_ids and hit_ids and old_hit_ids[0] == hit_ids[0]),
    }


async def _retrieve_explicit(
    retriever: Retriever,
    case: dict[str, Any],
    expected_ids: list[str],
) -> tuple[list[dict[str, Any]], list[str], str, dict[str, Any]]:
    probe = _as_dict(case.get("probe"))
    query = _case_query(case)
    limit = max(1, int(probe.get("limit") or 8))
    search_mode = str(probe.get("search_mode") or "semantic")
    if search_mode == "grep":
        return [], [], "grep_not_replayed", {}
    actual_top_k = max(limit, EXPLICIT_VECTOR_TOP_K)
    hits = await retriever.retrieve(
        query,
        memory_types=_memory_types(case),
        top_k=actual_top_k,
        # recall_memory tool does not pass scope into ExplicitRetrievalRequest.
        scope_channel=None,
        scope_chat_id=None,
        require_scope_match=False,
        score_threshold=EXPLICIT_VECTOR_SCORE_THRESHOLD,
        keyword_enabled=True,
    )
    output_hits = list(hits)[:limit]
    lane_trace = await _trace_lanes(
        retriever,
        query=query,
        fusion_hits=output_hits,
        actual_top_k=actual_top_k,
        memory_types=_memory_types(case),
        scope_channel=None,
        scope_chat_id=None,
        require_scope_match=False,
        score_threshold=EXPLICIT_VECTOR_SCORE_THRESHOLD,
        keyword_enabled=True,
        expected_ids=expected_ids,
    )
    return output_hits, [], "explicit_semantic_direct", lane_trace


async def _retrieve_context(
    retriever: Retriever,
    case: dict[str, Any],
    expected_ids: list[str],
) -> tuple[list[dict[str, Any]], list[str], str, dict[str, Any]]:
    query = _case_query(case)
    top_k = max(1, int(_as_dict(case.get("probe")).get("top_k") or 8))
    channel, chat_id = _scope(case)
    hits = await retriever.retrieve(
        query,
        top_k=top_k,
        scope_channel=channel,
        scope_chat_id=chat_id,
        require_scope_match=False,
        keyword_enabled=True,
    )
    _text_block, injected_ids = retriever.build_injection_block(list(hits))
    lane_trace = await _trace_lanes(
        retriever,
        query=query,
        fusion_hits=list(hits),
        actual_top_k=top_k,
        memory_types=None,
        scope_channel=channel,
        scope_chat_id=chat_id,
        require_scope_match=False,
        score_threshold=None,
        keyword_enabled=True,
        expected_ids=expected_ids,
    )
    return list(hits), injected_ids, "context_retrieve_direct", lane_trace


async def _eval_sticky_case(
    retriever: Retriever,
    case: dict[str, Any],
) -> dict[str, Any]:
    memory_id = str(_as_dict(case.get("probe")).get("memory_id") or "").strip()
    sample_turns = [
        item for item in (_as_dict(case.get("observed")).get("sample_turns") or [])
        if isinstance(item, dict)
    ]
    per_turn: list[dict[str, Any]] = []
    retrieved_presence = 0
    injected_presence = 0
    for turn in sample_turns:
        query = str(turn.get("user_text") or "").strip()
        if not query:
            continue
        synthetic_case = {
            "probe": {"query": query, "top_k": 8},
            "turn": {
                "channel": turn.get("channel") or "",
                "chat_id": turn.get("chat_id") or "",
            },
        }
        hits, injected_ids, _mode, lane_trace = await _retrieve_context(
            retriever,
            synthetic_case,
            expected_ids=[],
        )
        ids = _hit_ids(hits)
        retrieved = memory_id in ids
        injected = memory_id in injected_ids
        retrieved_presence += int(retrieved)
        injected_presence += int(injected)
        per_turn.append(
            {
                "turn_id": turn.get("turn_id"),
                "query": query,
                "hit_ids": ids,
                "injected_ids": injected_ids,
                "target_retrieved": retrieved,
                "target_injected": injected,
                "lane_trace": lane_trace,
            }
        )
    count = len(per_turn)
    return {
        "hits": [],
        "hit_ids": [],
        "injected_ids": [],
        "mode": "sticky_sample_context_retrieve",
        "metrics": {
            "memory_id": memory_id,
            "sample_turn_count": count,
            "sample_retrieved_presence_count": retrieved_presence,
            "sample_injected_presence_count": injected_presence,
            "sample_retrieved_presence_rate": round(retrieved_presence / count, 6) if count else 0.0,
            "sample_injected_presence_rate": round(injected_presence / count, 6) if count else 0.0,
            "observed_injected_context_count": _as_dict(case.get("observed")).get("injected_context_count"),
        },
        "sticky_sample_turns": per_turn,
    }


async def _eval_case(retriever: Retriever, case_path: Path) -> dict[str, Any]:
    case = _read_json(case_path)
    case_id = str(case.get("case_id") or case_path.stem)
    failure_type = str(case.get("failure_type") or "")
    task = str(case.get("task") or "")
    result: dict[str, Any] = {
        "kind": "eval_case",
        "case_id": case_id,
        "failure_type": failure_type,
        "task": task,
        "case_path": str(case_path),
        "query": _case_query(case),
    }
    expected_ids = [
        str(item_id)
        for item_id in _as_dict(case.get("expected")).get("memory_ids") or []
        if str(item_id).strip()
    ]

    try:
        if task == "retrieve_explicit":
            hits, injected_ids, mode, lane_trace = await _retrieve_explicit(
                retriever,
                case,
                expected_ids,
            )
            metrics: dict[str, Any] = {}
        elif task == "retrieve_context":
            hits, injected_ids, mode, lane_trace = await _retrieve_context(
                retriever,
                case,
                expected_ids,
            )
            expected = _as_dict(case.get("expected"))
            max_injected = expected.get("max_injected_count")
            metrics = {
                "injected_count": len(injected_ids),
                "hit_count": len(hits),
            }
            if isinstance(max_injected, int):
                metrics["max_injected_count"] = max_injected
                metrics["pass_injection_gate"] = len(injected_ids) <= max_injected
        elif task == "injection_frequency_review":
            sticky_result = await _eval_sticky_case(retriever, case)
            result.update(sticky_result)
            result["status"] = "ok"
            return result
        else:
            hits, injected_ids, mode = [], [], "unsupported_task"
            lane_trace = {}
            metrics = {"unsupported_task": task}

        ids = _hit_ids(hits)
        metrics.update(_rank_metrics(ids, expected_ids))
        metrics.update(_observed_overlap_metrics(ids, case))
        if failure_type == "explicit_empty_recall":
            metrics["still_empty"] = len(ids) == 0

        result.update(
            {
                "status": "ok",
                "mode": mode,
                "expected_memory_ids": expected_ids,
                "hit_ids": ids,
                "injected_ids": injected_ids,
                "hits": _compact_hits(hits),
                "metrics": metrics,
                "lane_trace": lane_trace,
            }
        )
        return result
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "metrics": {},
                "hit_ids": [],
                "injected_ids": [],
            }
        )
        return result


def _summarize(results: list[dict[str, Any]], embedder: EvalEmbedder, sandbox_workspace: Path) -> dict[str, Any]:
    by_type = Counter(str(item.get("failure_type") or "") for item in results)
    status_counts = Counter(str(item.get("status") or "") for item in results)

    expected_results = [
        item for item in results
        if item.get("expected_memory_ids")
    ]
    positive = [
        item for item in results
        if item.get("failure_type") == "positive_control"
    ]
    short_query = [
        item for item in results
        if item.get("failure_type") == "short_query_over_recall"
    ]
    explicit_empty = [
        item for item in results
        if item.get("failure_type") == "explicit_empty_recall"
    ]
    sticky = [
        item for item in results
        if item.get("failure_type") == "sticky_memory"
    ]

    def avg_metric(items: list[dict[str, Any]], key: str) -> float:
        vals = [
            float(_as_dict(item.get("metrics")).get(key))
            for item in items
            if isinstance(_as_dict(item.get("metrics")).get(key), int | float)
        ]
        return round(sum(vals) / len(vals), 6) if vals else 0.0

    def avg_lane_metric(items: list[dict[str, Any]], lane: str, key: str) -> float:
        vals: list[float] = []
        for item in items:
            lane_metrics = _as_dict(_as_dict(item.get("lane_trace")).get("metrics"))
            value = _as_dict(lane_metrics.get(lane)).get(key)
            if isinstance(value, int | float):
                vals.append(float(value))
        return round(sum(vals) / len(vals), 6) if vals else 0.0

    def avg_contribution_rate(items: list[dict[str, Any]], key: str) -> float:
        vals: list[float] = []
        for item in items:
            contribution = _as_dict(_as_dict(item.get("lane_trace")).get("contribution"))
            fusion_count = contribution.get("fusion_count")
            value = contribution.get(key)
            if isinstance(value, int | float) and isinstance(fusion_count, int | float) and fusion_count:
                vals.append(float(value) / float(fusion_count))
        return round(sum(vals) / len(vals), 6) if vals else 0.0

    recall_summary = {
        f"recall@{k}": avg_metric(expected_results, f"recall@{k}")
        for k in (1, 3, 5, 8)
    }
    positive_summary = {
        f"recall@{k}": avg_metric(positive, f"recall@{k}")
        for k in (1, 3, 5, 8)
    }
    positive_summary["mrr"] = avg_metric(positive, "mrr")

    lane_expected_metrics = {
        lane: {
            "mrr": avg_lane_metric(expected_results, lane, "mrr"),
            **{
                f"recall@{k}": avg_lane_metric(expected_results, lane, f"recall@{k}")
                for k in (1, 3, 5, 8)
            },
            **{
                f"hit@{k}": avg_lane_metric(expected_results, lane, f"hit@{k}")
                for k in (1, 3, 5, 8)
            },
        }
        for lane in ("fusion", "dense", "keyword", "bm25_summary")
    }

    traced_results = [
        item for item in results
        if _as_dict(item.get("lane_trace")).get("contribution")
    ]
    lane_fusion_contribution = {
        "dense_in_fusion_rate": avg_contribution_rate(
            traced_results,
            "dense_in_fusion_count",
        ),
        "keyword_in_fusion_rate": avg_contribution_rate(
            traced_results,
            "keyword_in_fusion_count",
        ),
        "bm25_in_fusion_rate": avg_contribution_rate(
            traced_results,
            "bm25_in_fusion_count",
        ),
        "dense_only_in_fusion_rate": avg_contribution_rate(
            traced_results,
            "dense_only_in_fusion_count",
        ),
        "keyword_only_in_fusion_rate": avg_contribution_rate(
            traced_results,
            "keyword_only_in_fusion_count",
        ),
        "dense_and_keyword_in_fusion_rate": avg_contribution_rate(
            traced_results,
            "dense_and_keyword_in_fusion_count",
        ),
    }

    short_still_over = sum(
        1
        for item in short_query
        if _as_dict(item.get("metrics")).get("pass_injection_gate") is False
    )
    sticky_with_samples = [
        item for item in sticky
        if _as_dict(item.get("metrics")).get("sample_turn_count")
    ]

    return {
        "kind": "summary",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sandbox_workspace": str(sandbox_workspace),
        "total_cases": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "failure_type_counts": dict(sorted(by_type.items())),
        "expected_memory_case_count": len(expected_results),
        "expected_memory_metrics": recall_summary,
        "lane_expected_metrics": lane_expected_metrics,
        "lane_fusion_contribution": lane_fusion_contribution,
        "positive_control_count": len(positive),
        "positive_control_metrics": positive_summary,
        "short_query_over_recall": {
            "count": len(short_query),
            "still_over_injected_count": short_still_over,
            "avg_injected_count": avg_metric(short_query, "injected_count"),
        },
        "explicit_empty_recall": {
            "count": len(explicit_empty),
            "still_empty_count": sum(
                1
                for item in explicit_empty
                if _as_dict(item.get("metrics")).get("still_empty") is True
            ),
        },
        "sticky_memory": {
            "count": len(sticky),
            "sampled_count": len(sticky_with_samples),
            "avg_sample_retrieved_presence_rate": avg_metric(
                sticky_with_samples,
                "sample_retrieved_presence_rate",
            ),
            "avg_sample_injected_presence_rate": avg_metric(
                sticky_with_samples,
                "sample_injected_presence_rate",
            ),
        },
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay daily memory-engine badcases against an isolated sandbox copy."
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--badcase-dir", type=Path, default=DEFAULT_BADCASE_DIR)
    parser.add_argument("--report", type=Path, default=None)
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
        default="auto",
        help="auto/live use configured embedding until it fails; keyword-only skips vector lane.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser


async def _amain() -> None:
    args = _build_parser().parse_args()
    if not args.verbose:
        logging.getLogger("memory2.retriever").setLevel(logging.ERROR)
        logging.getLogger("memory2.store").setLevel(logging.ERROR)
    source_workspace = args.workspace.expanduser().resolve()
    stamp = _now_stamp()
    report_path = args.report or DEFAULT_REPORT_DIR / f"daily_eval_{stamp}.jsonl"
    sandbox_workspace = args.sandbox_root / f"daily_{stamp}"
    badcase_dir = args.badcase_dir

    cases = sorted(badcase_dir.glob("*.json"))
    if args.limit:
        cases = cases[: max(0, args.limit)]
    if not cases:
        raise FileNotFoundError(f"no case json files under {badcase_dir}")

    db_path = _copy_memory_db_to_sandbox(source_workspace, sandbox_workspace)
    embedding_settings = _load_embedding_settings(args.config)
    retrieval_settings = _load_retrieval_settings(args.memory_config)
    embedder = EvalEmbedder(mode=args.embedding_mode, settings=embedding_settings)
    store, retriever = _build_retriever(db_path, embedder, retrieval_settings)

    results: list[dict[str, Any]] = []
    try:
        for index, case_path in enumerate(cases, 1):
            result = await _eval_case(retriever, case_path)
            result["index"] = index
            result["embedding_effective_mode"] = embedder.effective_mode
            _append_jsonl(report_path, result)
            results.append(result)
    finally:
        store.close()
        await embedder.aclose()

    summary = _summarize(results, embedder, sandbox_workspace)
    _append_jsonl(report_path, summary)
    print(f"report: {report_path}")
    print(f"sandbox_workspace: {sandbox_workspace}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
