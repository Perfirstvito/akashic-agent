from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.config_models import Config
from core.net.http import SharedHttpResources
from memory2.embedder import Embedder
from plugins.akasha.config import load_akasha_config, resolve_akasha_db_path
from plugins.akasha.core import SourceMessage
from plugins.akasha.store import AkashaStore


@dataclass(frozen=True)
class CacheStats:
    candidates: int = 0
    cached: int = 0
    skipped_existing: int = 0
    skipped_policy: int = 0
    skipped_empty: int = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate Akasha message embedding cache from sessions.db."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--workspace",
        default=str(Path.home() / ".akashic" / "workspace"),
        help="Akashic workspace path.",
    )
    parser.add_argument("--sessions-db", default="")
    parser.add_argument("--db-path", default="")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-skipped", action="store_true")
    return parser.parse_args()


def _policy_skip(content: str, extra_raw: object) -> bool:
    if content.startswith("[后台任务完成]"):
        return True
    try:
        parsed: object = json.loads(str(extra_raw or "{}"))
    except json.JSONDecodeError:
        parsed = {}
    extra = parsed if isinstance(parsed, dict) else {}
    return bool(extra.get("proactive")) or bool(extra.get("skip_post_memory"))


def _iter_messages(
    sessions_db: Path,
    *,
    include_skipped: bool,
) -> Iterable[tuple[SourceMessage, bool]]:
    with closing(sqlite3.connect(str(sessions_db))) as db:
        rows = db.execute(
            """
            SELECT id, session_key, seq, role, content, ts, extra
            FROM messages
            WHERE role IN ('user', 'assistant')
            ORDER BY ts, session_key, seq
            """
        )
        for row in rows:
            message = SourceMessage(
                id=str(row[0]),
                session_key=str(row[1]),
                seq=int(row[2]),
                role=str(row[3] or ""),
                content=str(row[4] or ""),
                ts=str(row[5] or ""),
            )
            skipped = _policy_skip(message.content, row[6])
            if skipped and not include_skipped:
                yield message, True
            else:
                yield message, False


async def _populate() -> CacheStats:
    args = _parse_args()
    workspace = Path(str(args.workspace)).expanduser()
    sessions_db = (
        Path(str(args.sessions_db)).expanduser()
        if str(args.sessions_db or "").strip()
        else workspace / "sessions.db"
    )
    if not sessions_db.exists():
        raise FileNotFoundError(f"sessions.db not found: {sessions_db}")

    config = Config.load(str(args.config))
    embedding_config = config.memory.embedding
    if not embedding_config.api_key:
        raise ValueError("memory.embedding.api_key is empty")
    if not embedding_config.base_url:
        raise ValueError("memory.embedding.base_url is empty")

    akasha_config = load_akasha_config()
    if str(args.db_path or "").strip():
        from dataclasses import replace

        akasha_config = replace(akasha_config, db_path=str(args.db_path))
    db_path = resolve_akasha_db_path(
        workspace=workspace,
        akasha_config=akasha_config,
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)

    http_resources = SharedHttpResources()
    store = AkashaStore(db_path)
    embedder = Embedder(
        base_url=embedding_config.base_url,
        api_key=embedding_config.api_key,
        model=embedding_config.model,
        requester=http_resources.external_default,
    )
    batch_size = max(1, min(int(args.batch_size), Embedder.MAX_BATCH))
    limit = max(0, int(args.limit))
    model = embedding_config.model

    candidates = 0
    cached = 0
    skipped_existing = 0
    skipped_policy = 0
    skipped_empty = 0
    pending: list[SourceMessage] = []

    async def flush() -> None:
        nonlocal cached, pending
        if not pending:
            return
        current = pending
        pending = []
        embeddings = await embedder.embed_batch([item.content for item in current])
        for message, embedding in zip(current, embeddings, strict=False):
            store.upsert_cached_embedding(
                message=message,
                model=model,
                embedding=embedding,
            )
            cached += 1
        print(
            f"cached={cached} skipped_existing={skipped_existing} "
            f"skipped_policy={skipped_policy}",
            flush=True,
        )

    try:
        for message, skipped_by_policy in _iter_messages(
            sessions_db,
            include_skipped=bool(args.include_skipped),
        ):
            if skipped_by_policy:
                skipped_policy += 1
                continue
            if not message.content.strip():
                skipped_empty += 1
                continue
            if limit and candidates >= limit:
                break
            candidates += 1
            if store.get_cached_embedding(message=message, model=model) is not None:
                skipped_existing += 1
                continue
            pending.append(message)
            if len(pending) >= batch_size:
                await flush()
        await flush()
    finally:
        await embedder.aclose()
        await http_resources.aclose()
        store.close()

    return CacheStats(
        candidates=candidates,
        cached=cached,
        skipped_existing=skipped_existing,
        skipped_policy=skipped_policy,
        skipped_empty=skipped_empty,
    )


def main() -> None:
    stats = asyncio.run(_populate())
    print(
        "Akasha embedding cache populated: "
        f"candidates={stats.candidates} "
        f"cached={stats.cached} "
        f"skipped_existing={stats.skipped_existing} "
        f"skipped_policy={stats.skipped_policy} "
        f"skipped_empty={stats.skipped_empty}"
    )


if __name__ == "__main__":
    main()
