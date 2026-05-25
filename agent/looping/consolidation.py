"""ConsolidationService — bridge for the LongMemEval benchmark.

Wraps _MarkdownConsolidationWorker (core.memory.markdown) with the
API expected by eval/longmemeval/runtime.py.
"""

from __future__ import annotations

import logging
from typing import Any

from core.memory.markdown import _MarkdownConsolidationWorker

logger = logging.getLogger(__name__)


class ConsolidationService:
    """Thin wrapper around _MarkdownConsolidationWorker.

    The benchmark runtime creates this manually so it can control
    consolidation timing during ingest (one consolidate() call per
    haystack session).
    """

    def __init__(
        self,
        *,
        memory_port: Any = None,
        profile_maint: Any,
        provider: Any,
        model: str,
        keep_count: int,
        profile_extractor: Any = None,
        recent_context_provider: Any | None = None,
        recent_context_model: str | None = None,
    ) -> None:
        self._memory_port = memory_port
        self._profile_maint = profile_maint
        self._worker = _MarkdownConsolidationWorker(
            profile_maint=profile_maint,
            provider=provider,
            model=model,
            keep_count=keep_count,
            recent_context_provider=recent_context_provider,
            recent_context_model=recent_context_model,
        )

    async def consolidate(self, session, archive_all: bool = False) -> None:
        """Run one consolidation pass on *session*."""
        draft = await self._worker.prepare_consolidation(
            session, archive_all=archive_all
        )
        if draft is None:
            logger.debug(
                "consolidation skipped: session=%s messages=%d archive_all=%s",
                getattr(session, "key", "?"),
                len(getattr(session, "messages", [])),
                archive_all,
            )
            return
        total = len(session.messages)
        session.last_consolidated = total
        logger.info(
            "consolidation committed: session=%s messages=%d archive_all=%s",
            getattr(session, "key", "?"),
            total,
            archive_all,
        )
