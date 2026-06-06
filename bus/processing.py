"""
bus/processing.py — 会话级处理仲裁

AgentLoop 在处理每条入站消息时调用 acquire/enter/exit；
ProactiveLoop/AgentTick 在发送前调用 is_busy(session_key) 检查目标会话，
SchedulerService 使用同一个 acquire(session_key) 串行化定时任务与普通 turn。

设计约束：
- 只在单一 asyncio 事件循环中使用。
- session_key 作用域：A 会话 busy 不影响 B 会话的 proactive/scheduler 判断。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class ProcessingState:
    """会话级处理计数器与互斥锁。"""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def enter(self, session_key: str) -> None:
        """标记 session_key 开始处理消息。"""
        self._counts[session_key] = self._counts.get(session_key, 0) + 1

    def exit(self, session_key: str) -> None:
        """标记 session_key 完成处理消息。"""
        self._counts[session_key] = max(0, self._counts.get(session_key, 0) - 1)

    def is_busy(self, session_key: str) -> bool:
        """返回目标会话当前是否正在处理回复。"""
        return self._counts.get(session_key, 0) > 0

    def _lock_for(self, session_key: str) -> asyncio.Lock:
        lock = self._locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_key] = lock
        return lock

    @asynccontextmanager
    async def acquire(self, session_key: str) -> AsyncIterator[None]:
        """串行化同一 session_key 的处理，并维护 busy 计数。"""
        lock = self._lock_for(session_key)
        async with lock:
            self.enter(session_key)
            try:
                yield
            finally:
                self.exit(session_key)
