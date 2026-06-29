"""SleepContext — 睡眠状态上下文数据模型。

proactive 链路用睡眠状态修正打断度（夜间/午睡降低打扰概率）。
当前 fitbit 实现已移除，本模块仅保留与具体手表解耦的上下文数据结构，
作为未来睡眠数据源（手表 / 其他设备）接入的可插拔契约：
任何 provider 只需实现 `get() -> SleepContext | None` 并注入 ProactiveLoop。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SleepContext:
    state: str  # sleeping | awake | uncertain | unknown
    prob: float | None  # 0-1，None=无数据
    prob_source: str  # ml | heuristic | unavailable
    data_lag_min: int | None
    fetched_at: float  # time.time()
    available: bool  # False=服务不可达或数据过期
    sleeping_modifier: float = 0.15

    @property
    def sleep_modifier(self) -> float:
        """
        用于乘以 interrupt_factor 的修正系数。
        不设为 0，保留 chat 的概率可能性。
        """
        if not self.available:
            return 1.0  # 降级：不影响现有行为
        if self.state == "sleeping":
            return self.sleeping_modifier
        if self.state == "uncertain":
            # 睡眠高概率的 uncertain 也按睡眠保护处理，减少夜间/午睡打扰。
            if (
                self.prob is not None
                and self.prob >= 0.60
                and (self.data_lag_min is None or self.data_lag_min <= 15)
            ):
                return 0.20
            return 0.50  # 普通 uncertain：chat 概率降约 50%
        if self.state == "awake":
            return 1.0
        return 0.88  # unknown：轻微保守


_FALLBACK = SleepContext(
    state="unknown",
    prob=None,
    prob_source="unavailable",
    data_lag_min=None,
    sleeping_modifier=0.15,
    fetched_at=0.0,
    available=False,
)
