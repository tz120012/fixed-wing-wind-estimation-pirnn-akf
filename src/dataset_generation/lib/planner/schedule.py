"""论文版采集计划表（160 轮 / 800 段）。

迁移自 ``generate_dataset.py:184-215``。本模块为纯函数，无副作用。

轮次分配（共 160 轮）：
    - 110 轮 train   → 550 段，风速 [0.8, 3.2] m/s
    - 30  轮 val     → 150 段，风速 [0.8, 3.2] m/s（提升方向覆盖均匀性）
    - 10  轮 test_id → 50  段，风速 [1.0, 3.2] m/s（in-distribution 泛化测试）
    - 10  轮 test_ood→ 50  段，风速 [3.5, 5.0] m/s（与 train 严格无重叠）
"""

from __future__ import annotations

from typing import List, Tuple


SEGMENTS_PER_RUN = 5

ScheduleEntry = Tuple[str, int, List[str]]


def build_paper_run_schedule() -> List[ScheduleEntry]:
    """返回论文版 160 轮的 ``(dataset_type, base_flight_id, segment_types)`` 列表。

    ``segment_types`` 为长 5 的列表，每项 ``'steady' | 'gust_id' | 'gust_ood'``。

    分布说明：
        - 110 轮 train：``steady, gust_id, gust_id, steady, gust_id``
        - 30  轮 val：同上（增至 30 轮以改善方向覆盖均匀性）
        - 10  轮 test_id：``steady, gust_id ×4``（保留 1 段稳态作 sanity check）
        - 10  轮 test_ood：``steady, gust_ood ×4``（风速 [3.5, 5.0] 与 train [0.8, 3.2] 严格无重叠）
    """
    schedule: List[ScheduleEntry] = []
    for r in range(110):
        schedule.append(("train", 1 + r * 5, ["steady", "gust_id", "gust_id", "steady", "gust_id"]))
    for r in range(30):
        schedule.append(("val", 1 + r * 5, ["steady", "gust_id", "gust_id", "steady", "gust_id"]))
    for r in range(10):
        schedule.append(("test_id", 1 + r * 5, ["steady", "gust_id", "gust_id", "gust_id", "gust_id"]))
    for r in range(10):
        schedule.append(("test_ood", 1 + r * 5, ["steady", "gust_ood", "gust_ood", "gust_ood", "gust_ood"]))
    return schedule


def build_80_run_schedule() -> List[ScheduleEntry]:
    """向后兼容旧函数名；当前返回论文版 160 轮计划表。"""
    return build_paper_run_schedule()
