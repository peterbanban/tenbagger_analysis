from __future__ import annotations

from typing import Optional


def industry_ceiling_score(industry: str) -> float:
    s = (industry or "").strip()
    high = [
        "半导体",
        "软件",
        "互联网",
        "通信",
        "计算机",
        "医疗",
        "生物",
        "创新药",
        "军工",
        "自动化",
        "机器人",
        "电池",
        "新能源",
        "光伏",
        "储能",
        "风电",
        "电网设备",
        "汽车零部件",
    ]
    mid = ["化工", "化学", "机械", "专用设备", "通用设备", "仪器仪表", "消费电子", "家电", "食品饮料"]
    low = ["煤炭", "钢铁", "建材", "银行", "地产", "公用事业", "港口", "航运"]
    if any(k in s for k in high):
        return 85.0
    if any(k in s for k in mid):
        return 60.0
    if any(k in s for k in low):
        return 35.0
    return 50.0


def overseas_score(overseas_share: Optional[float]) -> float:
    if overseas_share is None:
        return 0.0
    if overseas_share >= 0.5:
        return 100.0
    if overseas_share >= 0.3:
        return 80.0
    if overseas_share >= 0.2:
        return 65.0
    if overseas_share >= 0.1:
        return 45.0
    if overseas_share > 0:
        return 25.0
    return 0.0


def hhi(shares: list[float]) -> Optional[float]:
    if not shares:
        return None
    s = sum(shares)
    if s <= 0:
        return None
    return sum((x / s) ** 2 for x in shares if x > 0)


def diversification_score(top1_share: Optional[float], hhi_value: Optional[float], seg_cnt: int) -> float:
    score = 0.0
    if seg_cnt >= 4:
        score += 20
    elif seg_cnt == 3:
        score += 15
    elif seg_cnt == 2:
        score += 8

    if top1_share is not None:
        if top1_share <= 0.55:
            score += 45
        elif top1_share <= 0.70:
            score += 30
        elif top1_share <= 0.85:
            score += 15
        else:
            score += 5

    if hhi_value is not None:
        if hhi_value <= 0.35:
            score += 35
        elif hhi_value <= 0.50:
            score += 22
        elif hhi_value <= 0.65:
            score += 12
        else:
            score += 5
    return min(100.0, score)


def rd_score(rd_ratio: Optional[float]) -> float:
    if rd_ratio is None:
        return 0.0
    if rd_ratio >= 0.12:
        return 100.0
    if rd_ratio >= 0.08:
        return 80.0
    if rd_ratio >= 0.05:
        return 55.0
    if rd_ratio >= 0.03:
        return 35.0
    return 15.0


def capex_score(capex_ratio: Optional[float]) -> float:
    if capex_ratio is None:
        return 0.0
    if capex_ratio >= 0.12:
        return 85.0
    if capex_ratio >= 0.07:
        return 65.0
    if capex_ratio >= 0.03:
        return 45.0
    if capex_ratio > 0:
        return 25.0
    return 0.0


def total_score_framework_v2(*, base: float, ceiling_industry: float, ceiling_overseas: float, second_total: float) -> float:
    ceiling_total = 0.6 * ceiling_industry + 0.4 * ceiling_overseas
    return 0.50 * base + 0.25 * ceiling_total + 0.25 * second_total

