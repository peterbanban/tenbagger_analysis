from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


def parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")


def _ma(closes: list[float], i: int, window: int) -> Optional[float]:
    if window <= 0 or i < window - 1:
        return None
    seg = closes[i - (window - 1) : i + 1]
    if not seg:
        return None
    return sum(seg) / window


def momentum_from_series(
    dates: list[str], closes: list[float], asof_date: str
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        i = dates.index(asof_date)
    except ValueError:
        return None, None, None
    if i < 260:
        return None, None, None
    c0 = closes[i]
    c120 = closes[i - 120]
    c250 = closes[i - 250]
    ret120 = (c0 / c120 - 1.0) if c120 > 0 else None
    ret250 = (c0 / c250 - 1.0) if c250 > 0 else None
    high250 = max(closes[i - 250 : i + 1])
    dd = (c0 / high250 - 1.0) if high250 > 0 else None
    return ret120, ret250, dd


@dataclass(frozen=True)
class InflectionSignal:
    ok: bool
    ma20: Optional[float]
    ma60: Optional[float]
    ma120: Optional[float]
    ma60_slope_20d: Optional[float]
    pos_from_250_low: Optional[float]


def inflection_signal(dates: list[str], closes: list[float], asof_date: str) -> InflectionSignal:
    """
    Attempts to catch "low-cost but turning" entries:
    - Trend turns up (MA20 > MA60 and price > MA20)
    - MA60 is rising (vs 20 trading days ago) to reduce dead money
    - Still not too far from 250d low
    """
    try:
        i = dates.index(asof_date)
    except ValueError:
        return InflectionSignal(False, None, None, None, None, None)
    if i < 260:
        return InflectionSignal(False, None, None, None, None, None)

    c0 = closes[i]
    ma20 = _ma(closes, i, 20)
    ma60 = _ma(closes, i, 60)
    ma120 = _ma(closes, i, 120)
    ma60_prev = _ma(closes, i - 20, 60) if i >= 80 else None
    ma60_slope = (ma60 / ma60_prev - 1.0) if (ma60 is not None and ma60_prev is not None and ma60_prev > 0) else None

    low250 = min(closes[i - 250 : i + 1])
    pos_from_low = (c0 / low250 - 1.0) if low250 > 0 else None

    trend_ok = (ma20 is not None and ma60 is not None and c0 > ma20 and ma20 > ma60)
    slope_ok = (ma60_slope is not None and ma60_slope > 0)
    low_cost_ok = (pos_from_low is not None and pos_from_low <= 0.60)

    ok = bool(trend_ok and slope_ok and low_cost_ok)
    return InflectionSignal(ok, ma20, ma60, ma120, ma60_slope, pos_from_low)


def score_quality_value_core(
    mktcap_yi: Optional[float],
    pe: Optional[float],
    pb: Optional[float],
    nm: Optional[float],
    roe: Optional[float],
    debt_ratio: Optional[float],
    dd250: Optional[float],
) -> float:
    """
    A single "quality/value + stress" core score (0..100).
    """
    score = 0.0
    # Size (20)
    if mktcap_yi is not None:
        if 10 <= mktcap_yi <= 50:
            score += 20
        elif 50 < mktcap_yi <= 120:
            score += 17
        elif 5 <= mktcap_yi < 10:
            score += 14
        elif 120 < mktcap_yi <= 250:
            score += 10
        elif 250 < mktcap_yi <= 600:
            score += 6
        else:
            score += 3
    # Profitability (25)
    if nm is not None:
        if nm >= 0.15:
            score += 25
        elif nm >= 0.10:
            score += 20
        elif nm >= 0.05:
            score += 12
        elif nm > 0:
            score += 6
    # ROE proxy (10)
    if roe is not None:
        if roe >= 0.20:
            score += 10
        elif roe >= 0.15:
            score += 8
        elif roe >= 0.10:
            score += 6
        elif roe >= 0.05:
            score += 3
    # Valuation (25)
    if pe is not None and pe > 0:
        if pe < 12:
            score += 14
        elif pe < 18:
            score += 12
        elif pe < 28:
            score += 9
        elif pe < 45:
            score += 5
    if pb is not None and pb > 0:
        if pb < 1.5:
            score += 11
        elif pb < 2.5:
            score += 8
        elif pb < 4:
            score += 6
        elif pb < 8:
            score += 3
    # Stress (drawdown) (10) – deeper drawdown is a plus if fundamentals are ok
    if dd250 is not None:
        if dd250 <= -0.70:
            score += 10
        elif dd250 <= -0.55:
            score += 8
        elif dd250 <= -0.40:
            score += 6
        elif dd250 <= -0.25:
            score += 3
    # Balance sheet safety (10)
    if debt_ratio is not None:
        if debt_ratio < 0.55:
            score += 10
        elif debt_ratio < 0.65:
            score += 7
        elif debt_ratio < 0.75:
            score += 4
        else:
            score += 1
    return min(100.0, score)


def _cagr(old: Optional[float], new: Optional[float], years: int) -> Optional[float]:
    if old is None or new is None or years <= 0:
        return None
    if old <= 0 or new <= 0:
        return None
    try:
        return (new / old) ** (1.0 / years) - 1.0
    except Exception:
        return None


def growth_score(rev_cagr: Optional[float], profit_cagr: Optional[float]) -> float:
    """
    Proxy for market expansion + company growth quality (multi-year, not single spikes).
    """
    score = 0.0
    c = rev_cagr
    if c is not None:
        if c >= 0.35:
            score += 55
        elif c >= 0.25:
            score += 48
        elif c >= 0.15:
            score += 38
        elif c >= 0.08:
            score += 28
        elif c > 0:
            score += 18
        else:
            score += 6
    p = profit_cagr
    if p is not None:
        if p >= 0.45:
            score += 45
        elif p >= 0.30:
            score += 38
        elif p >= 0.18:
            score += 28
        elif p >= 0.08:
            score += 18
        elif p > 0:
            score += 10
        else:
            score += 4
    return min(100.0, score)


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


@dataclass(frozen=True)
class HardGate:
    ok: bool
    ceiling_total: float
    second_total: float
    growth_total: float


def hard_gate(
    *,
    ceiling_total: float,
    second_total: float,
    growth_total: float,
    min_ceiling_total: float = 55.0,
    min_second_total: float = 55.0,
    min_growth_total: float = 35.0,
) -> HardGate:
    """
    Tenbagger hard gate:
    - Must have (ceiling >= threshold OR second_curve_proxy >= threshold)
    - Must have non-trivial multi-year growth (proxy)
    """
    if growth_total < min_growth_total:
        return HardGate(False, ceiling_total, second_total, growth_total)
    if ceiling_total >= min_ceiling_total or second_total >= min_second_total:
        return HardGate(True, ceiling_total, second_total, growth_total)
    return HardGate(False, ceiling_total, second_total, growth_total)


def score_fused_total(
    *,
    core_score: float,
    gate: HardGate,
    signal: InflectionSignal,
    ret120: Optional[float],
    ret250: Optional[float],
    dd250: Optional[float],
    second_total: float,
) -> float:
    """
    Final fused score: "can become big" (gate metrics) + "turning now" (signal) + "cheap enough" + core quality/value.
    """
    turning = 0.0
    if signal.ok:
        turning += 60
    else:
        if ret120 is not None and ret120 > 0:
            turning += 20
        if signal.ma60_slope_20d is not None and signal.ma60_slope_20d > 0:
            turning += 15
    if ret250 is not None and ret250 > 0:
        turning += 10
    turning = min(100.0, turning)

    low_cost = 0.0
    if signal.pos_from_250_low is not None:
        if signal.pos_from_250_low <= 0.25:
            low_cost += 50
        elif signal.pos_from_250_low <= 0.45:
            low_cost += 35
        elif signal.pos_from_250_low <= 0.60:
            low_cost += 22
        else:
            low_cost += 8
    if dd250 is not None:
        if dd250 <= -0.55:
            low_cost += 25
        elif dd250 <= -0.35:
            low_cost += 18
        elif dd250 <= -0.20:
            low_cost += 10
        elif dd250 <= -0.05:
            low_cost += 4
    low_cost = min(100.0, low_cost)

    # Gate components: ceiling/growth/second_curve are used as "alpha prerequisites" and as score drivers.
    prereq = 0.0
    prereq += 0.40 * gate.ceiling_total
    prereq += 0.30 * gate.growth_total
    prereq += 0.30 * second_total

    total = 0.38 * core_score + 0.22 * turning + 0.15 * low_cost + 0.25 * prereq
    return float(min(100.0, max(0.0, total)))

