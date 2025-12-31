from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple


def parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")


def momentum_from_series(
    dates: List[str], closes: List[float], asof_date: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
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


def entry_ok_ma(dates: List[str], closes: List[float], asof_date: str) -> Tuple[bool, Optional[float], Optional[float]]:
    try:
        i = dates.index(asof_date)
    except ValueError:
        return False, None, None
    if i < 60:
        return False, None, None
    ma20 = sum(closes[i - 19 : i + 1]) / 20
    ma60 = sum(closes[i - 59 : i + 1]) / 60
    c0 = closes[i]
    ok = (ma20 > ma60) and (c0 > ma20)
    return ok, ma20, ma60


def score_model_a(
    mktcap_yi: Optional[float],
    pe: Optional[float],
    pb: Optional[float],
    nm: Optional[float],
    roe: Optional[float],
    debt_ratio: Optional[float],
    dd: Optional[float],
) -> float:
    score = 0.0
    # Size (25)
    if mktcap_yi is not None:
        if 10 <= mktcap_yi <= 50:
            score += 25
        elif 50 < mktcap_yi <= 100:
            score += 22
        elif 5 <= mktcap_yi < 10:
            score += 18
        elif 100 < mktcap_yi <= 200:
            score += 14
        elif 200 < mktcap_yi <= 500:
            score += 8
        else:
            score += 4
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
    # Valuation (20)
    if pe is not None and pe > 0:
        if pe < 12:
            score += 12
        elif pe < 18:
            score += 10
        elif pe < 28:
            score += 7
        elif pe < 45:
            score += 4
    if pb is not None and pb > 0:
        if pb < 1.5:
            score += 8
        elif pb < 2.5:
            score += 6
        elif pb < 4:
            score += 4
        elif pb < 8:
            score += 2
    # Stress (drawdown) (10) – deeper drawdown is a plus if fundamentals are ok
    if dd is not None:
        if dd <= -0.70:
            score += 10
        elif dd <= -0.55:
            score += 8
        elif dd <= -0.40:
            score += 6
        elif dd <= -0.25:
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


def score_model_b(
    mktcap_yi: Optional[float],
    pe: Optional[float],
    pb: Optional[float],
    nm: Optional[float],
    roe: Optional[float],
    ret120: Optional[float],
    ret250: Optional[float],
    dd: Optional[float],
) -> float:
    score = 0.0
    # Size (15)
    if mktcap_yi is not None:
        if 10 <= mktcap_yi <= 200:
            score += 15
        elif mktcap_yi <= 500:
            score += 10
        else:
            score += 5
    # Profitability (25)
    if nm is not None:
        if nm >= 0.15:
            score += 25
        elif nm >= 0.10:
            score += 20
        elif nm >= 0.05:
            score += 12
    if roe is not None:
        if roe >= 0.20:
            score += 8
        elif roe >= 0.12:
            score += 6
        elif roe >= 0.08:
            score += 3
    # Valuation sanity (12)
    if pe is not None and pe > 0:
        if 8 <= pe <= 35:
            score += 10
        elif pe <= 50:
            score += 6
    if pb is not None and 0 < pb <= 8:
        score += 2
    # Trend strength (36)
    if ret120 is not None:
        if ret120 > 0:
            score += 10
        if ret120 > 0.3:
            score += 4
    if ret250 is not None:
        if ret250 > 0:
            score += 10
        if ret250 > 0.5:
            score += 4
    if dd is not None and dd > -0.25:
        score += 8
    return min(100.0, score)

