from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .http import http_get_bytes


SINA_PROFIT_URL = (
    "https://money.finance.sina.com.cn/corp/go.php/"
    "vFD_ProfitStatement/stockid/{code}/ctrl/{year}/displaytype/4.phtml"
)
SINA_BALANCE_URL = (
    "https://money.finance.sina.com.cn/corp/go.php/"
    "vFD_BalanceSheet/stockid/{code}/ctrl/{year}/displaytype/4.phtml"
)

# Conservative: annual report available after Apr-30.
ANNUAL_REPORT_AVAILABLE_MMDD = "0430"


def _last_available_annual_year(asof_date: str) -> int:
    y = int(asof_date[:4])
    mmdd = asof_date[5:7] + asof_date[8:10]
    if mmdd <= ANNUAL_REPORT_AVAILABLE_MMDD:
        return y - 2
    return y - 1


def _sina_stmt_value(html: str, *, label_contains: str, date_str: str) -> Optional[float]:
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return None
    stmt = None
    for t in tables:
        if "报表日期" in " ".join(map(str, t.values.flatten())):
            stmt = t
            break
    if stmt is None:
        return None
    dates = [str(x).strip() for x in stmt.iloc[0, 1:].tolist()]
    col = None
    for i, dt in enumerate(dates, start=1):
        if dt == date_str:
            col = i
            break
    if col is None:
        return None
    labels = stmt.iloc[:, 0].astype(str)
    hits = labels[labels.str.contains(label_contains, regex=False, na=False)].index
    if len(hits) == 0:
        return None
    raw = str(stmt.iloc[hits[0], col]).strip().replace(",", "")
    if raw in ("", "--", "nan", "NaN"):
        return None
    try:
        return float(raw)
    except Exception:
        return None


@dataclass(frozen=True)
class Fundamentals:
    year: int
    revenue_yuan: Optional[float]
    net_profit_yuan: Optional[float]
    equity_yuan: Optional[float]
    assets_yuan: Optional[float]
    liabilities_yuan: Optional[float]


def get_fundamentals_last_annual(code: str, asof_date: str, cache_dir: Path) -> Fundamentals:
    y = _last_available_annual_year(asof_date)
    return get_fundamentals_annual(code=code, year=y, cache_dir=cache_dir)


def get_fundamentals_annual(*, code: str, year: int, cache_dir: Path) -> Fundamentals:
    date_str = f"{year}-12-31"
    cache_dir.mkdir(parents=True, exist_ok=True)

    profit = cache_dir / f"profit_{code}_{year}.html"
    balance = cache_dir / f"balance_{code}_{year}.html"
    if not profit.exists():
        profit.write_bytes(http_get_bytes(SINA_PROFIT_URL.format(code=code, year=year)))
    if not balance.exists():
        balance.write_bytes(http_get_bytes(SINA_BALANCE_URL.format(code=code, year=year)))

    profit_html = profit.read_bytes().decode("gb18030", errors="ignore")
    bal_html = balance.read_bytes().decode("gb18030", errors="ignore")

    revenue_wy = _sina_stmt_value(profit_html, label_contains="营业总收入", date_str=date_str)
    np_wy = _sina_stmt_value(profit_html, label_contains="归属于母公司所有者的净利润", date_str=date_str)
    equity_wy = _sina_stmt_value(bal_html, label_contains="归属于母公司股东权益合计", date_str=date_str)
    assets_wy = _sina_stmt_value(bal_html, label_contains="资产总计", date_str=date_str)
    liab_wy = _sina_stmt_value(bal_html, label_contains="负债合计", date_str=date_str)

    def wy_to_yuan(x: Optional[float]) -> Optional[float]:
        return x * 10000 if x is not None else None

    return Fundamentals(
        year=year,
        revenue_yuan=wy_to_yuan(revenue_wy),
        net_profit_yuan=wy_to_yuan(np_wy),
        equity_yuan=wy_to_yuan(equity_wy),
        assets_yuan=wy_to_yuan(assets_wy),
        liabilities_yuan=wy_to_yuan(liab_wy),
    )


def get_fundamentals_series_last_n_years(*, code: str, asof_date: str, n: int, cache_dir: Path) -> list[Fundamentals]:
    """
    Returns latest N annual fundamentals available as-of asof_date, ordered from old->new.
    """
    if n <= 0:
        return []
    last_y = _last_available_annual_year(asof_date)
    years = list(range(last_y - (n - 1), last_y + 1))
    out: list[Fundamentals] = []
    for y in years:
        out.append(get_fundamentals_annual(code=code, year=y, cache_dir=cache_dir))
    return out
