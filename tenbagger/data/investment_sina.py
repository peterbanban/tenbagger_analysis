from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .http import http_get_bytes
from .fundamentals_sina import SINA_PROFIT_URL


SINA_CASHFLOW_URL = (
    "https://money.finance.sina.com.cn/corp/go.php/"
    "vFD_CashFlow/stockid/{code}/ctrl/{year}/displaytype/4.phtml"
)


def _sina_fetch_html_cached(url: str, cache_path: Path, encoding: str = "gb18030") -> str:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        cache_path.write_bytes(http_get_bytes(url))
    return cache_path.read_bytes().decode(encoding, errors="ignore")


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


def rd_and_capex_ratio(
    *,
    code: str,
    year: int,
    revenue_yuan: Optional[float],
    cache_dir: Path,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (rd_ratio, capex_ratio) for the specified annual year.
    Uses:
    - 研发费用 from profit statement
    - 购建固定资产、无形资产和其他长期资产所支付的现金 from cashflow
    """
    if revenue_yuan is None or revenue_yuan <= 0:
        return None, None

    date_str = f"{year}-12-31"

    profit_html = _sina_fetch_html_cached(
        SINA_PROFIT_URL.format(code=code, year=year), cache_dir / f"profit_{code}_{year}.html"
    )
    cash_html = _sina_fetch_html_cached(
        SINA_CASHFLOW_URL.format(code=code, year=year), cache_dir / f"cash_{code}_{year}.html"
    )

    rd_wy = _sina_stmt_value(profit_html, label_contains="研发费用", date_str=date_str)
    capex_wy = _sina_stmt_value(
        cash_html,
        label_contains="购建固定资产、无形资产和其他长期资产所支付的现金",
        date_str=date_str,
    )
    rd = rd_wy * 10000 if rd_wy is not None else None
    capex = capex_wy * 10000 if capex_wy is not None else None

    rd_ratio = (rd / revenue_yuan) if (rd is not None and revenue_yuan > 0) else None
    capex_ratio = (capex / revenue_yuan) if (capex is not None and revenue_yuan > 0) else None
    return rd_ratio, capex_ratio

