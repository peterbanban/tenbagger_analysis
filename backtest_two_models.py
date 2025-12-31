#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


START_DATE = "2019-01-01"
END_DATE = "2025-12-31"
ANNUAL_REPORT_AVAILABLE_MMDD = "0430"

EM_CLLIST_URL = (
    "https://push2.eastmoney.com/api/qt/clist/get"
    "?pn={pn}&pz=100&po=1&np=1&fltt=2&invt=2&fid=f3"
    "&fs=m:1+t:2,m:1+t:23,m:0+t:6,m:0+t:80"
    "&fields=f12,f14,f13,f100"
)

EM_STOCK_GET_URL = (
    "https://push2.eastmoney.com/api/qt/stock/get"
    "?secid={secid}&fields=f57"
)

EM_KLINE_URL = (
    "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    "?secid={secid}&klt=101&fqt={fqt}&beg={beg}&end={end}"
    "&fields1=f1,f2,f3,f4,f5"
    "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
)

EM_CAPITAL_URL = (
    "https://emweb.securities.eastmoney.com/PC_HSF10/CapitalStockStructure/PageAjax"
    "?code={mkt}{code}"
)

SINA_PROFIT_URL = (
    "https://money.finance.sina.com.cn/corp/go.php/"
    "vFD_ProfitStatement/stockid/{code}/ctrl/{year}/displaytype/4.phtml"
)
SINA_BALANCE_URL = (
    "https://money.finance.sina.com.cn/corp/go.php/"
    "vFD_BalanceSheet/stockid/{code}/ctrl/{year}/displaytype/4.phtml"
)


def _http_get_json(url: str, *, retries: int = 6, timeout: int = 30) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            text = raw.decode("utf-8", errors="ignore")
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end >= 0:
                text = text[start : end + 1]
            return json.loads(text)
        except (HTTPError, URLError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep((2**attempt) * (0.15 + random.random() * 0.25))
    raise RuntimeError(f"failed GET {url}: {last_err}")


def _http_get_bytes(url: str, *, retries: int = 6, timeout: int = 30) -> bytes:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (HTTPError, URLError) as e:
            last_err = e
            time.sleep((2**attempt) * (0.15 + random.random() * 0.25))
    raise RuntimeError(f"failed GET {url}: {last_err}")


def _parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")


@dataclass(frozen=True)
class Stock:
    secid: str
    code: str
    name: str
    industry: str


@dataclass(frozen=True)
class RangeResult:
    stock: Stock
    low_date: str
    low_close_fq: float
    hit_date: Optional[str]
    hit_close_fq: Optional[float]


@dataclass(frozen=True)
class TenbaggerRow:
    code: str
    name: str
    industry: str
    low_date: str
    hit_date: str


def get_all_a_shares() -> List[Stock]:
    out: List[Stock] = []
    pn = 1
    total: Optional[int] = None
    while True:
        j = _http_get_json(EM_CLLIST_URL.format(pn=pn))
        data = j.get("data") or {}
        if total is None:
            try:
                total = int(data.get("total") or 0)
            except Exception:
                total = 0
        diff = data.get("diff") or []
        if not diff:
            break
        for it in diff:
            code = str(it.get("f12") or "").strip()
            name = str(it.get("f14") or "").strip()
            mkt = it.get("f13")
            industry = str(it.get("f100") or "").strip()
            if not code or not name or mkt not in (0, 1):
                continue
            out.append(Stock(secid=f"{mkt}.{code}", code=code, name=name, industry=industry))
        if total and pn * 100 >= total:
            break
        pn += 1
        time.sleep(0.15)
    return out


def _kline_closes_in_period(secid: str, *, fqt: int) -> List[Tuple[str, float]]:
    j = _http_get_json(EM_KLINE_URL.format(secid=secid, fqt=fqt, beg="20190101", end="20251231"))
    data = j.get("data") or {}
    kl = data.get("klines") or []
    out: List[Tuple[str, float]] = []
    for row in kl:
        parts = row.split(",")
        if len(parts) < 3:
            continue
        d = parts[0]
        if d < START_DATE or d > END_DATE:
            continue
        try:
            c = float(parts[2])
        except Exception:
            continue
        out.append((d, c))
    return out


def compute_low_and_first_10x(stock: Stock) -> Optional[RangeResult]:
    closes = _kline_closes_in_period(stock.secid, fqt=1)
    if len(closes) < 50:
        return None
    min_close = float("inf")
    min_date = ""
    hit_date: Optional[str] = None
    hit_close: Optional[float] = None
    for d, c in closes:
        if c <= 0:
            continue
        if c < min_close:
            min_close = c
            min_date = d
            continue
        if min_date and hit_date is None and c / min_close >= 10.0:
            hit_date = d
            hit_close = c
            break
    if not min_date:
        return None
    return RangeResult(
        stock=stock,
        low_date=min_date,
        low_close_fq=min_close,
        hit_date=hit_date,
        hit_close_fq=hit_close,
    )


def _resolve_secid(code: str) -> str:
    # Try matching by listing board.
    candidates = ["1", "0"] if code.startswith(("6", "9")) else ["0", "1"]
    for mkt in candidates:
        secid = f"{mkt}.{code}"
        try:
            j = _http_get_json(EM_STOCK_GET_URL.format(secid=secid), retries=2)
            if (j.get("data") or {}).get("f57"):
                return secid
        except Exception:
            continue
    return f"{candidates[0]}.{code}"


def _shares_at_date(changes: List[Dict[str, Any]], d: str) -> Optional[float]:
    target = _parse_date(d).date()
    best: Optional[Tuple[datetime, float]] = None
    earliest: Optional[Tuple[datetime, float]] = None
    for it in changes:
        end_date = it.get("END_DATE")
        total = it.get("TOTAL_SHARES")
        if not end_date or total in (None, "", "-"):
            continue
        try:
            end_dt = datetime.strptime(end_date.split(" ")[0], "%Y-%m-%d")
            total_shares = float(total)
        except Exception:
            continue
        if earliest is None or end_dt < earliest[0]:
            earliest = (end_dt, total_shares)
        if end_dt.date() <= target:
            if best is None or end_dt > best[0]:
                best = (end_dt, total_shares)
    if best:
        return best[1]
    return earliest[1] if earliest else None


def _get_total_shares(code: str, mkt_prefix: str, d: str, cache_dir: Path) -> Optional[float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"capital_{mkt_prefix}{code}.json"
    if not cache.exists():
        cap = _http_get_json(EM_CAPITAL_URL.format(mkt=mkt_prefix, code=code))
        cache.write_text(json.dumps(cap, ensure_ascii=False), encoding="utf-8")
    else:
        cap = json.loads(cache.read_text("utf-8", errors="ignore"))
    changes = cap.get("lngbbd") or []
    return _shares_at_date(changes, d)


def _get_raw_close_on(secid: str, d: str) -> Optional[float]:
    # minimal window: year-1 start to date
    y = int(d[:4]) - 1
    j = _http_get_json(EM_KLINE_URL.format(secid=secid, fqt=0, beg=f"{y}0101", end=d.replace("-", "")))
    data = j.get("data") or {}
    kl = data.get("klines") or []
    for row in reversed(kl):
        parts = row.split(",")
        if len(parts) >= 3 and parts[0] == d:
            try:
                return float(parts[2])
            except Exception:
                return None
    return None


def _sina_stmt_value(html: str, *, label_contains: str, date_str: str) -> Optional[float]:
    try:
        tables = pd.read_html(html)
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


def _last_available_annual_year(asof_date: str) -> int:
    """
    Avoid obvious lookahead:
    - If asof is before (or on) April 30 of year Y, assume last available annual report is Y-2.
    - Else use Y-1.
    """
    y = int(asof_date[:4])
    mmdd = asof_date[5:7] + asof_date[8:10]
    if mmdd <= ANNUAL_REPORT_AVAILABLE_MMDD:
        return y - 2
    return y - 1


def _get_fundamentals_last_annual(code: str, asof_date: str, cache_dir: Path) -> Fundamentals:
    y = _last_available_annual_year(asof_date)
    date_str = f"{y}-12-31"
    cache_dir.mkdir(parents=True, exist_ok=True)

    profit = cache_dir / f"profit_{code}_{y}.html"
    balance = cache_dir / f"balance_{code}_{y}.html"
    if not profit.exists():
        profit.write_bytes(_http_get_bytes(SINA_PROFIT_URL.format(code=code, year=y)))
    if not balance.exists():
        balance.write_bytes(_http_get_bytes(SINA_BALANCE_URL.format(code=code, year=y)))

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
        year=y,
        revenue_yuan=wy_to_yuan(revenue_wy),
        net_profit_yuan=wy_to_yuan(np_wy),
        equity_yuan=wy_to_yuan(equity_wy),
        assets_yuan=wy_to_yuan(assets_wy),
        liabilities_yuan=wy_to_yuan(liab_wy),
    )


def _momentum_from_series(
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


def _entry_date_trend_confirmed_from_series(
    dates: List[str], closes: List[float], low_date: str
) -> Optional[str]:
    # Find earliest date after low_date where:
    # - MA20 > MA60
    # - close > MA20
    # - ret120 > 0
    # - dd_from_250d_high > -0.25
    try:
        low_idx = dates.index(low_date)
    except ValueError:
        return None
    if len(closes) < 300:
        return None
    max_i = min(len(dates) - 1, low_idx + 252)
    for i in range(low_idx + 1, max_i + 1):
        if i < 260:
            continue
        ma20 = sum(closes[i - 19 : i + 1]) / 20
        ma60 = sum(closes[i - 59 : i + 1]) / 60
        c0 = closes[i]
        if not (ma20 > ma60 and c0 > ma20):
            continue
        c120 = closes[i - 120]
        ret120 = (c0 / c120 - 1.0) if c120 > 0 else None
        high250 = max(closes[i - 250 : i + 1])
        dd = (c0 / high250 - 1.0) if high250 > 0 else None
        if ret120 is None or dd is None:
            continue
        if ret120 <= 0:
            continue
        if dd <= -0.25:
            continue
        return dates[i]
    return None


def _kline_cached(
    *,
    secid: str,
    fqt: int,
    beg: str,
    end: str,
    cache_dir: Path,
) -> Tuple[List[str], List[float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"kline_{secid.replace('.', '_')}_fqt{fqt}_{beg}_{end}.csv"
    p = cache_dir / key
    if p.exists():
        dates: List[str] = []
        closes: List[float] = []
        for line in p.read_text("utf-8", errors="ignore").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            d = parts[0].strip()
            try:
                c = float(parts[1])
            except Exception:
                continue
            dates.append(d)
            closes.append(c)
        return dates, closes

    j = _http_get_json(EM_KLINE_URL.format(secid=secid, fqt=fqt, beg=beg, end=end))
    data = j.get("data") or {}
    kl = data.get("klines") or []
    rows: List[Tuple[str, float]] = []
    for row in kl:
        parts = row.split(",")
        if len(parts) < 3:
            continue
        d = parts[0]
        try:
            c = float(parts[2])
        except Exception:
            continue
        rows.append((d, c))
    rows.sort(key=lambda x: x[0])
    p.write_text("\n".join(f"{d},{c}" for d, c in rows), encoding="utf-8")
    return [d for d, _ in rows], [c for _, c in rows]


@dataclass(frozen=True)
class ModelScore:
    code: str
    name: str
    industry: str
    label_low_10x: int
    low_date: str
    hit_date: Optional[str]
    dv_score: Optional[float]
    tc_score: Optional[float]
    tc_entry_date: Optional[str]
    label_entry_10x: Optional[int]
    error: str


def _score_dv(mktcap_yi: Optional[float], pe: Optional[float], pb: Optional[float], nm: Optional[float], roe: Optional[float], debt_ratio: Optional[float], ret250: Optional[float], dd: Optional[float]) -> float:
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
    # Stress (drawdown) (10) – deeper drawdown is a *plus* if fundamentals are ok
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


def _score_tc(mktcap_yi: Optional[float], pe: Optional[float], pb: Optional[float], nm: Optional[float], roe: Optional[float], ret120: Optional[float], ret250: Optional[float], dd: Optional[float]) -> float:
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


def _series_min_in_window(
    dates: List[str], closes: List[float], start: str, end: str
) -> Optional[Tuple[str, float]]:
    min_c = float("inf")
    min_d: Optional[str] = None
    for d, c in zip(dates, closes):
        if d < start or d > end:
            continue
        if c <= 0:
            continue
        if c < min_c:
            min_c = c
            min_d = d
    if min_d is None:
        return None
    return min_d, min_c


def _hits_10x_after(
    dates: List[str], closes: List[float], base_date: str, base_close: float, end: str
) -> bool:
    try:
        i0 = dates.index(base_date)
    except ValueError:
        return False
    target = base_close * 10.0
    for d, c in zip(dates[i0 + 1 :], closes[i0 + 1 :]):
        if d > end:
            break
        if c >= target:
            return True
    return False


def score_one(rr: RangeResult, cache_dir: Path) -> ModelScore:
    try:
        secid = _resolve_secid(rr.stock.code)
        mkt = secid.split(".", 1)[0]
        mkt_prefix = "SH" if mkt == "1" else "SZ"

        # DV at low_date
        raw_dates, raw_closes = _kline_cached(
            secid=secid, fqt=0, beg="20190101", end="20251231", cache_dir=cache_dir / "kline"
        )
        raw_map = dict(zip(raw_dates, raw_closes))
        low_raw = raw_map.get(rr.low_date)
        shares_low = _get_total_shares(rr.stock.code, mkt_prefix, rr.low_date, cache_dir / "capital")
        mktcap_low = (low_raw * shares_low) if (low_raw is not None and shares_low is not None) else None
        mktcap_low_yi = (mktcap_low / 1e8) if mktcap_low is not None else None

        f_low = _get_fundamentals_last_annual(rr.stock.code, rr.low_date, cache_dir / "sina")
        nm = None
        roe = None
        debt_ratio = None
        if f_low.revenue_yuan and f_low.net_profit_yuan is not None and f_low.revenue_yuan > 0:
            nm = f_low.net_profit_yuan / f_low.revenue_yuan
        if f_low.equity_yuan and f_low.net_profit_yuan is not None and f_low.equity_yuan > 0:
            roe = f_low.net_profit_yuan / f_low.equity_yuan
        if f_low.assets_yuan and f_low.liabilities_yuan and f_low.assets_yuan > 0:
            debt_ratio = f_low.liabilities_yuan / f_low.assets_yuan
        pe = (mktcap_low / f_low.net_profit_yuan) if (mktcap_low is not None and f_low.net_profit_yuan and f_low.net_profit_yuan > 0) else None
        pb = (mktcap_low / f_low.equity_yuan) if (mktcap_low is not None and f_low.equity_yuan and f_low.equity_yuan > 0) else None
        fq_dates, fq_closes = _kline_cached(
            secid=secid, fqt=1, beg="20170101", end="20251231", cache_dir=cache_dir / "kline"
        )
        ret120, ret250, dd = _momentum_from_series(fq_dates, fq_closes, rr.low_date)
        dv_score = _score_dv(mktcap_low_yi, pe, pb, nm, roe, debt_ratio, ret250, dd)

        # TC: entry date then score at that date
        entry = _entry_date_trend_confirmed_from_series(fq_dates, fq_closes, rr.low_date)
        tc_score = None
        label_entry_10x: Optional[int] = None
        if entry:
            entry_raw = raw_map.get(entry)
            shares_entry = _get_total_shares(rr.stock.code, mkt_prefix, entry, cache_dir / "capital")
            mktcap_entry = (entry_raw * shares_entry) if (entry_raw is not None and shares_entry is not None) else None
            mktcap_entry_yi = (mktcap_entry / 1e8) if mktcap_entry is not None else None
            f_entry = _get_fundamentals_last_annual(rr.stock.code, entry, cache_dir / "sina")
            nm_e = (f_entry.net_profit_yuan / f_entry.revenue_yuan) if (f_entry.net_profit_yuan is not None and f_entry.revenue_yuan and f_entry.revenue_yuan > 0) else None
            roe_e = (f_entry.net_profit_yuan / f_entry.equity_yuan) if (f_entry.net_profit_yuan is not None and f_entry.equity_yuan and f_entry.equity_yuan > 0) else None
            pe_e = (mktcap_entry / f_entry.net_profit_yuan) if (mktcap_entry is not None and f_entry.net_profit_yuan and f_entry.net_profit_yuan > 0) else None
            pb_e = (mktcap_entry / f_entry.equity_yuan) if (mktcap_entry is not None and f_entry.equity_yuan and f_entry.equity_yuan > 0) else None
            r120_e, r250_e, dd_e = _momentum_from_series(fq_dates, fq_closes, entry)
            tc_score = _score_tc(mktcap_entry_yi, pe_e, pb_e, nm_e, roe_e, r120_e, r250_e, dd_e)
            # entry-based label: can it still reach 10x from entry close?
            try:
                entry_idx = fq_dates.index(entry)
                entry_close = fq_closes[entry_idx]
            except Exception:
                entry_close = None
            if entry_close is not None and entry_close > 0:
                label_entry_10x = 1 if _hits_10x_after(fq_dates, fq_closes, entry, entry_close, END_DATE) else 0

        return ModelScore(
            code=rr.stock.code,
            name=rr.stock.name,
            industry=rr.stock.industry,
            label_low_10x=1 if rr.hit_date else 0,
            low_date=rr.low_date,
            hit_date=rr.hit_date,
            dv_score=dv_score,
            tc_score=tc_score,
            tc_entry_date=entry,
            label_entry_10x=label_entry_10x,
            error="",
        )
    except Exception as e:
        msg = str(e)
        if len(msg) > 300:
            msg = msg[:300] + "..."
        return ModelScore(
            code=rr.stock.code,
            name=rr.stock.name,
            industry=rr.stock.industry,
            label_low_10x=1 if rr.hit_date else 0,
            low_date=rr.low_date,
            hit_date=rr.hit_date,
            dv_score=None,
            tc_score=None,
            tc_entry_date=None,
            label_entry_10x=None,
            error=msg,
        )


def _precision_at_k(items: List[Tuple[float, int]], k: int) -> float:
    top = items[:k]
    if not top:
        return 0.0
    return sum(lbl for _, lbl in top) / len(top)


def evaluate(scores: List[ModelScore]) -> Dict[str, Any]:
    positives_low = sum(1 for s in scores if s.label_low_10x == 1)
    base_rate_low = positives_low / len(scores) if scores else 0.0
    entries = [s for s in scores if s.label_entry_10x is not None]
    positives_entry = sum(1 for s in entries if s.label_entry_10x == 1)
    base_rate_entry = positives_entry / len(entries) if entries else 0.0

    def eval_model(get_score, get_label, pool: List[ModelScore], base_rate: float):
        pairs = [(get_score(s), get_label(s)) for s in pool if get_score(s) is not None and get_label(s) is not None]
        pairs.sort(key=lambda x: x[0], reverse=True)
        cover = len(pairs)
        out = {
            "coverage": cover,
            "base_rate": base_rate,
            "precision@50": _precision_at_k(pairs, 50),
            "precision@100": _precision_at_k(pairs, 100),
            "precision@200": _precision_at_k(pairs, 200),
            "precision@500": _precision_at_k(pairs, 500),
        }
        if cover:
            sc = [p[0] for p in pairs]
            out["score_median"] = statistics.median(sc)
            out["score_p90"] = sc[int(0.9 * (len(sc) - 1))]
        return out

    return {
        "count": len(scores),
        "positives_low": positives_low,
        "dv_low10x": eval_model(lambda s: s.dv_score, lambda s: s.label_low_10x, scores, base_rate_low),
        "tc_entry10x": eval_model(lambda s: s.tc_score, lambda s: s.label_entry_10x, entries, base_rate_entry),
        "tc_entry_coverage_count": len(entries),
        "positives_entry": positives_entry,
    }

def _read_tenbaggers(path: Path) -> List[TenbaggerRow]:
    rows: List[TenbaggerRow] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        for it in r:
            code = str(it.get("code") or it.get("股票代码") or "").strip()
            name = str(it.get("name") or it.get("股票名称") or "").strip()
            industry = str(it.get("industry") or it.get("行业") or "").strip()
            low_date = str(it.get("low_date") or it.get("低点日期") or "").strip()
            hit_date = str(it.get("hit_date") or it.get("首次十倍日期") or it.get("首次十倍日期") or "").strip()
            if not code or not low_date or not hit_date:
                continue
            rows.append(
                TenbaggerRow(
                    code=code,
                    name=name,
                    industry=industry,
                    low_date=low_date,
                    hit_date=hit_date,
                )
            )
    return rows


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--neg-multiple", type=int, default=3, help="negatives sampled per positive")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="debug: limit number of stocks (negatives) for scoring")
    ap.add_argument("--tenbaggers", default="", help="CSV path (tenbaggers list). If provided, skip full scan.")
    ap.add_argument("--exclude-st", action="store_true", help="exclude *ST/ST stocks by name")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    stocks = get_all_a_shares()
    if args.exclude_st:
        stocks = [s for s in stocks if "ST" not in s.name.upper()]

    stock_by_code = {s.code: s for s in stocks}

    positives: List[RangeResult] = []
    pos_codes: set[str] = set()
    if args.tenbaggers:
        tb_rows = _read_tenbaggers(Path(args.tenbaggers))
        if args.exclude_st:
            tb_rows = [t for t in tb_rows if "ST" not in t.name.upper()]
        for t in tb_rows:
            st = stock_by_code.get(t.code) or Stock(secid=_resolve_secid(t.code), code=t.code, name=t.name, industry=t.industry)
            positives.append(
                RangeResult(
                    stock=st,
                    low_date=t.low_date,
                    low_close_fq=float("nan"),
                    hit_date=t.hit_date,
                    hit_close_fq=None,
                )
            )
            pos_codes.add(t.code)
        print(f"loaded tenbaggers={len(positives)} from {args.tenbaggers}")
    else:
        # Fallback: full scan (slow).
        print(f"scanning universe={len(stocks)} workers={args.workers}")
        results: List[RangeResult] = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(compute_low_and_first_10x, s) for s in stocks]
            for fut in as_completed(futs):
                try:
                    rr = fut.result()
                except Exception:
                    rr = None
                if rr:
                    results.append(rr)
        positives = [r for r in results if r.hit_date]
        pos_codes = {r.stock.code for r in positives}
        print(f"positives={len(positives)} from full scan")

    # Sample negatives stratified by industry to keep distribution similar
    all_neg_candidates = [s for s in stocks if s.code not in pos_codes]
    if args.limit:
        all_neg_candidates = all_neg_candidates[: args.limit]
    neg_needed = min(len(all_neg_candidates), len(positives) * args.neg_multiple)
    by_ind: Dict[str, List[RangeResult]] = {}
    # desired per industry proportional to positives (use tenbagger industries)
    pos_by_ind: Dict[str, int] = {}
    for p in positives:
        pos_by_ind[p.stock.industry or "未知"] = pos_by_ind.get(p.stock.industry or "未知", 0) + 1
    # build negative pools per industry (just Stock for now)
    neg_by_ind_stock: Dict[str, List[Stock]] = {}
    for s in all_neg_candidates:
        neg_by_ind_stock.setdefault(s.industry or "未知", []).append(s)

    sampled_negs: List[RangeResult] = []
    for ind, cnt in pos_by_ind.items():
        take = int(round(cnt * args.neg_multiple))
        pool = neg_by_ind_stock.get(ind, [])
        if not pool:
            continue
        random.shuffle(pool)
        for s in pool[: min(take, len(pool))]:
            sampled_negs.append(RangeResult(stock=s, low_date="", low_close_fq=float("nan"), hit_date=None, hit_close_fq=None))
    if len(sampled_negs) < neg_needed:
        rest = [s for s in all_neg_candidates if s.code not in {r.stock.code for r in sampled_negs}]
        random.shuffle(rest)
        for s in rest[: (neg_needed - len(sampled_negs))]:
            sampled_negs.append(RangeResult(stock=s, low_date="", low_close_fq=float("nan"), hit_date=None, hit_close_fq=None))
    sampled_negs = sampled_negs[:neg_needed]

    # Fill negative low_date by computing min close within window
    def fill_negative_low(rr: RangeResult) -> Optional[RangeResult]:
        secid = _resolve_secid(rr.stock.code)
        fq_dates, fq_closes = _kline_cached(
            secid=secid, fqt=1, beg="20190101", end="20251231", cache_dir=cache_dir / "kline_scan"
        )
        m = _series_min_in_window(fq_dates, fq_closes, START_DATE, END_DATE)
        if not m:
            return None
        low_d, low_c = m
        return RangeResult(stock=rr.stock, low_date=low_d, low_close_fq=low_c, hit_date=None, hit_close_fq=None)

    filled_negs: List[RangeResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fill_negative_low, rr) for rr in sampled_negs]
        for fut in as_completed(futs):
            try:
                v = fut.result()
            except Exception:
                v = None
            if v:
                filled_negs.append(v)

    sample = positives + filled_negs
    random.shuffle(sample)
    print(f"sample size={len(sample)} (pos={len(positives)} neg={len(filled_negs)})")

    # Step2: compute model scores for sample
    scores: List[ModelScore] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(score_one, rr, cache_dir) for rr in sample]
        for fut in as_completed(futs):
            try:
                scores.append(fut.result())
            except Exception:
                continue

    scores.sort(key=lambda s: (s.label_low_10x, s.code))
    out_csv = out_dir / "backtest_two_models_scores.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "股票代码",
                "股票名称",
                "行业",
                "标签_低点后十倍股(1/0)",
                "低点日期",
                "首次十倍日期",
                "模型A_抄底分(0-100)",
                "模型B_趋势确认分(0-100)",
                "模型B_入场日期",
                "标签_入场后十倍股(1/0)",
                "错误(如有)",
            ]
        )
        for s in scores:
            w.writerow(
                [
                    s.code,
                    s.name,
                    s.industry,
                    s.label_low_10x,
                    s.low_date,
                    s.hit_date or "",
                    "" if s.dv_score is None else f"{s.dv_score:.1f}",
                    "" if s.tc_score is None else f"{s.tc_score:.1f}",
                    s.tc_entry_date or "",
                    "" if s.label_entry_10x is None else s.label_entry_10x,
                    s.error,
                ]
            )

    summary = evaluate(scores)
    (out_dir / "backtest_two_models_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
