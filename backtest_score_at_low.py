#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


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

EM_STOCK_GET_URL = (
    "https://push2.eastmoney.com/api/qt/stock/get"
    "?secid={secid}&fields=f57,f58"
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


def _parse_ymd(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _resolve_secid(code: str) -> str:
    # Try SH first, then SZ/BJ. We only need a secid that the kline endpoint accepts.
    candidates = []
    if code.startswith(("6", "9")):
        candidates = ["1", "0"]
    elif code.startswith(("0", "3")):
        candidates = ["0", "1"]
    else:
        candidates = ["0", "1"]
    for mkt in candidates:
        secid = f"{mkt}.{code}"
        try:
            j = _http_get_json(EM_STOCK_GET_URL.format(secid=secid), retries=2)
            if (j.get("data") or {}).get("f57"):
                return secid
        except Exception:
            continue
    return f"0.{code}"


def _get_kline_closes(secid: str, *, fqt: int, beg: str, end: str) -> List[Tuple[str, float]]:
    j = _http_get_json(EM_KLINE_URL.format(secid=secid, fqt=fqt, beg=beg, end=end))
    data = j.get("data") or {}
    kl = data.get("klines") or []
    out: List[Tuple[str, float]] = []
    for row in kl:
        parts = row.split(",")
        if len(parts) < 3:
            continue
        d = parts[0]
        try:
            close = float(parts[2])
        except Exception:
            continue
        out.append((d, close))
    return out


def _shares_at_date(changes: List[Dict[str, Any]], d: str) -> Optional[float]:
    target = _parse_ymd(d).date()
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


def _get_total_shares_on(code: str, mkt_prefix: str, d: str) -> Optional[float]:
    cap = _http_get_json(EM_CAPITAL_URL.format(mkt=mkt_prefix, code=code))
    changes = cap.get("lngbbd") or []
    return _shares_at_date(changes, d)


def _sina_statement_value(
    html: str, *, label_contains: str, date_str: str
) -> Optional[float]:
    # html is gb18030 typically; but we decode outside. Here we parse tables.
    tables = pd.read_html(html)
    stmt = None
    for t in tables:
        flat = " ".join(map(str, t.values.flatten()))
        if "报表日期" in flat:
            stmt = t
            break
    if stmt is None:
        return None

    # find which column matches date_str (YYYY-12-31)
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


def _get_last_annual_financials(code: str, low_date: str, cache_dir: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # Returns (revenue_yuan, net_profit_yuan, equity_yuan) using latest complete annual report before low_date.
    y = int(low_date[:4]) - 1
    date_str = f"{y}-12-31"

    cache_dir.mkdir(parents=True, exist_ok=True)
    profit_cache = cache_dir / f"profit_{code}_{y}.html"
    bal_cache = cache_dir / f"balance_{code}_{y}.html"

    if not profit_cache.exists():
        b = _http_get_bytes(SINA_PROFIT_URL.format(code=code, year=y))
        profit_cache.write_bytes(b)
    if not bal_cache.exists():
        b = _http_get_bytes(SINA_BALANCE_URL.format(code=code, year=y))
        bal_cache.write_bytes(b)

    profit_html = profit_cache.read_bytes().decode("gb18030", errors="ignore")
    bal_html = bal_cache.read_bytes().decode("gb18030", errors="ignore")

    revenue_wy = _sina_statement_value(profit_html, label_contains="营业总收入", date_str=date_str)
    np_parent_wy = _sina_statement_value(profit_html, label_contains="归属于母公司所有者的净利润", date_str=date_str)
    equity_parent_wy = _sina_statement_value(bal_html, label_contains="归属于母公司股东权益合计", date_str=date_str)

    if revenue_wy is not None:
        revenue_wy *= 10000  # 万元 -> 元
    if np_parent_wy is not None:
        np_parent_wy *= 10000
    if equity_parent_wy is not None:
        equity_parent_wy *= 10000
    return revenue_wy, np_parent_wy, equity_parent_wy


def _momentum_asof(secid: str, low_date: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # Use forward-adjusted closes for momentum.
    # Pull roughly 2 years before low_date to ensure enough trading days.
    dt = _parse_ymd(low_date)
    beg = f"{dt.year-2}0101"
    end = low_date.replace("-", "")
    kl = _get_kline_closes(secid, fqt=1, beg=beg, end=end)
    closes = [c for _, c in kl if c > 0]
    if len(closes) < 260:
        return None, None, None
    c0 = closes[-1]
    c120 = closes[-121]
    c250 = closes[-251]
    ret120 = (c0 / c120 - 1.0) if c120 > 0 else None
    ret250 = (c0 / c250 - 1.0) if c250 > 0 else None
    high250 = max(closes[-251:])
    dd = (c0 / high250 - 1.0) if high250 > 0 else None
    return ret120, ret250, dd


@dataclass(frozen=True)
class ScoredRow:
    code: str
    name: str
    industry: str
    low_date: str
    score: float
    mktcap_yi: Optional[float]
    pe_approx: Optional[float]
    pb_approx: Optional[float]
    revenue_yi: Optional[float]
    net_profit_yi: Optional[float]
    net_margin: Optional[float]
    ret_120d: Optional[float]
    ret_250d: Optional[float]
    dd_from_250d_high: Optional[float]
    flags: str


def score_at_low(code: str, name: str, industry: str, low_date: str, cache_dir: Path) -> ScoredRow:
    flags: List[str] = []
    secid = _resolve_secid(code)
    mkt = secid.split(".", 1)[0]
    mkt_prefix = "SH" if mkt == "1" else "SZ"

    # Market cap at low_date: raw close * shares
    raw_kl = _get_kline_closes(
        secid,
        fqt=0,
        beg=f"{int(low_date[:4])-1}0101",
        end=low_date.replace("-", ""),
    )
    raw_close_map = {d: c for d, c in raw_kl}
    raw_close = raw_close_map.get(low_date)
    if raw_close is None:
        flags.append("缺少不复权收盘价")

    shares = _get_total_shares_on(code, mkt_prefix, low_date)
    if shares is None:
        flags.append("缺少总股本")

    mktcap_yuan = None
    if raw_close is not None and shares is not None:
        mktcap_yuan = raw_close * shares
    mktcap_yi = (mktcap_yuan / 1e8) if mktcap_yuan is not None else None

    # Latest complete annual before low_date as a proxy for TTM fundamentals around low_date.
    rev, np, equity = _get_last_annual_financials(code, low_date, cache_dir / "sina_cache")
    revenue_yi = (rev / 1e8) if rev is not None else None
    net_profit_yi = (np / 1e8) if np is not None else None
    net_margin = (np / rev) if (np is not None and rev and rev > 0) else None

    pe = (mktcap_yuan / np) if (mktcap_yuan is not None and np and np > 0) else None
    pb = (mktcap_yuan / equity) if (mktcap_yuan is not None and equity and equity > 0) else None

    if np is None:
        flags.append("缺少净利润(上一年年报)")
    elif np <= 0:
        flags.append("上一年净利润<=0")
    if rev is None:
        flags.append("缺少年营收(上一年年报)")

    ret120, ret250, dd = _momentum_asof(secid, low_date)
    if ret250 is None:
        flags.append("动量数据不足")

    # Scoring: re-use the idea of hard_screen but strictly using low_date-timepoint inputs.
    score = 0.0
    # size sweet spot at low_date
    if mktcap_yi is not None:
        if 10 <= mktcap_yi <= 100:
            score += 10
        elif 5 <= mktcap_yi < 10 or 100 < mktcap_yi <= 200:
            score += 6
    else:
        flags.append("缺少市值")

    # profitability
    if np is not None and np > 0:
        score += 8
    if net_margin is not None:
        if net_margin >= 0.10:
            score += 6
        elif net_margin >= 0.05:
            score += 3
        elif net_margin < 0:
            flags.append("净利率<0")

    # valuation sanity (approx PE/PB at low_date)
    if pe is not None:
        if 5 <= pe <= 50:
            score += 6
        elif pe > 80:
            flags.append("PE偏高")
    else:
        flags.append("PE不可算")
    if pb is not None and 0 < pb <= 10:
        score += 3

    # momentum at low_date
    if ret250 is not None:
        if ret250 > 0:
            score += 5
        if ret250 > 0.5:
            score += 3
    if dd is not None and dd > -0.25:
        score += 3

    return ScoredRow(
        code=code,
        name=name,
        industry=industry,
        low_date=low_date,
        score=score,
        mktcap_yi=mktcap_yi,
        pe_approx=pe,
        pb_approx=pb,
        revenue_yi=revenue_yi,
        net_profit_yi=net_profit_yi,
        net_margin=net_margin,
        ret_120d=ret120,
        ret_250d=ret250,
        dd_from_250d_high=dd,
        flags=";".join(flags),
    )


def _read_input_csv(path: Path) -> List[Tuple[str, str, str, str]]:
    # Support both English and Chinese headers.
    raw = path.read_text("utf-8", errors="ignore").splitlines()
    if not raw:
        return []
    header = raw[0].split(",")
    cn = "股票代码" in header and "低点日期" in header

    rows: List[Tuple[str, str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if cn:
                code = (row.get("股票代码") or "").strip()
                name = (row.get("股票名称") or "").strip()
                industry = (row.get("行业") or "").strip()
                low_date = (row.get("低点日期") or "").strip()
            else:
                code = (row.get("code") or "").strip()
                name = (row.get("name") or "").strip()
                industry = (row.get("industry") or "").strip()
                low_date = (row.get("low_date") or "").strip()
            if not code or not low_date:
                continue
            rows.append((code, name, industry, low_date))
    return rows


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="tenbaggers csv (your exported file)")
    ap.add_argument("--out", required=True, help="output csv path")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-code", action="append", default=[], help="only score specific stock code(s)")
    args = ap.parse_args(argv)

    inp = Path(args.input)
    out = Path(args.out)
    cache_dir = out.parent / ".cache_backtest"
    cache_dir.mkdir(parents=True, exist_ok=True)

    items = _read_input_csv(inp)
    if args.only_code:
        allow = {c.strip() for c in args.only_code if c.strip()}
        items = [x for x in items if x[0] in allow]
    if args.limit:
        items = items[: args.limit]
    print(f"scoring items={len(items)} workers={args.workers}", file=sys.stderr)

    results: List[ScoredRow] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(score_at_low, code, name, industry, low_date, cache_dir)
            for code, name, industry, low_date in items
        ]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                continue

    results.sort(key=lambda r: (r.low_date, r.code))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "股票代码",
                "股票名称",
                "行业",
                "低点日期",
                "模型分值(低点时点)",
                "低点市值(亿元,估算)",
                "低点PE(估算)",
                "低点PB(估算)",
                "上一年营收(亿元)",
                "上一年归母净利(亿元)",
                "上一年净利率",
                "低点时近120日涨跌幅(前复权)",
                "低点时近250日涨跌幅(前复权)",
                "低点时距250日高点回撤(前复权)",
                "标记",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.code,
                    r.name,
                    r.industry,
                    r.low_date,
                    f"{r.score:.2f}",
                    "" if r.mktcap_yi is None else f"{r.mktcap_yi:.2f}",
                    "" if r.pe_approx is None else f"{r.pe_approx:.2f}",
                    "" if r.pb_approx is None else f"{r.pb_approx:.2f}",
                    "" if r.revenue_yi is None else f"{r.revenue_yi:.2f}",
                    "" if r.net_profit_yi is None else f"{r.net_profit_yi:.2f}",
                    "" if r.net_margin is None else f"{r.net_margin:.4f}",
                    "" if r.ret_120d is None else f"{r.ret_120d:.4f}",
                    "" if r.ret_250d is None else f"{r.ret_250d:.4f}",
                    "" if r.dd_from_250d_high is None else f"{r.dd_from_250d_high:.4f}",
                    r.flags,
                ]
            )

    print(f"wrote {out} rows={len(results)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
