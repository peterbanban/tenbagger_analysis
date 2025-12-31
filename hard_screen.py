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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EM_CLLIST_URL = (
    "https://push2.eastmoney.com/api/qt/clist/get"
    "?pn={pn}&pz=100&po=1&np=1&fltt=2&invt=2&fid=f3"
    "&fs=m:1+t:2,m:1+t:23,m:0+t:6,m:0+t:80"
    "&fields=f12,f14,f13,f100,f40,f9,f23"
)

EM_STOCK_GET_URL = (
    "https://push2.eastmoney.com/api/qt/stock/get"
    "?secid={secid}&fields=f57,f58,f103,f104,f105,f106,f107,f108,f109,f110,f111"
)

EM_KLINE_URL = (
    "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    "?secid={secid}&klt=101&fqt=1&beg={beg}&end={end}"
    "&fields1=f1,f2,f3,f4,f5"
    "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
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


def _today_ymd() -> str:
    # Use server-independent approx; Eastmoney ignores future dates anyway.
    return time.strftime("%Y%m%d")


@dataclass(frozen=True)
class ListedStock:
    secid: str
    code: str
    name: str
    industry: str
    mktcap_yuan: Optional[float]
    pe_ttm: Optional[float]
    pb: Optional[float]


@dataclass(frozen=True)
class Enriched:
    stock: ListedStock
    revenue_yuan: Optional[float]
    net_profit_yuan: Optional[float]
    net_margin: Optional[float]
    ret_120d: Optional[float]
    ret_250d: Optional[float]
    dd_from_250d_high: Optional[float]
    score: float
    flags: str


def get_all_a_shares() -> List[ListedStock]:
    out: List[ListedStock] = []
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
            secid = f"{mkt}.{code}"

            def f(x):
                try:
                    return float(x)
                except Exception:
                    return None

            out.append(
                ListedStock(
                    secid=secid,
                    code=code,
                    name=name,
                    industry=industry,
                    mktcap_yuan=f(it.get("f40")),
                    pe_ttm=f(it.get("f9")),
                    pb=f(it.get("f23")),
                )
            )
        if total and pn * 100 >= total:
            break
        pn += 1
        time.sleep(0.15)
    return out


def _get_financials(secid: str) -> Tuple[Optional[float], Optional[float]]:
    j = _http_get_json(EM_STOCK_GET_URL.format(secid=secid))
    data = j.get("data") or {}
    rev = data.get("f104")  # heuristic: revenue (latest annual/TTM, yuan)
    np = data.get("f109")  # heuristic: net profit attributable? (yuan)
    try:
        rev_f = float(rev) if rev not in (None, "-", "") else None
    except Exception:
        rev_f = None
    try:
        np_f = float(np) if np not in (None, "-", "") else None
    except Exception:
        np_f = None
    return rev_f, np_f


def _get_momentum(secid: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # Use last ~1.2 years to compute 120d/250d return & drawdown from 250d high.
    # Kline endpoint returns trading days within date range.
    end = _today_ymd()
    beg = "20190101"
    j = _http_get_json(EM_KLINE_URL.format(secid=secid, beg=beg, end=end))
    data = j.get("data") or {}
    kl = data.get("klines") or []
    closes: List[float] = []
    for row in kl:
        parts = row.split(",")
        if len(parts) < 3:
            continue
        try:
            closes.append(float(parts[2]))
        except Exception:
            continue
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


def enrich_one(stock: ListedStock) -> Enriched:
    flags: List[str] = []
    rev, np = _get_financials(stock.secid)
    net_margin = None
    if rev and rev > 0 and np is not None:
        net_margin = np / rev
    ret120, ret250, dd = _get_momentum(stock.secid)

    # Score = quality-ish + momentum-ish + tenbagger-prior
    score = 0.0
    # size sweet spot
    if stock.mktcap_yuan is not None:
        mkt_yi = stock.mktcap_yuan / 1e8
        if 10 <= mkt_yi <= 100:
            score += 10
        elif 5 <= mkt_yi < 10 or 100 < mkt_yi <= 200:
            score += 6
        else:
            score += 0
    # profitability
    if np is not None and np > 0:
        score += 8
    else:
        flags.append("亏损/未知利润")
    if net_margin is not None:
        if net_margin >= 0.10:
            score += 6
        elif net_margin >= 0.05:
            score += 3
        elif net_margin < 0:
            flags.append("净利率<0")
    # valuation sanity
    if stock.pe_ttm is not None:
        if 5 <= stock.pe_ttm <= 50:
            score += 6
        elif stock.pe_ttm <= 0 or stock.pe_ttm > 80:
            flags.append("PE异常")
    else:
        flags.append("PE缺失")
    if stock.pb is not None and 0 < stock.pb <= 10:
        score += 3

    # momentum
    if ret250 is not None:
        if ret250 > 0:
            score += 5
        if ret250 > 0.5:
            score += 3
    else:
        flags.append("K线不足")
    if dd is not None and dd > -0.25:
        score += 3  # close to highs implies trend strength

    return Enriched(
        stock=stock,
        revenue_yuan=rev,
        net_profit_yuan=np,
        net_margin=net_margin,
        ret_120d=ret120,
        ret_250d=ret250,
        dd_from_250d_high=dd,
        score=score,
        flags=";".join(flags),
    )


def load_industry_prior(summary_path: Path, top_n: int = 20) -> Dict[str, float]:
    if not summary_path.exists():
        return {}
    j = json.loads(summary_path.read_text("utf-8", errors="ignore"))
    top = j.get("industry_top20") or []
    out: Dict[str, float] = {}
    for i, (ind, cnt) in enumerate(top[:top_n]):
        # simple prior: higher rank -> higher boost
        out[ind] = max(0.0, 2.5 - i * 0.1)
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--exclude-st", action="store_true", default=True)
    ap.add_argument("--min-mktcap-yi", type=float, default=10.0)
    ap.add_argument("--max-mktcap-yi", type=float, default=100.0)
    ap.add_argument("--min-pe", type=float, default=5.0)
    ap.add_argument("--max-pe", type=float, default=60.0)
    ap.add_argument("--min-net-margin", type=float, default=0.05)
    ap.add_argument("--min-revenue-yi", type=float, default=5.0)
    ap.add_argument("--min-net-profit-yi", type=float, default=0.5)
    ap.add_argument("--min-ret-250d", type=float, default=-0.10)
    ap.add_argument("--min-dd-from-250d-high", type=float, default=-0.30)
    ap.add_argument("--min-score", type=float, default=18.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: limit number of stocks")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prior = load_industry_prior(Path("a_share_tenbagger/out/summary_no_st.json"))

    stocks = get_all_a_shares()
    if args.exclude_st:
        stocks = [s for s in stocks if "ST" not in s.name.upper()]
    if args.limit:
        stocks = stocks[: args.limit]

    print(f"screening stocks={len(stocks)} workers={args.workers}", file=sys.stderr)
    enriched: List[Enriched] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(enrich_one, s): s for s in stocks}
        for fut in as_completed(futs):
            try:
                enriched.append(fut.result())
            except Exception:
                continue

    # Apply hard filters
    candidates: List[Enriched] = []
    for e in enriched:
        mkt = e.stock.mktcap_yuan
        if mkt is None:
            continue
        mkt_yi = mkt / 1e8
        if not (args.min_mktcap_yi <= mkt_yi <= args.max_mktcap_yi):
            continue
        if e.stock.pe_ttm is None or not (args.min_pe <= e.stock.pe_ttm <= args.max_pe):
            continue
        if e.revenue_yuan is None or e.revenue_yuan < args.min_revenue_yi * 1e8:
            continue
        if e.net_profit_yuan is None or e.net_profit_yuan < args.min_net_profit_yi * 1e8:
            continue
        if e.net_margin is None or e.net_margin < args.min_net_margin:
            continue
        if e.ret_250d is None or e.ret_250d < args.min_ret_250d:
            continue
        if e.dd_from_250d_high is None or e.dd_from_250d_high < args.min_dd_from_250d_high:
            continue
        # add prior boost
        score = e.score + prior.get(e.stock.industry, 0.0)
        if score < args.min_score:
            continue
        candidates.append(
            Enriched(
                stock=e.stock,
                revenue_yuan=e.revenue_yuan,
                net_profit_yuan=e.net_profit_yuan,
                net_margin=e.net_margin,
                ret_120d=e.ret_120d,
                ret_250d=e.ret_250d,
                dd_from_250d_high=e.dd_from_250d_high,
                score=score,
                flags=e.flags,
            )
        )

    candidates.sort(key=lambda x: x.score, reverse=True)

    out_csv = out_dir / "candidates.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "code",
                "name",
                "industry",
                "mktcap_yi",
                "pe_ttm",
                "pb",
                "revenue_yi",
                "net_profit_yi",
                "net_margin",
                "ret_120d",
                "ret_250d",
                "dd_from_250d_high",
                "score",
                "flags",
            ]
        )
        for e in candidates:
            mkt_yi = (e.stock.mktcap_yuan or 0) / 1e8
            w.writerow(
                [
                    e.stock.code,
                    e.stock.name,
                    e.stock.industry,
                    f"{mkt_yi:.2f}",
                    "" if e.stock.pe_ttm is None else f"{e.stock.pe_ttm:.2f}",
                    "" if e.stock.pb is None else f"{e.stock.pb:.2f}",
                    "" if e.revenue_yuan is None else f"{e.revenue_yuan/1e8:.2f}",
                    "" if e.net_profit_yuan is None else f"{e.net_profit_yuan/1e8:.2f}",
                    "" if e.net_margin is None else f"{e.net_margin:.4f}",
                    "" if e.ret_120d is None else f"{e.ret_120d:.4f}",
                    "" if e.ret_250d is None else f"{e.ret_250d:.4f}",
                    "" if e.dd_from_250d_high is None else f"{e.dd_from_250d_high:.4f}",
                    f"{e.score:.2f}",
                    e.flags,
                ]
            )

    summary = {
        "input_count": len(stocks),
        "candidate_count": len(candidates),
        "params": {
            "min_mktcap_yi": args.min_mktcap_yi,
            "max_mktcap_yi": args.max_mktcap_yi,
            "min_pe": args.min_pe,
            "max_pe": args.max_pe,
            "min_net_margin": args.min_net_margin,
            "min_revenue_yi": args.min_revenue_yi,
            "min_net_profit_yi": args.min_net_profit_yi,
            "min_ret_250d": args.min_ret_250d,
            "min_dd_from_250d_high": args.min_dd_from_250d_high,
            "min_score": args.min_score,
            "exclude_st": args.exclude_st,
        },
        "top10": [
            {
                "code": e.stock.code,
                "name": e.stock.name,
                "industry": e.stock.industry,
                "mktcap_yi": (e.stock.mktcap_yuan or 0) / 1e8,
                "score": e.score,
            }
            for e in candidates[:10]
        ],
    }
    (out_dir / "candidates_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
