#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from backtest_two_models import (  # type: ignore
    END_DATE,
    START_DATE,
    Stock,
    _get_fundamentals_last_annual,
    _get_total_shares,
    _kline_cached,
    _momentum_from_series,
    _resolve_secid,
    _score_dv,
    _score_tc,
    get_all_a_shares,
)

from sqlite_market import get_dates_and_closes, list_stocks, open_db


def _ma(closes: list[float], end_idx: int, window: int) -> Optional[float]:
    if end_idx - window + 1 < 0:
        return None
    w = closes[end_idx - window + 1 : end_idx + 1]
    if len(w) != window:
        return None
    return sum(w) / window


def _entry_ok_today(dates: list[str], closes: list[float], asof_date: str) -> Tuple[bool, Optional[float], Optional[float]]:
    try:
        i = dates.index(asof_date)
    except ValueError:
        return False, None, None
    ma20 = _ma(closes, i, 20)
    ma60 = _ma(closes, i, 60)
    if ma20 is None or ma60 is None:
        return False, ma20, ma60
    c0 = closes[i]
    ok = (ma20 > ma60) and (c0 > ma20)
    return ok, ma20, ma60


def _last_trading_date(dates: list[str]) -> Optional[str]:
    if not dates:
        return None
    return max(dates)


@dataclass(frozen=True)
class Scored:
    code: str
    name: str
    industry: str
    asof_date: str
    score_a: Optional[float]
    score_b: Optional[float]
    score_composite_strict: Optional[float]
    score_composite_loose: Optional[float]
    entry_ok: Optional[int]
    ma20: Optional[float]
    ma60: Optional[float]
    mktcap_yi: Optional[float]
    pe: Optional[float]
    pb: Optional[float]
    nm: Optional[float]
    roe: Optional[float]
    debt_ratio: Optional[float]
    ret120: Optional[float]
    ret250: Optional[float]
    dd250: Optional[float]
    error: str


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _score_one(stock: Stock, cache_dir: Path) -> Scored:
    try:
        secid = _resolve_secid(stock.code)
        mkt = secid.split(".", 1)[0]
        mkt_prefix = "SH" if mkt == "1" else "SZ"

        fq_dates, fq_closes = _kline_cached(
            secid=secid, fqt=1, beg="20170101", end=END_DATE.replace("-", ""), cache_dir=cache_dir / "kline"
        )
        asof_date = _last_trading_date(fq_dates)
        if not asof_date:
            raise RuntimeError("no kline data")

        ret120, ret250, dd = _momentum_from_series(fq_dates, fq_closes, asof_date)
        entry_ok, ma20, ma60 = _entry_ok_today(fq_dates, fq_closes, asof_date)

        raw_dates, raw_closes = _kline_cached(
            secid=secid, fqt=0, beg="20170101", end=END_DATE.replace("-", ""), cache_dir=cache_dir / "kline"
        )
        raw_map = dict(zip(raw_dates, raw_closes))
        close_raw = raw_map.get(asof_date)
        shares = _get_total_shares(stock.code, mkt_prefix, asof_date, cache_dir / "capital")
        mktcap = (close_raw * shares) if (close_raw is not None and shares is not None) else None
        mktcap_yi = (mktcap / 1e8) if mktcap is not None else None

        f = _get_fundamentals_last_annual(stock.code, asof_date, cache_dir / "sina")
        nm = _safe_div(f.net_profit_yuan, f.revenue_yuan) if (f.revenue_yuan and f.revenue_yuan > 0) else None
        roe = _safe_div(f.net_profit_yuan, f.equity_yuan) if (f.equity_yuan and f.equity_yuan > 0) else None
        debt_ratio = _safe_div(f.liabilities_yuan, f.assets_yuan) if (f.assets_yuan and f.assets_yuan > 0) else None
        pe = _safe_div(mktcap, f.net_profit_yuan) if (f.net_profit_yuan and f.net_profit_yuan > 0) else None
        pb = _safe_div(mktcap, f.equity_yuan) if (f.equity_yuan and f.equity_yuan > 0) else None

        score_a = _score_dv(mktcap_yi, pe, pb, nm, roe, debt_ratio, ret250, dd)
        # Always compute model-B score; use entry_ok as the separate "can enter now" flag.
        score_b = _score_tc(mktcap_yi, pe, pb, nm, roe, ret120, ret250, dd)
        entry_ok_int = 1 if entry_ok else 0

        def composite(sa: Optional[float], sb: Optional[float]) -> Optional[float]:
            if sa is None and sb is None:
                return None
            if sa is None:
                return sb
            if sb is None:
                return sa
            return 0.5 * sa + 0.5 * sb

        score_composite_loose = composite(score_a, score_b)
        score_composite_strict = composite(score_a, score_b) if entry_ok else score_a

        return Scored(
            code=stock.code,
            name=stock.name,
            industry=stock.industry,
            asof_date=asof_date,
            score_a=score_a,
            score_b=score_b,
            score_composite_strict=score_composite_strict,
            score_composite_loose=score_composite_loose,
            entry_ok=entry_ok_int,
            ma20=ma20,
            ma60=ma60,
            mktcap_yi=mktcap_yi,
            pe=pe,
            pb=pb,
            nm=nm,
            roe=roe,
            debt_ratio=debt_ratio,
            ret120=ret120,
            ret250=ret250,
            dd250=dd,
            error="",
        )
    except Exception as e:
        msg = str(e)
        if len(msg) > 300:
            msg = msg[:300] + "..."
        return Scored(
            code=stock.code,
            name=stock.name,
            industry=stock.industry,
            asof_date="",
            score_a=None,
            score_b=None,
            score_composite_strict=None,
            score_composite_loose=None,
            entry_ok=None,
            ma20=None,
            ma60=None,
            mktcap_yi=None,
            pe=None,
            pb=None,
            nm=None,
            roe=None,
            debt_ratio=None,
            ret120=None,
            ret250=None,
            dd250=None,
            error=msg,
        )


def _fmt(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.4f}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude-st", action="store_true")
    ap.add_argument(
        "--db",
        default="",
        help="sqlite db path for offline stocks+kline. default: <project>/data/tenbagger_analysis_market.sqlite if exists",
    )
    args = ap.parse_args(argv)

    random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Prefer local SQLite market db.
    db_path = Path(args.db).expanduser() if args.db else (Path(__file__).resolve().parent / "data" / "tenbagger_analysis_market.sqlite")
    db_conn = None
    stocks: list[Stock] = []
    if db_path.exists():
        db_conn = open_db(db_path)
        stocks = [Stock(secid=s.secid, code=s.code, name=s.name, industry=s.industry) for s in list_stocks(db_conn)]
        if args.exclude_st:
            stocks = [s for s in stocks if "ST" not in s.name.upper()]
        print(f"using sqlite db: {db_path}")
    else:
        stocks = get_all_a_shares()
        if args.exclude_st:
            stocks = [s for s in stocks if "ST" not in s.name.upper()]
        print("using network universe (no sqlite db found)")

    print(f"universe={len(stocks)} workers={args.workers}")

    def score_one_with_series(s: Stock) -> Scored:
        try:
            secid = s.secid or _resolve_secid(s.code)
            mkt = secid.split(".", 1)[0]
            mkt_prefix = "SH" if mkt == "1" else "SZ"

            if db_conn is not None:
                fq_dates, fq_closes = get_dates_and_closes(db_conn, s.code, 1)
                raw_dates, raw_closes = get_dates_and_closes(db_conn, s.code, 0)
            else:
                fq_dates, fq_closes = _kline_cached(
                    secid=secid, fqt=1, beg="20170101", end=END_DATE.replace("-", ""), cache_dir=cache_dir / "kline"
                )
                raw_dates, raw_closes = _kline_cached(
                    secid=secid, fqt=0, beg="20170101", end=END_DATE.replace("-", ""), cache_dir=cache_dir / "kline"
                )

            asof_date = _last_trading_date(fq_dates)
            if not asof_date:
                raise RuntimeError("no kline data")

            ret120, ret250, dd = _momentum_from_series(fq_dates, fq_closes, asof_date)
            entry_ok, ma20, ma60 = _entry_ok_today(fq_dates, fq_closes, asof_date)

            raw_map = dict(zip(raw_dates, raw_closes))
            close_raw = raw_map.get(asof_date)
            shares = _get_total_shares(s.code, mkt_prefix, asof_date, cache_dir / "capital")
            mktcap = (close_raw * shares) if (close_raw is not None and shares is not None) else None
            mktcap_yi = (mktcap / 1e8) if mktcap is not None else None

            f = _get_fundamentals_last_annual(s.code, asof_date, cache_dir / "sina")
            nm = _safe_div(f.net_profit_yuan, f.revenue_yuan) if (f.revenue_yuan and f.revenue_yuan > 0) else None
            roe = _safe_div(f.net_profit_yuan, f.equity_yuan) if (f.equity_yuan and f.equity_yuan > 0) else None
            debt_ratio = _safe_div(f.liabilities_yuan, f.assets_yuan) if (f.assets_yuan and f.assets_yuan > 0) else None
            pe = (mktcap / f.net_profit_yuan) if (mktcap is not None and f.net_profit_yuan and f.net_profit_yuan > 0) else None
            pb = (mktcap / f.equity_yuan) if (mktcap is not None and f.equity_yuan and f.equity_yuan > 0) else None

            score_a = _score_dv(mktcap_yi, pe, pb, nm, roe, debt_ratio, ret250, dd)
            score_b = _score_tc(mktcap_yi, pe, pb, nm, roe, ret120, ret250, dd)

            def composite(sa: Optional[float], sb: Optional[float]) -> Optional[float]:
                if sa is None and sb is None:
                    return None
                if sa is None:
                    return sb
                if sb is None:
                    return sa
                return 0.5 * sa + 0.5 * sb

            score_composite_loose = composite(score_a, score_b)
            score_composite_strict = composite(score_a, score_b) if entry_ok else score_a

            return Scored(
                code=s.code,
                name=s.name,
                industry=s.industry,
                asof_date=asof_date,
                score_a=score_a,
                score_b=score_b,
                score_composite_strict=score_composite_strict,
                score_composite_loose=score_composite_loose,
                entry_ok=1 if entry_ok else 0,
                ma20=ma20,
                ma60=ma60,
                mktcap_yi=mktcap_yi,
                pe=pe,
                pb=pb,
                nm=nm,
                roe=roe,
                debt_ratio=debt_ratio,
                ret120=ret120,
                ret250=ret250,
                dd250=dd,
                error="",
            )
        except Exception as e:
            msg = str(e)
            if len(msg) > 300:
                msg = msg[:300] + "..."
            return Scored(
                code=s.code,
                name=s.name,
                industry=s.industry,
                asof_date="",
                score_a=None,
                score_b=None,
                score_composite_strict=None,
                score_composite_loose=None,
                entry_ok=None,
                ma20=None,
                ma60=None,
                mktcap_yi=None,
                pe=None,
                pb=None,
                nm=None,
                roe=None,
                debt_ratio=None,
                ret120=None,
                ret250=None,
                dd250=None,
                error=msg,
            )

    scored: list[Scored] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(score_one_with_series, s) for s in stocks]
        for fut in as_completed(futs):
            scored.append(fut.result())

    # Keep only those with at least one score.
    scored_ok = [s for s in scored if s.score_composite_strict is not None]

    scored_ok.sort(key=lambda s: (s.score_composite_strict or -1), reverse=True)
    top = scored_ok[: args.top]

    out_csv = out_dir / "top30_two_models.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "排名",
                "股票代码",
                "股票名称",
                "行业",
                "截至日期",
                "综合得分(严格入场)",
                "综合得分(不严格)",
                "模型A_抄底分",
                "模型B_趋势确认分",
                "趋势入场条件(1/0)",
                "MA20",
                "MA60",
                "总市值(亿)",
                "PE(估算)",
                "PB(估算)",
                "净利率(估算)",
                "ROE(估算)",
                "资产负债率(估算)",
                "120日收益",
                "250日收益",
                "距250日高点回撤",
                "错误(如有)",
            ]
        )
        for i, s in enumerate(top, start=1):
            w.writerow(
                [
                    i,
                    s.code,
                    s.name,
                    s.industry,
                    s.asof_date,
                    "" if s.score_composite_strict is None else f"{s.score_composite_strict:.1f}",
                    "" if s.score_composite_loose is None else f"{s.score_composite_loose:.1f}",
                    "" if s.score_a is None else f"{s.score_a:.1f}",
                    "" if s.score_b is None else f"{s.score_b:.1f}",
                    "" if s.entry_ok is None else s.entry_ok,
                    _fmt(s.ma20),
                    _fmt(s.ma60),
                    "" if s.mktcap_yi is None else f"{s.mktcap_yi:.1f}",
                    _fmt(s.pe),
                    _fmt(s.pb),
                    _fmt(s.nm),
                    _fmt(s.roe),
                    _fmt(s.debt_ratio),
                    _fmt(s.ret120),
                    _fmt(s.ret250),
                    _fmt(s.dd250),
                    s.error,
                ]
            )

    summary = {
        "universe": len(stocks),
        "scored": len(scored_ok),
        "score_composite_strict_median": statistics.median(
            [s.score_composite_strict for s in scored_ok if s.score_composite_strict is not None]
        )
        if scored_ok
        else None,
        "out_csv": str(out_csv),
    }
    (out_dir / "top30_two_models_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
