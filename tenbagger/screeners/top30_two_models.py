#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tenbagger.data.capital import get_total_shares
from tenbagger.data.fundamentals_sina import Fundamentals, get_fundamentals_last_annual
from tenbagger.data.paths import default_market_db_path
from tenbagger.data.sqlite_market import get_dates_and_closes, open_db
from tenbagger.data.universe import Stock, load_universe_from_sqlite
from tenbagger.models.two_models import entry_ok_ma, momentum_from_series, score_model_a, score_model_b


END_DATE = "2025-12-31"


def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    return f"{x:.{nd}f}"


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


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


def _last_trading_date(dates: list[str]) -> Optional[str]:
    return max(dates) if dates else None


def _score_one(stock: Stock, *, db_conn, cache_dir: Path) -> Scored:
    try:
        fq_dates, fq_closes = get_dates_and_closes(db_conn, stock.code, 1, start_date="2017-01-01", end_date=END_DATE)
        raw_dates, raw_closes = get_dates_and_closes(db_conn, stock.code, 0, start_date="2017-01-01", end_date=END_DATE)
        asof_date = _last_trading_date(fq_dates)
        if not asof_date:
            raise RuntimeError("no kline")

        ret120, ret250, dd = momentum_from_series(fq_dates, fq_closes, asof_date)
        entry_ok, ma20, ma60 = entry_ok_ma(fq_dates, fq_closes, asof_date)

        raw_map = dict(zip(raw_dates, raw_closes))
        close_raw = raw_map.get(asof_date)
        mkt_prefix = "SH" if stock.secid.startswith("1.") else "SZ"
        shares = get_total_shares(stock.code, mkt_prefix=mkt_prefix, asof_date=asof_date, cache_dir=cache_dir / "capital")
        mktcap = (close_raw * shares) if (close_raw is not None and shares is not None) else None
        mktcap_yi = (mktcap / 1e8) if mktcap is not None else None

        f: Fundamentals = get_fundamentals_last_annual(stock.code, asof_date, cache_dir / "sina")
        nm = _safe_div(f.net_profit_yuan, f.revenue_yuan) if (f.revenue_yuan and f.revenue_yuan > 0) else None
        roe = _safe_div(f.net_profit_yuan, f.equity_yuan) if (f.equity_yuan and f.equity_yuan > 0) else None
        debt_ratio = _safe_div(f.liabilities_yuan, f.assets_yuan) if (f.assets_yuan and f.assets_yuan > 0) else None
        pe = _safe_div(mktcap, f.net_profit_yuan) if (mktcap is not None and f.net_profit_yuan and f.net_profit_yuan > 0) else None
        pb = _safe_div(mktcap, f.equity_yuan) if (mktcap is not None and f.equity_yuan and f.equity_yuan > 0) else None

        score_a = score_model_a(mktcap_yi, pe, pb, nm, roe, debt_ratio, dd)
        score_b = score_model_b(mktcap_yi, pe, pb, nm, roe, ret120, ret250, dd)

        def composite(sa: Optional[float], sb: Optional[float]) -> Optional[float]:
            if sa is None and sb is None:
                return None
            if sa is None:
                return sb
            if sb is None:
                return sa
            return 0.5 * sa + 0.5 * sb

        score_loose = composite(score_a, score_b)
        score_strict = composite(score_a, score_b) if entry_ok else score_a

        return Scored(
            code=stock.code,
            name=stock.name,
            industry=stock.industry,
            asof_date=asof_date,
            score_a=score_a,
            score_b=score_b,
            score_composite_strict=score_strict,
            score_composite_loose=score_loose,
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


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--exclude-st", action="store_true", default=True)
    ap.add_argument("--db", default="", help="sqlite db path, default: project/data/tenbagger_analysis_market.sqlite")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.db).expanduser() if args.db else default_market_db_path()
    stocks = load_universe_from_sqlite(db_path, exclude_st=args.exclude_st)
    db_conn = open_db(db_path)

    scored = [_score_one(s, db_conn=db_conn, cache_dir=cache_dir) for s in stocks]
    scored_ok = [s for s in scored if s.score_composite_strict is not None]
    scored_ok.sort(key=lambda s: s.score_composite_strict or -1, reverse=True)
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
                    _fmt(s.ma20, 4),
                    _fmt(s.ma60, 4),
                    _fmt(s.mktcap_yi, 1),
                    _fmt(s.pe, 4),
                    _fmt(s.pb, 4),
                    _fmt(s.nm, 4),
                    _fmt(s.roe, 4),
                    _fmt(s.debt_ratio, 4),
                    _fmt(s.ret120, 4),
                    _fmt(s.ret250, 4),
                    _fmt(s.dd250, 4),
                    s.error,
                ]
            )

    summary = {
        "universe": len(stocks),
        "scored": len(scored_ok),
        "score_strict_median": statistics.median([s.score_composite_strict for s in scored_ok if s.score_composite_strict is not None])
        if scored_ok
        else None,
        "out_csv": str(out_csv),
        "db": str(db_path),
    }
    (out_dir / "top30_two_models_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
