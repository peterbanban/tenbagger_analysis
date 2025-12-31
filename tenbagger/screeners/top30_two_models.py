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
from tenbagger.data.fundamentals_sina import Fundamentals, get_fundamentals_last_annual, get_fundamentals_series_last_n_years
from tenbagger.data.investment_sina import rd_and_capex_ratio
from tenbagger.data.paths import default_market_db_path
from tenbagger.data.sqlite_market import get_dates_and_closes, open_db
from tenbagger.data.universe import Stock, load_universe_from_sqlite
from tenbagger.models.framework_v2 import capex_score, growth_score, hard_gate_ok, industry_ceiling_score, rd_score
from tenbagger.models.two_models import InflectionSignal, inflection_signal, momentum_from_series, score_combined_inflection, score_model_a


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


def _cagr(old: Optional[float], new: Optional[float], years: int) -> Optional[float]:
    if old is None or new is None or years <= 0:
        return None
    if old <= 0 or new <= 0:
        return None
    return (new / old) ** (1.0 / years) - 1.0


@dataclass(frozen=True)
class Scored:
    code: str
    name: str
    industry: str
    asof_date: str
    score_combined: Optional[float]
    gate_ok: Optional[int]
    signal_ok: Optional[int]
    ma20: Optional[float]
    ma60: Optional[float]
    ma60_slope_20d: Optional[float]
    pos_from_250_low: Optional[float]
    mktcap_yi: Optional[float]
    pe: Optional[float]
    pb: Optional[float]
    nm: Optional[float]
    roe: Optional[float]
    debt_ratio: Optional[float]
    rev_cagr_3y: Optional[float]
    profit_cagr_3y: Optional[float]
    growth_total: Optional[float]
    ceiling_industry: Optional[float]
    ceiling_total: Optional[float]
    second_rd: Optional[float]
    second_capex: Optional[float]
    second_total: Optional[float]
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
        signal: InflectionSignal = inflection_signal(fq_dates, fq_closes, asof_date)

        raw_map = dict(zip(raw_dates, raw_closes))
        close_raw = raw_map.get(asof_date)
        mkt_prefix = "SH" if stock.secid.startswith("1.") else "SZ"
        shares = get_total_shares(stock.code, mkt_prefix=mkt_prefix, asof_date=asof_date, cache_dir=cache_dir / "capital")
        mktcap = (close_raw * shares) if (close_raw is not None and shares is not None) else None
        mktcap_yi = (mktcap / 1e8) if mktcap is not None else None

        f: Fundamentals = get_fundamentals_last_annual(stock.code, asof_date, cache_dir / "sina")
        series = get_fundamentals_series_last_n_years(code=stock.code, asof_date=asof_date, n=4, cache_dir=cache_dir / "sina")
        rev_cagr_3y = None
        profit_cagr_3y = None
        if len(series) >= 4:
            rev_cagr_3y = _cagr(series[0].revenue_yuan, series[-1].revenue_yuan, 3)
            profit_cagr_3y = _cagr(series[0].net_profit_yuan, series[-1].net_profit_yuan, 3)
        growth_total = growth_score(rev_cagr_3y, profit_cagr_3y)

        nm = _safe_div(f.net_profit_yuan, f.revenue_yuan) if (f.revenue_yuan and f.revenue_yuan > 0) else None
        roe = _safe_div(f.net_profit_yuan, f.equity_yuan) if (f.equity_yuan and f.equity_yuan > 0) else None
        debt_ratio = _safe_div(f.liabilities_yuan, f.assets_yuan) if (f.assets_yuan and f.assets_yuan > 0) else None
        pe = _safe_div(mktcap, f.net_profit_yuan) if (mktcap is not None and f.net_profit_yuan and f.net_profit_yuan > 0) else None
        pb = _safe_div(mktcap, f.equity_yuan) if (mktcap is not None and f.equity_yuan and f.equity_yuan > 0) else None

        score_a = score_model_a(mktcap_yi, pe, pb, nm, roe, debt_ratio, dd)

        rd_ratio, capex_ratio = rd_and_capex_ratio(
            code=stock.code, year=f.year, revenue_yuan=f.revenue_yuan, cache_dir=cache_dir / "sina"
        )
        second_rd = rd_score(rd_ratio)
        second_capex = capex_score(capex_ratio)
        second_total = 0.6 * second_rd + 0.4 * second_capex

        ceiling_ind = industry_ceiling_score(stock.industry)
        ceiling_total = ceiling_ind  # no overseas in this light screener
        gate = hard_gate_ok(
            ceiling_total=ceiling_total,
            second_total=second_total,
            growth_total=growth_total,
        )

        return Scored(
            code=stock.code,
            name=stock.name,
            industry=stock.industry,
            asof_date=asof_date,
            score_combined=score_combined_inflection(
                score_a=score_a,
                ret120=ret120,
                ret250=ret250,
                dd250=dd,
                signal=signal,
                growth_total=growth_total,
            ),
            gate_ok=1 if gate else 0,
            signal_ok=1 if signal.ok else 0,
            ma20=signal.ma20,
            ma60=signal.ma60,
            ma60_slope_20d=signal.ma60_slope_20d,
            pos_from_250_low=signal.pos_from_250_low,
            mktcap_yi=mktcap_yi,
            pe=pe,
            pb=pb,
            nm=nm,
            roe=roe,
            debt_ratio=debt_ratio,
            rev_cagr_3y=rev_cagr_3y,
            profit_cagr_3y=profit_cagr_3y,
            growth_total=growth_total,
            ceiling_industry=ceiling_ind,
            ceiling_total=ceiling_total,
            second_rd=second_rd,
            second_capex=second_capex,
            second_total=second_total,
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
            score_combined=None,
            gate_ok=None,
            signal_ok=None,
            ma20=None,
            ma60=None,
            ma60_slope_20d=None,
            pos_from_250_low=None,
            mktcap_yi=None,
            pe=None,
            pb=None,
            nm=None,
            roe=None,
            debt_ratio=None,
            rev_cagr_3y=None,
            profit_cagr_3y=None,
            growth_total=None,
            ceiling_industry=None,
            ceiling_total=None,
            second_rd=None,
            second_capex=None,
            second_total=None,
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
    ap.add_argument("--require-entry", action="store_true", default=True, help="only keep turning-point (inflection) signals")
    ap.add_argument("--disable-gate", action="store_true", default=False, help="disable tenbagger hard gate")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.db).expanduser() if args.db else default_market_db_path()
    stocks = load_universe_from_sqlite(db_path, exclude_st=args.exclude_st)
    db_conn = open_db(db_path)

    scored = [_score_one(s, db_conn=db_conn, cache_dir=cache_dir) for s in stocks]
    scored_ok = [s for s in scored if s.score_combined is not None]
    if not args.disable_gate:
        scored_ok = [s for s in scored_ok if s.gate_ok == 1]
    if args.require_entry:
        scored_ok = [s for s in scored_ok if s.signal_ok == 1]
    scored_ok.sort(key=lambda s: s.score_combined or -1, reverse=True)
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
                "综合得分(拐点融合模型)",
                "硬门槛通过(1/0)",
                "拐点信号通过(1/0)",
                "MA20",
                "MA60",
                "MA60_20日斜率",
                "距250日低点涨幅",
                "总市值(亿)",
                "PE(估算)",
                "PB(估算)",
                "净利率(估算)",
                "ROE(估算)",
                "资产负债率(估算)",
                "收入CAGR_3年(估算)",
                "利润CAGR_3年(估算)",
                "增长得分",
                "行业天花板得分",
                "天花板总分(轻量)",
                "第二曲线_R&D得分",
                "第二曲线_Capex得分",
                "第二曲线总分(轻量)",
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
                    "" if s.score_combined is None else f"{s.score_combined:.1f}",
                    "" if s.gate_ok is None else s.gate_ok,
                    "" if s.signal_ok is None else s.signal_ok,
                    _fmt(s.ma20, 4),
                    _fmt(s.ma60, 4),
                    _fmt(s.ma60_slope_20d, 4),
                    _fmt(s.pos_from_250_low, 4),
                    _fmt(s.mktcap_yi, 1),
                    _fmt(s.pe, 4),
                    _fmt(s.pb, 4),
                    _fmt(s.nm, 4),
                    _fmt(s.roe, 4),
                    _fmt(s.debt_ratio, 4),
                    _fmt(s.rev_cagr_3y, 4),
                    _fmt(s.profit_cagr_3y, 4),
                    _fmt(s.growth_total, 1),
                    _fmt(s.ceiling_industry, 1),
                    _fmt(s.ceiling_total, 1),
                    _fmt(s.second_rd, 1),
                    _fmt(s.second_capex, 1),
                    _fmt(s.second_total, 1),
                    _fmt(s.ret120, 4),
                    _fmt(s.ret250, 4),
                    _fmt(s.dd250, 4),
                    s.error,
                ]
            )

    summary = {
        "universe": len(stocks),
        "scored": len(scored_ok),
        "score_median": statistics.median([s.score_combined for s in scored_ok if s.score_combined is not None]) if scored_ok else None,
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
