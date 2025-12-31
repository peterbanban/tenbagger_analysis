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

from tenbagger.data.business_analysis import business_snapshot_asof
from tenbagger.data.capital import get_total_shares
from tenbagger.data.fundamentals_sina import Fundamentals, get_fundamentals_last_annual, get_fundamentals_series_last_n_years
from tenbagger.data.investment_sina import rd_and_capex_ratio
from tenbagger.data.paths import default_market_db_path
from tenbagger.data.sqlite_market import get_dates_and_closes, open_db
from tenbagger.data.universe import Stock, load_universe_from_sqlite
from tenbagger.models.framework_v2 import (
    capex_score,
    diversification_score,
    growth_score,
    hard_gate_ok,
    industry_ceiling_score,
    overseas_score,
    rd_score,
    total_score_framework_v3,
)
from tenbagger.models.two_models import InflectionSignal, inflection_signal, momentum_from_series, score_model_a


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
class Row:
    code: str
    name: str
    industry: str
    asof_date: str
    score_total: float
    score_base: float
    score_a: float
    gate_ok: int
    signal_ok: int
    ma60_slope_20d: Optional[float]
    pos_from_250_low: Optional[float]
    ceiling_industry: float
    ceiling_overseas: float
    ceiling_total: float
    growth_total: float
    rev_cagr_3y: Optional[float]
    profit_cagr_3y: Optional[float]
    second_div: float
    second_rd: float
    second_capex: float
    second_total: float
    overseas_share: Optional[float]
    product_top1_share: Optional[float]
    product_hhi: Optional[float]
    product_seg_cnt: int
    mktcap_yi: Optional[float]
    pe: Optional[float]
    pb: Optional[float]
    nm: Optional[float]
    roe: Optional[float]
    debt_ratio: Optional[float]
    rd_ratio: Optional[float]
    capex_ratio: Optional[float]
    error: str


def _last_trading_date(dates: list[str]) -> Optional[str]:
    return max(dates) if dates else None


def score_one(stock: Stock, *, db_conn, cache_dir: Path) -> Row:
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
        # Base: keep quality/value core; use inflection as a bonus to avoid dead money.
        inflection_bonus = 0.0
        if signal.ok:
            inflection_bonus += 15.0
        if signal.pos_from_250_low is not None:
            if signal.pos_from_250_low <= 0.25:
                inflection_bonus += 8.0
            elif signal.pos_from_250_low <= 0.45:
                inflection_bonus += 4.0
        if ret120 is not None and ret120 > 0:
            inflection_bonus += 4.0
        score_base = min(100.0, score_a + inflection_bonus)

        snap = business_snapshot_asof(code=stock.code, mkt_prefix=mkt_prefix, asof_date=asof_date, cache_dir=cache_dir / "eastmoney")
        overseas_share = snap.overseas_share if snap else None
        product_top1_share = snap.product_top1_share if snap else None
        product_hhi = snap.product_hhi if snap else None
        product_seg_cnt = snap.product_seg_cnt if snap else 0

        ceiling_ind = industry_ceiling_score(stock.industry)
        ceiling_ov = overseas_score(overseas_share)
        ceiling_total = 0.6 * ceiling_ind + 0.4 * ceiling_ov

        rd_ratio, capex_ratio = rd_and_capex_ratio(
            code=stock.code, year=f.year, revenue_yuan=f.revenue_yuan, cache_dir=cache_dir / "sina"
        )
        second_div = diversification_score(product_top1_share, product_hhi, product_seg_cnt)
        second_rd = rd_score(rd_ratio)
        second_capex = capex_score(capex_ratio)
        second_total = 0.5 * second_div + 0.3 * second_rd + 0.2 * second_capex

        gate_ok = hard_gate_ok(ceiling_total=ceiling_total, second_total=second_total, growth_total=growth_total)

        score_total = total_score_framework_v3(
            base=score_base,
            ceiling_industry=ceiling_ind,
            ceiling_overseas=ceiling_ov,
            second_total=second_total,
            growth_total=growth_total,
        )

        return Row(
            code=stock.code,
            name=stock.name,
            industry=stock.industry,
            asof_date=asof_date,
            score_total=score_total,
            score_base=score_base,
            score_a=score_a,
            gate_ok=1 if gate_ok else 0,
            signal_ok=1 if signal.ok else 0,
            ma60_slope_20d=signal.ma60_slope_20d,
            pos_from_250_low=signal.pos_from_250_low,
            ceiling_industry=ceiling_ind,
            ceiling_overseas=ceiling_ov,
            ceiling_total=ceiling_total,
            growth_total=growth_total,
            rev_cagr_3y=rev_cagr_3y,
            profit_cagr_3y=profit_cagr_3y,
            second_div=second_div,
            second_rd=second_rd,
            second_capex=second_capex,
            second_total=second_total,
            overseas_share=overseas_share,
            product_top1_share=product_top1_share,
            product_hhi=product_hhi,
            product_seg_cnt=product_seg_cnt,
            mktcap_yi=mktcap_yi,
            pe=pe,
            pb=pb,
            nm=nm,
            roe=roe,
            debt_ratio=debt_ratio,
            rd_ratio=rd_ratio,
            capex_ratio=capex_ratio,
            error="",
        )
    except Exception as e:
        msg = str(e)
        if len(msg) > 250:
            msg = msg[:250] + "..."
        return Row(
            code=stock.code,
            name=stock.name,
            industry=stock.industry,
            asof_date="",
            score_total=-1.0,
            score_base=0.0,
            score_a=0.0,
            gate_ok=0,
            signal_ok=0,
            ma60_slope_20d=None,
            pos_from_250_low=None,
            ceiling_industry=0.0,
            ceiling_overseas=0.0,
            ceiling_total=0.0,
            growth_total=0.0,
            rev_cagr_3y=None,
            profit_cagr_3y=None,
            second_div=0.0,
            second_rd=0.0,
            second_capex=0.0,
            second_total=0.0,
            overseas_share=None,
            product_top1_share=None,
            product_hhi=None,
            product_seg_cnt=0,
            mktcap_yi=None,
            pe=None,
            pb=None,
            nm=None,
            roe=None,
            debt_ratio=None,
            rd_ratio=None,
            capex_ratio=None,
            error=msg,
        )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--exclude-st", action="store_true", default=True)
    ap.add_argument("--db", default="", help="sqlite db path, default: project/data/tenbagger_analysis_market.sqlite")
    ap.add_argument("--candidates", type=int, default=800, help="preselect top-N by base score before F10 enrichment")
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

    # Stage1: base scores only (without F10), for speed
    base_rows: list[tuple[float, Stock]] = []
    for s in stocks:
        fq_dates, fq_closes = get_dates_and_closes(db_conn, s.code, 1, start_date="2017-01-01", end_date=END_DATE)
        raw_dates, raw_closes = get_dates_and_closes(db_conn, s.code, 0, start_date="2017-01-01", end_date=END_DATE)
        asof_date = _last_trading_date(fq_dates)
        if not asof_date:
            continue
        ret120, _, dd = momentum_from_series(fq_dates, fq_closes, asof_date)
        signal = inflection_signal(fq_dates, fq_closes, asof_date)
        if args.require_entry and not signal.ok:
            continue
        raw_map = dict(zip(raw_dates, raw_closes))
        close_raw = raw_map.get(asof_date)
        mkt_prefix = "SH" if s.secid.startswith("1.") else "SZ"
        shares = get_total_shares(s.code, mkt_prefix=mkt_prefix, asof_date=asof_date, cache_dir=cache_dir / "capital")
        mktcap = (close_raw * shares) if (close_raw is not None and shares is not None) else None
        mktcap_yi = (mktcap / 1e8) if mktcap is not None else None
        f = get_fundamentals_last_annual(s.code, asof_date, cache_dir / "sina")
        nm = _safe_div(f.net_profit_yuan, f.revenue_yuan) if (f.revenue_yuan and f.revenue_yuan > 0) else None
        roe = _safe_div(f.net_profit_yuan, f.equity_yuan) if (f.equity_yuan and f.equity_yuan > 0) else None
        debt_ratio = _safe_div(f.liabilities_yuan, f.assets_yuan) if (f.assets_yuan and f.assets_yuan > 0) else None
        pe = _safe_div(mktcap, f.net_profit_yuan) if (mktcap is not None and f.net_profit_yuan and f.net_profit_yuan > 0) else None
        pb = _safe_div(mktcap, f.equity_yuan) if (mktcap is not None and f.equity_yuan and f.equity_yuan > 0) else None
        score_a = score_model_a(mktcap_yi, pe, pb, nm, roe, debt_ratio, dd)
        base = score_a + (10.0 if (signal.ok and (ret120 is not None and ret120 > 0)) else 0.0)
        base_rows.append((base, s))

    base_rows.sort(key=lambda x: x[0], reverse=True)
    cand = [s for _, s in base_rows[: args.candidates]]

    rows = [score_one(s, db_conn=db_conn, cache_dir=cache_dir) for s in cand]
    rows_ok = [r for r in rows if r.score_total >= 0]
    if not args.disable_gate:
        rows_ok = [r for r in rows_ok if r.gate_ok == 1]
    if args.require_entry:
        rows_ok = [r for r in rows_ok if r.signal_ok == 1]
    rows_ok.sort(key=lambda r: r.score_total, reverse=True)
    top = rows_ok[: args.top]

    out_csv = out_dir / "top30_framework_v2.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "排名",
                "股票代码",
                "股票名称",
                "行业",
                "截至日期",
                "总分(框架V3+硬门槛)",
                "基础分(质量/估值+拐点加分)",
                "模型A_质量/估值分",
                "硬门槛通过(1/0)",
                "拐点信号通过(1/0)",
                "MA60_20日斜率",
                "距250日低点涨幅",
                "天花板分_行业(代理)",
                "天花板分_出海(代理)",
                "天花板分_合计",
                "增长得分(3年CAGR代理)",
                "收入CAGR_3年(估算)",
                "利润CAGR_3年(估算)",
                "第二曲线分_多元化(代理)",
                "第二曲线分_研发(代理)",
                "第二曲线分_资本开支(代理)",
                "第二曲线分_合计",
                "境外收入占比(如可得)",
                "产品Top1占比(如可得)",
                "产品HHI(如可得)",
                "产品分部数(如可得)",
                "总市值(亿)",
                "PE(估算)",
                "PB(估算)",
                "净利率(估算)",
                "ROE(估算)",
                "资产负债率(估算)",
                "研发/营收(估算)",
                "资本开支/营收(估算)",
                "错误(如有)",
            ]
        )
        for i, r in enumerate(top, start=1):
            w.writerow(
                [
                    i,
                    r.code,
                    r.name,
                    r.industry,
                    r.asof_date,
                    _fmt(r.score_total, 1),
                    _fmt(r.score_base, 1),
                    _fmt(r.score_a, 1),
                    r.gate_ok,
                    r.signal_ok,
                    _fmt(r.ma60_slope_20d, 4),
                    _fmt(r.pos_from_250_low, 4),
                    _fmt(r.ceiling_industry, 1),
                    _fmt(r.ceiling_overseas, 1),
                    _fmt(r.ceiling_total, 1),
                    _fmt(r.growth_total, 1),
                    _fmt(r.rev_cagr_3y, 4),
                    _fmt(r.profit_cagr_3y, 4),
                    _fmt(r.second_div, 1),
                    _fmt(r.second_rd, 1),
                    _fmt(r.second_capex, 1),
                    _fmt(r.second_total, 1),
                    _fmt(r.overseas_share, 4),
                    _fmt(r.product_top1_share, 4),
                    _fmt(r.product_hhi, 4),
                    r.product_seg_cnt,
                    _fmt(r.mktcap_yi, 1),
                    _fmt(r.pe, 4),
                    _fmt(r.pb, 4),
                    _fmt(r.nm, 4),
                    _fmt(r.roe, 4),
                    _fmt(r.debt_ratio, 4),
                    _fmt(r.rd_ratio, 4),
                    _fmt(r.capex_ratio, 4),
                    r.error,
                ]
            )

    summary = {
        "universe": len(stocks),
        "candidates": len(cand),
        "scored": len(rows_ok),
        "top": args.top,
        "out_csv": str(out_csv),
        "db": str(db_path),
    }
    (out_dir / "top30_framework_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
