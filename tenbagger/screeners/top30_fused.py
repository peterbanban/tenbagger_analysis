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
from tenbagger.models.fused import (
    HardGate,
    InflectionSignal,
    capex_score,
    cash_quality_score,
    diversification_score,
    financial_quality_total,
    growth_score,
    hard_gate,
    industry_ceiling_score,
    inflection_signal,
    interest_coverage_score,
    momentum_from_series,
    ops_quality_score,
    overseas_score,
    rd_score,
    refi_pressure_score,
    score_fused_total,
    score_quality_value_core,
)


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
    total_score: float
    gate_ok: int
    entry_ok: int
    core_score: float
    turning_score_proxy: float
    low_cost_proxy: float
    ceiling_industry: float
    overseas_share: Optional[float]
    ceiling_overseas: float
    ceiling_total: float
    rev_cagr_3y: Optional[float]
    profit_cagr_3y: Optional[float]
    growth_total: float
    product_top1_share: Optional[float]
    product_hhi: Optional[float]
    product_seg_cnt: int
    second_div: float
    rd_ratio: Optional[float]
    capex_ratio: Optional[float]
    second_rd: float
    second_capex: float
    second_total: float
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
    cfo_yuan: Optional[float]
    cfo_to_profit: Optional[float]
    cash_yuan: Optional[float]
    short_debt_yuan: Optional[float]
    long_debt_yuan: Optional[float]
    net_debt_yuan: Optional[float]
    short_debt_to_cash: Optional[float]
    ar_to_rev: Optional[float]
    inv_to_rev: Optional[float]
    ar_to_rev_delta: Optional[float]
    inv_to_rev_delta: Optional[float]
    fin_quality_total: float
    fin_cash_score: float
    fin_ops_score: float
    fin_refi_score: float
    fin_cov_score: float
    ret120: Optional[float]
    ret250: Optional[float]
    dd250: Optional[float]
    error: str


def _last_trading_date(dates: list[str]) -> Optional[str]:
    return max(dates) if dates else None


def _row_stub(stock: Stock, msg: str) -> Row:
    if len(msg) > 250:
        msg = msg[:250] + "..."
    return Row(
        code=stock.code,
        name=stock.name,
        industry=stock.industry,
        asof_date="",
        total_score=-1.0,
        gate_ok=0,
        entry_ok=0,
        core_score=0.0,
        turning_score_proxy=0.0,
        low_cost_proxy=0.0,
        ceiling_industry=0.0,
        overseas_share=None,
        ceiling_overseas=0.0,
        ceiling_total=0.0,
        rev_cagr_3y=None,
        profit_cagr_3y=None,
        growth_total=0.0,
        product_top1_share=None,
        product_hhi=None,
        product_seg_cnt=0,
        second_div=0.0,
        rd_ratio=None,
        capex_ratio=None,
        second_rd=0.0,
        second_capex=0.0,
        second_total=0.0,
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
        cfo_yuan=None,
        cfo_to_profit=None,
        cash_yuan=None,
        short_debt_yuan=None,
        long_debt_yuan=None,
        net_debt_yuan=None,
        short_debt_to_cash=None,
        ar_to_rev=None,
        inv_to_rev=None,
        ar_to_rev_delta=None,
        inv_to_rev_delta=None,
        fin_quality_total=0.0,
        fin_cash_score=0.0,
        fin_ops_score=0.0,
        fin_refi_score=0.0,
        fin_cov_score=0.0,
        ret120=None,
        ret250=None,
        dd250=None,
        error=msg,
    )


def _compute_base_fast(stock: Stock, *, db_conn, cache_dir: Path) -> Optional[tuple[float, str]]:
    """
    Fast stage-1 score for preselection: avoids F10 business snapshot.
    """
    fq_dates, fq_closes = get_dates_and_closes(db_conn, stock.code, 1, start_date="2017-01-01", end_date=END_DATE)
    raw_dates, raw_closes = get_dates_and_closes(db_conn, stock.code, 0, start_date="2017-01-01", end_date=END_DATE)
    asof_date = _last_trading_date(fq_dates)
    if not asof_date:
        return None

    ret120, ret250, dd250 = momentum_from_series(fq_dates, fq_closes, asof_date)
    signal = inflection_signal(fq_dates, fq_closes, asof_date)

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

    core = score_quality_value_core(mktcap_yi, pe, pb, nm, roe, debt_ratio, dd250)

    series = get_fundamentals_series_last_n_years(code=stock.code, asof_date=asof_date, n=4, cache_dir=cache_dir / "sina")
    rev_cagr_3y = None
    profit_cagr_3y = None
    if len(series) >= 4:
        rev_cagr_3y = _cagr(series[0].revenue_yuan, series[-1].revenue_yuan, 3)
        profit_cagr_3y = _cagr(series[0].net_profit_yuan, series[-1].net_profit_yuan, 3)
    growth_total = growth_score(rev_cagr_3y, profit_cagr_3y)

    rd_ratio, capex_ratio = rd_and_capex_ratio(code=stock.code, year=f.year, revenue_yuan=f.revenue_yuan, cache_dir=cache_dir / "sina")
    second_total_light = 0.6 * rd_score(rd_ratio) + 0.4 * capex_score(capex_ratio)

    # Financial quality light proxy (uses year-end balances; no F10)
    ar_to_rev = _safe_div(f.accounts_receivable_yuan, f.revenue_yuan) if (f.revenue_yuan and f.revenue_yuan > 0) else None
    inv_to_rev = _safe_div(f.inventory_yuan, f.revenue_yuan) if (f.revenue_yuan and f.revenue_yuan > 0) else None
    ar_to_rev_delta = None
    inv_to_rev_delta = None
    if len(series) >= 2 and series[-2].revenue_yuan and series[-2].revenue_yuan > 0 and series[-1].revenue_yuan and series[-1].revenue_yuan > 0:
        ar_prev = _safe_div(series[-2].accounts_receivable_yuan, series[-2].revenue_yuan)
        inv_prev = _safe_div(series[-2].inventory_yuan, series[-2].revenue_yuan)
        if ar_prev is not None and ar_to_rev is not None:
            ar_to_rev_delta = ar_to_rev - ar_prev
        if inv_prev is not None and inv_to_rev is not None:
            inv_to_rev_delta = inv_to_rev - inv_prev

    fin_total = financial_quality_total(
        cfo_yuan=f.cfo_yuan,
        net_profit_yuan=f.net_profit_yuan,
        ar_to_rev=ar_to_rev,
        inv_to_rev=inv_to_rev,
        ar_to_rev_delta=ar_to_rev_delta,
        inv_to_rev_delta=inv_to_rev_delta,
        cash_yuan=f.cash_yuan,
        short_debt_yuan=f.short_debt_yuan,
        long_debt_yuan=f.long_debt_yuan,
        operating_profit_yuan=f.operating_profit_yuan,
        interest_expense_yuan=f.interest_expense_yuan,
    )

    ceiling_total_light = industry_ceiling_score(stock.industry)
    gate = hard_gate(ceiling_total=ceiling_total_light, second_total=second_total_light, growth_total=growth_total)

    turning_bonus = 0.0
    if signal.ok:
        turning_bonus += 18.0
    if ret120 is not None and ret120 > 0:
        turning_bonus += 6.0
    if signal.pos_from_250_low is not None and signal.pos_from_250_low <= 0.45:
        turning_bonus += 6.0

    base = 0.50 * core + 0.22 * growth_total + 0.18 * second_total_light + 0.10 * fin_total + turning_bonus
    if not gate.ok:
        base -= 25.0
    return float(base), asof_date


def score_one(stock: Stock, *, db_conn, cache_dir: Path) -> Row:
    try:
        fq_dates, fq_closes = get_dates_and_closes(db_conn, stock.code, 1, start_date="2017-01-01", end_date=END_DATE)
        raw_dates, raw_closes = get_dates_and_closes(db_conn, stock.code, 0, start_date="2017-01-01", end_date=END_DATE)
        asof_date = _last_trading_date(fq_dates)
        if not asof_date:
            raise RuntimeError("no kline")

        ret120, ret250, dd250 = momentum_from_series(fq_dates, fq_closes, asof_date)
        signal: InflectionSignal = inflection_signal(fq_dates, fq_closes, asof_date)

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

        core_score = score_quality_value_core(mktcap_yi, pe, pb, nm, roe, debt_ratio, dd250)

        series = get_fundamentals_series_last_n_years(code=stock.code, asof_date=asof_date, n=4, cache_dir=cache_dir / "sina")
        rev_cagr_3y = None
        profit_cagr_3y = None
        if len(series) >= 4:
            rev_cagr_3y = _cagr(series[0].revenue_yuan, series[-1].revenue_yuan, 3)
            profit_cagr_3y = _cagr(series[0].net_profit_yuan, series[-1].net_profit_yuan, 3)
        growth_total = growth_score(rev_cagr_3y, profit_cagr_3y)

        snap = business_snapshot_asof(code=stock.code, mkt_prefix=mkt_prefix, asof_date=asof_date, cache_dir=cache_dir / "eastmoney")
        overseas_share = snap.overseas_share if snap else None
        product_top1_share = snap.product_top1_share if snap else None
        product_hhi = snap.product_hhi if snap else None
        product_seg_cnt = snap.product_seg_cnt if snap else 0

        ceiling_ind = industry_ceiling_score(stock.industry)
        ceiling_ov = overseas_score(overseas_share)
        ceiling_total = 0.6 * ceiling_ind + 0.4 * ceiling_ov

        rd_ratio, capex_ratio = rd_and_capex_ratio(code=stock.code, year=f.year, revenue_yuan=f.revenue_yuan, cache_dir=cache_dir / "sina")
        second_div = diversification_score(product_top1_share, product_hhi, product_seg_cnt)
        second_rd = rd_score(rd_ratio)
        second_capex = capex_score(capex_ratio)
        second_total = 0.5 * second_div + 0.3 * second_rd + 0.2 * second_capex

        # Financial/ops quality (annual statement proxies)
        cfo_to_profit = _safe_div(f.cfo_yuan, f.net_profit_yuan) if (f.net_profit_yuan and f.net_profit_yuan > 0) else None
        cash = f.cash_yuan
        sd = f.short_debt_yuan
        ld = f.long_debt_yuan
        debt = (sd or 0.0) + (ld or 0.0)
        net_debt_yuan = (debt - (cash or 0.0)) if (cash is not None or debt > 0) else None
        short_debt_to_cash = _safe_div(sd, cash) if (sd is not None and cash is not None and cash > 0) else None

        ar_to_rev = _safe_div(f.accounts_receivable_yuan, f.revenue_yuan) if (f.revenue_yuan and f.revenue_yuan > 0) else None
        inv_to_rev = _safe_div(f.inventory_yuan, f.revenue_yuan) if (f.revenue_yuan and f.revenue_yuan > 0) else None
        ar_to_rev_delta = None
        inv_to_rev_delta = None
        if len(series) >= 2:
            prev = series[-2]
            ar_prev = _safe_div(prev.accounts_receivable_yuan, prev.revenue_yuan) if (prev.revenue_yuan and prev.revenue_yuan > 0) else None
            inv_prev = _safe_div(prev.inventory_yuan, prev.revenue_yuan) if (prev.revenue_yuan and prev.revenue_yuan > 0) else None
            if ar_prev is not None and ar_to_rev is not None:
                ar_to_rev_delta = ar_to_rev - ar_prev
            if inv_prev is not None and inv_to_rev is not None:
                inv_to_rev_delta = inv_to_rev - inv_prev

        fin_cash_score = cash_quality_score(cfo_yuan=f.cfo_yuan, net_profit_yuan=f.net_profit_yuan)
        fin_ops_score = ops_quality_score(
            ar_to_rev=ar_to_rev, inv_to_rev=inv_to_rev, ar_to_rev_delta=ar_to_rev_delta, inv_to_rev_delta=inv_to_rev_delta
        )
        fin_refi_score = refi_pressure_score(cash_yuan=cash, short_debt_yuan=sd, long_debt_yuan=ld)
        fin_cov_score = interest_coverage_score(
            operating_profit_yuan=f.operating_profit_yuan, interest_expense_yuan=f.interest_expense_yuan
        )

        fin_quality_total_score = financial_quality_total(
            cfo_yuan=f.cfo_yuan,
            net_profit_yuan=f.net_profit_yuan,
            ar_to_rev=ar_to_rev,
            inv_to_rev=inv_to_rev,
            ar_to_rev_delta=ar_to_rev_delta,
            inv_to_rev_delta=inv_to_rev_delta,
            cash_yuan=cash,
            short_debt_yuan=sd,
            long_debt_yuan=ld,
            operating_profit_yuan=f.operating_profit_yuan,
            interest_expense_yuan=f.interest_expense_yuan,
        )

        gate: HardGate = hard_gate(ceiling_total=ceiling_total, second_total=second_total, growth_total=growth_total)
        total = score_fused_total(
            core_score=core_score,
            gate=gate,
            signal=signal,
            ret120=ret120,
            ret250=ret250,
            dd250=dd250,
            second_total=second_total,
            fin_quality=fin_quality_total_score,
        )

        # purely informational proxies for debug
        turning_proxy = 100.0 if signal.ok else 0.0
        low_cost_proxy = 0.0 if signal.pos_from_250_low is None else max(0.0, 100.0 - 100.0 * signal.pos_from_250_low)

        return Row(
            code=stock.code,
            name=stock.name,
            industry=stock.industry,
            asof_date=asof_date,
            total_score=total,
            gate_ok=1 if gate.ok else 0,
            entry_ok=1 if signal.ok else 0,
            core_score=core_score,
            turning_score_proxy=turning_proxy,
            low_cost_proxy=low_cost_proxy,
            ceiling_industry=ceiling_ind,
            overseas_share=overseas_share,
            ceiling_overseas=ceiling_ov,
            ceiling_total=ceiling_total,
            rev_cagr_3y=rev_cagr_3y,
            profit_cagr_3y=profit_cagr_3y,
            growth_total=growth_total,
            product_top1_share=product_top1_share,
            product_hhi=product_hhi,
            product_seg_cnt=product_seg_cnt,
            second_div=second_div,
            rd_ratio=rd_ratio,
            capex_ratio=capex_ratio,
            second_rd=second_rd,
            second_capex=second_capex,
            second_total=second_total,
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
            cfo_yuan=f.cfo_yuan,
            cfo_to_profit=cfo_to_profit,
            cash_yuan=cash,
            short_debt_yuan=sd,
            long_debt_yuan=ld,
            net_debt_yuan=net_debt_yuan,
            short_debt_to_cash=short_debt_to_cash,
            ar_to_rev=ar_to_rev,
            inv_to_rev=inv_to_rev,
            ar_to_rev_delta=ar_to_rev_delta,
            inv_to_rev_delta=inv_to_rev_delta,
            fin_quality_total=fin_quality_total_score,
            fin_cash_score=fin_cash_score,
            fin_ops_score=fin_ops_score,
            fin_refi_score=fin_refi_score,
            fin_cov_score=fin_cov_score,
            ret120=ret120,
            ret250=ret250,
            dd250=dd250,
            error="",
        )
    except Exception as e:
        return _row_stub(stock, str(e))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--exclude-st", action="store_true", default=True)
    ap.add_argument("--db", default="", help="sqlite db path, default: project/data/tenbagger_analysis_market.sqlite")
    ap.add_argument("--candidates", type=int, default=800, help="preselect top-N by fast score before F10 enrichment")
    ap.add_argument("--no-require-entry", action="store_true", default=False, help="do NOT require turning-point (inflection) signals")
    ap.add_argument("--disable-gate", action="store_true", default=False, help="disable tenbagger hard gate")
    args = ap.parse_args(argv)
    require_entry = not args.no_require_entry

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.db).expanduser() if args.db else default_market_db_path()
    stocks = load_universe_from_sqlite(db_path, exclude_st=args.exclude_st)
    db_conn = open_db(db_path)

    # Stage1: fast preselect
    base_rows: list[tuple[float, Stock]] = []
    for s in stocks:
        try:
            res = _compute_base_fast(s, db_conn=db_conn, cache_dir=cache_dir)
            if not res:
                continue
            base, asof = res
            if require_entry:
                fq_dates, fq_closes = get_dates_and_closes(db_conn, s.code, 1, start_date="2017-01-01", end_date=END_DATE)
                if not inflection_signal(fq_dates, fq_closes, asof).ok:
                    continue
            base_rows.append((base, s))
        except Exception:
            continue

    base_rows.sort(key=lambda x: x[0], reverse=True)
    cand = [s for _, s in base_rows[: args.candidates]]

    rows = [score_one(s, db_conn=db_conn, cache_dir=cache_dir) for s in cand]
    rows_ok = [r for r in rows if r.total_score >= 0]
    if not args.disable_gate:
        rows_ok = [r for r in rows_ok if r.gate_ok == 1]
    if require_entry:
        rows_ok = [r for r in rows_ok if r.entry_ok == 1]

    rows_ok.sort(key=lambda r: r.total_score, reverse=True)
    top = rows_ok[: args.top]

    out_csv = out_dir / "top30_fused.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "排名",
                "股票代码",
                "股票名称",
                "行业",
                "截至日期",
                "总分(融合模型)",
                "硬门槛通过(1/0)",
                "拐点信号通过(1/0)",
                "核心分(质量/估值/压力)",
                "财务运营质量总分",
                "现金含金量得分(CFO≈利润)",
                "营运质量得分(应收/存货)",
                "再融资压力得分(现金/短债)",
                "利息保障得分",
                "经营现金流CFO(元)",
                "CFO/净利润(估算)",
                "货币资金(元)",
                "有息负债_短债(元)",
                "有息负债_长债(元)",
                "净负债(元)",
                "短债/现金(估算)",
                "应收/收入(估算)",
                "存货/收入(估算)",
                "应收/收入_变化(估算)",
                "存货/收入_变化(估算)",
                "行业天花板得分",
                "境外收入占比(如可得)",
                "出海得分",
                "天花板总分",
                "收入CAGR_3年(估算)",
                "利润CAGR_3年(估算)",
                "增长得分(3年CAGR代理)",
                "产品Top1占比(如可得)",
                "产品HHI(如可得)",
                "产品分部数(如可得)",
                "第二曲线_多元化得分",
                "研发/营收(估算)",
                "资本开支/营收(估算)",
                "第二曲线_R&D得分",
                "第二曲线_Capex得分",
                "第二曲线总分",
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
                "120日收益",
                "250日收益",
                "距250日高点回撤",
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
                    _fmt(r.total_score, 1),
                    r.gate_ok,
                    r.entry_ok,
                    _fmt(r.core_score, 1),
                    _fmt(r.fin_quality_total, 1),
                    _fmt(r.fin_cash_score, 1),
                    _fmt(r.fin_ops_score, 1),
                    _fmt(r.fin_refi_score, 1),
                    _fmt(r.fin_cov_score, 1),
                    _fmt(r.cfo_yuan, 0),
                    _fmt(r.cfo_to_profit, 4),
                    _fmt(r.cash_yuan, 0),
                    _fmt(r.short_debt_yuan, 0),
                    _fmt(r.long_debt_yuan, 0),
                    _fmt(r.net_debt_yuan, 0),
                    _fmt(r.short_debt_to_cash, 4),
                    _fmt(r.ar_to_rev, 4),
                    _fmt(r.inv_to_rev, 4),
                    _fmt(r.ar_to_rev_delta, 4),
                    _fmt(r.inv_to_rev_delta, 4),
                    _fmt(r.ceiling_industry, 1),
                    _fmt(r.overseas_share, 4),
                    _fmt(r.ceiling_overseas, 1),
                    _fmt(r.ceiling_total, 1),
                    _fmt(r.rev_cagr_3y, 4),
                    _fmt(r.profit_cagr_3y, 4),
                    _fmt(r.growth_total, 1),
                    _fmt(r.product_top1_share, 4),
                    _fmt(r.product_hhi, 4),
                    r.product_seg_cnt,
                    _fmt(r.second_div, 1),
                    _fmt(r.rd_ratio, 4),
                    _fmt(r.capex_ratio, 4),
                    _fmt(r.second_rd, 1),
                    _fmt(r.second_capex, 1),
                    _fmt(r.second_total, 1),
                    _fmt(r.ma20, 4),
                    _fmt(r.ma60, 4),
                    _fmt(r.ma60_slope_20d, 4),
                    _fmt(r.pos_from_250_low, 4),
                    _fmt(r.mktcap_yi, 1),
                    _fmt(r.pe, 4),
                    _fmt(r.pb, 4),
                    _fmt(r.nm, 4),
                    _fmt(r.roe, 4),
                    _fmt(r.debt_ratio, 4),
                    _fmt(r.ret120, 4),
                    _fmt(r.ret250, 4),
                    _fmt(r.dd250, 4),
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
        "require_entry": bool(require_entry),
        "gate_enabled": bool(not args.disable_gate),
        "score_median": statistics.median([x.total_score for x in rows_ok]) if rows_ok else None,
    }
    (out_dir / "top30_fused_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
