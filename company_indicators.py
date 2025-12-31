#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from backtest_two_models import (  # type: ignore
    END_DATE,
    SINA_BALANCE_URL,
    SINA_PROFIT_URL,
    _get_raw_close_on,
    _get_total_shares,
    _http_get_bytes,
    _kline_cached,
    _resolve_secid,
    _sina_stmt_value,
)


SINA_CASHFLOW_URL = (
    "https://money.finance.sina.com.cn/corp/go.php/"
    "vFD_CashFlow/stockid/{code}/ctrl/{year}/displaytype/4.phtml"
)


def _wy_to_yuan(x: Optional[float]) -> Optional[float]:
    return x * 10000 if x is not None else None


def _to_yi(yuan: Optional[float]) -> Optional[float]:
    return yuan / 1e8 if yuan is not None else None


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    return f"{x:.{nd}f}"


def _sina_fetch_html_cached(url: str, cache_path: Path, encoding: str = "gb18030") -> str:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        cache_path.write_bytes(_http_get_bytes(url))
    return cache_path.read_bytes().decode(encoding, errors="ignore")


def _stmt_value_any(html: str, date_str: str, labels: list[str]) -> Optional[float]:
    for lab in labels:
        v = _sina_stmt_value(html, label_contains=lab, date_str=date_str)
        if v is not None:
            return v
    return None


@dataclass(frozen=True)
class YearRow:
    year: int
    revenue_yi: Optional[float]
    revenue_yoy: Optional[float]
    net_profit_yi: Optional[float]
    net_profit_yoy: Optional[float]
    net_margin: Optional[float]
    cogs_yi: Optional[float]
    rd_yi: Optional[float]
    cfo_yi: Optional[float]
    capex_yi: Optional[float]
    fcf_yi: Optional[float]
    cfo_np: Optional[float]
    equity_yi: Optional[float]
    assets_yi: Optional[float]
    liabilities_yi: Optional[float]
    debt_ratio: Optional[float]
    roe: Optional[float]
    ar_yi: Optional[float]
    inventory_yi: Optional[float]
    ar_days: Optional[float]
    inv_days: Optional[float]
    overseas_rev_yi: Optional[float]
    overseas_rev_share: Optional[float]


def load_year(code: str, year: int, cache_dir: Path) -> YearRow:
    date_str = f"{year}-12-31"
    profit_html = _sina_fetch_html_cached(
        SINA_PROFIT_URL.format(code=code, year=year), cache_dir / f"profit_{code}_{year}.html"
    )
    balance_html = _sina_fetch_html_cached(
        SINA_BALANCE_URL.format(code=code, year=year), cache_dir / f"balance_{code}_{year}.html"
    )
    cash_html = _sina_fetch_html_cached(
        SINA_CASHFLOW_URL.format(code=code, year=year), cache_dir / f"cash_{code}_{year}.html"
    )

    revenue_wy = _stmt_value_any(profit_html, date_str, ["营业总收入", "营业收入"])
    np_wy = _stmt_value_any(profit_html, date_str, ["归属于母公司所有者的净利润", "净利润"])
    cogs_wy = _stmt_value_any(profit_html, date_str, ["营业成本"])
    rd_wy = _stmt_value_any(profit_html, date_str, ["研发费用"])

    # Some companies disclose overseas revenue in the statement; try a few common labels.
    overseas_wy = _stmt_value_any(
        profit_html,
        date_str,
        [
            "其中:境外营业收入",
            "其中：境外营业收入",
            "其中:境外收入",
            "其中：境外收入",
            "境外营业收入",
            "境外收入",
            "海外收入",
        ],
    )

    equity_wy = _stmt_value_any(balance_html, date_str, ["归属于母公司股东权益合计", "所有者权益(或股东权益)合计"])
    assets_wy = _stmt_value_any(balance_html, date_str, ["资产总计"])
    liab_wy = _stmt_value_any(balance_html, date_str, ["负债合计"])
    ar_wy = _stmt_value_any(balance_html, date_str, ["应收账款"])
    inv_wy = _stmt_value_any(balance_html, date_str, ["存货"])

    cfo_wy = _stmt_value_any(cash_html, date_str, ["经营活动产生的现金流量净额"])
    capex_wy = _stmt_value_any(cash_html, date_str, ["购建固定资产、无形资产和其他长期资产所支付的现金"])

    revenue = _wy_to_yuan(revenue_wy)
    np_ = _wy_to_yuan(np_wy)
    cogs = _wy_to_yuan(cogs_wy)
    rd = _wy_to_yuan(rd_wy)
    overseas = _wy_to_yuan(overseas_wy)

    equity = _wy_to_yuan(equity_wy)
    assets = _wy_to_yuan(assets_wy)
    liab = _wy_to_yuan(liab_wy)
    ar = _wy_to_yuan(ar_wy)
    inv = _wy_to_yuan(inv_wy)

    cfo = _wy_to_yuan(cfo_wy)
    capex = _wy_to_yuan(capex_wy)
    fcf = (cfo - capex) if (cfo is not None and capex is not None) else None

    net_margin = _safe_div(np_, revenue) if (revenue and revenue > 0) else None
    debt_ratio = _safe_div(liab, assets) if (assets and assets > 0) else None
    roe = _safe_div(np_, equity) if (equity and equity > 0) else None
    cfo_np = _safe_div(cfo, np_) if (np_ and np_ != 0) else None

    ar_days = _safe_div(ar, revenue) * 365 if (ar is not None and revenue and revenue > 0) else None
    inv_days = _safe_div(inv, cogs) * 365 if (inv is not None and cogs and cogs > 0) else None

    overseas_share = _safe_div(overseas, revenue) if (overseas is not None and revenue and revenue > 0) else None

    return YearRow(
        year=year,
        revenue_yi=_to_yi(revenue),
        revenue_yoy=None,
        net_profit_yi=_to_yi(np_),
        net_profit_yoy=None,
        net_margin=net_margin,
        cogs_yi=_to_yi(cogs),
        rd_yi=_to_yi(rd),
        cfo_yi=_to_yi(cfo),
        capex_yi=_to_yi(capex),
        fcf_yi=_to_yi(fcf),
        cfo_np=cfo_np,
        equity_yi=_to_yi(equity),
        assets_yi=_to_yi(assets),
        liabilities_yi=_to_yi(liab),
        debt_ratio=debt_ratio,
        roe=roe,
        ar_yi=_to_yi(ar),
        inventory_yi=_to_yi(inv),
        ar_days=ar_days,
        inv_days=inv_days,
        overseas_rev_yi=_to_yi(overseas),
        overseas_rev_share=overseas_share,
    )


def add_yoy(rows: list[YearRow]) -> list[YearRow]:
    rows = sorted(rows, key=lambda r: r.year)
    out: list[YearRow] = []
    prev = None
    for r in rows:
        if prev is None:
            out.append(r)
        else:
            rev_yoy = _safe_div((r.revenue_yi - prev.revenue_yi), prev.revenue_yi) if (r.revenue_yi is not None and prev.revenue_yi) else None
            np_yoy = _safe_div((r.net_profit_yi - prev.net_profit_yi), prev.net_profit_yi) if (r.net_profit_yi is not None and prev.net_profit_yi) else None
            out.append(
                YearRow(
                    **{
                        **r.__dict__,
                        "revenue_yoy": rev_yoy,
                        "net_profit_yoy": np_yoy,
                    }
                )
            )
        prev = r
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="A-share code, e.g. 605305")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--start-year", type=int, default=2021)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--asof", default="", help="valuation as-of date (YYYY-MM-DD). default: last trading date in cache")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    code = args.code.strip()
    secid = _resolve_secid(code)
    mkt = secid.split(".", 1)[0]
    mkt_prefix = "SH" if mkt == "1" else "SZ"

    years = list(range(args.start_year, args.end_year + 1))
    rows = [load_year(code, y, cache_dir / "sina") for y in years]
    rows = add_yoy(rows)

    out_csv = out_dir / f"{code}_财务指标_{args.start_year}_{args.end_year}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "年份",
                "营收(亿)",
                "营收同比",
                "归母净利润(亿)",
                "归母净利润同比",
                "净利率",
                "营业成本(亿)",
                "研发费用(亿)",
                "经营现金流净额(亿)",
                "资本开支现金(亿)",
                "自由现金流(亿)=经营-资本开支",
                "经营现金流/净利润",
                "归母权益(亿)",
                "总资产(亿)",
                "总负债(亿)",
                "资产负债率",
                "ROE(净利润/归母权益)",
                "应收账款(亿)",
                "存货(亿)",
                "应收/营收天数(粗略)",
                "存货/营业成本天数(粗略)",
                "境外收入(亿,如有)",
                "境外收入占比(如有)",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.year,
                    _fmt(r.revenue_yi, 2),
                    _fmt(r.revenue_yoy, 4),
                    _fmt(r.net_profit_yi, 2),
                    _fmt(r.net_profit_yoy, 4),
                    _fmt(r.net_margin, 4),
                    _fmt(r.cogs_yi, 2),
                    _fmt(r.rd_yi, 2),
                    _fmt(r.cfo_yi, 2),
                    _fmt(r.capex_yi, 2),
                    _fmt(r.fcf_yi, 2),
                    _fmt(r.cfo_np, 4),
                    _fmt(r.equity_yi, 2),
                    _fmt(r.assets_yi, 2),
                    _fmt(r.liabilities_yi, 2),
                    _fmt(r.debt_ratio, 4),
                    _fmt(r.roe, 4),
                    _fmt(r.ar_yi, 2),
                    _fmt(r.inventory_yi, 2),
                    _fmt(r.ar_days, 1),
                    _fmt(r.inv_days, 1),
                    _fmt(r.overseas_rev_yi, 2),
                    _fmt(r.overseas_rev_share, 4),
                ]
            )

    # Valuation as-of date (market cap via raw close * shares; pe/pb via last annual in cache rules)
    fq_dates, fq_closes = _kline_cached(secid=secid, fqt=1, beg="20170101", end=END_DATE.replace("-", ""), cache_dir=cache_dir / "kline")
    asof = args.asof.strip() or (max(fq_dates) if fq_dates else "")
    if asof:
        raw_close = _get_raw_close_on(secid, asof)
        shares = _get_total_shares(code, mkt_prefix, asof, cache_dir / "capital")
        mktcap_yuan = (raw_close * shares) if (raw_close is not None and shares is not None) else None
        # latest annual as-of date
        f_last = None
        try:
            from backtest_two_models import _get_fundamentals_last_annual  # type: ignore

            f_last = _get_fundamentals_last_annual(code, asof, cache_dir / "sina")
        except Exception:
            f_last = None
        pe = _safe_div(mktcap_yuan, f_last.net_profit_yuan) if (mktcap_yuan is not None and f_last and f_last.net_profit_yuan and f_last.net_profit_yuan > 0) else None
        pb = _safe_div(mktcap_yuan, f_last.equity_yuan) if (mktcap_yuan is not None and f_last and f_last.equity_yuan and f_last.equity_yuan > 0) else None

        out_val = out_dir / f"{code}_估值_截至{asof}.csv"
        with out_val.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["股票代码", "截至日期", "收盘价(不复权)", "总股本(股)", "总市值(亿)", "PE(估算)", "PB(估算)", "估值使用年报"])
            w.writerow(
                [
                    code,
                    asof,
                    _fmt(raw_close, 2),
                    "" if shares is None else f"{int(shares)}",
                    _fmt(_to_yi(mktcap_yuan), 2),
                    _fmt(pe, 4),
                    _fmt(pb, 4),
                    "" if f_last is None else f_last.year,
                ]
            )
    else:
        out_val = None

    # Simple kline facts (listing since 2021 for this stock)
    if fq_dates and fq_closes:
        # use first date in fq_dates as listing
        start_d = min(fq_dates)
        end_d = max(fq_dates)
        # max drawdown
        peak = fq_closes[0]
        peak_d = fq_dates[0]
        max_dd = 0.0
        dd_start = peak_d
        dd_end = peak_d
        for d, c in zip(fq_dates, fq_closes):
            if c > peak:
                peak = c
                peak_d = d
            dd = c / peak - 1.0
            if dd < max_dd:
                max_dd = dd
                dd_start = peak_d
                dd_end = d
        out_k = out_dir / f"{code}_股价结构_前复权.csv"
        with out_k.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["股票代码", "起始日期(有数据)", "结束日期", "最大回撤", "回撤起点", "回撤终点"])
            w.writerow([code, start_d, end_d, _fmt(max_dd, 4), dd_start, dd_end])
    else:
        out_k = None

    summary = {
        "code": code,
        "secid": secid,
        "out_financial_csv": str(out_csv),
        "out_valuation_csv": str(out_val) if asof else "",
        "out_kline_csv": str(out_k) if out_k else "",
        "notes": [
            "客户集中度/前五大客户、订单/在手订单、海外收入按地区拆分、产品结构/毛利按业务：通常需要年报附注文本/经营讨论(MD&A)；本脚本只抓取三大表中的可结构化字段。",
            "境外收入字段并非所有公司会直接在利润表披露，因此可能为空。",
        ],
    }
    (out_dir / f"{code}_指标汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

