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
from typing import Any, Dict, Optional, Tuple

import requests

from backtest_two_models import (  # type: ignore
    END_DATE,
    Fundamentals,
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


EM_BUSINESS_ANALYSIS_URL = "https://emweb.securities.eastmoney.com/PC_HSF10/BusinessAnalysis/PageAjax?code={code}"


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


def _industry_ceiling_score(industry: str) -> float:
    """
    Proxy for business ceiling. This is a heuristic mapping, not a true TAM estimate.
    Returns 0..100.
    """
    s = (industry or "").strip()
    # High-ceiling long-duration themes (heuristic)
    high = [
        "半导体",
        "软件",
        "互联网",
        "通信",
        "计算机",
        "医疗",
        "生物",
        "创新药",
        "军工",
        "自动化",
        "机器人",
        "电池",
        "新能源",
        "光伏",
        "储能",
        "风电",
        "电网设备",
        "汽车零部件",
    ]
    mid = ["化工", "化学", "机械", "专用设备", "通用设备", "仪器仪表", "消费电子", "家电", "食品饮料"]
    low = ["煤炭", "钢铁", "建材", "银行", "地产", "公用事业", "港口", "航运"]

    if any(k in s for k in high):
        return 85.0
    if any(k in s for k in mid):
        return 60.0
    if any(k in s for k in low):
        return 35.0
    return 50.0


def _overseas_score(overseas_share: Optional[float]) -> float:
    if overseas_share is None:
        return 0.0
    if overseas_share >= 0.5:
        return 100.0
    if overseas_share >= 0.3:
        return 80.0
    if overseas_share >= 0.2:
        return 65.0
    if overseas_share >= 0.1:
        return 45.0
    if overseas_share > 0:
        return 25.0
    return 0.0


def _hhi(shares: list[float]) -> Optional[float]:
    if not shares:
        return None
    s = sum(shares)
    if s <= 0:
        return None
    return sum((x / s) ** 2 for x in shares if x > 0)


def _diversification_score(top1_share: Optional[float], hhi: Optional[float], seg_cnt: int) -> float:
    """
    Proxy for "second curve / product platform". 0..100.
    Lower concentration and more segments => higher.
    """
    score = 0.0
    if seg_cnt >= 4:
        score += 20
    elif seg_cnt == 3:
        score += 15
    elif seg_cnt == 2:
        score += 8

    if top1_share is not None:
        if top1_share <= 0.55:
            score += 45
        elif top1_share <= 0.70:
            score += 30
        elif top1_share <= 0.85:
            score += 15
        else:
            score += 5

    if hhi is not None:
        if hhi <= 0.35:
            score += 35
        elif hhi <= 0.50:
            score += 22
        elif hhi <= 0.65:
            score += 12
        else:
            score += 5
    return min(100.0, score)


def _rd_score(rd_ratio: Optional[float]) -> float:
    if rd_ratio is None:
        return 0.0
    if rd_ratio >= 0.12:
        return 100.0
    if rd_ratio >= 0.08:
        return 80.0
    if rd_ratio >= 0.05:
        return 55.0
    if rd_ratio >= 0.03:
        return 35.0
    return 15.0


def _capex_score(capex_ratio: Optional[float]) -> float:
    if capex_ratio is None:
        return 0.0
    if capex_ratio >= 0.12:
        return 85.0
    if capex_ratio >= 0.07:
        return 65.0
    if capex_ratio >= 0.03:
        return 45.0
    if capex_ratio > 0:
        return 25.0
    return 0.0


def _http_get_json_cached(url: str, cache_path: Path, *, timeout: int = 25) -> Dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text("utf-8", errors="ignore"))
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    cache_path.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.05)
    return j


@dataclass(frozen=True)
class BusinessSnapshot:
    report_date: str
    overseas_share: Optional[float]
    product_top1_share: Optional[float]
    product_hhi: Optional[float]
    product_seg_cnt: int


def _parse_business_snapshot(j: Dict[str, Any]) -> Optional[BusinessSnapshot]:
    zygcfx = j.get("zygcfx") or []
    if not zygcfx:
        return None

    # Use latest annual if possible, else latest report_date.
    dates = sorted({str(r.get("REPORT_DATE") or "") for r in zygcfx if r.get("REPORT_DATE")})
    if not dates:
        return None
    annual = [d for d in dates if d.startswith("2024-12-31") or d.endswith("12-31 00:00:00")]
    pick = (sorted(annual)[-1] if annual else dates[-1])

    rows = [r for r in zygcfx if str(r.get("REPORT_DATE") or "") == pick]

    # MAINOP_TYPE: 2=by product, 3=by region (外销/内销), 1=by industry (e.g., 风电行业)
    product = [r for r in rows if str(r.get("MAINOP_TYPE") or "") == "2"]
    region = [r for r in rows if str(r.get("MAINOP_TYPE") or "") == "3"]

    def income(r):
        try:
            return float(r.get("MAIN_BUSINESS_INCOME") or 0.0)
        except Exception:
            return 0.0

    total_income = sum(income(r) for r in rows) or None

    overseas_income = None
    if region and total_income:
        overseas_income = sum(income(r) for r in region if "外" in str(r.get("ITEM_NAME") or ""))

    overseas_share = _safe_div(overseas_income, total_income) if (overseas_income is not None and total_income) else None

    prod_incomes = [income(r) for r in product if income(r) > 0]
    prod_total = sum(prod_incomes) if prod_incomes else 0.0
    prod_shares = [(x / prod_total) for x in prod_incomes] if prod_total > 0 else []
    prod_shares.sort(reverse=True)

    top1_share = prod_shares[0] if prod_shares else None
    hhi = _hhi(prod_shares) if prod_shares else None
    seg_cnt = len(prod_shares)

    return BusinessSnapshot(
        report_date=pick.split(" ")[0],
        overseas_share=overseas_share,
        product_top1_share=top1_share,
        product_hhi=hhi,
        product_seg_cnt=seg_cnt,
    )


@dataclass(frozen=True)
class BaseScore:
    code: str
    name: str
    industry: str
    asof_date: str
    mktcap_yi: Optional[float]
    pe: Optional[float]
    pb: Optional[float]
    nm: Optional[float]
    roe: Optional[float]
    debt_ratio: Optional[float]
    rd_ratio: Optional[float]
    capex_ratio: Optional[float]
    ret120: Optional[float]
    ret250: Optional[float]
    dd250: Optional[float]
    entry_ok: int
    score_a: Optional[float]
    score_b: Optional[float]
    score_base: Optional[float]
    error: str


def score_base(stock: Stock, cache_dir: Path, db_conn) -> BaseScore:
    try:
        secid = _resolve_secid(stock.code)
        mkt = secid.split(".", 1)[0]
        mkt_prefix = "SH" if mkt == "1" else "SZ"

        if db_conn is not None:
            fq_dates, fq_closes = get_dates_and_closes(db_conn, stock.code, 1)
            raw_dates, raw_closes = get_dates_and_closes(db_conn, stock.code, 0)
        else:
            fq_dates, fq_closes = _kline_cached(
                secid=secid, fqt=1, beg="20170101", end=END_DATE.replace("-", ""), cache_dir=cache_dir / "kline"
            )
            raw_dates, raw_closes = _kline_cached(
                secid=secid, fqt=0, beg="20170101", end=END_DATE.replace("-", ""), cache_dir=cache_dir / "kline"
            )
        asof_date = _last_trading_date(fq_dates)
        if not asof_date:
            raise RuntimeError("no kline")

        ret120, ret250, dd = _momentum_from_series(fq_dates, fq_closes, asof_date)
        entry_ok, _, _ = _entry_ok_today(fq_dates, fq_closes, asof_date)

        # Raw close and shares for market cap
        raw_map = dict(zip(raw_dates, raw_closes))
        close_raw = raw_map.get(asof_date)
        shares = _get_total_shares(stock.code, mkt_prefix, asof_date, cache_dir / "capital")
        mktcap = (close_raw * shares) if (close_raw is not None and shares is not None) else None
        mktcap_yi = (mktcap / 1e8) if mktcap is not None else None

        f: Fundamentals = _get_fundamentals_last_annual(stock.code, asof_date, cache_dir / "sina")
        revenue = f.revenue_yuan
        np_ = f.net_profit_yuan
        equity = f.equity_yuan
        assets = f.assets_yuan
        liab = f.liabilities_yuan

        nm = _safe_div(np_, revenue) if (revenue and revenue > 0) else None
        roe = _safe_div(np_, equity) if (equity and equity > 0) else None
        debt_ratio = _safe_div(liab, assets) if (assets and assets > 0) else None
        pe = _safe_div(mktcap, np_) if (np_ and np_ > 0 and mktcap is not None) else None
        pb = _safe_div(mktcap, equity) if (equity and equity > 0 and mktcap is not None) else None

        # R&D and capex from Sina statements (annual as-of year)
        # Reuse parsing from company_indicators-like approach with cached HTML if already present.
        y = f.year
        profit_html_path = (cache_dir / "sina" / f"profit_{stock.code}_{y}.html")
        cash_html_path = (cache_dir / "sina" / f"cash_{stock.code}_{y}.html")
        # best-effort: if cash html missing, skip capex ratio.
        rd_ratio = None
        capex_ratio = None
        try:
            if profit_html_path.exists():
                html = profit_html_path.read_bytes().decode("gb18030", errors="ignore")
                import pandas as pd  # local import to avoid global dependency complaints

                # simple: search "研发费用" on same date col
                from backtest_two_models import _sina_stmt_value  # type: ignore

                rd_wy = _sina_stmt_value(html, label_contains="研发费用", date_str=f"{y}-12-31")
                rd = (rd_wy * 10000) if rd_wy is not None else None
                rd_ratio = _safe_div(rd, revenue) if (rd is not None and revenue and revenue > 0) else None
        except Exception:
            rd_ratio = None

        try:
            if cash_html_path.exists():
                html = cash_html_path.read_bytes().decode("gb18030", errors="ignore")
                from backtest_two_models import _sina_stmt_value  # type: ignore

                capex_wy = _sina_stmt_value(html, label_contains="购建固定资产、无形资产和其他长期资产所支付的现金", date_str=f"{y}-12-31")
                capex = (capex_wy * 10000) if capex_wy is not None else None
                capex_ratio = _safe_div(capex, revenue) if (capex is not None and revenue and revenue > 0) else None
        except Exception:
            capex_ratio = None

        score_a = _score_dv(mktcap_yi, pe, pb, nm, roe, debt_ratio, ret250, dd)
        score_b = _score_tc(mktcap_yi, pe, pb, nm, roe, ret120, ret250, dd)

        # Base composite: require entry_ok for B in strict mode
        score_base = (0.5 * score_a + 0.5 * score_b) if entry_ok else score_a

        return BaseScore(
            code=stock.code,
            name=stock.name,
            industry=stock.industry,
            asof_date=asof_date,
            mktcap_yi=mktcap_yi,
            pe=pe,
            pb=pb,
            nm=nm,
            roe=roe,
            debt_ratio=debt_ratio,
            rd_ratio=rd_ratio,
            capex_ratio=capex_ratio,
            ret120=ret120,
            ret250=ret250,
            dd250=dd,
            entry_ok=1 if entry_ok else 0,
            score_a=score_a,
            score_b=score_b,
            score_base=score_base,
            error="",
        )
    except Exception as e:
        msg = str(e)
        if len(msg) > 250:
            msg = msg[:250] + "..."
        return BaseScore(
            code=stock.code,
            name=stock.name,
            industry=stock.industry,
            asof_date="",
            mktcap_yi=None,
            pe=None,
            pb=None,
            nm=None,
            roe=None,
            debt_ratio=None,
            rd_ratio=None,
            capex_ratio=None,
            ret120=None,
            ret250=None,
            dd250=None,
            entry_ok=0,
            score_a=None,
            score_b=None,
            score_base=None,
            error=msg,
        )


@dataclass(frozen=True)
class FinalRow:
    code: str
    name: str
    industry: str
    asof_date: str
    score_total: float
    score_base: float
    score_a: float
    score_b: float
    entry_ok: int
    ceiling_industry: float
    ceiling_overseas: float
    ceiling_total: float
    second_diversification: float
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


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--exclude-st", action="store_true")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--candidates", type=int, default=800, help="top-N by base score to enrich with F10 business data")
    ap.add_argument("--seed", type=int, default=7)
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

    base_scores: list[BaseScore] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(score_base, s, cache_dir, db_conn) for s in stocks]
        for fut in as_completed(futs):
            base_scores.append(fut.result())

    base_ok = [b for b in base_scores if b.score_base is not None and not b.error]
    base_ok.sort(key=lambda x: x.score_base or -1, reverse=True)
    cand = base_ok[: args.candidates]

    print(f"base scored={len(base_ok)} candidates={len(cand)}")

    def enrich_one(b: BaseScore) -> FinalRow:
        # ceiling & second curve proxies
        ceiling_ind = _industry_ceiling_score(b.industry)
        # business analysis
        secid = _resolve_secid(b.code)
        mkt = secid.split(".", 1)[0]
        mkt_prefix = "SH" if mkt == "1" else "SZ"
        ba_code = f"{mkt_prefix}{b.code}"
        try:
            j = _http_get_json_cached(
                EM_BUSINESS_ANALYSIS_URL.format(code=ba_code),
                cache_dir / "eastmoney" / f"business_{ba_code}.json",
            )
            snap = _parse_business_snapshot(j)
        except Exception as e:
            snap = None

        overseas_share = snap.overseas_share if snap else None
        ceiling_ov = _overseas_score(overseas_share)
        ceiling_total = 0.6 * ceiling_ind + 0.4 * ceiling_ov

        top1 = snap.product_top1_share if snap else None
        hhi = snap.product_hhi if snap else None
        seg_cnt = snap.product_seg_cnt if snap else 0
        second_div = _diversification_score(top1, hhi, seg_cnt)
        second_rd = _rd_score(b.rd_ratio)
        second_capex = _capex_score(b.capex_ratio)
        second_total = 0.5 * second_div + 0.3 * second_rd + 0.2 * second_capex

        # total score 0..100
        score_a = b.score_a or 0.0
        score_b = b.score_b or 0.0
        score_base = b.score_base or 0.0
        # Weighting: base 50%, ceiling 25%, second-curve 25%
        score_total = 0.50 * score_base + 0.25 * ceiling_total + 0.25 * second_total

        return FinalRow(
            code=b.code,
            name=b.name,
            industry=b.industry,
            asof_date=b.asof_date,
            score_total=score_total,
            score_base=score_base,
            score_a=score_a,
            score_b=score_b,
            entry_ok=b.entry_ok,
            ceiling_industry=ceiling_ind,
            ceiling_overseas=ceiling_ov,
            ceiling_total=ceiling_total,
            second_diversification=second_div,
            second_rd=second_rd,
            second_capex=second_capex,
            second_total=second_total,
            overseas_share=overseas_share,
            product_top1_share=top1,
            product_hhi=hhi,
            product_seg_cnt=seg_cnt,
            mktcap_yi=b.mktcap_yi,
            pe=b.pe,
            pb=b.pb,
            nm=b.nm,
            roe=b.roe,
            debt_ratio=b.debt_ratio,
            rd_ratio=b.rd_ratio,
            capex_ratio=b.capex_ratio,
            error=b.error,
        )

    finals: list[FinalRow] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, 12)) as ex:
        futs = [ex.submit(enrich_one, b) for b in cand]
        for fut in as_completed(futs):
            finals.append(fut.result())

    finals.sort(key=lambda r: r.score_total, reverse=True)
    top = finals[: args.top]

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
                "总分(新框架)",
                "基础分(估值+质量+趋势)",
                "模型A_抄底分",
                "模型B_趋势分",
                "趋势入场(1/0)",
                "天花板分_行业(代理)",
                "天花板分_出海(代理)",
                "天花板分_合计",
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
                    _fmt(r.score_b, 1),
                    r.entry_ok,
                    _fmt(r.ceiling_industry, 1),
                    _fmt(r.ceiling_overseas, 1),
                    _fmt(r.ceiling_total, 1),
                    _fmt(r.second_diversification, 1),
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
                ]
            )

    summary = {
        "universe": len(stocks),
        "base_scored": len(base_ok),
        "candidates_enriched": len(finals),
        "top": args.top,
        "out_csv": str(out_csv),
        "notes": [
            "‘天花板/第二曲线’为代理指标：行业关键词映射 + F10主营构成(产品/地区)集中度 + 研发/资本开支强度；不等同于真实TAM测算。",
            "如需更严谨的TAM：需要引入行业装机/渗透率/单价等外部数据或手工假设。",
        ],
    }
    (out_dir / "top30_framework_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
