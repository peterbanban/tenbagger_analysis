#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backtest_two_models import END_DATE, _kline_cached, _resolve_secid  # type: ignore
from sqlite_market import get_dates_and_closes, open_db


def _parse_float(x: str) -> Optional[float]:
    s = (x or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    return f"{x:.{nd}f}"


@dataclass(frozen=True)
class Constituent:
    code: str
    name: str
    score_total: float
    weight: float


def load_constituents(path: Path) -> List[Constituent]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise RuntimeError(f"empty csv: {path}")

    codes: List[str] = []
    names: Dict[str, str] = {}
    scores: Dict[str, float] = {}
    for row in rows:
        code = (row.get("股票代码") or row.get("code") or "").strip()
        name = (row.get("股票名称") or row.get("name") or "").strip()
        sc = _parse_float(row.get("总分(新框架)") or row.get("score_total") or "")
        if not code or sc is None:
            continue
        codes.append(code)
        names[code] = name
        scores[code] = sc

    if len(scores) == 0:
        raise RuntimeError(f"no valid rows in: {path}")
    total = sum(scores.values())
    if total <= 0:
        raise RuntimeError("sum score <=0, cannot weight")

    out = [
        Constituent(code=c, name=names.get(c, ""), score_total=scores[c], weight=scores[c] / total)
        for c in sorted(scores.keys())
    ]
    # sanity: normalize
    s = sum(x.weight for x in out)
    out = [Constituent(code=x.code, name=x.name, score_total=x.score_total, weight=x.weight / s) for x in out]
    return out


def month_end_dates(dates: List[str]) -> List[str]:
    out: List[str] = []
    last_by_month: Dict[str, str] = {}
    for d in dates:
        ym = d[:7]
        if ym not in last_by_month or d > last_by_month[ym]:
            last_by_month[ym] = d
    out = sorted(last_by_month.values())
    return out


def percentiles(xs: List[float], ps: List[float]) -> Dict[str, float]:
    if not xs:
        return {}
    xs2 = sorted(xs)
    n = len(xs2)
    out: Dict[str, float] = {}
    for p in ps:
        if n == 1:
            out[str(p)] = xs2[0]
            continue
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            out[str(p)] = xs2[int(k)]
        else:
            out[str(p)] = xs2[f] + (xs2[c] - xs2[f]) * (k - f)
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basket", required=True, help="top30_framework_v2.csv path")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--kline-cache", default="", help="path to existing .cache directory (optional)")
    ap.add_argument(
        "--db",
        default="",
        help="sqlite db path for offline kline. Default: <project>/data/tenbagger_analysis_market.sqlite if exists.",
    )
    ap.add_argument("--horizon", type=int, default=252, help="trading days horizon")
    ap.add_argument("--start", default="2019-01-01", help="min formation date")
    ap.add_argument("--end", default="2024-12-31", help="max formation date (needs horizon after it)")
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=0.9,
        help="min weight coverage required to include a sample date when some constituents lack data",
    )
    args = ap.parse_args(argv)

    basket_path = Path(args.basket)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    constituents = load_constituents(basket_path)
    codes = [c.code for c in constituents]
    weights = {c.code: c.weight for c in constituents}

    db_conn = None
    db_path = (
        Path(args.db).expanduser()
        if args.db
        else (Path(__file__).resolve().parent / "data" / "tenbagger_analysis_market.sqlite")
    )
    if db_path.exists():
        db_conn = open_db(db_path)

    cache_dir = Path(args.kline_cache) if args.kline_cache else (out_dir / ".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load kline for each code from cache; fqt=1 for total-return-ish series.
    series: Dict[str, Tuple[List[str], List[float]]] = {}
    for code in codes:
        if db_conn is not None:
            dates, closes = get_dates_and_closes(
                db_conn,
                code,
                1,
                start_date=args.start,
                end_date=END_DATE,
            )
        else:
            secid = _resolve_secid(code)
            dates, closes = _kline_cached(
                secid=secid, fqt=1, beg="20170101", end=END_DATE.replace("-", ""), cache_dir=cache_dir / "kline"
            )
        series[code] = (dates, closes)

    # Use month-end dates from the union of trading dates; later filter by coverage.
    all_dates: set[str] = set()
    for code in codes:
        all_dates |= {d for d in series[code][0] if args.start <= d <= args.end}
    if not all_dates:
        raise RuntimeError("no trading dates in range")
    formation_dates = month_end_dates(sorted(all_dates))

    # Prebuild index maps for speed.
    idx_map: Dict[str, Dict[str, int]] = {}
    for code in codes:
        dates = series[code][0]
        idx_map[code] = {d: i for i, d in enumerate(dates)}

    horizon = int(args.horizon)
    rows: List[Tuple[str, float, float, float, int, int]] = []  # (d, ret_cash, ret_renorm, coverage, n_ok, n_total)
    for d in formation_dates:
        pr_cash = 0.0
        ok_weight = 0.0
        n_ok = 0
        for code in codes:
            dates, closes = series[code]
            i = idx_map[code].get(d)
            if i is None or i + horizon >= len(closes):
                continue
            p0 = closes[i]
            p1 = closes[i + horizon]
            if p0 <= 0 or p1 <= 0:
                continue
            w = weights[code]
            pr_cash += w * (p1 / p0 - 1.0)
            ok_weight += w
            n_ok += 1
        if ok_weight >= args.min_coverage:
            pr_renorm = pr_cash / ok_weight if ok_weight > 0 else 0.0
            rows.append((d, pr_cash, pr_renorm, ok_weight, n_ok, len(codes)))

    if not rows:
        raise RuntimeError("no formation dates meet min_coverage with available horizon data")

    rets_cash = [r[1] for r in rows]
    rets_renorm = [r[2] for r in rows]
    pct_cash = percentiles(rets_cash, [0.1, 0.25, 0.5, 0.75, 0.9])
    pct_renorm = percentiles(rets_renorm, [0.1, 0.25, 0.5, 0.75, 0.9])
    summary = {
        "basket": str(basket_path),
        "n_constituents": len(codes),
        "weighting": "proportional_to_total_score",
        "formation_rule": "month_end_any_trading_day",
        "horizon_trading_days": horizon,
        "sample_count": len(rows),
        "formation_date_min": min(r[0] for r in rows),
        "formation_date_max": max(r[0] for r in rows),
        "min_coverage": args.min_coverage,
        "coverage_mean": statistics.mean([r[3] for r in rows]),
        "coverage_median": statistics.median([r[3] for r in rows]),
        "n_ok_mean": statistics.mean([r[4] for r in rows]),
        "return_cash_mean": statistics.mean(rets_cash),
        "return_cash_median": statistics.median(rets_cash),
        "return_cash_p10": pct_cash.get("0.1"),
        "return_cash_p50": pct_cash.get("0.5"),
        "return_cash_p90": pct_cash.get("0.9"),
        "return_renorm_mean": statistics.mean(rets_renorm),
        "return_renorm_median": statistics.median(rets_renorm),
        "return_renorm_p10": pct_renorm.get("0.1"),
        "return_renorm_p50": pct_renorm.get("0.5"),
        "return_renorm_p90": pct_renorm.get("0.9"),
        "notes": [
            "这是对‘固定这30只股票及其权重’的历史滚动1年收益分布估计，并非对未来1年的确定预测。",
            "使用前复权收盘价(近似含分红送转影响)；未计交易成本/税费。",
            "由于这30只股票很多是近年上市/样本期较短，难以找到“所有成分都同时具备未来252交易日数据”的形成日；因此本工具允许部分成分缺失。",
            "ret_cash：缺失成分视作现金(该权重当期收益=0)；ret_renorm：仅对有数据成分按覆盖权重再归一化后的收益。",
        ],
    }

    out_csv = out_dir / "portfolio_1y_return_samples.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["形成日期", "未来252交易日收益_ret_cash", "未来252交易日收益_ret_renorm", "权重覆盖率", "有效成分数", "总成分数"])
        for d, r1, r2, cov, n_ok, n_total in rows:
            w.writerow([d, _fmt(r1, 6), _fmt(r2, 6), _fmt(cov, 6), n_ok, n_total])

    out_w = out_dir / "portfolio_constituents_weights.csv"
    with out_w.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["股票代码", "股票名称", "总分(新框架)", "权重"])
        for c in sorted(constituents, key=lambda x: x.weight, reverse=True):
            w.writerow([c.code, c.name, _fmt(c.score_total, 2), _fmt(c.weight, 6)])

    (out_dir / "portfolio_1y_return_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote: {out_csv}")
    print(f"wrote: {out_w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
