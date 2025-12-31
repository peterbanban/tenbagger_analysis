#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


START_DATE = "2019-01-01"
END_DATE = "2025-12-31"

EM_CLLIST_URL = (
    "https://push2.eastmoney.com/api/qt/clist/get"
    "?pn={pn}&pz={pz}&po=1&np=1&fltt=2&invt=2&fid=f3"
    "&fs=m:1+t:2,m:1+t:23,m:0+t:6,m:0+t:80"
    "&fields=f12,f14,f13,f100,f40"
)

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


def _http_get_json(url: str, *, retries: int = 5, timeout: int = 30) -> Dict[str, Any]:
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
            # jittered backoff
            time.sleep((2**attempt) * (0.2 + random.random() * 0.3))
    raise RuntimeError(f"failed GET {url}: {last_err}")


def _parse_date(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


@dataclass(frozen=True)
class Stock:
    secid: str  # e.g. 1.600309 or 0.300750
    code: str
    name: str
    industry: str
    current_mktcap_yuan: Optional[float]


@dataclass(frozen=True)
class TenBaggerHit:
    secid: str
    code: str
    name: str
    industry: str
    low_date: str
    low_close_fq: float  # 前复权收盘价（用于10x判定）
    hit_date: str
    hit_close_fq: float
    ratio: float
    days: int
    low_close_raw: Optional[float] = None  # 不复权收盘价（用于市值估算）
    hit_close_raw: Optional[float] = None
    low_total_shares: Optional[float] = None
    hit_total_shares: Optional[float] = None
    low_mktcap_yuan: Optional[float] = None
    hit_mktcap_yuan: Optional[float] = None


def get_all_a_shares() -> List[Stock]:
    stocks: List[Stock] = []
    # Eastmoney clist endpoint caps page size at 100, regardless of pz.
    page_size = 100
    pn = 1
    total: Optional[int] = None
    while True:
        url = EM_CLLIST_URL.format(pn=pn, pz=page_size)
        j = _http_get_json(url)
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
            mktcap = it.get("f40")
            secid = f"{mkt}.{code}"
            mktcap_val = None
            try:
                mktcap_val = float(mktcap) if mktcap not in (None, "-", "") else None
            except ValueError:
                mktcap_val = None
            if not code or not name or mkt not in (0, 1):
                continue
            stocks.append(
                Stock(
                    secid=secid,
                    code=code,
                    name=name,
                    industry=industry,
                    current_mktcap_yuan=mktcap_val,
                )
            )
        if total and pn * page_size >= total:
            break
        pn += 1
        time.sleep(0.2)
    return stocks


def get_kline_closes(secid: str, fqt: int) -> List[Tuple[str, float]]:
    url = EM_KLINE_URL.format(
        secid=secid, fqt=fqt, beg="20190101", end="20251231"
    )
    j = _http_get_json(url)
    data = j.get("data") or {}
    klines = data.get("klines") or []
    out: List[Tuple[str, float]] = []
    for row in klines:
        parts = row.split(",")
        if len(parts) < 3:
            continue
        d = parts[0]
        try:
            close = float(parts[2])
        except ValueError:
            continue
        if d < START_DATE or d > END_DATE:
            continue
        out.append((d, close))
    return out


def find_first_tenbagger(st: Stock) -> Optional[TenBaggerHit]:
    closes = get_kline_closes(st.secid, fqt=1)
    if len(closes) < 50:
        return None
    min_close = float("inf")
    min_date = ""
    for d, c in closes:
        if c <= 0:
            continue
        if c < min_close:
            min_close = c
            min_date = d
            continue
        if min_date:
            ratio = c / min_close
            if ratio >= 10.0:
                days = (_parse_date(d) - _parse_date(min_date)).days
                return TenBaggerHit(
                    secid=st.secid,
                    code=st.code,
                    name=st.name,
                    industry=st.industry,
                    low_date=min_date,
                    low_close_fq=min_close,
                    hit_date=d,
                    hit_close_fq=c,
                    ratio=ratio,
                    days=days,
                )
    return None


def _shares_at_date(changes: List[Dict[str, Any]], d: str) -> Optional[float]:
    target = datetime.strptime(d, "%Y-%m-%d").date()
    best: Optional[Tuple[date, float]] = None
    earliest: Optional[Tuple[date, float]] = None
    for it in changes:
        end_date = it.get("END_DATE")
        total = it.get("TOTAL_SHARES")
        if not end_date or total in (None, "", "-"):
            continue
        try:
            end_dt = datetime.strptime(end_date.split(" ")[0], "%Y-%m-%d").date()
            total_shares = float(total)
        except Exception:
            continue
        if earliest is None or end_dt < earliest[0]:
            earliest = (end_dt, total_shares)
        if end_dt <= target:
            if best is None or end_dt > best[0]:
                best = (end_dt, total_shares)
    if best:
        return best[1]
    # If target is earlier than the first recorded change, fall back to the earliest known shares.
    return earliest[1] if earliest else None


def enrich_with_mktcap(hit: TenBaggerHit) -> TenBaggerHit:
    mkt, code = hit.secid.split(".", 1)
    mkt_prefix = "SH" if mkt == "1" else "SZ"

    # shares history
    cap = _http_get_json(EM_CAPITAL_URL.format(mkt=mkt_prefix, code=code))
    changes = cap.get("lngbbd") or []
    low_sh = _shares_at_date(changes, hit.low_date)
    hi_sh = _shares_at_date(changes, hit.hit_date)

    # raw closes (for market cap)
    raw_closes = dict(get_kline_closes(hit.secid, fqt=0))
    low_raw = raw_closes.get(hit.low_date)
    hi_raw = raw_closes.get(hit.hit_date)

    low_mktcap = None
    hi_mktcap = None
    if low_raw is not None and low_sh is not None:
        low_mktcap = low_raw * low_sh
    if hi_raw is not None and hi_sh is not None:
        hi_mktcap = hi_raw * hi_sh

    return dataclasses.replace(
        hit,
        low_close_raw=low_raw,
        hit_close_raw=hi_raw,
        low_total_shares=low_sh,
        hit_total_shares=hi_sh,
        low_mktcap_yuan=low_mktcap,
        hit_mktcap_yuan=hi_mktcap,
    )


def write_csv(path: Path, hits: List[TenBaggerHit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "code",
                "name",
                "industry",
                "low_date",
                "hit_date",
                "days",
                "low_close_fq",
                "hit_close_fq",
                "ratio",
                "low_close_raw",
                "hit_close_raw",
                "low_total_shares",
                "hit_total_shares",
                "low_mktcap_yuan",
                "hit_mktcap_yuan",
            ]
        )
        for h in hits:
            w.writerow(
                [
                    h.code,
                    h.name,
                    h.industry,
                    h.low_date,
                    h.hit_date,
                    h.days,
                    f"{h.low_close_fq:.4f}",
                    f"{h.hit_close_fq:.4f}",
                    f"{h.ratio:.4f}",
                    "" if h.low_close_raw is None else f"{h.low_close_raw:.4f}",
                    "" if h.hit_close_raw is None else f"{h.hit_close_raw:.4f}",
                    "" if h.low_total_shares is None else f"{h.low_total_shares:.0f}",
                    "" if h.hit_total_shares is None else f"{h.hit_total_shares:.0f}",
                    "" if h.low_mktcap_yuan is None else f"{h.low_mktcap_yuan:.2f}",
                    "" if h.hit_mktcap_yuan is None else f"{h.hit_mktcap_yuan:.2f}",
                ]
            )


def summarize(hits: List[TenBaggerHit]) -> Dict[str, Any]:
    def bucket_mktcap(v: float) -> str:
        # RMB
        b = v / 1e8  # 亿元
        if b < 20:
            return "<20亿"
        if b < 50:
            return "20-50亿"
        if b < 100:
            return "50-100亿"
        if b < 200:
            return "100-200亿"
        if b < 500:
            return "200-500亿"
        return ">=500亿"

    by_industry: Dict[str, int] = {}
    by_bucket: Dict[str, int] = {}
    days_list: List[int] = []
    for h in hits:
        by_industry[h.industry or "未知"] = by_industry.get(h.industry or "未知", 0) + 1
        if h.low_mktcap_yuan is not None:
            k = bucket_mktcap(h.low_mktcap_yuan)
            by_bucket[k] = by_bucket.get(k, 0) + 1
        days_list.append(h.days)

    days_list.sort()
    def pct(p: float) -> Optional[int]:
        if not days_list:
            return None
        idx = int(round((len(days_list) - 1) * p))
        return days_list[idx]

    return {
        "count": len(hits),
        "days_p25": pct(0.25),
        "days_median": pct(0.50),
        "days_p75": pct(0.75),
        "industry_top20": sorted(by_industry.items(), key=lambda x: x[1], reverse=True)[:20],
        "mktcap_bucket_by_low": sorted(by_bucket.items(), key=lambda x: x[0]),
    }


def cmd_scan(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"
    hits_path = out_dir / "hits.jsonl"

    done: set[str] = set()
    if args.resume and progress_path.exists():
        for line in progress_path.read_text("utf-8", errors="ignore").splitlines():
            try:
                done.add(json.loads(line)["secid"])
            except Exception:
                continue

    stocks = get_all_a_shares()
    if args.limit:
        stocks = stocks[: args.limit]

    targets = [s for s in stocks if s.secid not in done]
    print(f"stocks total={len(stocks)} pending={len(targets)} workers={args.workers}", file=sys.stderr)

    new_hits: List[TenBaggerHit] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_map = {ex.submit(find_first_tenbagger, s): s for s in targets}
        for fut in as_completed(fut_map):
            st = fut_map[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = None
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"secid": st.secid, "ts": time.time()}) + "\n")
            if res:
                new_hits.append(res)
                with hits_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(dataclasses.asdict(res), ensure_ascii=False) + "\n")

    # load all basic hits from jsonl (resume-safe), then enrich them
    all_basic: Dict[str, TenBaggerHit] = {}
    if hits_path.exists():
        for line in hits_path.read_text("utf-8", errors="ignore").splitlines():
            try:
                obj = json.loads(line)
                secid = obj.get("secid")
                if not secid:
                    continue
                all_basic[secid] = TenBaggerHit(**obj)
            except Exception:
                continue

    basic_list = list(all_basic.values())
    enriched: List[TenBaggerHit] = []
    if basic_list:
        print(f"enriching {len(basic_list)} total hits with shares+raw close...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=min(args.workers, 8)) as ex:
            futs = [ex.submit(enrich_with_mktcap, h) for h in basic_list]
            for fut in as_completed(futs):
                try:
                    enriched.append(fut.result())
                except Exception:
                    continue

    all_hits = enriched
    all_hits.sort(key=lambda h: (h.low_date, h.code))
    write_csv(out_dir / "tenbaggers.csv", all_hits)
    summary = summarize(all_hits)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    csv_path = out_dir / "tenbaggers.csv"
    if not csv_path.exists():
        print(f"missing {csv_path}", file=sys.stderr)
        return 2
    hits: List[TenBaggerHit] = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                hits.append(
                    TenBaggerHit(
                        secid="",
                        code=row["code"],
                        name=row["name"],
                        industry=row["industry"],
                        low_date=row["low_date"],
                        low_close_fq=float(row["low_close_fq"]),
                        hit_date=row["hit_date"],
                        hit_close_fq=float(row["hit_close_fq"]),
                        ratio=float(row["ratio"]),
                        days=int(row["days"]),
                        low_close_raw=float(row["low_close_raw"]) if row["low_close_raw"] else None,
                        hit_close_raw=float(row["hit_close_raw"]) if row["hit_close_raw"] else None,
                        low_total_shares=float(row["low_total_shares"]) if row["low_total_shares"] else None,
                        hit_total_shares=float(row["hit_total_shares"]) if row["hit_total_shares"] else None,
                        low_mktcap_yuan=float(row["low_mktcap_yuan"]) if row["low_mktcap_yuan"] else None,
                        hit_mktcap_yuan=float(row["hit_mktcap_yuan"]) if row["hit_mktcap_yuan"] else None,
                    )
                )
            except Exception:
                continue
    summary = summarize(hits)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("scan")
    sp.add_argument("--out", required=True)
    sp.add_argument("--workers", type=int, default=10)
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--limit", type=int, default=0, help="debug: limit number of stocks")
    sp.set_defaults(func=cmd_scan)

    sp2 = sub.add_parser("summarize")
    sp2.add_argument("--out", required=True)
    sp2.set_defaults(func=cmd_summarize)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
