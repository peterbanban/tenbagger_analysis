#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# Eastmoney universe list (A-shares)
EM_CLLIST_URL = (
    "https://push2.eastmoney.com/api/qt/clist/get"
    "?pn={pn}&pz=100&po=1&np=1&fltt=2&invt=2&fid=f3"
    "&fs=m:1+t:2,m:1+t:23,m:0+t:6,m:0+t:80"
    "&fields=f12,f14,f13,f100"
)

# Eastmoney Kline (daily)
EM_KLINE_URL = (
    "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    "?secid={secid}&klt=101&fqt={fqt}&beg={beg}&end={end}"
    "&fields1=f1,f2,f3,f4,f5"
    "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
)

# Eastmoney stock/get (secid validation)
EM_STOCK_GET_URL = "https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f57"


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


@dataclass(frozen=True)
class Stock:
    code: str
    secid: str
    name: str
    industry: str


def _resolve_secid(code: str) -> str:
    candidates = ["1", "0"] if code.startswith(("6", "9")) else ["0", "1"]
    for mkt in candidates:
        secid = f"{mkt}.{code}"
        try:
            j = _http_get_json(EM_STOCK_GET_URL.format(secid=secid), retries=2)
            if (j.get("data") or {}).get("f57"):
                return secid
        except Exception:
            continue
    return f"{candidates[0]}.{code}"


def get_all_a_shares() -> List[Stock]:
    out: List[Stock] = []
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
            if not code or not name or mkt not in (0, 1):
                continue
            out.append(Stock(code=code, secid=f"{mkt}.{code}", name=name, industry=industry))
        if total and pn * 100 >= total:
            break
        pn += 1
        time.sleep(0.12)
    return out


def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _next_day_yyyymmdd(s: str) -> str:
    d = _parse_yyyy_mm_dd(s)
    return _yyyymmdd(d + timedelta(days=1))


def db_connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")  # ~200MB
    return conn


def db_init(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS stocks (
          code TEXT PRIMARY KEY,
          secid TEXT NOT NULL,
          name TEXT,
          industry TEXT,
          updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS kline_daily (
          code TEXT NOT NULL,
          fqt INTEGER NOT NULL,
          date TEXT NOT NULL,
          open REAL,
          close REAL,
          high REAL,
          low REAL,
          volume REAL,
          amount REAL,
          PRIMARY KEY (code, fqt, date)
        );

        CREATE INDEX IF NOT EXISTS idx_kline_date ON kline_daily(date);
        """
    )
    conn.commit()


def db_upsert_stocks(conn: sqlite3.Connection, stocks: List[Stock]) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    rows = [(s.code, s.secid, s.name, s.industry, now) for s in stocks]
    conn.executemany(
        """
        INSERT INTO stocks(code, secid, name, industry, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(code) DO UPDATE SET
          secid=excluded.secid,
          name=excluded.name,
          industry=excluded.industry,
          updated_at=excluded.updated_at
        """,
        rows,
    )
    conn.commit()


def db_list_codes(conn: sqlite3.Connection) -> List[Tuple[str, str, str]]:
    return list(conn.execute("SELECT code, secid, COALESCE(name,'') FROM stocks ORDER BY code"))


def db_last_kline_date(conn: sqlite3.Connection, code: str, fqt: int) -> Optional[str]:
    row = conn.execute(
        "SELECT MAX(date) FROM kline_daily WHERE code=? AND fqt=?",
        (code, fqt),
    ).fetchone()
    return row[0] if row and row[0] else None


def fetch_kline(secid: str, *, fqt: int, beg: str, end: str) -> List[Tuple[str, float, float, float, float, float, float]]:
    j = _http_get_json(EM_KLINE_URL.format(secid=secid, fqt=fqt, beg=beg, end=end))
    data = j.get("data") or {}
    kl = data.get("klines") or []
    out: List[Tuple[str, float, float, float, float, float, float]] = []
    for row in kl:
        parts = row.split(",")
        if len(parts) < 7:
            continue
        d = parts[0]
        try:
            o = float(parts[1])
            c = float(parts[2])
            h = float(parts[3])
            l = float(parts[4])
            vol = float(parts[5])
            amt = float(parts[6])
        except Exception:
            continue
        out.append((d, o, c, h, l, vol, amt))
    return out


def db_upsert_kline(
    conn: sqlite3.Connection,
    code: str,
    fqt: int,
    rows: List[Tuple[str, float, float, float, float, float, float]],
) -> None:
    conn.executemany(
        """
        INSERT INTO kline_daily(code, fqt, date, open, close, high, low, volume, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(code, fqt, date) DO UPDATE SET
          open=excluded.open,
          close=excluded.close,
          high=excluded.high,
          low=excluded.low,
          volume=excluded.volume,
          amount=excluded.amount
        """,
        [(code, fqt, d, o, c, h, l, vol, amt) for (d, o, c, h, l, vol, amt) in rows],
    )


def cmd_init(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser()
    conn = db_connect(db_path)
    db_init(conn)
    print(f"ok: {db_path}")
    return 0


def cmd_update_universe(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser()
    conn = db_connect(db_path)
    db_init(conn)
    stocks = get_all_a_shares()
    if args.exclude_st:
        stocks = [s for s in stocks if "ST" not in s.name.upper()]
    # normalize secid via resolve if requested
    if args.resolve_secid:
        fixed: List[Stock] = []
        for s in stocks:
            fixed.append(Stock(code=s.code, secid=_resolve_secid(s.code), name=s.name, industry=s.industry))
        stocks = fixed
    db_upsert_stocks(conn, stocks)
    print(f"universe={len(stocks)} wrote into {db_path}")
    return 0


def cmd_download_kline(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser()
    conn = db_connect(db_path)
    db_init(conn)

    codes = db_list_codes(conn)
    if not codes:
        raise RuntimeError("stocks table empty; run update-universe first")

    # Download range
    end = args.end or date.today().strftime("%Y-%m-%d")
    end_yyyymmdd = end.replace("-", "")
    beg_yyyymmdd = args.beg.replace("-", "")
    fqt_list = [0, 1] if args.with_fqt1 else [0]

    total = len(codes)
    done = 0

    # sequential is slower but gentler to upstream; you can run multiple processes if needed.
    for code, secid, name in codes:
        done += 1
        if args.limit and done > args.limit:
            break
        if done % 50 == 0:
            print(f"progress {done}/{total} ({done/total:.1%})")

        for fqt in fqt_list:
            last = db_last_kline_date(conn, code, fqt)
            beg = beg_yyyymmdd if not last else _next_day_yyyymmdd(last)
            if beg > end_yyyymmdd:
                continue
            try:
                rows = fetch_kline(secid, fqt=fqt, beg=beg, end=end_yyyymmdd)
            except Exception as e:
                if args.verbose:
                    print(f"warn: {code} {name} fqt={fqt} fetch failed: {e}")
                continue
            if not rows:
                continue
            db_upsert_kline(conn, code, fqt, rows)
            conn.commit()
        time.sleep(args.sleep)

    print(f"done. db={db_path}")
    return 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="sqlite db path")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="init sqlite schema")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("update-universe", help="download A-share universe list into stocks table")
    sp.add_argument("--exclude-st", action="store_true")
    sp.add_argument("--resolve-secid", action="store_true", help="validate secid via stock/get (slower)")
    sp.set_defaults(func=cmd_update_universe)

    sp = sub.add_parser("download-kline", help="download daily kline into sqlite (20y initial + incremental)")
    sp.add_argument("--beg", default="2005-01-01", help="begin date (YYYY-MM-DD)")
    sp.add_argument("--end", default="", help="end date (YYYY-MM-DD), default: today")
    sp.add_argument("--with-fqt1", action="store_true", help="also store fqt=1 (前复权) rows")
    sp.add_argument("--limit", type=int, default=0, help="debug limit number of stocks")
    sp.add_argument("--sleep", type=float, default=0.02, help="sleep seconds between stocks")
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=cmd_download_kline)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

