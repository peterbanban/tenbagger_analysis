#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class StockRow:
    code: str
    secid: str
    name: str
    industry: str


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def list_stocks(conn: sqlite3.Connection) -> List[StockRow]:
    rows = conn.execute(
        "SELECT code, secid, COALESCE(name,'') AS name, COALESCE(industry,'') AS industry FROM stocks ORDER BY code"
    ).fetchall()
    return [StockRow(code=r["code"], secid=r["secid"], name=r["name"], industry=r["industry"]) for r in rows]


def get_dates_and_closes(
    conn: sqlite3.Connection,
    code: str,
    fqt: int,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[List[str], List[float]]:
    sql = "SELECT date, close FROM kline_daily WHERE code=? AND fqt=?"
    params: List[object] = [code, int(fqt)]
    if start_date:
        sql += " AND date>=?"
        params.append(start_date)
    if end_date:
        sql += " AND date<=?"
        params.append(end_date)
    sql += " ORDER BY date"
    rows = conn.execute(sql, params).fetchall()
    dates: List[str] = []
    closes: List[float] = []
    for r in rows:
        dates.append(r["date"])
        closes.append(float(r["close"]) if r["close"] is not None else float("nan"))
    return dates, closes


def get_close_on(conn: sqlite3.Connection, code: str, fqt: int, d: str) -> Optional[float]:
    row = conn.execute(
        "SELECT close FROM kline_daily WHERE code=? AND fqt=? AND date=?",
        (code, int(fqt), d),
    ).fetchone()
    if not row or row[0] is None:
        return None
    return float(row[0])


def next_trading_date(conn: sqlite3.Connection, code: str, fqt: int, d: str) -> Optional[str]:
    row = conn.execute(
        "SELECT date FROM kline_daily WHERE code=? AND fqt=? AND date>=? ORDER BY date LIMIT 1",
        (code, int(fqt), d),
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def max_date(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute("SELECT MAX(date) FROM kline_daily").fetchone()
    return str(row[0]) if row and row[0] else None

