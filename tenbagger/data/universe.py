from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .paths import default_market_db_path
from .sqlite_market import list_stocks, open_db


@dataclass(frozen=True)
class Stock:
    code: str
    secid: str
    name: str
    industry: str


def load_universe_from_sqlite(db_path: Optional[Path] = None, *, exclude_st: bool = True) -> List[Stock]:
    p = (db_path or default_market_db_path()).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"sqlite db not found: {p}")
    conn = open_db(p)
    rows = list_stocks(conn)
    out = [Stock(code=r.code, secid=r.secid, name=r.name, industry=r.industry) for r in rows]
    if exclude_st:
        out = [s for s in out if "ST" not in s.name.upper()]
    return out

