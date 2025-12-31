from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .http import http_get_json


EM_CAPITAL_URL = (
    "https://emweb.securities.eastmoney.com/PC_HSF10/CapitalStockStructure/PageAjax"
    "?code={mkt}{code}"
)


def _parse_date_ymd(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _shares_at_date(changes: List[Dict[str, Any]], d: str) -> Optional[float]:
    target = _parse_date_ymd(d).date()
    best: Optional[Tuple[datetime, float]] = None
    earliest: Optional[Tuple[datetime, float]] = None
    for it in changes:
        end_date = it.get("END_DATE")
        total = it.get("TOTAL_SHARES")
        if not end_date or total in (None, "", "-"):
            continue
        try:
            end_dt = _parse_date_ymd(end_date.split(" ")[0])
            total_shares = float(total)
        except Exception:
            continue
        if earliest is None or end_dt < earliest[0]:
            earliest = (end_dt, total_shares)
        if end_dt.date() <= target:
            if best is None or end_dt > best[0]:
                best = (end_dt, total_shares)
    if best:
        return best[1]
    return earliest[1] if earliest else None


def get_total_shares(code: str, *, mkt_prefix: str, asof_date: str, cache_dir: Path) -> Optional[float]:
    """
    Returns TOTAL_SHARES as-of date using Eastmoney HSF10 capital stock structure.
    mkt_prefix: 'SH' or 'SZ'.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"capital_{mkt_prefix}{code}.json"
    if not cache.exists():
        cap = http_get_json(EM_CAPITAL_URL.format(mkt=mkt_prefix, code=code))
        cache.write_text(json.dumps(cap, ensure_ascii=False), encoding="utf-8")
    else:
        cap = json.loads(cache.read_text("utf-8", errors="ignore"))
    changes = cap.get("lngbbd") or []
    return _shares_at_date(changes, asof_date)

