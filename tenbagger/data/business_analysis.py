from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .http import http_get_json
from ..models.fused import hhi


EM_BUSINESS_ANALYSIS_URL = "https://emweb.securities.eastmoney.com/PC_HSF10/BusinessAnalysis/PageAjax?code={code}"


@dataclass(frozen=True)
class BusinessSnapshot:
    report_date: str
    overseas_share: Optional[float]
    product_top1_share: Optional[float]
    product_hhi: Optional[float]
    product_seg_cnt: int


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _income(r: Dict[str, Any]) -> float:
    try:
        return float(r.get("MAIN_BUSINESS_INCOME") or 0.0)
    except Exception:
        return 0.0


def _load_json_cached(url: str, cache_path: Path) -> Dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text("utf-8", errors="ignore"))
    j = http_get_json(url)
    cache_path.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    return j


def business_snapshot_asof(
    *,
    code: str,
    mkt_prefix: str,
    asof_date: str,
    cache_dir: Path,
) -> Optional[BusinessSnapshot]:
    """
    Uses Eastmoney F10 BusinessAnalysis `zygcfx` to derive:
    - overseas_share: based on MAINOP_TYPE==3 rows with ITEM_NAME containing '外'
    - product concentration: MAINOP_TYPE==2 rows (product)
    """
    ba_code = f"{mkt_prefix}{code}"
    j = _load_json_cached(
        EM_BUSINESS_ANALYSIS_URL.format(code=ba_code),
        cache_dir / f"business_{ba_code}.json",
    )
    zygcfx = j.get("zygcfx") or []
    if not zygcfx:
        return None
    asof = asof_date + " 23:59:59"
    dates = sorted({str(r.get("REPORT_DATE") or "") for r in zygcfx if r.get("REPORT_DATE")})
    cand = [d for d in dates if d <= asof]
    if not cand:
        return None
    pick = cand[-1]
    rows = [r for r in zygcfx if str(r.get("REPORT_DATE") or "") == pick]

    product = [r for r in rows if str(r.get("MAINOP_TYPE") or "") == "2"]
    region = [r for r in rows if str(r.get("MAINOP_TYPE") or "") == "3"]

    total_income = sum(_income(r) for r in rows) or None
    overseas_income = None
    if region and total_income:
        overseas_income = sum(_income(r) for r in region if "外" in str(r.get("ITEM_NAME") or ""))
    overseas_share = _safe_div(overseas_income, total_income) if (overseas_income is not None and total_income) else None

    prod_incomes = [x for x in (_income(r) for r in product) if x > 0]
    prod_total = sum(prod_incomes)
    prod_shares: List[float] = [(x / prod_total) for x in prod_incomes] if prod_total > 0 else []
    prod_shares.sort(reverse=True)

    top1 = prod_shares[0] if prod_shares else None
    hhi_value = hhi(prod_shares) if prod_shares else None

    return BusinessSnapshot(
        report_date=pick.split(" ")[0],
        overseas_share=overseas_share,
        product_top1_share=top1,
        product_hhi=hhi_value,
        product_seg_cnt=len(prod_shares),
    )
