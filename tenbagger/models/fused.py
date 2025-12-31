from __future__ import annotations

"""
融合十倍股打分模型（单一版本）

目标：
- 先用“硬门槛”过滤：没有足够业务天花板/第二曲线/增长势能的公司，理论上很难承载 10x 级别估值扩张与利润扩张。
- 再用“拐点信号”尽量在底部/拐点附近介入，减少资金沉默成本（避免长期横盘等待）。
- 同时保留“质量/估值/压力”核心分，避免只追题材/只追走势。

数据口径（由上层 screener 负责取数）：
- 行情：前复权收盘（fqt=1）用于收益/趋势；不复权收盘 + 总股本用于估算市值/估值。
- 财务：Sina 年报（保守：截至日期 <= 4/30 时取 Y-2，否则取 Y-1）。
- 业务结构：东方财富 F10 BusinessAnalysis（海外收入占比、产品集中度等）。

分数结构（0..100）：
1) core_score：质量/估值/压力（“能拿得住”）
2) ceiling_total：行业天花板 + 出海天花板（“能做多大”）
3) growth_total：3年 CAGR 代理（“能否持续扩容/增长”）
4) second_total：第二曲线代理（多元化/研发/Capex）（“是否在培育新增量”）
5) turning/low_cost：拐点与低成本买入代理（“减少沉默成本、避免追高”）

最终输出：
- hard_gate(...)：硬门槛是否通过（前置过滤）
- score_fused_total(...)：融合总分（用于 Top30 排序）
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


def parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")


def _ma(closes: list[float], i: int, window: int) -> Optional[float]:
    if window <= 0 or i < window - 1:
        return None
    seg = closes[i - (window - 1) : i + 1]
    if not seg:
        return None
    return sum(seg) / window


def momentum_from_series(
    dates: list[str], closes: list[float], asof_date: str
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        i = dates.index(asof_date)
    except ValueError:
        return None, None, None
    if i < 260:
        return None, None, None
    c0 = closes[i]
    c120 = closes[i - 120]
    c250 = closes[i - 250]
    ret120 = (c0 / c120 - 1.0) if c120 > 0 else None
    ret250 = (c0 / c250 - 1.0) if c250 > 0 else None
    high250 = max(closes[i - 250 : i + 1])
    dd = (c0 / high250 - 1.0) if high250 > 0 else None
    return ret120, ret250, dd


@dataclass(frozen=True)
class InflectionSignal:
    ok: bool
    ma20: Optional[float]
    ma60: Optional[float]
    ma120: Optional[float]
    ma60_slope_20d: Optional[float]
    pos_from_250_low: Optional[float]


def inflection_signal(dates: list[str], closes: list[float], asof_date: str) -> InflectionSignal:
    """
    Attempts to catch "low-cost but turning" entries:
    - Trend turns up (MA20 > MA60 and price > MA20)
    - MA60 is rising (vs 20 trading days ago) to reduce dead money
    - Still not too far from 250d low
    """
    try:
        i = dates.index(asof_date)
    except ValueError:
        return InflectionSignal(False, None, None, None, None, None)
    if i < 260:
        return InflectionSignal(False, None, None, None, None, None)

    c0 = closes[i]
    ma20 = _ma(closes, i, 20)
    ma60 = _ma(closes, i, 60)
    ma120 = _ma(closes, i, 120)
    ma60_prev = _ma(closes, i - 20, 60) if i >= 80 else None
    ma60_slope = (ma60 / ma60_prev - 1.0) if (ma60 is not None and ma60_prev is not None and ma60_prev > 0) else None

    low250 = min(closes[i - 250 : i + 1])
    pos_from_low = (c0 / low250 - 1.0) if low250 > 0 else None

    trend_ok = (ma20 is not None and ma60 is not None and c0 > ma20 and ma20 > ma60)
    slope_ok = (ma60_slope is not None and ma60_slope > 0)
    low_cost_ok = (pos_from_low is not None and pos_from_low <= 0.60)

    ok = bool(trend_ok and slope_ok and low_cost_ok)
    return InflectionSignal(ok, ma20, ma60, ma120, ma60_slope, pos_from_low)


def score_quality_value_core(
    mktcap_yi: Optional[float],
    pe: Optional[float],
    pb: Optional[float],
    nm: Optional[float],
    roe: Optional[float],
    debt_ratio: Optional[float],
    dd250: Optional[float],
) -> float:
    """
    核心分（0..100）：质量/估值/压力（偏“价值投资底层”）

    维度与分值（合计上限 100）：
    - 规模（20）：偏好 10~50亿（更容易 10x），过大/过小扣分
    - 盈利能力（25）：净利率 nm 越高越好
    - ROE 代理（10）：用 年报净利润/归母权益 近似
    - 估值（25）：PE、PB 越“合理/便宜”加分（避免买太贵）
    - 压力/回撤（10）：距 250日高点回撤更深在“基本面OK”前提下反而加分（便宜/悲观）
    - 杠杆安全（10）：资产负债率越低越好
    """
    score = 0.0
    # Size (20)
    if mktcap_yi is not None:
        if 10 <= mktcap_yi <= 50:
            score += 20
        elif 50 < mktcap_yi <= 120:
            score += 17
        elif 5 <= mktcap_yi < 10:
            score += 14
        elif 120 < mktcap_yi <= 250:
            score += 10
        elif 250 < mktcap_yi <= 600:
            score += 6
        else:
            score += 3
    # Profitability (25)
    if nm is not None:
        if nm >= 0.15:
            score += 25
        elif nm >= 0.10:
            score += 20
        elif nm >= 0.05:
            score += 12
        elif nm > 0:
            score += 6
    # ROE proxy (10)
    if roe is not None:
        if roe >= 0.20:
            score += 10
        elif roe >= 0.15:
            score += 8
        elif roe >= 0.10:
            score += 6
        elif roe >= 0.05:
            score += 3
    # Valuation (25)
    if pe is not None and pe > 0:
        if pe < 12:
            score += 14
        elif pe < 18:
            score += 12
        elif pe < 28:
            score += 9
        elif pe < 45:
            score += 5
    if pb is not None and pb > 0:
        if pb < 1.5:
            score += 11
        elif pb < 2.5:
            score += 8
        elif pb < 4:
            score += 6
        elif pb < 8:
            score += 3
    # Stress (drawdown) (10) – deeper drawdown is a plus if fundamentals are ok
    if dd250 is not None:
        if dd250 <= -0.70:
            score += 10
        elif dd250 <= -0.55:
            score += 8
        elif dd250 <= -0.40:
            score += 6
        elif dd250 <= -0.25:
            score += 3
    # Balance sheet safety (10)
    if debt_ratio is not None:
        if debt_ratio < 0.55:
            score += 10
        elif debt_ratio < 0.65:
            score += 7
        elif debt_ratio < 0.75:
            score += 4
        else:
            score += 1
    return min(100.0, score)


def _cagr(old: Optional[float], new: Optional[float], years: int) -> Optional[float]:
    if old is None or new is None or years <= 0:
        return None
    if old <= 0 or new <= 0:
        return None
    try:
        return (new / old) ** (1.0 / years) - 1.0
    except Exception:
        return None


def growth_score(rev_cagr: Optional[float], profit_cagr: Optional[float]) -> float:
    """
    增长得分（0..100）：用 3年 CAGR 作为“市场扩容+公司成长质量”的代理。

    - rev_cagr：收入 3年CAGR（更接近行业空间/份额提升）
    - profit_cagr：利润 3年CAGR（更接近经营杠杆/产品力/价格权）

    注意：这里只是“代理”，真实的行业天花板与第二曲线仍以 F10/业务信息为主。
    """
    score = 0.0
    c = rev_cagr
    if c is not None:
        if c >= 0.35:
            score += 55
        elif c >= 0.25:
            score += 48
        elif c >= 0.15:
            score += 38
        elif c >= 0.08:
            score += 28
        elif c > 0:
            score += 18
        else:
            score += 6
    p = profit_cagr
    if p is not None:
        if p >= 0.45:
            score += 45
        elif p >= 0.30:
            score += 38
        elif p >= 0.18:
            score += 28
        elif p >= 0.08:
            score += 18
        elif p > 0:
            score += 10
        else:
            score += 4
    return min(100.0, score)


def industry_ceiling_score(industry: str) -> float:
    """
    行业天花板（0..100）：基于行业字符串的启发式映射。

    - 高：软件/半导体/新能源/电池/机器人等（更可能容纳高估值）
    - 中：化工/机械/消费电子等（可出十倍，但更依赖公司竞争力与扩张）
    - 低：煤钢银行地产公用等（通常更偏周期/分红属性）
    """
    s = (industry or "").strip()
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


def overseas_score(overseas_share: Optional[float]) -> float:
    """
    出海天花板（0..100）：海外收入占比越高，通常意味着可拓展市场更大/定价体系更多元（启发式）。
    """
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


def hhi(shares: list[float]) -> Optional[float]:
    if not shares:
        return None
    s = sum(shares)
    if s <= 0:
        return None
    return sum((x / s) ** 2 for x in shares if x > 0)


def diversification_score(top1_share: Optional[float], hhi_value: Optional[float], seg_cnt: int) -> float:
    """
    多元化得分（0..100）：第二曲线的“业务结构代理”。

    - seg_cnt：收入分部/产品条目数（越多越可能有第二曲线，但也可能是杂）
    - top1_share：第一大产品占比（过高说明过度依赖单一品类）
    - hhi：集中度指标（越低越分散）
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

    if hhi_value is not None:
        if hhi_value <= 0.35:
            score += 35
        elif hhi_value <= 0.50:
            score += 22
        elif hhi_value <= 0.65:
            score += 12
        else:
            score += 5
    return min(100.0, score)


def rd_score(rd_ratio: Optional[float]) -> float:
    """
    研发强度（0..100）：研发/营收。
    """
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


def capex_score(capex_ratio: Optional[float]) -> float:
    """
    资本开支强度（0..100）：资本开支/营收。
    对制造业/产能扩张型第二曲线更相关；对轻资产软件类可能偏低但不一定是坏事。
    """
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


def cash_quality_score(*, cfo_yuan: Optional[float], net_profit_yuan: Optional[float]) -> float:
    """
    现金含金量（0..100）：经营现金流与利润的匹配程度（CFO≈利润）。
    典型“好公司”：利润增长时，CFO 不明显掉队。
    """
    if net_profit_yuan is None or net_profit_yuan == 0 or cfo_yuan is None:
        return 0.0
    if net_profit_yuan < 0:
        return 0.0
    ratio = cfo_yuan / net_profit_yuan
    if ratio >= 1.3:
        return 100.0
    if ratio >= 1.0:
        return 90.0
    if ratio >= 0.8:
        return 75.0
    if ratio >= 0.6:
        return 55.0
    if ratio >= 0.4:
        return 35.0
    if ratio > 0:
        return 20.0
    return 0.0


def ops_quality_score(
    *,
    ar_to_rev: Optional[float],
    inv_to_rev: Optional[float],
    ar_to_rev_delta: Optional[float],
    inv_to_rev_delta: Optional[float],
) -> float:
    """
    营运质量（0..100）：用“应收/收入、存货/收入”及其变化做代理。
    - 比例越低越好（资金占用少、回款/周转更好）
    - 比例上升（delta>0）扣分（可能是压货/回款变差）
    """
    score = 0.0
    if ar_to_rev is not None:
        if ar_to_rev <= 0.10:
            score += 30
        elif ar_to_rev <= 0.18:
            score += 22
        elif ar_to_rev <= 0.28:
            score += 14
        else:
            score += 6
    if inv_to_rev is not None:
        if inv_to_rev <= 0.12:
            score += 30
        elif inv_to_rev <= 0.22:
            score += 22
        elif inv_to_rev <= 0.35:
            score += 14
        else:
            score += 6
    if ar_to_rev_delta is not None:
        if ar_to_rev_delta <= -0.03:
            score += 20
        elif ar_to_rev_delta <= 0.00:
            score += 14
        elif ar_to_rev_delta <= 0.03:
            score += 8
        else:
            score += 2
    if inv_to_rev_delta is not None:
        if inv_to_rev_delta <= -0.03:
            score += 20
        elif inv_to_rev_delta <= 0.00:
            score += 14
        elif inv_to_rev_delta <= 0.03:
            score += 8
        else:
            score += 2
    return min(100.0, score)


def refi_pressure_score(
    *,
    cash_yuan: Optional[float],
    short_debt_yuan: Optional[float],
    long_debt_yuan: Optional[float],
) -> float:
    """
    再融资/偿债压力（0..100）：用货币资金与有息负债结构做代理。
    - 净现金更好
    - 短债/现金比越低越好（短期压力低）
    """
    cash = cash_yuan if cash_yuan is not None else 0.0
    sd = short_debt_yuan if short_debt_yuan is not None else 0.0
    ld = long_debt_yuan if long_debt_yuan is not None else 0.0
    debt = sd + ld
    if cash <= 0 and debt <= 0:
        return 0.0

    net_debt = debt - cash
    short_cov = (sd / cash) if cash > 0 else None

    score = 0.0
    if net_debt <= 0:
        score += 55
    else:
        # net debt ratio proxy
        if debt > 0:
            nd_ratio = net_debt / debt
            if nd_ratio <= 0.25:
                score += 40
            elif nd_ratio <= 0.50:
                score += 28
            elif nd_ratio <= 0.75:
                score += 18
            else:
                score += 10

    if short_cov is not None:
        if short_cov <= 0.3:
            score += 45
        elif short_cov <= 0.6:
            score += 35
        elif short_cov <= 1.0:
            score += 24
        elif short_cov <= 2.0:
            score += 12
        else:
            score += 5
    else:
        if cash > 0 and sd <= 0:
            score += 30

    return min(100.0, score)


def interest_coverage_score(*, operating_profit_yuan: Optional[float], interest_expense_yuan: Optional[float]) -> float:
    """
    利息保障（0..100）：用营业利润/利息支出（或财务费用正值代理）做近似。
    """
    if operating_profit_yuan is None or interest_expense_yuan is None:
        return 0.0
    if operating_profit_yuan <= 0 or interest_expense_yuan <= 0:
        return 0.0
    cov = operating_profit_yuan / interest_expense_yuan
    if cov >= 12:
        return 100.0
    if cov >= 8:
        return 85.0
    if cov >= 5:
        return 70.0
    if cov >= 3:
        return 50.0
    if cov >= 1.5:
        return 30.0
    return 10.0


def financial_quality_total(
    *,
    cfo_yuan: Optional[float],
    net_profit_yuan: Optional[float],
    ar_to_rev: Optional[float],
    inv_to_rev: Optional[float],
    ar_to_rev_delta: Optional[float],
    inv_to_rev_delta: Optional[float],
    cash_yuan: Optional[float],
    short_debt_yuan: Optional[float],
    long_debt_yuan: Optional[float],
    operating_profit_yuan: Optional[float],
    interest_expense_yuan: Optional[float],
) -> float:
    """
    财务/运营质量综合分（0..100）：把“现金含金量、营运质量、偿债/再融资压力、利息保障”合并。
    """
    s_cash = cash_quality_score(cfo_yuan=cfo_yuan, net_profit_yuan=net_profit_yuan)
    s_ops = ops_quality_score(
        ar_to_rev=ar_to_rev, inv_to_rev=inv_to_rev, ar_to_rev_delta=ar_to_rev_delta, inv_to_rev_delta=inv_to_rev_delta
    )
    s_refi = refi_pressure_score(cash_yuan=cash_yuan, short_debt_yuan=short_debt_yuan, long_debt_yuan=long_debt_yuan)
    s_cov = interest_coverage_score(operating_profit_yuan=operating_profit_yuan, interest_expense_yuan=interest_expense_yuan)
    return 0.35 * s_cash + 0.25 * s_ops + 0.25 * s_refi + 0.15 * s_cov


@dataclass(frozen=True)
class HardGate:
    ok: bool
    ceiling_total: float
    second_total: float
    growth_total: float


def hard_gate(
    *,
    ceiling_total: float,
    second_total: float,
    growth_total: float,
    min_ceiling_total: float = 55.0,
    min_second_total: float = 55.0,
    min_growth_total: float = 35.0,
) -> HardGate:
    """
    Tenbagger hard gate:
    - Must have (ceiling >= threshold OR second_curve_proxy >= threshold)
    - Must have non-trivial multi-year growth (proxy)

    默认阈值解释：
    - min_growth_total=35：至少要有“非平庸”的 3年增长势能，否则很难走出 10x 的长期主升浪
    - min_ceiling_total=55：行业/出海维度至少不差（空间不低）
    - min_second_total=55：或者第二曲线代理足够强（业务可能能再做大一轮）
    """
    if growth_total < min_growth_total:
        return HardGate(False, ceiling_total, second_total, growth_total)
    if ceiling_total >= min_ceiling_total or second_total >= min_second_total:
        return HardGate(True, ceiling_total, second_total, growth_total)
    return HardGate(False, ceiling_total, second_total, growth_total)


def score_fused_total(
    *,
    core_score: float,
    gate: HardGate,
    signal: InflectionSignal,
    ret120: Optional[float],
    ret250: Optional[float],
    dd250: Optional[float],
    second_total: float,
    fin_quality: float,
) -> float:
    """
    融合总分（0..100），用于排序选股。

    权重（可理解为四大维度）：
    - 32% core_score：质量/估值/压力（更稳的底座）
    - 20% turning：拐点/趋势确认（减少沉默成本）
    - 13% low_cost：低成本买入代理（离 250日低点更近、回撤更深更加分）
    - 25% prereq：十倍股前置条件（天花板/增长/第二曲线）
    - 10% fin_quality：财务/运营质量（现金含金量、营运质量、再融资压力等）
    """
    turning = 0.0
    if signal.ok:
        turning += 60
    else:
        if ret120 is not None and ret120 > 0:
            turning += 20
        if signal.ma60_slope_20d is not None and signal.ma60_slope_20d > 0:
            turning += 15
    if ret250 is not None and ret250 > 0:
        turning += 10
    turning = min(100.0, turning)

    low_cost = 0.0
    if signal.pos_from_250_low is not None:
        if signal.pos_from_250_low <= 0.25:
            low_cost += 50
        elif signal.pos_from_250_low <= 0.45:
            low_cost += 35
        elif signal.pos_from_250_low <= 0.60:
            low_cost += 22
        else:
            low_cost += 8
    if dd250 is not None:
        if dd250 <= -0.55:
            low_cost += 25
        elif dd250 <= -0.35:
            low_cost += 18
        elif dd250 <= -0.20:
            low_cost += 10
        elif dd250 <= -0.05:
            low_cost += 4
    low_cost = min(100.0, low_cost)

    # prereq：十倍股“前置条件”本身也参与打分（不是只做过滤）。
    prereq = 0.0
    prereq += 0.40 * gate.ceiling_total
    prereq += 0.30 * gate.growth_total
    prereq += 0.30 * second_total

    total = 0.32 * core_score + 0.20 * turning + 0.13 * low_cost + 0.25 * prereq + 0.10 * fin_quality
    return float(min(100.0, max(0.0, total)))
