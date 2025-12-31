# tenbagger_analysis：A股十倍股研究与选股

本项目把“数据 / 模型 / 选股 / 回测”拆分成清晰的模块，支持先把近几年A股日K落到本地SQLite，后续脚本优先离线查询，减少全市场重复拉取与限流问题。

## 目录结构（核心）
- `tenbagger/data/`：数据层（SQLite行情库 + 网络补全：股本/财务/F10等）
- `tenbagger/models/`：模型与打分逻辑（融合模型：硬门槛 + 天花板/第二曲线 + 增长 + 拐点）
- `tenbagger/screeners/`：选股脚本（输出Top30等）
- `tenbagger/backtests/`：回测脚本（组合收益分布估计等）
- `tenbagger/tools/`：一次性工具（本地行情库构建、十倍股扫描等）
- `data/`：本地SQLite行情库（大文件，已在 `.gitignore` 忽略）

说明：根目录的 `local_market_db.py` / `tenbagger_scan.py` 是薄入口，具体实现放在 `tenbagger/` 下，便于复用与维护。

## 1) 本地数据（推荐先做）
近5年 + 前复权日K落库（会使用网络，首次耗时较长）：
```bash
python3 local_market_db.py --db data/tenbagger_analysis_market.sqlite init
python3 local_market_db.py --db data/tenbagger_analysis_market.sqlite update-universe --exclude-st
python3 local_market_db.py --db data/tenbagger_analysis_market.sqlite download-kline --beg 2021-01-01 --with-fqt1
```

说明见：
- `LOCAL_DATA.md`
- `MARKET_DB_EXPERIMENT.md`

## 2) 选股（Top30）
融合模型Top30（默认：硬门槛 + 拐点信号，尽量降低沉默成本）：
```bash
python3 screen_top30.py --out out_screen --exclude-st
```

可选：关闭硬门槛 / 关闭拐点过滤（不推荐）：
`python3 screen_top30.py --out out_screen --exclude-st --disable-gate`
`python3 screen_top30.py --out out_screen --exclude-st --no-require-entry`

输出字段说明（核心）：
- `总分(融合模型)`：排序用总分（见 `tenbagger/models/fused.py`）
- `硬门槛通过(1/0)`：十倍股“前置条件”过滤是否通过（行业天花板/第二曲线/增长）
- `拐点信号通过(1/0)`：尽量在“低成本拐点”介入，降低沉默成本

模型实现：
- 打分模型：`tenbagger/models/fused.py`
- Top30脚本：`tenbagger/screeners/top30_fused.py`

## 3) 回测（固定篮子1年收益分布估计）
按“CSV篮子（按总分分权）”做滚动1年收益分布估计：
```bash
python3 portfolio_expected_return_1y.py --basket out_screen/top30_fused.csv --out out_portfolio_1y
```

## 4) 十倍股扫描（可选，联网）
区间十倍股扫描（2019-01-01..2025-12-31，前复权收盘价判定10x）：
```bash
python3 tenbagger_scan.py scan --out out_tenbagger_scan --resume
```

## 研究评分卡（人工）
下面这张表是“做单家公司 360 度研究”的检查清单，用于补足模型难以自动化抓取的数据（订单/客户/产品力/行业格局等）。  
融合模型会尽量用可获取的公开数据做“代理指标”，但**不等同于**完整的行业/公司研究。

| 因子（人工研究） | 建议权重(分) | 模型覆盖情况（代理指标） | 证据/数据来源举例 |
|---|---:|---|---|
| 行业天花板/空间（渗透率/扩容清晰） | 10 | 部分：`industry_ceiling_score()`（行业启发式）+ `growth_score()`（3年CAGR代理） | 政策/产业趋势/海外对标/渗透率曲线 |
| 第二增长曲线是否清晰（新增） | 10 | 部分：`diversification_score()` + `rd_score()` + `capex_score()`（第二曲线代理）+ `hard_gate()` 作为硬门槛 | 新品/新业务收入占比、扩产项目、研发路线图 |
| 行业格局（集中度提升与赢家份额） | 10 | 未覆盖（缺CR3/CR5/份额等） | CR3/CR5/龙头份额/供给约束 |
| 产品力（性能/成本/良率/交付优势） | 10 | 未覆盖 | 良率/成本曲线/性能参数/交付数据 |
| 客户粘性（认证壁垒/切换成本/渠道） | 10 | 未覆盖 | 头部客户、认证周期、复购、渠道结构 |
| 技术迭代（研发→产品化效率） | 5 | 部分：研发强度 `rd_score()`（不等同于产品化效率） | 新品推出节奏、平台能力、研发转化 |
| 领先指标（订单/出货/利用率） | 10 | 未覆盖 | 在手订单、交付周期、产能利用率 |
| 扩张兑现（扩产/出海/渠道复制） | 10 | 部分：出海 `overseas_score()`；扩产 `capex_score()`（但不代表爬坡成功） | 扩产进度、爬坡、海外订单 |
| 现金含金量（CFO≈利润） | 5 | 未覆盖 | 经营现金流、回款、应收变化 |
| 回报能力（ROIC/ROE趋势） | 5 | 部分：ROE 代理（未做趋势/ROIC） | ROIC/ROE趋势、与同行对比 |
| 资产负债（杠杆与再融资压力） | 5 | 部分：资产负债率（未做净负债/短债/利息保障） | 净负债、短债、利息保障倍数 |
| 运营质量（应收/存货周转） | 5 | 未覆盖 | 周转天数、结构变化 |
| 估值与预期差（不贵且预期低） | 10 | 部分：PE/PB + 回撤（情绪/一致预期未覆盖） | 相对估值、预期差、交易拥挤度 |
| 催化（可验证事件路径） | 5 | 未覆盖 | 订单、产品、政策、海外突破等 |

## 注意
- SQLite行情库（`data/*.sqlite`）默认不提交Git。
- 估值用到的“股本/财务/F10”等仍需联网抓取（已做本地cache，避免重复下载）。
