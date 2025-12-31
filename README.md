# A股 2019-2025 十倍股扫描（收盘价、前复权）

按你的口径：
- 价格：收盘价
- 复权：前复权（`fqt=1`）
- 区间：`2019-01-01` 到 `2025-12-31`，任意“区间内低点 → 后续高点”首次达到 `10x`
- 市值：优先用公开接口的“当日总市值”（东方财富日K不直接给），因此本工具采用兜底：`当日收盘价(不复权) × 当日总股本`（总股本来自东方财富 HSF10 股本变动）

## 运行

首次运行（会拉取全A股列表并开始扫描，时间较长，建议让它跑完）：
```bash
python3 a_share_tenbagger/tenbagger_scan.py scan --out a_share_tenbagger/out
```

扫描完成后生成：
- `a_share_tenbagger/out/tenbaggers.csv`：所有在区间内达到过10倍的股票清单（含低点/达成日/用时/市值等）
- `a_share_tenbagger/out/summary.json`：汇总统计（市值分布、行业分布、用时分布等）

中断后续跑（断点续跑）：
```bash
python3 a_share_tenbagger/tenbagger_scan.py scan --out a_share_tenbagger/out --resume
```

只做汇总（不重新抓数据）：
```bash
python3 a_share_tenbagger/tenbagger_scan.py summarize --out a_share_tenbagger/out
```

## 半自动选股（硬筛→人工打分）

用东方财富公开接口做一轮“硬筛”，把全A股缩小到可人工研究的候选池（默认排除`ST/*ST`）：
```bash
python3 a_share_tenbagger/hard_screen.py --out a_share_tenbagger/out_screen
```

为了把候选池进一步压缩到约`~200`只以内（更适合手工打分），可用更严格参数（示例）：
```bash
python3 a_share_tenbagger/hard_screen.py --out a_share_tenbagger/out_screen_strict --workers 24 \
  --min-mktcap-yi 15 --max-mktcap-yi 80 \
  --min-pe 8 --max-pe 45 \
  --min-revenue-yi 10 --min-net-profit-yi 1.0 --min-net-margin 0.08 \
  --min-ret-250d 0.10 --min-dd-from-250d-high -0.20 \
  --min-score 20
```

输出：
- `a_share_tenbagger/out_screen/candidates.csv`：候选清单（市值、PE/PB、营收/净利、简单动量、综合分）
- `a_share_tenbagger/out_screen/candidates_summary.json`：参数与Top10

然后用打分卡进行人工深度分析：
- `a_share_tenbagger/scorecard_template.md`
- `a_share_tenbagger/scorecard_template.csv`

## 注意
- 数据源：东方财富 `push2/clist` + `push2his/kline` + `PC_HSF10/CapitalStockStructure`
- 访问频率过高可能触发限流；脚本内置重试与随机退避，但如果你网络不稳定，可降低并发：`--workers 6`
