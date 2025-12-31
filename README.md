# tenbagger_analysis：A股十倍股研究与选股

本项目把“数据 / 模型 / 选股 / 回测”拆分成清晰的模块，支持先把近几年A股日K落到本地SQLite，后续脚本优先离线查询，减少全市场重复拉取与限流问题。

## 目录结构（核心）
- `tenbagger/data/`：数据层（SQLite行情库 + 网络补全：股本/财务/F10等）
- `tenbagger/models/`：模型与打分逻辑（两模型 + 框架V2天花板/第二曲线代理）
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

## 注意
- SQLite行情库（`data/*.sqlite`）默认不提交Git。
- 估值用到的“股本/财务/F10”等仍需联网抓取（已做本地cache，避免重复下载）。
