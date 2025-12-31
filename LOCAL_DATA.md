# 本地A股数据仓库（避免每次全市场拉取）

目标：一次性拉取并本地保存：
- A股全量股票列表（代码/名称/行业/市场标识）
- 近20年日K数据（默认 2005-01-01 至今）

这样你后续做“回到某个日期选股/回测/统计”时可以直接本地SQL查询，不必每次对东方财富/Sina 发起全市场请求。

## 1) 磁盘空间预估
近20年日K：约 5200 只 × ~5000 交易日 ≈ 2600万行。
SQLite占用取决于字段与索引，通常会在 **数GB** 级别。

## 2) 初始化与下载（一次性）
在终端执行（会使用网络，跑很久，建议夜间跑）：

1) 初始化数据库：
`python3 local_market_db.py --db ~/Documents/github/tenbagger_analysis_market.sqlite init`

2) 下载股票列表（A股全量）：
`python3 local_market_db.py --db ~/Documents/github/tenbagger_analysis_market.sqlite update-universe --exclude-st`

3) 下载近20年日K（不复权 fqt=0；如需要前复权也可加 --with-fqt1）：
`python3 local_market_db.py --db ~/Documents/github/tenbagger_analysis_market.sqlite download-kline --beg 2005-01-01`

可选：同时存前复权数据（体积更大）：
`python3 local_market_db.py --db ~/Documents/github/tenbagger_analysis_market.sqlite download-kline --beg 2005-01-01 --with-fqt1`

## 3) 增量更新（日常）
每次只需要拉取“最后缺失的日期之后”的数据（脚本会从数据库里 `MAX(date)` 自动续拉）：
`python3 local_market_db.py --db ~/Documents/github/tenbagger_analysis_market.sqlite download-kline --beg 2005-01-01`

## 4) 查询示例（SQLite）
用sqlite3：
`sqlite3 ~/Documents/github/tenbagger_analysis_market.sqlite`

示例：
- 查一只股票近1年不复权收盘：
`SELECT date, close FROM kline_daily WHERE code='605305' AND fqt=0 ORDER BY date DESC LIMIT 252;`

- 查2022-01-04（某股的第一交易日）总收益因子可用的前复权收盘：
`SELECT close FROM kline_daily WHERE code='605305' AND fqt=1 AND date='2022-01-04';`

## 5) 注意事项
- 数据源为公开接口（东方财富）；存在限流/失败重试，首次全量下载需要很长时间。
- SQLite不适合放进GitHub仓库（文件太大），建议只在本地存放；仓库里只放脚本与生成的CSV结果。

