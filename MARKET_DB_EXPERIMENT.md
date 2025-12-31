# 本地行情库实验说明（SQLite）

## 1. 实验目的
为“十倍股选股/回测”提供**可重复、可离线、可快速检索**的数据底座，避免每次运行都对公开接口执行“全市场拉取”，从而减少耗时、限流、失败重试对实验的干扰。

## 2. 数据来源与口径
- 股票池：东方财富 A 股列表接口（剔除 `*ST/ST`），字段包含 `股票代码/名称/行业/市场标识(secid)`。
- 日K数据：东方财富 `push2his` 日线接口（`klt=101`）。
  - `fqt=0`：不复权（价格用于“市值=不复权收盘×总股本”等计算更直观）
  - `fqt=1`：前复权（近似包含送转分红影响，适合做收益/回测）
- 时间范围：近5年（本次落库结果为 `2021-01-04` 至 `2025-12-31`，以接口可得为准）。

## 3. 数据库文件与位置
- SQLite 文件：`Documents/github/a_share_tenbagger_analysis_2019_2025/data/tenbagger_analysis_market.sqlite`
- 注意：该文件体积大（本次约 1.3GB），不应提交到Git仓库（已在 `.gitignore` 忽略）。

## 4. 表结构
### 4.1 `stocks`
每只股票一行：
- `code`：股票代码（主键）
- `secid`：东方财富市场标识（如 `1.600000` / `0.000001`）
- `name`：股票名称
- `industry`：行业（东方财富字段）
- `updated_at`：更新时间

### 4.2 `kline_daily`
每个交易日一行，主键：`(code, fqt, date)`：
- `code`：股票代码
- `fqt`：复权类型（0/1）
- `date`：交易日（`YYYY-MM-DD`）
- `open/close/high/low`：开收高低
- `volume`：成交量
- `amount`：成交额

## 5. 生成过程（本次实验执行记录）
使用脚本：`Documents/github/a_share_tenbagger_analysis_2019_2025/local_market_db.py`
（入口脚本；实现位于 `tenbagger/tools/local_market_db.py`）

执行步骤：
1) 初始化：创建SQLite与表
2) 更新股票池：拉取A股全量、剔除ST后写入 `stocks`
3) 下载近5年日K：对每只股票写入 `kline_daily`（同时写 `fqt=0` 与 `fqt=1`）

生成后可用以下SQL快速验证：
```sql
SELECT count(*) FROM stocks;
SELECT fqt, count(*) FROM kline_daily GROUP BY fqt;
SELECT min(date), max(date) FROM kline_daily;
```

## 6. 局限性与注意事项
- 公开接口可能存在：短暂不可用、限流、返回字段变化；脚本做了重试但不能保证100%成功率。
- 本库仅包含日K与股票池信息；如果你的策略需要**总股本序列/财务报表/主营构成/客户集中度**等，仍需另建表或单独抓取落库。
- 前复权价格是“收益回测友好”的近似，但不同数据源的复权因子可能略有差异；跨源对比时要注意口径一致。

## 7. 如何更新（增量）
脚本会按每只股票、每个 `fqt` 的 `MAX(date)` 自动从下一天开始续拉：
```bash
python3 Documents/github/a_share_tenbagger_analysis_2019_2025/local_market_db.py \
  --db Documents/github/a_share_tenbagger_analysis_2019_2025/data/tenbagger_analysis_market.sqlite \
  download-kline --beg 2021-01-01 --with-fqt1
```

## 8. 如何在策略里使用
建议把“取K线/取股票池”做成一个数据层：
- 优先查SQLite（快、稳定、可重复）
- 只有SQLite缺失数据时再走网络补全（并写回SQLite）
