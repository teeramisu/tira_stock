# 美股股票回测系统

前后端分离的美股回测系统：**后端 FastAPI 只提供 REST API**，**前端单页**负责参数配置、发起回测、展示净值曲线与交易记录。

## 功能

- **市场与数据**：支持 **美股、A股、港股**。通过 yfinance 拉取 OHLCV，**支持最长 10 年区间**（6mo / 1y / 2y / 5y / 10y 或自定义起止日期）。代码格式：美股 AAPL；A股 600519（沪）/000001（深）；港股 0700/9988。
- **多策略回测**：
  - 双均线（SMA）金叉/死叉
  - 双均线（EMA）金叉/死叉
  - RSI 超卖超买（可设周期、超卖/超买阈值）
  - MACD 金叉死叉（可设快/慢/信号线周期）
  - 布林带（触及下轨买、上轨卖）
  - 买入持有（基准）
- **指标**：总收益、年化收益、夏普比率、最大回撤、波动率、交易次数、胜率。
- **ETF 评估**：多只 ETF/股票同一区间买入持有对比，总收益/年化/波动/夏普/最大回撤及归一化净值曲线（起点=100）。
- **精选策略库**：已导入 JoinQuant 年度精选策略源码、100 分综合评分、适用度/特殊性等分项、标签和策略介绍，便于筛选后逐个适配到本项目回测引擎。
- **前端**：单页表单（策略与参数选择）+ 净值曲线图（Chart.js）+ 交易明细表 + ETF 对比评估。

## 项目结构

```
stock_backtest/
├── backend/
│   ├── app/
│   │   └── main.py          # FastAPI 入口与 /api 路由
│   ├── backtest/
│   │   ├── engine.py        # 回测引擎（可单测验证）
│   │   └── __init__.py
│   ├── services/
│   │   ├── data.py          # 拉取行情（yfinance）
│   │   └── __init__.py
│   └── tests/
│       └── test_engine.py    # 引擎单元测试
├── frontend/
│   └── index.html           # 单页前端
├── requirements.txt
├── run.sh
└── README.md
```

## 安装与运行

### 1. 安装依赖

推荐使用独立 Conda 环境，避免和系统 Python 或其他项目冲突：

```bash
cd stock_backtest
conda create -n stock-backtest python=3.11 -y
conda activate stock-backtest
pip install -r requirements.txt
```

备选：使用 Python venv：

```bash
cd stock_backtest
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 方式一
conda activate stock-backtest
./run.sh

# 方式二
conda activate stock-backtest
uvicorn main:app --app-dir backend/app --host 0.0.0.0 --port 8000
```

### 3. 使用

- 浏览器打开：**http://localhost:8000/** → 填写股票代码、日期/周期、均线周期后点击「运行回测」。
- API 文档：**http://localhost:8000/docs**

## API 说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/health | 健康检查 |
| GET | /api/strategies | 可用策略列表及参数名 |
| GET | /api/strategy-library | 查询已导入的 JoinQuant 策略库，参数：q, collection, tag, min_score, limit, offset |
| GET | /api/strategy-library/facets | 查询策略库集合、标签和评分口径 |
| GET | /api/strategy-library/{strategy_id} | 查询单个策略详情；可加 include_source=true 返回原始源码 |
| GET | /api/history | 查询历史 OHLCV，参数：symbol, start, end 或 period（含 10y） |
| POST | /api/backtest | 执行回测，Body：symbol, start/end/period(6mo~10y), strategy, initial_cash 及策略参数 |
| POST | /api/backtest/custom | **自定义 Python 策略**：Body 含 symbol, period, initial_cash, **code**。code 中须定义 `signal(df)`，返回与 df 同长的 0/1 序列（1=持多 0=空仓）。仅允许 pandas/numpy 及安全内置，执行限时 15 秒。 |
| POST | /api/etf/compare | **ETF 评估**：Body 含 symbols（2～20 个代码）、period、market。返回各标的买入持有指标及归一化净值曲线。 |
| POST | /api/backtest/portfolio | **全市场组合回测**（本地 DuckDB 行情库）：`momentum_top_k` / `etf_rotation_top1`，返回净值、基准、成本摘要、`run_metadata` |
| GET | /api/market/data-status | 本地行情库状态（不触发下载） |
| GET | /api/market/dataset-versions | 可复现数据版本列表 |
| GET | /api/market/universe | 指定交易日股票池与过滤诊断 |
| GET | /api/backtest/{run_id}/artifacts | 分页读取回测 artifact（trades、benchmark_curve、run_metadata 等） |
| POST | /api/market/refresh | 开发环境刷新本地行情（需 `ENABLE_MARKET_REFRESH_API=1`） |
| GET | /api/recommendations/top | 知识星球股票推荐榜；参数：limit、sector、listed_only |
| GET | /api/recommendations/by-sector | **按行业板块汇总**推荐；参数：limit（板块数）、per_sector（每板块股票数） |
| POST | /api/recommendations/refresh | 重新解析语料并评分；参数：use_llm、llm_top_n |

### 知识星球股票推荐挖掘

从知识星球调研帖中提取股票推荐，形成结构化榜单（推荐理由、推荐源数量、综合评分、所属板块、未来 1 周 / 1 月预期涨幅）。

```bash
# 1) 仅规则评分（秒级，离线可用）
python scripts/zsxq_data_cleaner.py

# 2) DeepSeek 大模型精评 Top120（需 .env 中配置 DEEPSEEK_API_KEY）
python scripts/zsxq_data_cleaner.py --llm --llm-top-n 120
```

处理链路：

1. **语料**：扫描 `zsxq_output/scrolled_topics.json`（约 4000 帖）及 `all_topics/topics.json`、`crawler/data/zsxq/topics_raw*.json`，按 `topic_id` 去重。
2. **标的识别**：以本地证券主数据（`data/market/market_data.duckdb` 的 `instruments`）和 `data/board_constituents.json` 股票名称为词典做最长匹配，避免错码；券商团队署名（如“天风电新”）等非股票词自动忽略。
3. **板块**：同花顺行业（`industry`）+ 概念（`concepts`），行业缺失时用概念成份股投票推断，仍无法判断则归入“未分类”。
4. **聚合**：按股票代码聚合，`source_count` 为提及该股的**不同帖子数**，`author_count` 为不同作者数，`strong_count` 为含“重点推荐/买入/首推”等强信号的帖子数。
5. **行情**：读取本地 DuckDB 近 6 个月日线，计算近 20/60 交易日涨跌、年化波动、最大回撤，不依赖 yfinance，避免限流。
6. **评分**：规则分（推荐强度 + 作者数 + 点赞 + 动量）先排序，取 Top N 交给 DeepSeek 输出 0-100 分、理由、未来 1 周 / 1 月预期涨跌幅；无 API Key 或调用失败自动回退规则分。
7. **产物**：`data/zsxq_stocks_cleaned.json`（未评分结构化数据）、`data/zsxq_stocks_scored.json`（含评分，读接口直接使用）、`data/zsxq_stocks_scored.csv`（Excel 可直接打开）。

字段说明（`/api/recommendations/top` 返回的 `stocks[]`）：

| 字段 | 说明 |
|------|------|
| stock_name / stock_code | 股票名称 / 6 位 A 股代码 |
| industry / concepts / sectors | 所属行业 / 概念题材 / 前端展示用标签集合 |
| source_count / author_count / strong_count | 提及帖子数 / 作者数 / 强推荐帖子数 |
| reasons_summary | 推荐理由摘要（原文句子，最多 3 条） |
| comprehensive_score / score_reason / score_source | 综合评分（0-100）/ 评分理由 / `llm` 或 `rule` |
| expected_weekly / expected_monthly | 未来 1 周 / 1 月预期涨跌幅（%） |
| performance | 本地行情摘要：`return_20d_pct`、`return_60d_pct`、`volatility_pct`、`max_drawdown_pct`、`last_date` 等 |
| listed | 是否可交易（无有效 A 股代码为 false，不参与板块汇总与回测） |

> 返回中的 `data_as_of` 为本地行情最后交易日；若早于当前日期，请先运行行情刷新脚本再刷新评分。
>
> DuckDB 同一时刻只允许一个进程写入：服务端运行时，请通过 `POST /api/recommendations/refresh` 刷新，或先停止服务再执行 CLI 脚本。

运行测试：

```bash
python backend/tests/test_stock_recommendation.py
```


### 全市场组合回测（本地行情库）

1. 初始化/刷新行情（示例）：

```bash
conda activate stock-backtest
python scripts/refresh_market_data.py --market cn --mode incremental --symbols 600000,000001 --limit 20
```

2. 运行组合回测 API（需已有 stable `dataset_version`）：

```bash
curl -X POST http://localhost:8000/api/backtest/portfolio \
  -H 'Content-Type: application/json' \
  -d '{"market":"cn","start":"2024-01-01","end":"2024-03-31","strategy":"momentum_top_k","max_positions":10,"lookback_days":20,"initial_cash":1000000}'
```

3. 运行测试：

```bash
python -m pytest backend/tests/test_marketdata.py backend/tests/test_portfolio_engine.py backend/tests/test_portfolio_api.py -q
```

## Discovery Loop（自进化投研：P0/P1）

评估分两级，**只有组合级结果才能作为晋升依据**：

1. **单票信号诊断**：只用来筛选/淘汰原始信号，不直接晋升。
2. **组合级正式评估**：选股 → 权重 → 调仓 → 成本 → 组合净值 → 基准比较，才是最终标准。

目标不是"跑出高收益"，而是先保证每个候选都经过可复现、抗过拟合的证伪流程。

### 模块

| 文件 | 作用 |
|---|---|
| `backend/discovery/stats.py` | Sharpe/Sortino/Calmar/MaxDD、PSR、Deflated Sharpe、CSCV-PBO、BH-FDR、Block Bootstrap |
| `backend/discovery/evaluator.py` | 单票 Purged Walk-Forward、CPCV、参数搜索、因果性/延迟/成本/容量检查 |
| `backend/discovery/portfolio.py` | **组合级**：横截面选股、TopK、权重、调仓、换手成本、等权基准、组合级 Purged WF 与闸门 |
| `backend/discovery/signals.py` | 单票因果信号（SMA/EMA/RSI/MACD/布林），仅作诊断/原语 |
| `backend/discovery/ledger.py` | SQLite 实验账本：候选、实验、DSR/PBO、失败记忆、血缘 |
| `scripts/run_discovery_cycle.py` | 一轮完整循环（默认组合级；`--mode signal` 为单票诊断） |

### 运行

```bash
# 组合级（默认）：流动性前 300 只，1 年训练 / 3 个月测试 / purge 10 / 成本 10bp
python scripts/run_discovery_cycle.py --mode portfolio --market cn \
  --start 2020-01-01 --end 2026-07-10 --universe-limit 300

# 单票信号诊断（不能作为晋升依据）
python scripts/run_discovery_cycle.py --mode signal --symbols 600519,300750,002594
```

组合级输出：组合 OOS Sharpe/CAGR/MaxDD/Calmar、等权基准、超额 CAGR、DSR、PBO、正 fold 比例、年换手、代表 Genome 与晋升结论。

### 组合级晋升闸门（默认）

| 检查 | 阈值 |
|---|---|
| 组合 OOS Sharpe | ≥ 1.0 |
| 超额 CAGR（相对等权基准） | > 0 |
| 组合最大回撤 | ≤ 20% |
| Deflated Sharpe | ≥ 0.95 |
| PBO | ≤ 0.2 |
| 正收益 fold 比例 | ≥ 60% |
| 年换手 | ≤ 30x |

单票诊断阈值（仅用于筛信号）：OOS Sharpe ≥ 0.8、DSR ≥ 0.95、PBO ≤ 0.2、正 fold ≥ 60%、MaxDD ≤ 35%。

> **数据质量状态**：2026-07-20 的复权断层已于 2026-10-26 修复（修复后横截面均值 -2.1%，无 >35% 异常）。当前 CN 数据已统一为「原始价 + 前复权价」，可用全窗口 `2020-01-01 ~ 2026-10-26` 做组合评估。

### API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/discovery/summary` | 账本总览与最佳实验 |
| GET | `/api/discovery/experiments` | 实验列表（strategy/status/min_dsr/limit） |
| GET | `/api/discovery/leaderboard` | 按 dsr/pbo/fdr 排序 |
| GET | `/api/discovery/candidates` | 候选 Genome 列表 |
| GET | `/api/discovery/candidates/{id}` | 候选详情与血缘 |
| GET | `/api/discovery/lessons` | 失败记忆 |

账本默认位置：`data/discovery/ledger.db`（SQLite，与 DuckDB 行情库解耦，行情刷新期间仍可读写）。

### 测试

```bash
python -m pytest backend/tests/test_discovery.py -q
```

## 行情数据约定（A股）

- `open/high/low/close`：**未复权**真实成交价，用于展示、执行与原始收益核对
- `adj_open/.../adj_close`：**前复权**价，统一以最新交易日为基准，用于回测与因子计算
- `adj_factor = adj_close / close`：复权比例，便于审计和还原
- 默认 `read_daily_bars(adjusted=True)` 使用复权价；`adjusted=False` 使用原始价
- 全历史修复脚本（统一复权基准）：

```bash
python scripts/repair_cn_daily_prices.py --workers 12
```

- **日常增量刷新**（保留原始价，并按上一次复权基准对齐，避免断层）：

```bash
python scripts/refresh_cn_incremental.py --workers 12 --end 2026-11-02
```

> 不要再用旧的 `refresh_market_data.py --market cn --mode incremental` 做 qfq 增量：
> 它每次都会以最新日期重新基准化，容易在拼接处产生复权断层。

- **美股增量刷新**（流动性池 6,308 只，raw+qfq）：

```bash
python scripts/refresh_us_incremental.py --workers 12 --end 2026-11-02
```

> 美股存量历史中 `adjustment='none'` 的部分是未复权数据；增量新行会以最后交易日为基准做前复权并对齐，
> 因此拼接处连续，但更早的历史仍建议后续做一次全历史 raw+qfq 修复。

## 宏观数据（NBS / PBOC）

把国家统计局与人民银行公布的宏观数据抓取、解析、标准化，输出长表/宽表。

### 数据源与适配器

| 适配器 | 覆盖内容 | 方式 |
|---|---|---|
| `macro/sources/official_html.py` | NBS PMI / 固定资产投资；PBOC 社融增量 | 网页表格 `pandas.read_html`、PBOC Excel/HTML 附件 |
| `macro/sources/akshare_macro.py` | CPI/PPI/PMI/GDP/社零/工业增加值/M0-M2/新增贷款/LPR/存准率 | akshare（东方财富镜像，标注 NBS/PBOC 来源） |

网页里的表格会原样落到 `macro_raw_tables`（URL、表序号、列、行 JSON），再做字段归一化；这样即使页面结构变化，原始数据也可追溯。

### 指标与存储

- 指标目录：**84 个**，覆盖中国 / 美国 / 日本
  - 中国（51 个看板系列）：NBS 价格/景气/增长/消费/投资 + PBOC 货币/社融/信贷/利率/存准 + 中美国债收益率 + 工业增加值/固投/用电量/70城房价/央行资产负债表/SHIBOR/两融/全A PE-PB 分位/北向资金 + M1-M2 剪刀差/实际利率
  - 美国：CPI/核心 CPI/核心 PCE、失业率、非农、GDP、ISM PMI、零售、工业产出、消费者信心、贸易帐、联邦基金利率、2Y/10Y/30Y 国债与期限利差、世界银行年度补充
  - 日本：CPI/核心 CPI、失业率、央行政策利率、领先指标、世界银行年度 GDP/通胀/就业/制造业/经常账户
- 长表：`macro_observations`（`series_id, period, value, unit, source, provider, url, raw_label`），当前约 **93,600 条**观测
- 分省表：`macro_region_observations`（指标 × 31 省 × 年度），约 2,000 条
- 宽表：`macro_wide_monthly`、`macro_wide_quarterly`
- 原始网页表：`macro_raw_tables`
- 刷新日志：`macro_fetch_log`
- 数据库：`data/macro/macro.duckdb`（与行情库解耦）

### 刷新

```bash
# 全部来源
python scripts/refresh_macro_data.py

# 只抓官方网页
python scripts/refresh_macro_data.py --sources nbs,pbc

# 只跑 akshare 序列
python scripts/refresh_macro_data.py --sources akshare
```

输出：

```
data/macro/exports/macro_long.csv
data/macro/exports/macro_wide_monthly.csv
data/macro/exports/macro_wide_quarterly.csv
data/macro/exports/macro_latest.csv
data/macro/exports/macro_data.xlsx
data/macro/exports/macro_provinces_long.csv
data/macro/exports/macro_provinces_latest.csv
```

### API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/macro/status` | 指标/观测/原始表数量、最近刷新 |
| GET | `/api/macro/indicators` | 指标目录与覆盖区间 |
| GET | `/api/macro/series/{series_id}` | 单指标序列（start/end） |
| GET | `/api/macro/latest` | 各指标最新值 |
| GET | `/api/macro/dashboard?years=3&countries=cn,us,jp` | **三国决策看板**：最新值/变化/同比/近 N 年序列 |
| GET | `/api/macro/wide?frequency=monthly` | period × indicator 宽表 |
| GET | `/api/macro/regions/indicators` | 分省指标目录 |
| GET | `/api/macro/regions/{indicator_id}` | 分省最新值、同比变化、排名与排名变化 |
| POST | `/api/macro/refresh` | 同步刷新（sources/export） |

### 前端「宏观罗盘」

顶部导航新增 **宏观罗盘** 页面：

- **三国核心指标对比**：GDP、CPI、失业率、政策/短端利率、10Y 国债、制造业景气、社融
- **KPI 卡片**：按国家精选关键指标，显示最新值、较前值、同比、日期与来源
- **趋势图**：3 / 5 / 10 年切换；不同单位自动左右双轴；可切换“归一化（起点=100）”
- **决策解读**：规则化输出增长/通胀/流动性/地产/估值/情绪六维评分、支撑因素、主要风险、观察指标，以及**未来 1–2 季度**的基准判断（区间震荡/震荡偏强/偏弱；不构成投资建议）
- **省级变化对比**：31 个省份的 GDP 总量/实际增速、社零、居民收入、财政收入、房地产投资、CPI 等；支持“最新总量 / 同比变化”切换、排名变化与明细表
- **完整数据表**：全部指标的最新值、环比、同比、频率与来源
- 图表使用本地 `frontend/js/echarts.min.js`，不依赖外部 CDN

### 注意

- 统计局 `data.stats.gov.cn` API 在当前网络下返回 403，因此官方部分走**最新发布 HTML 表格**；akshare 作为稳定序列兜底。
- `nbs_fai_yoy` 是**累计同比**（如 2026 年 1—9 月），不是单月同比。
- `nbs_industrial_production_yoy` 的 akshare 序列更新较慢（当前到 2025-08），其余核心月度指标已到 2026-09/10。
- 存准率是事件型序列，只在调整时新增。
- 美国部分日历型指标（失业率/ISM/非农等）最新实际值约到 2025-09；国债收益率、CPI 同比更新到 2026-10。日本 10Y 国债暂无稳定免费源，看板中留空；日本 GDP/制造业为世界银行年度数据。
- 看板顶部“数据截至”是所有指标的最大日期，**具体指标以表格行内日期为准**。
- 分省数据来自国家统计局新版 API（`data.stats.gov.cn/dg/...`），部分指标或省份尚未发布时，省级图表/表格只展示有数据的地区；GDP 总量与实际增速覆盖 31 省。

## 回测逻辑（可验证）

- 使用**前一日**的均线信号，**当日收盘**成交，避免未来数据。
- 仅做多：金叉全仓买入，死叉全仓卖出。
- 可选手续费率 `commission_rate`（按成交金额比例）。
- 导入策略库中的原始 JoinQuant 策略多数依赖 `jqdata`、基本面、行业、期货、分钟数据等外部能力；当前已作为可检索策略资产整合，真正进入 `/api/backtest` 前需要改写为本项目约定的 `signal(df)`。
- 单元测试见 `backend/tests/test_engine.py`，可直接运行验证：
  ```bash
  python backend/tests/test_engine.py
  ```

## 注意事项

- 数据来源为 yfinance（Yahoo Finance），仅供学习与回测研究。
- 回测结果不代表实盘表现，实盘需考虑滑点、流动性、交易成本等。
