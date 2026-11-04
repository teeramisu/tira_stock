# JoinQuant 精选策略整合 TODO

## 当前状态

- 已分析 `/home/tira/Downloads/DC42-2022年度精选策略` 下 689 个策略。
- 已生成评分结果：
  - `/home/tira/Downloads/DC42-2022年度精选策略/_analysis/strategy_llm_value_scores.csv`
  - `/home/tira/Downloads/DC42-2022年度精选策略/_analysis/strategy_llm_value_scores_summary.json`
- 已把原始策略源码复制进 `stock_backtest`：
  - `data/joinquant_strategies/`
- 已生成策略库元数据：
  - `data/joinquant_strategy_library.json`
- 已初步增加后端策略库服务：
  - `backend/services/strategy_library.py`
- 已初步增加 API：
  - `GET /api/strategy-library`
  - `GET /api/strategy-library/facets`
  - `GET /api/strategy-library/{strategy_id}`
- 已增加基础测试草稿：
  - `backend/tests/test_strategy_library.py`
- 已完成第一批可回测适配：
  - `jq_bbi_momentum_timing`：来自 `jq-0001`，宽基 ETF BBI 动量择时。
  - `jq_rsrs_timing`：来自 `jq-0004`，ETF RSRS 择时。
  - `jq_rsi_gap_reversal`：来自 `jq-0036`，RSI 衍生 ETF 择时。
- 已在前端单页增加策略库筛选、详情、源码查看和已适配策略回测入口。

## 明天优先做

1. 跑测试并修正问题
   - [x] 运行 `python backend/tests/test_strategy_library.py`
   - [x] 运行 `python backend/tests/test_engine.py`
   - [x] 通过 TestClient 检查 `/api/strategy-library?limit=10`

2. 检查新增 API 是否能正常加载大 JSON
   - [x] 确认本地路径 `data/joinquant_strategy_library.json` 可访问。
   - [x] 列表接口只返回摘要；源码读取放到详情接口 `include_source=true`。

3. 给前端增加策略库页面或区块
   - [x] 展示排名、总分、标签、简介、已适配状态。
   - [x] 支持关键词搜索、按集合筛选、按标签筛选、最低分过滤。
   - [x] 点击策略后展示评分拆分和原始源码。

4. 决定哪些策略要真正适配为可回测策略
   - [x] 优先挑 Top 20 中适合当前 OHLCV 引擎的策略。
   - [x] ETF/指数轮动、均线、RSRS/LLT、RSI、MACD 类优先。
   - 小市值、基本面、多因子、打板、集合竞价、期货/期权策略先只做资料库，不直接接入回测。

5. 设计“可运行适配器”格式
   - 当前项目自定义策略要求 `signal(df) -> 0/1`。
   - 为每个已适配策略增加：
     - `id`
     - `name`
     - `source_strategy_id`
     - `description`
     - `score`
     - `params`
     - `signal(df, **params)`
   - [x] 已新建 `backend/backtest/imported_strategies.py`。

## 推荐优先适配策略

1. `jq-0001`：宽基 ETF 轮动
   - 分数：75.2
   - 原因：适用度高，ETF/指数类更容易用当前行情数据复现。

2. `jq-0004`：ETF 动量轮动 + RSRS 择时
   - 分数：68.4
   - 原因：适合抽象为指数/ETF 择时信号。

3. `jq-0007`：行业 ETF 轮动 + 择时
   - 分数：66.6
   - 原因：多标的轮动价值高，但需要先扩展当前单标的回测引擎。

4. `jq-0008`：动量 ETF 轮动 RSRS
   - 分数：65.8
   - 原因：和第 2 类同源，可合并为一个 RSRS/动量轮动框架。

5. `2022年度精选策略/87.RSI衍生指标择时轮动A股ETF.txt`
   - 分数：58.3
   - 原因：RSI 类与当前内置 RSI 策略接近，迁移成本较低。

## 暂不建议直接适配的类型

- 小市值/微盘策略：依赖股票池、基本面过滤、ST/停牌/涨跌停过滤，当前 yfinance 数据不足。
- 打板/集合竞价/盘口/Tick 策略：当前引擎只有日线 OHLCV，无法真实模拟。
- 期货/期权/套利策略：资产类型、保证金、合约换月和交易规则都需要单独引擎。
- 机器学习策略：需要训练数据、特征工程、样本外验证和防过拟合流程。
- 聚宽 API 深度依赖策略：需要替换 `jqdata`、`get_fundamentals`、`get_price`、`run_daily` 等接口。

## 后续分阶段计划

### 第一阶段：策略资料库

- 完成后端 API 测试。
- 完成前端展示和筛选。
- 保持所有 689 个策略都能查到源码、评分和介绍。
- 本阶段已基本完成；部署环境路径还需实际容器/服务器验证。

### 第二阶段：可运行策略适配

- 已先适配 3 个 ETF/技术指标类策略。
- 已给适配后的策略加单元测试。
- 已在 `/api/strategies` 中标记来源评分和原策略链接。

### 第三阶段：多标的回测

- 当前回测引擎是单标的。
- ETF 轮动、行业轮动、多策略组合需要支持多标的价格矩阵。
- 需要新增组合净值、调仓记录、权重曲线、基准对比。
- 详细分层设计和实施清单见 `docs/FULL_MARKET_BACKTEST_TODO.md`。

### 第四阶段：A 股基础数据增强

- 增加 ST、停牌、涨跌停、上市天数、财务因子等数据源。
- 再考虑适配小市值、红利、基本面、多因子策略。

## 验收标准

- 策略库 API 可以稳定返回 689 个策略。
- 前端可以查看 Top 排名、评分拆分、简介、风险提示和源码。
- 至少 3 个高分策略被改写成当前引擎可运行的策略。
- 每个可运行策略都有单元测试和示例回测。
- 文档明确区分“已导入源码”和“已适配可回测”。
