# 全市场组合回测架构 TODO

## 目标

把当前“单标的 OHLCV 择时回测”升级为“可对全市场股票池执行策略的组合回测系统”。

当前单标的能力继续保留，用于 ETF、指数、个股择时。新增组合回测能力用于小市值、多因子、行业轮动、ETF 轮动、红利、动量、价值等策略。

核心原则：

- 回测时只读本地行情库，不实时逐只请求外部数据。
- 策略不能直接访问未来数据，所有因子和选股结果必须按交易日切片。
- 数据、股票池、策略、撮合、API、前端分层清晰，避免后续返工。
- 每一层先用小型 fixture 数据做完整 UT，再接真实全市场数据。

## 当前结论

- 当前 `backend/backtest/engine.py` 是单标的引擎，输入单个 `DataFrame`，输出单条净值曲线。
- 当前 `backend/services/data.py` 基于 yfinance 逐标的拉取，适合临时单票查询，不适合全市场。
- 当前 `.ohlcv_cache` 是 pickle 临时缓存，不具备增量刷新、股票池过滤、批量查询和数据质量校验能力。
- 全市场回测需要新增本地市场数据仓库和组合引擎，不应继续扩展单标的 `signal(df) -> 0/1` 接口。

## 架构决策

### 本地行情库

优先使用 DuckDB 单文件作为本地行情仓库：

- 适合千万级日线数据和较大规模分钟线数据的本地分析读取。
- 不需要额外数据库服务，开发和部署简单。
- 支持 SQL、列式扫描、Parquet 导入导出，后续迁移到 PostgreSQL/ClickHouse 也清晰。

保留现有 PostgreSQL/SQLAlchemy 作为用户、回测记录、权限等业务数据库。行情数据和用户业务数据不要混在一套表里。

默认文件路径：

- `data/market/market_data.duckdb`
- `data/market/parquet/` 作为可选冷备份和批量导入目录。

日内数据扩展约束：

- 日线、分钟线、Tick 不混表。日线用 `trade_date`，分钟线/Tick 用带时区语义的 `bar_time` / `tick_time`。
- 所有行情读取必须经过 `DataPortal`，策略和引擎不能直接依赖具体表名。
- 数据表必须保留 `frequency` 或按频率分表，避免未来 1m、5m、15m、30m、60m 混淆。
- A 股交易时段、午休、集合竞价、半日市等信息放进独立 `trading_sessions` 表，不写死在策略里。
- 分钟线优先落 DuckDB；Tick 数据量更大，后续可按月 Parquet 分区存储，DuckDB 只做查询入口。

### 多市场行情仓库

行情仓库从第一版就按多市场设计，不能只假设 A 股。第一阶段真实数据刷新先做 A 股小样本 + 美股小样本，验证同一 schema、DataPortal、CostModel、Calendar 可以支持 `cn/us`。

- `market` 是一级维度，例如 `cn`、`us`、`hk`。
- `exchange` 是市场内交易所维度，例如 `SSE`、`SZSE`、`BSE`、`NYSE`、`NASDAQ`、`AMEX`。
- 同一个 `symbol` 在不同市场可能冲突，内部主键和查询都必须使用 `market + symbol` 或全局 `instrument_id`。
- 交易日历、交易时段、交易成本、整数手规则、涨跌停规则都按 `market + exchange` 配置。
- 数据源适配器按市场拆分：A 股优先 AkShare，后续美股可接 yfinance/Stooq/Polygon/FMP/Databento 等。
- 回测请求必须显式传 `market`，不能根据代码猜市场。
- `dataset_version` 必须包含市场覆盖范围，例如只覆盖 `cn`，或同时覆盖 `cn/us`。

### A 股全市场范围

默认 A 股全市场股票池：

- 明确包含沪市、深市、创业板、科创板、北交所 A 股。
- 排除新三板挂牌标的。
- 默认排除退市整理、无法交易、缺失价格的标的。

已确认的产品口径：

- 默认排除 ST / *ST，并允许用户在高级参数中打开。
- 新股默认上市满 `60` 个交易日后进入股票池。
- 第一阶段真实数据范围：A 股小样本 + 美股小样本，先验证多市场 schema 和读取链路。
- 日线价格同时存原始价格和复权价格；策略默认使用前复权，原始价格和复权因子用于审计、敏感性和口径切换。

### 数据源

首选新增 AkShare A 股数据源，原因：

- 支持 A 股股票列表、历史日线、复权、指数和部分基础信息。
- 不需要 API Key，适合先做本地全市场 MVP。

后续可加 Tushare Pro 作为增强源：

- 财务因子、复权因子、停复牌、涨跌停、ST 状态等数据更规范。
- 需要 token 和积分，适合第二阶段接入。

### Dataset Versioning

全市场回测必须可复现。任何一次回测都必须绑定一个明确的数据快照版本，不能隐式使用“当前最新数据”。

约束：

- 每次全量初始化或增量刷新完成后生成 `dataset_version`。
- `dataset_version` 至少包含：行情源、刷新时间、覆盖日期、覆盖标的数量、schema 版本、质量检查摘要。
- 回测请求可以指定 `dataset_version`；未指定时使用最新 stable 版本，并在响应中返回实际版本。
- 回测记录必须保存 `dataset_version`，否则同一参数未来无法复现同一结果。
- 数据刷新失败或质量检查未通过的版本不能标记为 stable。
- 不允许回测过程中一边读取旧数据一边混入刷新中的新数据。

### DataPortal Contract

`DataPortal` 是策略唯一数据入口，也是防未来函数的核心边界。

必须支持的接口：

```python
data.history(symbols, fields, end_time, bars, frequency)
data.current(symbols, fields, as_of)
data.universe(as_of, config)
data.calendar(market, start, end, frequency)
data.sessions(market, exchange, trade_date)
```

硬约束：

- 策略不能直接读 DuckDB、Parquet、文件、网络或全局缓存。
- `DataPortal` 必须绑定 `dataset_version` 和当前仿真时间。
- `history(...)` 必须显式声明 `frequency`，日线默认 `1d`，日内必须传 `1m`、`5m` 等。
- 对每个字段执行 point-in-time 可见性检查，不能返回当前时点尚不可知的数据。
- 缺失数据必须以 `NaN` 或明确错误返回，不能静默填 0。
- 所有 DataPortal 行为必须可用 fixture 手工复核。

### Broker / Order Model

组合引擎不要把撮合、成本、现金和持仓逻辑揉成一个大函数，应拆成清晰子模块：

- `PortfolioEngine`：时间推进、调仓日判断、调用策略、汇总结果。
- `BrokerSimulator`：订单接收、撮合、现金、持仓、失败订单。
- `CostModel`：佣金、印花税、过户费、滑点、冲击成本。
- `OrderValidator`：涨跌停、停牌、100 股整数手、现金不足、权重上限。
- `MetricsCalculator`：净值、回撤、换手、成本、基准比较。

订单和成交模型必须显式：

- `Order`：目标标的、方向、目标股数/金额/权重、创建时间、期望执行时间。
- `Fill`：成交时间、成交价、成交数量、成本拆分、失败原因。
- `Position`：持仓数量、成本、最新市值、权重。
- 失败订单不是异常退出，而是进入 `rejected_orders` / `trade_logs`，用于前端解释。

### Point-in-Time Data

A 股全市场策略最大的风险是数据未来函数。除价格以外，任何会随时间修正或公布的数据都必须区分“归属期”和“可见时间”。

字段设计要求：

- 价格数据使用 `trade_date` / `bar_time`。
- 财务数据必须同时有 `period_end_date`、`announcement_date`、`effective_date`。
- ST 状态、停复牌、涨跌停、指数成分、行业分类必须有 `effective_date`。
- 股票上市、退市、名称变更必须按当时可见状态查询。
- 股票池不能使用今天才知道的未来成分、未来 ST 状态或未来退市信息。

验收标准：

- 任一策略在日期 `T` 查询到的数据，必须能解释“为什么在 `T` 已经可见”。
- 缺少 `known_date` / `effective_date` 的数据源不能进入策略，只能作为离线分析参考。
- 财务/基本面因子未完成 point-in-time 建模前，不适配红利、价值、小市值增强、多因子策略。

### Backtest Artifacts

回测结果不能只塞一个大 JSON。MVP 可以同步返回摘要，但架构上要支持结果 artifact。

建议拆分：

- `backtest_runs`：请求参数、策略、数据版本、状态、摘要指标。
- `backtest_artifacts`：大结果文件索引，例如净值曲线、每日持仓、订单、成交、调仓日志。
- `backtest_snapshots`：必要时保存可复现快照，包含代码版本、数据版本、参数 hash。

约束：

- API 默认返回摘要、净值曲线和少量交易记录。
- 大型明细，如每日全持仓和全订单，必须分页或通过 artifact 读取。
- 每次回测生成 `request_hash`，用于判断相同数据版本和相同参数是否可复用结果。

### Engineering Standards

本项目后续会承载数据刷新、回测计算、策略迁移和前端展示，必须把工程规范作为准入条件，而不是上线前补丁。

代码组织：

- 新功能按领域分包：`marketdata`、`backtest`、`services`、`app`、`database`，禁止跨层随意 import。
- 业务核心逻辑放纯 Python 模块，FastAPI handler 只做参数校验、调用服务、组装响应。
- 数据源适配器只负责拉取和标准化外部数据，不写回测逻辑。
- 回测引擎不能直接依赖 AkShare、yfinance、FastAPI、前端字段名。
- 公共 dataclass / Pydantic model 要稳定命名，避免 dict 到处传。

类型和接口：

- 新增核心模块必须使用 type hints。
- 公共函数参数和返回值必须显式标注。
- 外部输入使用 Pydantic model 校验，内部核心模型优先 dataclass。
- 策略协议、DataPortal、Broker、CostModel 要有明确接口，不依赖隐式字段。
- 浮点金额统一用 `float` 计算、输出按展示层格式化；如后续做实盘再改 Decimal。

配置管理：

- 所有路径、数据源开关、默认交易成本、超时、批量大小都必须走配置，不写死在业务函数中。
- 本地默认配置可用，生产配置必须能通过环境变量覆盖。
- API 请求参数优先于默认配置；回测响应必须返回实际使用的配置快照。
- 依赖新增必须写入 `requirements.txt`，并在文档中说明用途。

错误处理：

- 用户可修复错误返回 400，例如参数非法、数据版本不存在、股票池为空。
- 系统不可用返回 503，例如行情库未初始化、数据刷新中、外部源不可用。
- 代码 bug 返回 500，但日志必须保留 traceback。
- 核心层抛领域异常，例如 `MarketDataNotFound`、`UnstableDatasetVersion`、`InvalidOrder`，API 层统一转换。
- 不允许宽泛 `except Exception: pass` 吞掉错误；必须记录原因或返回诊断。

可观测性：

- 数据刷新必须记录开始、结束、耗时、成功/失败标的数、失败原因。
- 回测必须记录 `request_hash`、`dataset_version`、策略、日期范围、股票池大小、耗时。
- 关键路径日志使用结构化字段，避免只写自然语言。
- API 需要返回可解释 diagnostics，不把内部 traceback 暴露给前端。
- 性能测试输出应保存到文档或 artifact，便于比较优化前后差异。

测试规范：

- 每个核心模块必须有单元测试，使用 fixture 数据，不依赖网络。
- 数据源适配器需要 mock 外部响应，不能在 UT 中访问真实 AkShare/yfinance。
- API 测试使用 TestClient 和临时 DuckDB 文件。
- 引擎测试必须有可手工复核的小样本，覆盖现金、成本、停牌、涨跌停、调仓。
- 回归测试必须保证原有单标的 `/api/backtest` 行为不变。
- 第一阶段 CI 采用 `pytest + ruff`；`pytest` 跑快速 UT，`ruff` 执行基础 lint、导入顺序和风格检查。
- mypy 暂不作为第一阶段硬门禁，但核心接口仍必须写 type hints，后续再提升为严格类型检查。
- 性能测试单独标记，不和快速 UT 混跑。

迁移和兼容：

- DuckDB schema 需要 `schema_version`，变更必须有迁移脚本或重建说明。
- 数据库变更不能破坏已有 `backtest_runs` 业务表。
- API 新增字段保持向后兼容，删除/重命名字段必须先做弃用期。
- 文档、示例请求、测试 fixture 要和 schema 同步更新。

代码质量：

- 新增代码默认 ASCII，除中文文档和用户可见文案外不引入特殊字符。
- Python 代码遵循 Google Python Style Guide：模块/类/函数 docstring、命名、导入顺序、异常说明和测试命名都按该风格执行。
- 公共函数 docstring 使用 Google style，至少说明 `Args`、`Returns`、`Raises`。
- 测试命名采用 `test_<unit>_<behavior>_<expected_result>`，避免只写 `test_ok` 这类低信息名称。
- 函数保持单一职责，核心函数过长时拆分。
- 避免全局可变状态；缓存必须有清晰生命周期和失效策略。
- 不做无关重构，不把单标的引擎强行改造成组合引擎。
- 所有 TODO 必须指向明确后续任务，不能留下“以后优化”这类模糊注释。

### Quant Engineering Practices

以下实践来自成熟量化系统的常见工程要求，应尽量在 MVP 阶段就打好边界，避免回测结果“看起来能跑，但无法信任”。

测试金字塔：

- Unit tests 覆盖纯函数和核心模型：DataPortal、Universe、CostModel、OrderValidator、Broker、Metrics。
- Component tests 覆盖模块组合：fixture 行情 + 策略 + Broker + 引擎。
- API tests 覆盖 FastAPI 请求响应和错误映射。
- Golden tests 保存小型确定性回测结果，任何引擎改动都必须解释 golden diff。
- Property-based tests 用随机但受约束的数据验证不变量，例如现金不为负、权重不超过 100%、没有未来数据访问。
- Regression tests 固定历史 bug，尤其是涨跌停、停牌、最低佣金、复权和调仓日边界。

核心不变量：

- 任何时点：`cash + positions_market_value` 必须等于组合净值，误差在容忍范围内。
- 无杠杆策略：目标权重和实际权重不能超过配置上限。
- 不允许负股数，不允许非整数手买入 A 股。
- 成交后现金不能因漏算成本变成负数。
- 停牌、涨停买入、跌停卖出必须产生 rejected order，而不是静默成交。
- `DataPortal` 返回的数据时间戳不得晚于当前仿真可见时间。
- 同一 `dataset_version + request_hash + code_version` 必须得到确定性结果。

实验登记：

- 每个策略实验都应记录 `experiment_id`、策略版本、参数、数据版本、代码版本、运行时间和作者/来源。
- 参数搜索不能只保留最优结果，必须保留搜索空间和所有候选摘要，降低过拟合风险。
- 策略上线到“可适配”状态前，必须写清楚样本内/样本外区间和主要风险。
- JoinQuant 迁移策略必须记录原策略与本项目复现版本的差异。

风险和约束层：

- MVP 先实现回测前/下单前的硬约束：最大持仓数、单票最大权重、总仓位上限、最小成交额。
- 后续可加入行业暴露、风格暴露、流动性约束和成交额占比约束。
- 风险约束失败应进入 diagnostics，前端可解释为什么某只股票被裁掉或降权。
- 风险层不能由策略绕过，必须由引擎或 Broker 统一执行。

数据血缘：

- 每个字段需要能追溯到数据源、刷新任务、数据版本和质量检查状态。
- 派生因子要记录依赖字段、计算窗口、计算时间和 point-in-time 规则。
- 对外展示的指标要能从 artifact 追溯到输入参数和原始行情版本。
- 数据修正不能覆盖历史版本，应生成新的 `dataset_version`。

偏差检查：

- 默认检查 survivorship bias：股票池必须来自当时可见的上市/退市状态。
- 默认检查 lookahead bias：财务、ST、行业、成分股按 `effective_date` 可见。
- 默认检查 execution bias：涨跌停、停牌、新股一字板不能按理想价格成交。
- 默认检查 cost bias：所有回测默认带成本，允许用户显式设为 0，但响应中必须标记。
- 默认检查 overfitting risk：策略参数过多或样本过短时在 diagnostics 中提示。

### Research Platform Extensions

全市场组合回测后续会自然演进成量化研究平台。MVP 可以先只实现动量 Top K，但架构需要预留因子生产、批量实验、样本外验证、组合构建和容量评估。

因子层：

- 新增 `FactorPipeline`，因子计算与策略选股分离。
- 因子定义必须包含 `factor_name`、`version`、`lookback_window`、`dependencies`、`frequency`、`point_in_time_rule`。
- 因子计算接口：

```python
pipeline.compute(date, universe, data) -> factor_frame
```

- `factor_frame` 采用长表或宽表均可，但必须表达 `date x symbol x factor`。
- 因子结果可以落本地仓库复用，避免每次回测重复计算。
- 派生因子必须记录依赖字段、数据版本、计算代码版本和参数 hash。
- 支持标准预处理接口：winsorize、standardize、neutralize、missing_value_policy。
- 财务/基本面因子在 point-in-time 规则完整前不得进入可回测策略。

横截面数据契约：

- 同一交易日横截面必须使用同一 `dataset_version` 和同一可见时点。
- 股票池过滤顺序必须固定：基础 universe -> 可交易过滤 -> 数据可用过滤 -> 因子有效样本。
- 每次排名必须记录 `universe_size`、`tradable_size`、`factor_valid_count`。
- 缺失因子的处理必须由策略或因子配置显式声明，不能默认填 0。
- 横截面排序必须稳定处理并列值，保证同一输入得到确定性输出。

样本切分和 walk-forward：

- 每个策略实验必须声明 `train`、`validation`、`test` 区间。
- 支持 `walk_forward` 配置：训练窗口、验证窗口、测试窗口、步长、重训频率。
- 参数调优只能使用 train/validation，不能用 test 反复调参。
- 回测 artifact 必须保存 split 配置和每个 split 的指标。
- 样本过短、参数过多或换手过高时在 diagnostics 中提示过拟合风险。

因子评估：

- 预留因子评估模块，不只依赖组合回测结果。
- 指标包括：IC、Rank IC、ICIR、分层收益、覆盖率、换手率、因子自相关、多空组合收益。
- 后续支持行业中性、市值中性、风格暴露中性后的因子表现。
- 因子评估 artifact 需要保存分层组合、IC 时间序列和覆盖率时间序列。

Alpha / Portfolio Construction / Risk / Execution 分层：

- 长期目标不要让策略直接生成最终权重，而是拆成：
  - `AlphaModel`：输出股票 score 或 expected return。
  - `PortfolioConstructor`：把 score 转成目标权重。
  - `RiskModel`：计算暴露和约束。
  - `ExecutionModel`：把目标权重转成订单。
- MVP 可让 `PortfolioStrategy` 内部合并这些步骤，但接口和 artifact 要预留分层字段。
- 组合构建至少支持 Top K 等权、score 加权、最大权重约束和现金保留。
- 风险层约束失败必须记录 diagnostics，策略不能绕过。

容量和流动性：

- 小市值和高换手策略必须有容量评估，不得只看理想收益。
- 默认支持成交额过滤、ADV 过滤、单票成交额占比上限。
- 预留 `participation_rate_limit`：单日成交额不能超过标的近 N 日 ADV 的某个比例。
- 预留冲击成本模型：成本随成交额占比上升。
- 新股一字板、连续涨停、长期停牌必须显式标记为不可买。
- 输出容量报告：不同资金规模下收益、换手、成本、无法成交比例。

批量实验系统：

- 预留 `experiments` 和 `experiment_runs`。
- 支持参数网格、随机搜索和固定实验批次。
- 每个实验运行记录参数、数据版本、代码版本、状态、耗时、错误、摘要指标。
- 相同 `dataset_version + strategy_version + params_hash` 可以复用结果。
- 长任务后续改异步队列，MVP 先保留同步 API 和命令行入口。

数据质量监控：

- 单股票缺失交易日比例。
- 全市场覆盖率时间序列。
- 异常涨跌幅检测。
- 复权因子跳变检测。
- 成交额为 0 但价格变化异常检测。
- 每日新增、退市、停牌、ST 数量统计。
- 数据质量报告必须进入 `dataset_versions.quality_summary`。

研究解释性 artifact：

- 每期选股原因。
- 每个持仓的 score / factor breakdown。
- 每次调仓前后权重变化。
- 被过滤股票的原因统计。
- 因子贡献、行业贡献、个股贡献、交易成本贡献。
- 回测收益拆解：选股收益、择时收益、成本拖累、现金拖累。

### Investment Governance and Macro Risk

随着系统从单策略回测走向研究平台，需要把“投资过程可治理、可解释、可审计”纳入设计。策略不能只因为回测收益好看就进入候选池。

研究治理：

- 每个策略必须记录 `hypothesis`、`economic_rationale`、适用市场环境、失效条件和主要风险。
- 策略状态流转：`idea` -> `research` -> `validated` -> `paper_trading_ready` -> `deprecated`。
- 策略进入 `validated` 前必须有样本外表现、压力测试、成本敏感性和容量评估。
- JoinQuant 迁移策略必须说明原始逻辑、可复现部分、删减部分和潜在偏差。
- 不允许只用单一全样本收益率判断策略质量。

Regime / Scenario Framework：

- 回测指标必须支持按市场状态切片。
- A 股 regime 示例：牛市、熊市、震荡、高波动、低波动、流动性收缩、政策冲击。
- 美股 regime 示例：金融危机、疫情冲击、加息周期、科技股高波动、流动性宽松。
- Regime 标签可以先人工配置，后续再用波动率、趋势、宏观因子自动识别。
- artifact 保存每个 regime 的收益、回撤、波动、换手、成本和胜率。

压力测试：

- 支持历史压力窗口，例如 A 股 2015 股灾、2016 熔断、2020 疫情、2022 下跌。
- 支持美股历史压力窗口，例如 2008 金融危机、2020 疫情、2022 加息周期。
- 支持假设压力测试：成交成本上升、流动性下降、无法卖出、波动率翻倍、相关性上升。
- 压力测试结果进入策略 validation artifact。
- 压力测试失败不一定禁止策略，但必须在 diagnostics 中标记。

组合层风险预算：

- 多策略组合不能简单叠加权重，必须有组合层风险预算。
- 预留 `PortfolioRiskBudget`：目标波动率、最大回撤预算、单策略风险贡献、单资产风险贡献。
- 支持组合层约束：总仓位、净/毛敞口、单市场敞口、单行业敞口、相关性上限。
- 支持波动率缩放和回撤降风险。
- MVP 单策略可以先不启用，但接口必须能承载多策略组合。

资本分配层：

- 预留 `Allocator`，负责在多个策略或 alpha sleeve 之间分配资金。
- 支持等权、等风险贡献、波动率目标、回撤降权、regime-aware allocation。
- 资本分配结果必须记录到 artifact：输入指标、约束、目标权重、实际权重。
- 分配层不能修改底层策略信号，只能调整资金权重。

决策审计：

- 每次调仓必须能解释“为什么买/卖/不交易”。
- 审计记录包含：数据版本、当前仿真时间、universe、因子值、排名、风险约束、成本假设、目标权重、最终订单。
- 被过滤、降权、拒单的标的必须记录原因。
- 前端后续应能按某个交易日展开完整决策链路。

模型风险管理：

- 支持参数敏感性分析：关键参数上下浮动后表现是否稳定。
- 支持样本区间敏感性分析：不同起止日期、不同 regime 下表现是否稳定。
- 支持数据源敏感性分析：同一策略在不同数据源或复权口径下差异。
- 支持成本假设敏感性分析：佣金、滑点、冲击成本上升后的表现。
- 不稳定策略应标记为 high model risk，不能进入 validated 状态。

Backtest / Paper Trading Parity：

- 虽然当前暂不做实盘，但回测和未来模拟盘应共用策略接口、DataPortal、Broker 模型和 CostModel。
- 差异只允许存在于数据源和 Broker 实现：BacktestBroker、PaperBroker、LiveBroker。
- 策略代码不能判断“当前是回测还是模拟盘”后走不同逻辑。
- 所有 paper trading 结果也必须绑定数据版本、代码版本和参数版本。

## 分层实施 TODO

### 第 0 层：接口边界和目录

- [x] 新增 `backend/marketdata/` 包，市场数据相关逻辑不放进 `services/data.py`。
- [x] 新增 `backend/backtest/portfolio_engine.py`，只负责时间推进和策略调用，组合回测不塞进现有单标的 `engine.py`。
- [x] 新增 `backend/backtest/broker.py`，实现订单、成交、现金和持仓。
- [x] 新增 `backend/backtest/costs.py`，实现 `CostModel`。
- [x] 新增 `backend/backtest/order_validator.py`，实现停牌、涨跌停、整数手、现金约束。
- [x] 新增 `backend/backtest/portfolio_strategies.py`，放 MVP 组合策略。
- [x] 新增 `backend/tests/fixtures/market_data/`，所有组合引擎 UT 使用固定小数据。
- [x] 新增 `scripts/refresh_market_data.py`，命令行刷新本地行情库。
- [x] 保留现有 `/api/backtest`，新增 `/api/backtest/portfolio`。

验收：

- [x] 原有单标的测试全部保持通过。
- [x] 新增模块不改变现有 API 返回结构。
- [x] 策略、Broker、CostModel、DataPortal 可以分别单测，不依赖真实行情源。
- [x] 新增核心模块有 type hints、领域异常和对应 UT。
- [x] FastAPI handler 不包含回测核心逻辑。
- [x] 新增 Python 代码通过 scoped ruff 检查，并按 Google style 补充核心 docstring。
- [x] marketdata 核心模块已有正常路径、错误路径和边界条件 UT。

### 第 1 层：本地行情数据库

新增 DuckDB schema：

- [x] `instruments`：证券主数据。
  - `instrument_id`
  - `market`
  - `symbol`
  - `name`
  - `exchange`
  - `asset_type`
  - `currency`
  - `list_date`
  - `delist_date`
  - `is_active`
  - `is_st`
  - `source`
  - `updated_at`
- [x] `daily_bars`：日线行情。
  - `market`
  - `symbol`
  - `trade_date`
  - `open`
  - `high`
  - `low`
  - `close`
  - `pre_close`
  - `adj_open`
  - `adj_high`
  - `adj_low`
  - `adj_close`
  - `adjustment`
  - `volume`
  - `amount`
  - `adj_factor`
  - `paused`
  - `limit_up`
  - `limit_down`
  - `source`
  - `updated_at`
- [x] `intraday_bars`：日内 K 线，MVP 先建表预留，日线阶段可以不刷新。
  - `market`
  - `symbol`
  - `bar_time`
  - `trade_date`
  - `frequency`
  - `open`
  - `high`
  - `low`
  - `close`
  - `volume`
  - `amount`
  - `source`
  - `updated_at`
  - 主键：`market + symbol + bar_time + frequency`
- [ ] `ticks`：逐笔/Tick，后续可选，不进入首个 MVP。
  - `market`
  - `symbol`
  - `tick_time`
  - `trade_date`
  - `price`
  - `volume`
  - `amount`
  - `side`
  - `source`
  - `updated_at`
  - 推荐按月 Parquet 分区，DuckDB 外表查询。
- [x] `trading_calendar`：交易日历。
  - `trade_date`
  - `market`
  - `is_open`
- [x] `trading_sessions`：交易时段，给日内数据和撮合使用。
  - `market`
  - `exchange`
  - `session_type`
  - `start_time`
  - `end_time`
  - `timezone`
  - `is_auction`
- [x] `dataset_versions`：可复现数据快照版本。
  - `dataset_version`
  - `schema_version`
  - `markets`
  - `source`
  - `created_at`
  - `min_trade_date`
  - `max_trade_date`
  - `instrument_count`
  - `daily_bar_count`
  - `intraday_bar_count`
  - `quality_status`
  - `quality_summary`
  - `is_stable`
- [x] `refresh_runs`：数据刷新记录。
  - `id`
  - `dataset_version`
  - `job_type`
  - `started_at`
  - `finished_at`
  - `status`
  - `symbols_total`
  - `symbols_success`
  - `symbols_failed`
  - `error`

实现任务：

- [x] 新增 `backend/marketdata/db.py`：连接 DuckDB、初始化 schema、事务封装。
- [x] 新增 `backend/marketdata/models.py`：轻量 dataclass 或 typed dict，不引入 ORM。
- [x] 新增 `backend/marketdata/repository.py`：批量 upsert、按日期/股票池读取。
- [x] 新增 `backend/marketdata/sources/akshare_source.py`：股票列表、交易日历、日线增量拉取。
- [x] 新增 `backend/marketdata/refresh.py`：全量初始化和增量刷新编排。
- [x] 新增 `backend/marketdata/data_portal.py`：统一读取日线/分钟线/未来 Tick 数据。
- [x] 新增 `backend/marketdata/versioning.py`：生成、查询、锁定 `dataset_version`。
- [x] 刷新逻辑必须幂等，同一 `market + symbol + trade_date` 重跑不会重复插入。
- [x] 日线幂等键为 `market + symbol + trade_date`，分钟线幂等键为 `market + symbol + bar_time + frequency`。
- [x] 刷新失败要记录失败标的，允许下次只重试失败部分。
- [x] 回测读取只能使用 stable `dataset_version`，禁止读取刷新中的半成品版本。

验收：

- [x] 可初始化空 DuckDB 文件。
- [x] 可写入和覆盖更新 fixture 行情。
- [x] 可按日期区间读取多个 symbol 的价格矩阵。
- [x] 可通过同一 `DataPortal.history(...)` 接口读取日线和预置分钟线 fixture。
- [x] 可查询某日可交易股票列表。
- [x] 可创建 stable / failed 数据版本，并按版本读取一致数据。
- [x] 数据库缺失、表缺失、空数据时给出明确错误。
- [x] DuckDB schema 带 `schema_version`，schema 变更有迁移或重建说明。
- [x] 数据源适配器 UT 使用 mock，不访问真实网络。
- [x] 数据版本、字段来源和质量检查摘要可追溯。

UT：

- [x] `test_marketdata_db_initializes_schema`
- [x] `test_daily_bars_upsert_is_idempotent`
- [x] `test_read_price_matrix_filters_date_range`
- [x] `test_intraday_bars_upsert_uses_symbol_time_frequency_key`
- [x] `test_data_portal_reads_daily_and_intraday_by_frequency`
- [x] `test_trading_sessions_model_supports_midday_break`
- [x] `test_dataset_version_created_after_successful_refresh`
- [x] `test_backtest_reads_only_stable_dataset_version`
- [x] `test_refresh_run_records_success_and_failure`
- [x] `test_repository_rejects_duplicate_or_invalid_keys`

### 第 2 层：股票池和过滤器

新增股票池接口：

```python
get_universe(trade_date, config) -> list[str]
```

配置项：

- [x] `market`: 默认 `cn`
- [x] `include_exchanges`: 默认 `["SSE", "SZSE", "BSE"]`
- [x] `exclude_neeq`: 默认 `True`
- [x] `exclude_st`: 默认 `True`
- [x] `exclude_suspended`: 默认 `True`
- [x] `min_list_days`: 默认 `60`
- [x] `exclude_limit_up_buy`: 默认 `True`
- [x] `exclude_limit_down_sell`: 默认 `True`

实现任务：

- [x] 新增 `backend/marketdata/universe.py`。
- [x] 支持 A 股全市场股票池，包含北交所。
- [x] 预留美股 universe：`market=us`，支持 `NYSE/NASDAQ/AMEX` 交易所过滤。
- [x] 支持手工传入自定义股票池。
- [x] 支持按行业/标签扩展，但 MVP 先保留接口，不做行业源。
- [x] 每个过滤器返回被过滤数量，方便前端解释“为什么只剩 N 只”。

验收：

- [x] 能排除新三板。
- [x] A 股全市场默认包含北交所标的。
- [x] 能排除 ST。
- [x] 能排除上市不足 N 天股票。
- [x] 能排除当日停牌股票。
- [x] 缺少某些可选字段时降级明确，不静默误选。

UT：

- [x] `test_universe_excludes_neeq`
- [x] `test_cn_universe_includes_bse_by_default`
- [x] `test_us_universe_uses_us_exchanges_without_cn_filters`
- [x] `test_universe_excludes_st_by_default`
- [x] `test_universe_respects_min_list_days`
- [x] `test_universe_excludes_suspended_on_trade_date`
- [x] `test_universe_returns_filter_diagnostics`

### 第 3 层：组合策略接口

新增组合策略协议：

```python
class PortfolioStrategy:
    id: str
    name: str
    warmup_days: int
    rebalance_frequency: str

    def generate_targets(self, date, data, context) -> list[TargetPosition]:
        ...
```

目标持仓：

```python
TargetPosition(
    symbol="600519",
    weight=0.05,
    score=1.23,
    reason="20日动量排名"
)
```

设计约束：

- [x] 策略只读 `DataPortal`，不能直接读数据库或请求网络。
- [x] `DataPortal` 默认只暴露当前日期之前的数据。
- [x] `DataPortal.history(symbols, fields, end_time, bars, frequency)` 是唯一历史行情入口。
- [x] `frequency` 必须显式传入，默认只允许 `1d`；日内策略必须声明 `1m`、`5m` 等频率。
- [x] `end_time` 的可见性由引擎控制：日线策略在信号日收盘后可见当日收盘数据，日内策略只能看当前 bar 之前的数据。
- [x] 策略声明 `data_frequency` 和 `rebalance_frequency`，引擎用它决定交易日历和撮合时点。
- [x] 策略返回目标权重，总权重不能超过 `1.0`。
- [x] 策略可以返回空列表，表示清仓或空仓。
- [x] 所有策略必须声明 `warmup_days`，引擎在预热期不交易。

MVP 策略：

- [x] `momentum_top_k`：过去 `lookback_days` 收益率排名，买入 Top K，等权。
- [ ] `small_cap_top_k`：预留接口，等有市值数据再实现。
- [ ] `jq_style_etf_rotation`：预留接口，等多 ETF 数据稳定后实现。

验收：

- [x] 策略无法读取未来日期。
- [x] 日内策略无法读取当前 bar 尚未完成的数据。
- [x] 权重超过 100% 时引擎拒绝。
- [x] 标的不存在或当日不可交易时引擎过滤并记录原因。
- [x] 策略实现不直接 import 数据库、文件、网络或 FastAPI。
- [x] 策略实验记录参数、数据版本、代码版本和样本区间。

UT：

- [x] `test_data_portal_blocks_future_data`
- [x] `test_data_portal_blocks_current_unclosed_intraday_bar`
- [x] `test_strategy_must_declare_data_frequency`
- [x] `test_momentum_strategy_selects_top_k`
- [x] `test_strategy_weights_sum_must_not_exceed_one`
- [x] `test_empty_targets_mean_no_position`

### 第 4 层：组合回测引擎

新增引擎能力：

- [x] 多股票持仓。
- [x] 现金管理。
- [x] 显式订单、成交、持仓模型：`Order`、`Fill`、`Position`。
- [x] 定期调仓。
- [x] 按目标权重下单。
- [x] A 股 100 股整数手买入。
- [x] 交易成本模型作为超参数：佣金、印花税、过户费、最低佣金、滑点、冲击成本。
- [x] 停牌不可交易。
- [x] 涨停不可买。
- [x] 跌停不可卖。
- [x] 交易失败记录。
- [x] 失败订单必须记录失败原因，不得静默丢弃。
- [x] 每日组合净值。
- [x] 每日持仓权重。
- [x] 每次调仓明细。

默认撮合规则：

- 信号日期 `T` 使用 `T` 收盘后已知数据生成目标。
- 交易日期 `T+1` 执行。
- MVP 默认用 `T+1` 开盘价成交；如果没有开盘价，则用收盘价并标记为 fallback。
- 买入按 100 股取整。
- 卖出允许清仓已有股数。

交易成本模型：

- [x] 新增 `CostModel`，所有成本参数从回测请求或策略配置传入，不能写死在引擎里。
- [x] `buy_commission_rate`：买入佣金率，默认按 A 股常见值 `0.0003`，允许设为 `0`。
- [x] `sell_commission_rate`：卖出佣金率，默认同买入佣金率。
- [x] `stamp_tax_rate`：卖出印花税率，默认 `0.0005` 或按当期规则配置，买入不收。
- [x] `transfer_fee_rate`：过户费/经手费等按成交额比例计费项，默认 `0`，允许市场级默认值。
- [x] `min_commission`：单笔最低佣金，默认 `5.0`，允许设为 `0` 便于研究。
- [x] `slippage_bps`：滑点基点，买入成交价上调、卖出成交价下调。
- [x] `market_impact_bps`：冲击成本基点，MVP 默认 `0`，后续可与成交额/成交量占比挂钩。
- [x] 成本计算必须逐订单记录到 trade 明细：`commission`、`stamp_tax`、`transfer_fee`、`slippage_cost`、`total_cost`。
- [x] 现金不足时必须把成本一起纳入买入金额约束，不能只按裸成交额下单。
- [x] 组合指标中增加 `total_cost` 和 `turnover`，便于比较不同 cost 超参数。

日内扩展撮合约束：

- 日线引擎和日内引擎共用订单、持仓、费用模型，但时间推进器不同。
- 日线时间推进器按交易日迭代；日内时间推进器按 `trading_sessions` 生成 bar 时间轴。
- 日内策略默认下一 bar 成交，不能在生成信号的同一未完成 bar 上成交。
- 集合竞价、午休、临停等不在首个 MVP 实现，但接口必须能表达交易时段不可撮合。

输出结构：

- [x] `equity_curve`
- [x] `positions`
- [x] `trades`
- [x] `rebalance_logs`
- [x] `filter_diagnostics`
- [x] `metrics`
- [x] `benchmark_curve`
- [x] `cost_summary`

指标：

- [x] 总收益。
- [x] 年化收益。
- [x] 年化波动。
- [x] 夏普比率。
- [x] 最大回撤。
- [x] 最大回撤持续天数。
- [x] 换手率。
- [x] 总交易成本。
- [x] 交易成本占成交额比例。
- [x] 持仓数量均值。
- [x] 胜率。
- [x] 基准超额收益。

验收：

- [x] 引擎在固定 fixture 下结果可手工复核。
- [x] 无未来函数。
- [x] 调仓日和非调仓日行为明确。
- [x] 价格缺失、停牌、涨跌停、现金不足时都有确定行为。
- [x] 交易成本参数变化会影响成交股数、现金和最终净值，且结果可手工复核。
- [x] Broker、CostModel、OrderValidator 的核心逻辑必须可脱离 PortfolioEngine 单独测试。
- [x] 引擎日志和结果包含 `request_hash`、`dataset_version`、策略、耗时和股票池大小。
- [x] 引擎 golden test 固定小样本回测结果，结果变化必须可解释。
- [x] 引擎属性测试覆盖现金、仓位、成本和时间可见性不变量。

UT：

- [x] `test_portfolio_engine_buys_equal_weight_top_k`
- [x] `test_portfolio_engine_rebalances_on_schedule_only`
- [x] `test_portfolio_engine_uses_next_day_open_without_lookahead`
- [x] `test_portfolio_engine_respects_lot_size`
- [x] `test_broker_records_rejected_order_reason`
- [x] `test_order_validator_rejects_invalid_symbol_or_weight`
- [x] `test_portfolio_engine_blocks_limit_up_buy`
- [x] `test_portfolio_engine_blocks_limit_down_sell`
- [x] `test_portfolio_engine_keeps_cash_when_order_cannot_fill`
- [x] `test_cost_model_applies_min_commission`
- [x] `test_cost_model_charges_stamp_tax_on_sell_only`
- [x] `test_cost_model_applies_slippage_to_fill_price`
- [x] `test_buy_order_cash_check_includes_all_costs`
- [x] `test_total_cost_is_reported_in_trades_and_metrics`
- [x] `test_portfolio_metrics_match_manual_fixture`

### 第 5 层：API

新增请求模型：

- [x] `PortfolioBacktestRequest`
  - `market`
  - `universe`
  - `symbols`
  - `start`
  - `end`
  - `dataset_version`
  - `initial_cash`
  - `strategy`
  - `rebalance_frequency`
  - `buy_commission_rate`
  - `sell_commission_rate`
  - `stamp_tax_rate`
  - `transfer_fee_rate`
  - `min_commission`
  - `slippage_bps`
  - `market_impact_bps`
  - `max_positions`
  - `lookback_days`
  - `exclude_st`
  - `min_list_days`

新增接口：

- [x] `POST /api/backtest/portfolio`
- [x] `GET /api/market/universe`
- [x] `GET /api/market/data-status`
- [x] `GET /api/market/dataset-versions`
- [x] `GET /api/market/supported-markets`
- [x] `GET /api/backtest/{run_id}/artifacts`
- [x] `POST /api/market/refresh`，开发期可用，生产环境需权限保护。

验收：

- [x] 行情库不存在时返回明确提示，不触发实时全市场下载。
- [x] 参数非法时返回 400 和具体字段原因。
- [x] API 响应必须返回实际使用的 `dataset_version` 和 `request_hash`。
- [x] 大型结果通过 artifact 分页读取，默认响应不返回每日全持仓。
- [x] 回测任务耗时较长时，MVP 可以同步返回；后续改异步任务。
- [x] API 错误由领域异常统一映射，不向前端暴露内部 traceback。

UT：

- [x] `test_portfolio_backtest_api_returns_metrics`
- [x] `test_portfolio_backtest_api_rejects_missing_market_data`
- [x] `test_portfolio_backtest_api_returns_dataset_version`
- [x] `test_portfolio_backtest_api_rejects_unstable_dataset_version`
- [x] `test_market_data_status_api_reports_date_range`
- [x] `test_supported_markets_api_includes_cn_and_us`
- [x] `test_portfolio_backtest_api_validates_universe_params`

### 第 6 层：前端

前端新增模式：

- [x] 回测类型：`单标的` / `全市场组合` / `自定义股票池`。
- [x] 全市场模式隐藏单个股票代码输入。
- [x] 展示本地行情库状态：最新交易日、股票数量、日线行数、最近刷新时间。
- [x] 展示股票池过滤诊断：全市场 N 只，排除 ST M 只，排除新股 K 只，最终可交易 X 只。
- [x] 展示组合净值曲线。
- [x] 展示调仓记录。
- [x] 展示当前/末期持仓。
- [x] 展示失败订单原因。
- [ ] 前端信息架构重塑：参考 anthropic.com 风格，按“市场洞察 / 回测工作台 / 策略研究 / 结果复盘”分区，并提供快速锚点导航，降低页面平铺带来的使用成本。

验收：

- [x] 用户能明确知道当前跑的是单标的还是全市场。
- [x] 数据未刷新时前端提示先刷新数据。
- [x] 大结果不一次性渲染所有交易明细，默认分页或截断。

### 第 7 层：数据刷新策略

刷新命令：

```bash
python scripts/refresh_market_data.py --market cn --mode init
python scripts/refresh_market_data.py --market cn --mode incremental
python scripts/refresh_market_data.py --market cn --symbols 600519,000001
python scripts/refresh_market_data.py --market us --mode incremental --symbols AAPL,MSFT
```

实现要求：

- [x] 初始化先拉股票列表和交易日历。
- [x] 按 `market` 选择数据源适配器和交易日历，不同市场刷新互不影响。
- [x] 日线刷新按 symbol 增量拉取。
- [x] 日内刷新使用独立命令参数 `--frequency 1m|5m|15m|30m|60m`，不能和日线刷新逻辑混在同一个 upsert 键里。
- [x] 日内数据按交易日或月份分批刷新，避免一次请求/一次事务过大。
- [x] 单只失败不影响整体任务。
- [x] 支持断点续跑。
- [x] 支持 `--limit` 做开发调试。
- [x] 支持 `--start` / `--end` 指定区间。
- [x] 支持 `--market us` 的仓库结构和接口占位，MVP 可先只实现少量美股 symbol 刷新。
- [x] 刷新后自动跑数据质量检查。

数据质量检查：

- [x] OHLC 不能为负。
- [x] `high >= low`。
- [x] `high >= open/close`。
- [x] `low <= open/close`。
- [x] 成交量缺失允许，但要记录。
- [x] 同一股票日期不能重复。
- [x] 同一股票同一 `bar_time + frequency` 不能重复。
- [x] 分钟线 `bar_time` 必须落在 `trading_sessions` 定义的可交易时段内。
- [x] 分钟线同一交易日覆盖率低于阈值时报警。
- [x] 最近交易日覆盖率低于阈值时报警。

UT：

- [x] `test_refresh_skips_symbols_already_up_to_date`
- [x] `test_refresh_continues_after_one_symbol_failure`
- [x] `test_quality_check_rejects_invalid_ohlc`
- [x] `test_quality_check_rejects_intraday_bar_outside_session`
- [x] `test_quality_check_reports_coverage`

### 第 8 层：JoinQuant 策略迁移

迁移原则：

- [x] 单标的择时类继续使用 `signal(df) -> 0/1`。
- [x] 选股/轮动/多因子类迁移到 `PortfolioStrategy`。
- [ ] 依赖基本面、市值、行业、ST、停牌、涨跌停的策略，必须等对应数据层完成后再接入。
- [x] 每个迁移策略必须写明原始 `source_strategy_id`、可复现部分、不可复现差异。

优先迁移：

- [x] ETF/指数轮动：先用多 ETF 数据实现。
- [x] 动量 Top K：作为全市场组合引擎首个样例。
- [ ] 小市值：等市值数据和 ST/停牌过滤完成后实现。
- [ ] 红利/价值：等财务因子数据完成后实现。

### 第 9 层：性能和鲁棒性

性能目标：

- [x] fixture 测试秒级完成。（最新基准：`fixture_20_1y=0.661s`）
- [x] 100 只股票、3 年日线回测在 5 秒内完成。（最新基准：`2.312s`）
- [x] 5000 只股票、5 年日线回测在本地机器上可接受，首版目标小于 60 秒。（最新基准：`15.494s`）
- [x] 日内数据首版只承诺小股票池回测，不承诺全市场 1 分钟线秒级完成。（已在性能基准与文档中明确边界）

工程约束：

- [x] 回测引擎不能逐日逐股票反复 SQL 查询，应批量读取后在内存中计算。
- [x] 大矩阵读取只取需要字段和日期范围。
- [x] 分钟线读取必须按 `symbols + time_range + frequency + fields` 下推过滤。
- [x] Tick 级数据默认不进 API 响应，只能生成聚合结果或分页读取。
- [x] API 返回默认不包含完整每日全部持仓，避免响应过大。
- [x] 大结果后续可落盘为回测快照，再分页读取。

鲁棒性：

- [x] 所有外部数据源调用都必须有 timeout。
- [x] 所有刷新任务都必须记录失败原因。
- [x] 缺失数据不能静默填 0。
- [x] 策略遇到数据不足必须返回空仓或明确错误。
- [x] 每次回测必须绑定 stable `dataset_version`。
- [x] 每次回测必须生成 `request_hash`，包含策略、参数、数据版本和代码版本。
- [x] Point-in-time 字段缺少 `known_date` / `effective_date` 时不能进入策略计算。
- [x] 新依赖、配置项、命令行参数和 API 示例必须同步更新文档。
- [x] CI 默认运行快速 UT；网络测试、性能测试、真实数据测试必须单独标记。

### 第 10 层：研究平台扩展

因子和横截面：

- [x] 新增 `backend/research/factors.py` 或 `backend/research/factor_pipeline.py`。
- [x] 定义 `FactorDefinition`：名称、版本、依赖、窗口、频率、point-in-time 规则。
- [x] 定义 `FactorPipeline.compute(date, universe, data)`。
- [x] 支持 fixture 横截面因子计算。
- [x] 记录每个交易日的 `universe_size`、`tradable_size`、`factor_valid_count`。
- [x] 支持 winsorize、standardize、missing value policy 的接口占位。

实验和样本切分：

- [x] 新增 `ExperimentConfig`：策略、参数、数据版本、train/validation/test 区间。
- [x] 支持 walk-forward 配置结构。
- [x] `request_hash` 纳入 split 配置。
- [ ] artifact 保存每个 split 的指标。
- [ ] 参数搜索记录搜索空间和所有候选摘要。

组合构建和风险：

- [x] 预留 `AlphaModel`、`PortfolioConstructor`、`RiskModel`、`ExecutionModel` 接口。
- [x] MVP `momentum_top_k` 内部可合并实现，但输出 artifact 要包含 score 和目标权重来源。
- [x] 风险约束支持最大持仓数、单票最大权重、总仓位上限、最小成交额。
- [x] 约束失败进入 diagnostics。

容量和流动性：

- [x] 支持 ADV 字段或成交额滚动均值。
- [x] 支持单票成交额占比上限。
- [x] 支持资金规模参数，用于容量报告。
- [x] 输出无法成交比例、成本占比、换手率和容量敏感性。

因子评估：

- [x] 预留 IC / Rank IC 计算接口。
- [x] 预留分层收益计算接口。
- [x] 预留因子覆盖率和因子自相关计算接口。
- [x] 因子评估使用 fixture 数据做 golden test。

数据质量和解释性：

- [x] 数据质量报告进入 `dataset_versions.quality_summary`。
- [x] artifact 包含每期选股原因、factor breakdown、过滤原因统计。
- [x] artifact 包含收益拆解：选股、择时、成本、现金拖累。

UT：

- [x] `test_factor_pipeline_computes_point_in_time_cross_section`
- [x] `test_factor_pipeline_records_valid_sample_count`
- [x] `test_factor_pipeline_missing_values_follow_policy`
- [x] `test_experiment_config_hash_includes_splits`
- [x] `test_walk_forward_split_has_no_overlap`
- [x] `test_portfolio_constructor_respects_max_weight`
- [x] `test_liquidity_constraint_limits_participation_rate`
- [x] `test_capacity_report_worsens_with_larger_capital`
- [x] `test_factor_ic_matches_manual_fixture`
- [x] `test_explainability_artifact_contains_filter_reasons`

### 第 11 层：投资治理和宏观风险

策略治理：

- [x] 定义 `StrategyMetadata`：hypothesis、economic rationale、适用环境、失效条件、风险说明。
- [x] 定义策略状态：`idea`、`research`、`validated`、`paper_trading_ready`、`deprecated`。
- [x] 策略状态升级必须检查样本外、压力测试、成本敏感性、容量报告。
- [x] JoinQuant 迁移策略 artifact 记录原始策略和复现差异。

Regime 和场景：

- [x] 新增 `regime_labels` 数据结构或配置文件。
- [x] 支持人工配置历史 regime 窗口。
- [x] 回测指标按 regime 切片输出。
- [x] artifact 保存每个 regime 的收益、回撤、波动、换手、成本。

压力测试：

- [x] 新增 `StressScenario` 配置：历史窗口或假设冲击。
- [x] 支持成本上升、流动性下降、无法卖出、波动率放大、相关性上升。
- [x] 压力测试结果进入 validation artifact。
- [x] 压力测试失败进入 diagnostics。

组合风险预算和资本分配：

- [x] 预留 `PortfolioRiskBudget`：目标波动、最大回撤、风险贡献、敞口限制。
- [x] 预留 `Allocator`：等权、等风险贡献、波动率目标、回撤降权、regime-aware allocation。
- [x] 多策略组合必须通过 Allocator 分配资本，不能简单叠加。
- [x] 分配结果进入 artifact：输入指标、约束、目标权重、实际权重。

决策审计：

- [x] artifact 支持按交易日查询完整决策链路。
- [x] 调仓日志包含数据版本、仿真时间、universe、因子值、排名、风险约束、成本假设、目标权重、最终订单。
- [x] 被过滤、降权、拒单标的记录原因。
- [x] 前端后续可展示单日决策树。

模型风险管理：

- [x] 支持参数敏感性报告。
- [x] 支持样本区间敏感性报告。
- [x] 支持数据源/复权口径敏感性报告。
- [x] 支持成本假设敏感性报告。
- [x] 高模型风险策略不能进入 `validated` 状态。

Backtest / Paper Trading Parity：

- [x] 抽象 `Broker` 接口，允许 `BacktestBroker`、`PaperBroker`、`LiveBroker`。
- [x] 策略不得根据运行模式分支执行不同逻辑。
- [x] Paper trading 结果绑定数据版本、代码版本和参数版本。

UT：

- [x] `test_strategy_cannot_validate_without_out_of_sample_metrics`
- [x] `test_regime_metrics_slice_equity_curve_by_config`
- [x] `test_stress_scenario_increases_costs_and_lowers_return`
- [x] `test_allocator_respects_risk_budget`
- [x] `test_decision_audit_contains_filter_and_order_reasons`
- [x] `test_high_model_risk_blocks_validated_status`
- [x] `test_strategy_logic_does_not_depend_on_broker_mode`

## 分阶段里程碑

### M1：本地行情库 MVP

- [x] DuckDB schema。
- [x] `dataset_versions`。
- [x] `DataPortal` 基础契约。
- [x] schema 支持 `market=cn/us`，第一阶段真实数据刷新 A 股小样本 + 美股小样本。
- [x] fixture 写入读取。
- [x] AkShare 股票列表和少量日线刷新适配器。
- [x] yfinance 美股小样本日线刷新适配器。
- [x] 同时存原始 OHLC 和前复权 OHLC，策略默认读取前复权。
- [x] 第一阶段 CI 使用 `pytest + ruff`。
- [x] 数据状态 API。
- [x] 数据层 UT 全通过。

### M2：股票池 MVP

- [x] 全市场股票池查询。
- [x] A 股全市场包含北交所。
- [x] 排除新三板。
- [x] 排除 ST。
- [x] 排除新股。
- [x] 排除停牌。
- [x] 股票池 UT 全通过。

### M3：组合回测 MVP

- [x] `momentum_top_k` 策略。
- [x] 多持仓组合引擎。
- [x] `BrokerSimulator`、`CostModel`、`OrderValidator`。
- [x] 下一交易日成交。
- [x] 等权调仓。
- [x] 组合指标。
- [x] 组合引擎 UT 全通过。

### M4：API 和前端

- [x] `/api/backtest/portfolio`。
- [x] `/api/market/data-status`。
- [x] 前端全市场模式。
- [x] 前端展示净值、持仓、调仓、过滤诊断。
- [x] API UT 全通过。

### M5：真实数据验证

- [x] 拉取小样本真实 A 股数据。
- [x] 拉取小样本真实美股数据，验证同一 schema 可存储和查询。
- [x] 跑 100 只股票样例。
- [x] 跑全市场样例。
- [x] 输出性能和数据质量报告。
- [x] 更新 README 和 JoinQuant TODO。

### M6：研究平台基础

- [x] `FactorPipeline` fixture MVP。
- [x] 横截面因子 golden test。
- [x] 样本切分和 walk-forward 配置。
- [x] 因子 IC / 分层收益 fixture 评估。
- [x] 容量和流动性报告 MVP。
- [x] 实验登记和解释性 artifact MVP。

### M7：投资治理和宏观风险基础

- [x] `StrategyMetadata` 和策略状态流转。
- [x] Regime 配置和按 regime 指标切片。
- [x] 历史/假设压力测试 MVP。
- [x] 决策审计 artifact MVP。
- [x] 参数、样本区间、成本敏感性报告。
- [x] `Allocator` / `PortfolioRiskBudget` 接口占位。

## 暂不做

- 分钟线、Tick、盘口。
- 期货、期权、融资融券。
- 实盘交易。
- 用户自定义全市场 Python 策略沙箱。
- 机器学习训练管线。
- 财务因子和行业因子，等日线组合引擎稳定后再做。

## 开始实现前需要确认

- [x] 全市场包含北交所。
- [x] 默认排除 ST / *ST，高级参数可打开。
- [x] 新股默认排除 60 个交易日。
- [x] 第一阶段真实数据刷新 A 股小样本 + 美股小样本。
- [x] 日线同时存原始价格和复权价格，策略默认前复权。
- [x] 第一阶段工程工具使用 `pytest + ruff`。
- [x] 首个 MVP 策略采用 `momentum_top_k`。
- [x] 本地行情库接受新增 `duckdb` 和 `akshare` 依赖。
