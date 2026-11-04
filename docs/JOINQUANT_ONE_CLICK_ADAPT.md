# JoinQuant 策略「一键适配」说明

## 当前机制（不是自动翻译聚宽代码）

「一键回测」= **人工抽取可复现逻辑** + **注册适配器** + **前端填参并运行**。

```
聚宽源码 (.txt)
    → 人工分析可复现部分
    → backend/factors/primitives/   （可复用因子原语，优先写这里）
    → backend/backtest/imported_strategies.py  （单标的 signal）
    → backend/portfolio/jq_strategies.py       （组合 PortfolioStrategy）
    → source_strategy_id 关联 jq-00xx
    → 策略库 API 标记 adapted_backtest_ready
    → 前端「回测」按钮 → runAdaptedStrategy → 自动运行
```

## 因子库（L1 原语层）

路径：`backend/factors/primitives/`

| 模块 | 内容 | 典型消费者 |
|------|------|------------|
| `oscillators.py` | RSI | 择时 gate、单标的信号 |
| `trend.py` | BBI、EMA5 gate | jq-0001 / jq-0007 |
| `breadth.py` | 大盘广度 | jq-0001 / jq-0007 |
| `momentum.py` | polyfit 动量、WPETF 评分 | jq-0019 / jq-0009 |
| `timing.py` | 价值择时 gate、RSRS、成交量/价格情绪 | jq-0002 / jq-0020 |
| `etf_selection.py` | jq-0020 ETF 排序与过滤 | jq-0020 |

`backtest/jq_bbi.py`、`jq_risk_control.py`、`jq_momentum_scoring.py` 为薄 re-export，新代码请直接 `from factors.primitives...`。

研究层 `research/factor_pipeline.py` 的 `FactorDefinition.compute_fn` 也应调用上述原语，避免回测/研究两套实现。

## 已适配策略一览（截至 Top20 无财报批次）

### 单标的（`ADAPTED_STRATEGIES` → 0/1 信号）

| 适配 ID | 源 jq-id | 说明 |
|---------|----------|------|
| `jq_bbi_momentum_timing` | jq-0001 | BBI 动量择时（单标的版） |
| `jq_rsrs_timing` | jq-0004, jq-0008 | RSRS 择时 |
| `jq_rsi_gap_reversal` | jq-0036 | RSI 缺口衍生 |
| `jq_value_ma_rsi_timing` | jq-0002 | 价值投资大盘择时层 |
| `jq_etf_ma_cross_timing` | jq-0026 | MA 交叉 |
| `jq_etf_value_averaging_timing` | jq-0003 | 价值平均 proxy（回撤入场 + 止盈） |

### 组合（`ADAPTED_PORTFOLIO_STRATEGIES` → PortfolioEngine）

| 适配 ID | 源 jq-id | 说明 |
|---------|----------|------|
| `jq_wide_base_etf_bbi_rotation` | jq-0001 | 宽基 ETF BBI 完整轮动（策略库一键优先） |
| `jq_value_timing_top_k` | jq-0002 | 成交额 proxy + 大盘 gate |
| `jq_industry_etf_bbi_rotation` | jq-0007 | 行业 ETF BBI + 择时 |
| `jq_etf_polyfit_momentum_top1` | jq-0019 | polyfit 动量 Top1 |
| `jq_etf_momentum_rsrs_top1` | jq-0010, jq-0008 | 动量 + RSRS gate Top1 |
| `jq_small_cap_top_k` | jq-0006, jq-0012, jq-0022, jq-0023 | 小市值成交额 proxy |
| `jq_etf_momentum_volume_gate_top1` | jq-0020 | 13 日动量 + 成交量/价格情绪 gate |
| `jq_etf_value_averaging` | jq-0003 | 月度价值平均 proxy |
| `jq_wpetf_momentum_top1` | jq-0009 | 多策略整合之 ETF 腿 |

同一 `source_strategy_id` 若同时有单标的与组合适配，策略库 `_adapted_by_source()` **优先展示组合版**（`run_preset.mode == portfolio`）。

## 以 jq-0002「54.价值投资策略-大盘择时」为例

### 原策略两层结构

| 模块 | 聚宽依赖 | 本项目现状 |
|------|----------|------------|
| **大盘择时** `RiskControl` | `get_bars` / `talib` / 沪深300 | ✅ `jq_value_ma_rsi_timing` / `factors.primitives.timing` |
| **基本面选股** `get_stock_list` | `get_fundamentals` 多期财报 | ⚠️ proxy：`jq_value_timing_top_k`（成交额排序） |
| **多标的调仓** | `order_target_value` / 周频 | ✅ 组合引擎周频等权 |

### 完整复现原策略还需

1. Phase A：财务因子入库（ROE、FCF、营收增速等）
2. Phase B：`get_stock_list` 真实基本面过滤
3. Phase C：原策略止盈止损与月频申购细节

## 新增一个「一键适配」的标准步骤

1. **读源码**，划分：`OHLCV 可复现` / `需基本面` / `需 Tick` / `不可复现`
2. **写因子原语**（`factors/primitives/`，可单测）
3. **写 signal 或 PortfolioStrategy**：
   - 单标的 → `imported_strategies.py`
   - 组合 → `portfolio/jq_strategies.py` + `services/portfolio_backtest.py` 注册
4. **注册** `ADAPTED_STRATEGIES` / `ADAPTED_PORTFOLIO_STRATEGIES`，必填：
   - `id`、`name`、`source_strategy_id`（可多源 `source_strategy_ids`）
   - `reproducible_scope`、`non_reproducible_diff`
   - `default_params`、`run_preset`（`market` / `symbol` / `mode` / `symbols`）
5. **测试**：`tests/test_factors.py`、`tests/test_jq_adapters.py`
6. **前端**：`index.html` 下拉 + `js/backtest-strategy-guide.js` 算法说明
7. **changelog**：`docs/changelog/entries/` 追加一条

## 全自动适配（未来）

可对 Top 策略用 LLM 生成 `signal(df)` 草稿 + 人工 Review，但上线前必须：

- 因子原语有单元测试
- 明确 `non_reproducible_diff`
- 禁止直接 `exec` 聚宽源码（API 不兼容）
