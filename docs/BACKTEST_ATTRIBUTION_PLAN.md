# 回测收益/回撤归因与交易周期展示 — 实现规划

> 状态：**Phase 1 已完成**  
> 关联更新日志：[CHANGELOG.md](./CHANGELOG.md)  
> 目标读者：开发 / 产品 / 回测使用者

---

## 1. 背景与目标

当前全市场组合回测结果页已展示：组合级收益、回撤、胜率、净值曲线、交易记录、末期持仓、投资治理（Regime / 压力测试）、决策审计（单日调仓链路）。

**缺失能力：**

- 看不出**哪些个股贡献了最大收益**（Top 10）
- 看不出**哪些个股造成了最大回撤**（Top 10）
- 看不到单只股票的**完整交易周期**（买入 → 卖出 → 再买入 …）
- 标的展示仅有 **代码**，缺少 **名称**

**本期目标：**

在组合回测结果中新增「收益归因 / 回撤归因」模块，展示 Top 10 个股及其完整交易周期时间线，并统一展示 `代码 + 名称`。

---

## 2. 现状盘点

### 2.1 已有数据（可直接复用）

| 数据源 | 位置 | 字段要点 |
|--------|------|----------|
| 成交明细 `trades` | `PortfolioEngine` → `execute_portfolio_backtest` artifacts | `trade_date`, `symbol`, `side`, `filled_shares`, `fill_price`, `total_cost`, `status` |
| 持仓快照 `positions` | 同上 | 每日 `symbol`, `shares`, `market_value`, `weight` |
| 调仓日志 `rebalance_logs` | 同上 | `signal_date`, `trade_date`, `targets[]` |
| 证券主数据 `instruments` | DuckDB `instruments` 表 | `symbol`, `name`, `exchange`, `market` |
| 治理 artifacts | `research/portfolio_governance.py` | Regime / 压力测试 / 溯源（**不含**个股归因） |

### 2.2 缺口

| 缺口 | 说明 |
|------|------|
| 无个股 PnL 聚合 | `trades` 只有逐笔成交，未按 symbol 汇总已实现/未实现盈亏 |
| 无个股回撤序列 | `positions` 有每日市值，但未计算单票贡献的组合回撤 |
| 无 round-trip 周期 | 未将买卖配对为「持仓周期」 |
| 前端无名称 | `renderPortfolioResult` / `renderDecisionAudit` 仅展示 `symbol` |
| API 无归因 artifact | `governance_artifacts` 不含 `profit_attribution` / `drawdown_attribution` |

---

## 3. 指标定义（口径先固定，避免争议）

### 3.1 最大收益来源 Top 10

**定义：** 回测区间内，按**个股总贡献盈亏**降序排列的前 10 只。

**个股总贡献盈亏（推荐口径）：**

```
symbol_pnl = realized_pnl + unrealized_pnl_at_end
```

- `realized_pnl`：由已平仓 round-trip 累计（卖出收入 − 买入成本 − 费用）
- `unrealized_pnl_at_end`：回测末日仍持仓部分的浮盈浮亏（按末日收盘价/market_value 估算）

**排序字段：** `total_pnl`（金额）或 `total_return_pct`（相对该票投入成本），默认按 **金额** 排序。

**展示字段：**

| 字段 | 说明 |
|------|------|
| `rank` | 1–10 |
| `symbol` | 代码 |
| `name` | 名称（来自 `instruments.name`，缺失时回退为代码） |
| `total_pnl` | 总盈亏金额 |
| `total_return_pct` | 总收益率 |
| `realized_pnl` | 已实现盈亏 |
| `unrealized_pnl` | 期末浮盈浮亏 |
| `trade_count` | 成交笔数（filled） |
| `holding_days` | 累计持仓交易日 |
| `cycles` | 交易周期列表（见 3.3） |

### 3.2 最大回撤来源 Top 10

**定义：** 回测区间内，按**个股对组合回撤的贡献**降序排列的前 10 只。

**推荐口径（MVP）：**

1. 由 `positions` 构建每只股票的**每日市值序列** `mv_t(symbol)`
2. 计算组合每日权益 `equity_t`
3. 单票每日贡献权重 `w_t = mv_t / equity_t`
4. 组合回撤序列 `dd_t = 1 - equity_t / peak_t`
5. 在组合**最大回撤窗口** `[peak_date, trough_date]` 内，估算单票贡献：

```
symbol_dd_contrib = max_over_window( w_t * dd_t )  或
symbol_dd_contrib = trough_mv - peak_mv  （更简单、可解释）
```

**MVP 采用「峰值到谷底市值变化」口径**（实现简单、用户易理解）：

- 在组合最大回撤区间内，取该票持仓市值从峰值到谷底的变化量
- 仅统计回撤窗口内**有持仓**的标的
- 按 `drawdown_contribution`（负值越大 = 拖累越多）排序，取绝对值 Top 10

**展示字段：** 同 3.1，额外含 `drawdown_contribution`, `drawdown_window_start`, `drawdown_window_end`, `peak_date`, `trough_date`。

### 3.3 交易周期（Round-trip Cycle）

**定义：** 同一 `symbol` 从**净持仓 0 → 建仓 → … → 净持仓回到 0** 为一个周期；若回测结束时仍持仓，则最后一个周期标记为 `open`。

**周期结构：**

```json
{
  "cycle_id": 1,
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "status": "closed",
  "entry_date": "2025-03-10",
  "exit_date": "2025-06-15",
  "holding_days": 68,
  "events": [
    {"date": "2025-03-10", "side": "buy",  "shares": 100, "price": 172.5, "cost": 5.2},
    {"date": "2025-04-01", "side": "buy",  "shares": 50,  "price": 168.0, "cost": 2.6},
    {"date": "2025-06-15", "side": "sell", "shares": 150, "price": 195.0, "cost": 5.8}
  ],
  "realized_pnl": 3120.4,
  "return_pct": 0.124
}
```

**事件来源：** 仅 `status=filled` 的 `trades`，按 `trade_date` 排序后做 FIFO 配对。

---

## 4. Artifact 设计

新增治理子 artifact（版本化，便于前端稳定解析）：

```json
{
  "artifact_type": "symbol_attribution_v1",
  "market": "us",
  "sample_start": "2025-01-10",
  "sample_end": "2026-05-11",
  "portfolio_drawdown": {
    "max_drawdown": 0.18,
    "peak_date": "2025-10-12",
    "trough_date": "2025-12-03"
  },
  "top_profit_contributors": [ /* Top 10, 含 cycles */ ],
  "top_drawdown_contributors": [ /* Top 10, 含 cycles */ ],
  "name_lookup_misses": ["SYM1"]
}
```

**落盘位置：**

- 写入 `governance_artifacts.symbol_attribution`
- 同步进 `data/backtests/{request_hash}.json` 快照
- 可通过现有 `GET /api/backtest/{run_id}/artifacts?artifact=symbol_attribution` 分页读取（若 rows 过大）

---

## 5. 后端实现方案

### 5.1 新模块

```
backend/research/symbol_attribution.py
├── build_trade_cycles(trades: pd.DataFrame) -> dict[str, list[Cycle]]
├── compute_symbol_pnl(trades, positions, end_date) -> pd.DataFrame
├── rank_top_profit_contributors(...) -> list[dict]
├── rank_top_drawdown_contributors(equity_curve, positions) -> list[dict]
├── enrich_with_instrument_names(repo, market, rows) -> list[dict]
└── build_symbol_attribution_artifact(...) -> dict
```

### 5.2 接入点

在 `services/portfolio_backtest.py` 的 `execute_portfolio_backtest()` 中，`build_portfolio_governance_artifacts()` 之后调用：

```python
symbol_attribution = build_symbol_attribution_artifact(
    trades=result.trades,
    positions=result.positions,
    equity_curve=result.equity_curve,
    repository=repo,
    market=req.market,
    ...
)
governance_artifacts["symbol_attribution"] = symbol_attribution
```

### 5.3 名称解析

- 主路径：`instruments` 表 `market + symbol → name`
- 回退：名称缺失时显示 `symbol`，并在 `name_lookup_misses` 记录
- 不实时调用 AkShare/yfinance（避免回测接口变慢）

### 5.4 性能注意

- 全市场 5000+ 标的、300+ 交易日：归因计算应在引擎结束后**一次性**向量化/in-memory 完成
- 仅 Top 10 附带完整 `cycles`；其余标的只保留汇总行（减小 JSON 体积）
- 目标：在现有 5000×5y 基准（~8s）上，归因模块增量 **< 2s**

---

## 6. 前端实现方案

### 6.1 展示位置

在「概览」Tab 的 `portfolioOverviewExtra` 区域新增两块（或新增子 Tab「归因分析」）：

```
┌─────────────────────────────────────────────┐
│ 最大收益来源 Top 10                          │
│ ┌──────┬────────┬────────┬────────┬────────┐ │
│ │ 排名 │ 代码   │ 名称   │ 总盈亏 │ 收益率 │ │
│ └──────┴────────┴────────┴────────┴────────┘ │
│ ▶ AAPL · Apple Inc.  展开交易周期              │
│   2025-03-10 买入 100股 @172.5               │
│   2025-06-15 卖出 150股 @195.0  盈亏 +3120   │
├─────────────────────────────────────────────┤
│ 最大回撤来源 Top 10                          │
│ （同上结构，附回撤窗口与贡献额）              │
└─────────────────────────────────────────────┘
```

### 6.2 交互

- 默认折叠周期详情，点击行展开 `events` 时间线
- 收益 Top 10 行点击可高亮净值图中对应区间（可选，Phase 2）
- 决策审计 Tab 的标的列表同步显示 `symbol · name`

### 6.3 改动文件

| 文件 | 改动 |
|------|------|
| `frontend/js/backtest-workspace.js` | 新增 `renderSymbolAttribution()` |
| `frontend/index.html` | 概览区 DOM 容器 + 样式 |
| `frontend/index.html` | `renderPortfolioResult` 调用归因渲染 |

---

## 7. 分阶段交付

### Phase 1 — MVP（优先）

- [x] `symbol_attribution.py` 核心计算 + 单元测试
- [x] 接入 `execute_portfolio_backtest` 响应
- [x] 前端 Top 10 收益 / 回撤表格 + 周期时间线
- [x] 名称从 `instruments` 表补齐
- [x] 文档与 CHANGELOG 更新

**验收：**

- 组合回测有成交时，Top 10 非空且含 `name`
- 单票多笔买卖可正确拆分为多个 cycle
- 空仓回测时展示空状态提示（非报错）

### Phase 2 — 体验增强

- [ ] 归因区与决策审计联动（点击 Top 票跳转到其调仓日）
- [ ] 图表标注（个股持仓区间 shading）
- [ ] artifacts 分页 API 支持 `symbol_attribution`
- [ ] 单标的回测模式复用同一套 cycle 展示

### Phase 3 — 治理扩展

- [ ] Regime 维度下的 Top 贡献票切片
- [ ] 导出 CSV / 下载归因报告
- [ ] 多策略组合 (`portfolio_multi`) 按子策略拆分归因

---

## 8. 测试计划

| 类型 | 用例 |
|------|------|
| 单元测试 | 单次买卖 cycle；加仓后一次卖出；多次往返；期末仍持仓 `open` cycle |
| 单元测试 | Top 10 排序稳定性；同盈亏 tie-break 按 symbol |
| 单元测试 | 名称 lookup 命中 / 缺失回退 |
| 集成测试 | `POST /api/backtest/portfolio` 响应含 `governance_artifacts.symbol_attribution` |
| 回归测试 | 现有 `test_portfolio_api.py` 不破坏 |
| 手工 | 前端展开周期、中文名称显示、空仓/单票/全市场三种场景 |

---

## 9. 风险与待定项

| 项 | 说明 | 建议 |
|----|------|------|
| 回撤归因口径 | 多种算法结果不同 | MVP 用「市值峰值→谷底变化」，文档写清 |
| 美股名称缺失 | `instruments.name` 可能等于 symbol | 刷新脚本后续补全 `name` 字段 |
| JSON 体积 | 全市场 trades 很多 | Top 10 才附 `cycles` 全量 |
| 单标的回测 | 当前为组合引擎专属 | Phase 2 评估是否复用 |

---

## 10. 任务拆解（开发 checklist）

```
后端
  [ ] research/symbol_attribution.py
  [ ] research/portfolio_governance.py 接入
  [ ] services/portfolio_backtest.py 写入 artifacts
  [ ] tests/test_symbol_attribution.py
  [ ] tests/test_portfolio_api.py 补充断言

前端
  [ ] backtest-workspace.js — renderSymbolAttribution
  [ ] index.html — 概览区布局
  [ ] 决策审计 / 交易记录 — symbol · name

文档
  [ ] CHANGELOG.md 记录各 Phase 完成项
  [ ] README.md 补充「归因分析」一句话说明（Phase 1 完成后）
```

---

*文档版本：v0.1 · 2026-06-18*
