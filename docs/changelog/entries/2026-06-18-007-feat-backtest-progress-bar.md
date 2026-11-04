# [feat] 回测进度条（组合真实进度 + 单股模拟）

- **日期**：2026-06-18
- **版本**：v0.2.0（未发布）
- **类型**：feat

## 背景

全市场组合回测耗时长，用户无法感知当前阶段与完成比例。

## 变更

### 后端

- `PortfolioEngine.run(on_progress=...)` — 按阶段与交易日推送进度（约 0–88% 模拟循环）
- `execute_portfolio_backtest(..., on_progress=...)` — 治理/归因阶段进度
- `iter_portfolio_backtest_stream` — NDJSON 流（progress / result / error）
- `execute_multi_strategy_portfolio` — 按 sleeve 聚合进度
- `iter_multi_strategy_portfolio_stream`
- 新 API：
  - `POST /api/backtest/portfolio/stream`
  - `POST /api/backtest/portfolio/multi/stream`

### 前端

- `#backtestProgressWrap` 进度条 UI
- 组合 / 多策略：消费 NDJSON 流更新真实进度
- 单股 / 自定义：模拟进度动画（请求较快）

## 相关文件

- `backend/portfolio/engine.py`
- `backend/services/portfolio_backtest.py`
- `backend/services/multi_strategy_portfolio.py`
- `backend/app/main.py`
- `frontend/index.html`

## 验证

1. 重启后端，硬刷新页面
2. 全市场组合回测：进度从「加载交易日历」→「模拟交易 N/M 日」→「生成治理指标」
3. 单股回测：显示模拟进度条
