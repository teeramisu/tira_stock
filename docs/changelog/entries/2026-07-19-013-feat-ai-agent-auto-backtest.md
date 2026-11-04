# [feat] AI Agent 产出自动因子回测

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：feat

## 背景

AI Agent 生成因子假设后仅做启发式评估（新颖性/可实现性/经济学逻辑/Regime匹配），从未运行真实的因子 IC 回测。promising 的因子只是"猜"出来的。

## 变更

- `backend/factors/backtest.py`：新增 `compute_factor(spec)` 和 `evaluate_factor(name)` 公开方法，支持单因子 IC 回测
- `backend/research/ai_agent.py`：新增 `backtest_hypothesis()` 方法，将假设名与 `FACTOR_CATALOG` 匹配并运行完整 IC → Rank IC → Spread 回测
- `run_full_research_cycle()` 增加 `run_backtest=True` 参数，自动对所有 promising 假设执行量化回测

## 相关文件

- `backend/factors/backtest.py`
- `backend/research/ai_agent.py`

## 验证

- 规则模板生成的因子假设自动获得 IC/rank IC/IC_IR 指标
- Rank IC > 0.02 的因子标记为 actionable
