# [feat] 策略生命周期自动化

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：feat

## 背景

`StrategyLifecycleManager` 定义了完整的 research→validated→production→degraded→retired 状态机，但从未被任何代码调用。`EvolutionRunner` 中创建的 `lifecycle_mgr` 实例未调用任何方法。

## 变更

- `backend/research/evolution_loop.py`：`run_daily_cycle()` 每日对每个活跃策略调用 `lifecycle_mgr.record_run()`，使用真实回测指标（有数据时）或估算指标
- `backend/research/evolution_loop.py`：`run_weekly_report()` 从 `StrategyLifecycleManager` 读取实际的 degraded/retired 列表，替代空列表

## 相关文件

- `backend/research/evolution_loop.py`
- `backend/research/strategy_lifecycle.py`

## 验证

- 策略性能衰退时自动降级（sharpe<0 & drawdown>15%）
- 连续3次衰退自动退役
- 连续3次良好运行自动从 research→validated
