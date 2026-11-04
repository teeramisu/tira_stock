# [fix] Evolution Loop 伪数据修复

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：fix

## 背景

`EvolutionRunner.run_weekly_report()` 中的策略指标（Sharpe/Return/MaxDrawdown）是硬编码的伪造数据：
```python
"sharpe_ratio": 0.5 if ev["regime_aligned"] else -0.2,
"total_return": 0.05 if ev["regime_aligned"] else -0.03,
```
整个"自进化"循环从未真正评估过策略表现。

## 变更

- `backend/research/evolution_loop.py`：新增 `_load_latest_backtest_metrics()` 优先从 `data/backtests/` 加载真实回测结果
- `backend/research/evolution_loop.py`：新增 `_run_quick_evals()` 作为备选——对活跃策略运行轻量回测
- 仅在两者均失败时回退到 regime 对齐估算

## 相关文件

- `backend/research/evolution_loop.py`

## 验证

- weekly report 中的策略指标来自真实回测
- 每次周报触发时自动运行 180 天快速回测
