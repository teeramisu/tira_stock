# [feat] jq-0002 价值投资大盘择时一键适配

- **日期**：2026-06-18
- **版本**：v0.2.0（未发布）
- **类型**：feat

## 背景

用户希望「54.价值投资策略-大盘择时」(jq-0002) 能像已适配策略一样一键回测。

## 变更

- 抽取原策略 `RiskControl.check_for_benchmark` → `jq_value_ma_rsi_timing_signal`
- 注册 `source_strategy_id: jq-0002`，推荐 `510300` 单标的择时回测
- 策略库「回测」按钮：填参 + 自动点击运行回测
- 文档：`docs/JOINQUANT_ONE_CLICK_ADAPT.md`

## 未包含

原策略基本面选股（`get_fundamentals`）与多标的周频调仓，需后续财务数据 + 组合策略。

## 验证

```bash
pytest backend/tests/test_engine.py backend/tests/test_strategy_library.py -q
# 策略库搜 jq-0002 → 应显示「已适配」→ 点回测
```
