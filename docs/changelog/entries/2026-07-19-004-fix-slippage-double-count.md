# [fix] 滑点报告重复计算修复

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：fix

## 背景

`CostModel.calculate()` 中滑点同时体现在两个地方：(1) fill_price 已包含滑点调整 (2) `CostBreakdown.slippage_cost` 又被加到 `total_cost`。导致回测报告中总成本被高估。

## 变更

- `backend/portfolio/models.py:57-59`：`total_cost` 不再包含 `slippage_cost`（滑点已通过 fill_price 反映在实际资金流中）

## 相关文件

- `backend/portfolio/models.py`

## 验证

- total_cost 只含 commission + stamp_tax + transfer_fee
- slippage_cost 保留在 CostBreakdown 中作为独立报告字段
