# [fix] 交易成本默认值修复

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：fix

## 背景

`PortfolioBacktestRequest` 中 `slippage_bps` 和 `market_impact_bps` 默认值为 0，导致回测假设零市场冲击，收益被系统性高估。

## 变更

- `backend/services/portfolio_backtest.py:71-72`：默认 `slippage_bps=5.0`（5 bps），`market_impact_bps=1.0`（1 bps）

## 相关文件

- `backend/services/portfolio_backtest.py`

## 验证

- 新的默认值更贴近 A 股实际交易成本
