# [fix] 美股组合回测股票池为空（收益恒为 0%）

- **日期**：2026-06-18
- **版本**：v0.2.0（未发布）
- **类型**：fix

## 背景

美股 `instruments.exchange` 大量为 `UNKNOWN`，默认 `NYSE/NASDAQ/AMEX` 过滤后 `universe_size=0`，组合回测无成交。

## 变更

- `marketdata/universe.py` — 美股默认 `include_exchanges` 兼容 `UNKNOWN`
- `marketdata/sources/yfinance_source.py` — 刷新时改进交易所解析
- `scripts/repair_us_instruments.py` — 批量修复已有 DB 中的 exchange 字段
- 组合引擎 / API 接入 `effective_include_exchanges()`

## 相关文件

- `backend/marketdata/universe.py`
- `backend/marketdata/sources/yfinance_source.py`
- `backend/portfolio/engine.py`
- `backend/services/portfolio_backtest.py`
- `backend/app/main.py`
- `scripts/repair_us_instruments.py`

## 验证

- 股票池由 0 → 约 17085
- 5 只样本组合回测产生成交记录
