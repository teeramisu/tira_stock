# [feat] 回测个股归因 Phase 1

- **日期**：2026-06-18
- **版本**：v0.2.0（未发布）
- **类型**：feat

## 背景

组合回测结果缺少「哪只股票贡献收益/回撤」的可解释性，需在治理 artifacts 中落地标的级归因。

## 变更

- 新增 `backend/research/symbol_attribution.py`，产出 `governance_artifacts.symbol_attribution`（schema `symbol_attribution_v1`）
- 能力：最大收益 Top 10、最大回撤 Top 10、买卖周期时间线、标的代码+名称映射
- 接入 `backend/services/portfolio_backtest.py`
- 前端 `backtest-workspace.js`：`renderSymbolAttribution`、概览区 `#symbolAttributionPanel`
- 交易/持仓/决策审计展示「代码 · 名称」

## 相关文件

- `backend/research/symbol_attribution.py`
- `backend/services/portfolio_backtest.py`
- `frontend/js/backtest-workspace.js`
- `frontend/index.html`
- `backend/tests/test_symbol_attribution.py`
- `docs/BACKTEST_ATTRIBUTION_PLAN.md`

## 验证

```bash
pytest backend/tests/test_symbol_attribution.py backend/tests/test_portfolio_api.py -q
```
