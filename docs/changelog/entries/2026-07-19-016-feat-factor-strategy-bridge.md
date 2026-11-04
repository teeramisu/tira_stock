# [feat] 因子→策略自动桥梁 + 前端因子扫描看板

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：feat

## 背景

因子扫描结果无人消费——强 IC 因子不会自动变成可交易策略。前端也无处查看因子扫描结果。

## 变更

### 因子→策略桥梁
- `backend/research/factor_scan.py`：新增 `_compute_strategy_recommendations()`，持续 3 期 Rank IC > 0.03 的因子自动映射到对应策略类型（momentum_top_k/mean_reversion_top_k/risk_adj_momentum_top_k/multi_factor_top_k 等）
- 策略推荐进入 `run_monthly_scan()` 输出，随月度报告一起呈现

### API 端点
- `POST /api/research/factor/scan` — 触发全因子扫描，返回 top 因子 + 新兴/衰减信号 + 策略推荐
- `GET /api/research/factor/history` — 查询扫描历史趋势
- `GET /api/research/factor/recommendations` — 查询因子派生策略推荐

### 前端
- `frontend/index.html`：新增因子扫描卡片（#evoFactorScanCard），含新兴信号、衰减警告、策略推荐三个子面板
- 新增「📡 因子扫描」按钮，关联到 `/api/research/factor/scan`

## 相关文件

- `backend/research/factor_scan.py`
- `backend/app/routers/backtest.py`
- `frontend/index.html`

## 验证

- 309 测试通过
- 前端点击因子扫描按钮可查看实时结果
