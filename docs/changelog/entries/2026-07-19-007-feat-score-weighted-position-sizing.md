# [feat] 策略改进：置信度加权 + 现金回退

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：feat

## 背景

所有 TopK 策略使用等权分配——排名第 1 和排名第 10 的股票仓位相同。并且没有任何策略在市场无机会时退回现金。

## 变更

- `backend/portfolio/strategies.py`：新增 `_score_weighted_targets()` 函数——根据 score 进行置信度加权（shifted softmax 风格），并支持现金回退
- `backend/portfolio/strategies.py`：MomentumTopK、MeanReversionTopK、MultiFactorTopK、RiskAdjustedMomentum、JqValueTimingTopK 均改为 score-weighted
- `backend/portfolio/strategies.py`：EtfRotationTop1 当最佳 ETF 收益 ≤ 0 时退回现金
- `backend/portfolio/engine.py`：`_momentum_targets_fast()` 快路径也改为 score-weighted

## 相关文件

- `backend/portfolio/strategies.py`
- `backend/portfolio/engine.py`

## 验证

- 高得分股票获得更大仓位
- 全市场无正收益标的时自动空仓
