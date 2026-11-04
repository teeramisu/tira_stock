# [fix] Regime detector + Features vol 窗口泄露修复

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：fix

## 背景

Regime detector 和 ML features 的 realized volatility 计算使用 `i - window + 1 : i + 1` 包含当天收盘价，造成微小但正式的前瞻偏差。

## 变更

- `backend/research/regime_detector.py:153`：`i - trend_period : i + 1` → `i - trend_period : i`
- `backend/ml/features.py:162`：`i - vol_window + 1 : i + 1` → `i - vol_window : i`

## 相关文件

- `backend/research/regime_detector.py`
- `backend/ml/features.py`

## 验证

- vol 计算只用到 T-1 及之前的数据
