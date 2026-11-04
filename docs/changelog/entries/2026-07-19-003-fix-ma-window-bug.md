# [fix] _judge_more_average MA 窗口错误修复

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：fix

## 背景

`_judge_more_average()` 函数中 `ma20` 实际只用了 15 个数据点，`ma30` 只用了 25 个点。变量名与实现不一致，导致 MA 信号失真。

## 变更

- `backend/portfolio/strategies.py:156-164`：修复窗口大小：ma20→20点，ma30→30点；最低数据要求从 25 提升到 30

## 相关文件

- `backend/portfolio/strategies.py`

## 验证

- MA 窗口与命名一致
