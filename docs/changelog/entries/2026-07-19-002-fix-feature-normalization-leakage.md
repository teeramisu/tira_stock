# [fix] 特征归一化未来数据泄露修复

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：fix

## 背景

`create_sequences()` 在全局特征矩阵上做 z-score 归一化，但 trainer 在归一化后才做训练/验证切分。验证集使用了包含未来数据的均值和标准差，造成数据泄露。

## 变更

- `backend/ml/features.py`：从 `create_sequences()` 移除归一化，改为返回 raw features
- `backend/ml/features.py`：新增 `normalize_features()` 函数，支持传入 mean/std 以便验证集使用训练集的统计量
- `backend/ml/trainer.py`：重写 walk-forward 训练流程——每个 fold 分别在训练集上计算 mean/std，应用到验证集
- `backend/ml/trainer.py`：保存最佳 fold 的归一化统计量到 checkpoint（替代全局统计量）

## 相关文件

- `backend/ml/features.py`
- `backend/ml/trainer.py`

## 验证

- 验证集不再受未来数据影响
