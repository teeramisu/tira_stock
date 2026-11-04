# [feat] Walk-Forward 输出自动应用到策略参数

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：feat

## 背景

Walk-Forward 优化器计算出 `best_overall_params` 后只返回给 API 调用者，从未被策略构造器消费。每次回测仍使用硬编码默认参数。

## 变更

- `backend/research/walk_forward.py`：新增 `WF_PARAMS_DIR` 目录和 `_save_best_params()` / `load_optimized_params()`，优化完成后自动持久化到 `data/walk_forward_params/{strategy_id}.json`
- `backend/portfolio/strategy_registry.py`：`build_portfolio_strategy()` 自动检查 WF 参数缓存，存在时覆盖请求参数

## 相关文件

- `backend/research/walk_forward.py`
- `backend/portfolio/strategy_registry.py`

## 验证

- 运行一次 WF 优化后，同策略回测自动使用优化参数
- 缓存不存在时回退到请求默认值
