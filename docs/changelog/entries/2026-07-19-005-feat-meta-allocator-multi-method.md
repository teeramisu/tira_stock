# [feat] Meta-Allocator 多方法支持

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：feat

## 背景

Meta-Allocator 模块定义了 5 种资金分配方法（等权/风险平价/动量加权/Regime自适应/Kelly），但 `governance.py` 的 `AllocatorMethod` 被硬编码为 `Literal["equal_weight"]`，`multi_strategy.py` 也拒绝非 equal_weight 的方法。5 种方法只有在 API 端点展示，实际回测从不使用。

## 变更

- `backend/research/governance.py`：`AllocatorMethod` 扩展为支持全部 5 种方法
- `backend/research/governance.py`：`allocate_capital()` 对非等权方法委托给 `MetaAllocator`，并接受 `strategy_metrics` 和 `regime` 参数
- `backend/research/multi_strategy.py`：`allocate_multi_strategy()` 和 `combine_strategy_sleeve_metrics()` 透传所有方法和额外参数
- `backend/services/multi_strategy_portfolio.py`：sleeve 回测使用子分配资金而非全额

## 相关文件

- `backend/research/governance.py`
- `backend/research/multi_strategy.py`
- `backend/services/multi_strategy_portfolio.py`

## 验证

- 可选择 risk_parity / momentum_weighted / regime_adaptive / kelly 分配方法
- 失败时自动回退到 equal_weight
