# [test] v0.3.0 修复针对性测试覆盖

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：test

## 变更

新增 12 个针对性测试，覆盖每项修复：

| 测试 | 验证内容 |
|------|---------|
| `test_ma_windows_use_correct_lengths` | MA 窗口长度正确（非旧版 15/25 bug） |
| `test_regime_vol_window_excludes_current_day` | Vol 计算不含当天 |
| `test_allocate_capital_supports_all_methods` | 5 种分配方法均可用 |
| `test_meta_allocator_risk_parity_prefers_low_vol` | 风险平价优先低波动 |
| `test_kelly_allocator_gives_higher_weight_to_better_win_rate` | Kelly 优先高胜率 |
| `test_score_weighted_targets_differs_from_equal` | 权重按 score 排序 |
| `test_score_weighted_targets_cash_fallback` | 全负分时空仓 |
| `test_score_weighted_targets_no_cash_fallback` | 关闭回退时返回负分 |
| `test_features_no_forward_normalization` | create_sequences 不归一化 |
| `test_multi_strategy_sleeve_uses_sub_allocated_cash` | Sleeve 资金正确拆分 |
| `test_slippage_not_double_counted_in_total_cost` | total_cost 不含滑点 |
| `test_cost_defaults_nonzero` | 成本默认值 > 0 |
| `test_sector_diversification_caps_single_sector` | 行业分散约束生效 |
| `test_sector_map_loading` | 行业映射正确加载 |
| `test_transformer_causal_mask_prevents_future_attendance` | 因果掩码不报错 |

## 相关文件

- `backend/tests/test_research.py`
- `backend/tests/test_portfolio_engine.py`
- `backend/tests/test_portfolio_strategies.py`

## 验证

- `306 passed`（不含 2 个环境锁定的 DuckDB 测试）
