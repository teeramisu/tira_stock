# [feat] 行业分散约束

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：feat

## 背景

所有 TopK 策略可能选中同一行业的股票，在行业崩盘时组合承受不成比例的回撤。

## 变更

- `backend/portfolio/engine.py`：新增 `max_sector_pct` 参数（默认 30%），限制单行业最大权重
- `backend/portfolio/engine.py`：新增 `_load_sector_map()` 从 `board_constituents.json` 加载 A 股行业映射
- 在 Regime 仓位调节后、Order 下达前，按行业压缩超限权重

## 相关文件

- `backend/portfolio/engine.py`

## 验证

- 行业权重超过 30% 时自动压缩并归一化
