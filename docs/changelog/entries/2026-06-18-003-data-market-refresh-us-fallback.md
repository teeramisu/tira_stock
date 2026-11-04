# [data] 行情刷新进展与美股 symbol 列表兜底

- **日期**：2026-06-18
- **版本**：v0.2.0（未发布）
- **类型**：data

## 背景

A 股增量全量刷新受网络/代理影响部分失败；美股 AkShare 全量 spot 接口失败时需能续跑。

## 变更

- A 股：生成 stable `akshare-cn-20260618075935`（至 2026-06-18），成功率约 49%
- 美股：全量列表失败时，`scripts/refresh_market_data.py` 从本地 DB `instruments` 读取 US symbol 列表兜底
- 美股 stable 仍可能为旧版 `us-auto-us-20260512090533`（视刷新是否完成）

## 相关文件

- `scripts/refresh_market_data.py`
- `data/market/market_data.duckdb`
- `.cursor/skills/refresh-market-data/SKILL.md`

## 验证

```bash
python scripts/refresh_market_data.py --market cn --mode incremental --full-fetch
python scripts/refresh_market_data.py --market us --mode incremental --full-fetch
```

## 后续

- A 股：`--retry-failed-only`
- 美股：确认新 stable 版本落库
