# [fix] 组合回测流式 API 404 与回退

- **日期**：2026-06-18
- **版本**：v0.2.0（未发布）
- **类型**：fix

## 背景

前端改用 `/api/backtest/portfolio/stream`，旧版后端未重启时返回 404，页面显示「请求失败: Not Found」。

## 变更

- 前端 `fetchBacktestStream`：流式 URL 404 时自动回退到 `/api/backtest/portfolio`（或 multi 非 stream），并用模拟进度条
- 需重启 `./run.sh` 以加载 stream 路由

## 相关文件

- `frontend/index.html`

## 验证

```bash
curl -X POST http://127.0.0.1:8000/api/backtest/portfolio/stream -H 'Content-Type: application/json' -d '...'
# 应 200，非 404
```
