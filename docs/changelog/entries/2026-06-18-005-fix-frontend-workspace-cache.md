# [fix] 前端 BacktestWorkspace 脚本缓存导致归因面板报错

- **日期**：2026-06-18
- **版本**：v0.2.0（未发布）
- **类型**：fix

## 背景

组合回测成功后报错：`BacktestWorkspace.renderSymbolAttribution is not a function`。原因为 `index.html` 已更新而浏览器缓存旧版 `backtest-workspace.js`。

## 变更

- 移除 `renderPortfolioDetails` 中对 `renderSymbolAttribution` 的重复调用（统一由 `afterPortfolioResult` 渲染）
- `backend/app/main.py`：首页注入 `__WORKSPACE_ASSET_VER__`（按 JS 文件 mtime）；`/assets` 与 `/` 返回 `Cache-Control: no-cache`
- `index.html`：meta no-cache；CSS/JS 带动态版本参数
- `afterPortfolioResult` 外包 try/catch，缓存异常时提示硬刷新

## 相关文件

- `frontend/index.html`
- `backend/app/main.py`
- `frontend/js/backtest-workspace.js`

## 验证

1. 重启 `./run.sh`
2. Ctrl+Shift+R 硬刷新
3. Network 中 `backtest-workspace.js?v=<mtime>` 且响应含 `renderSymbolAttribution`
