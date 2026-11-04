# 修改日志（Changelog）目录

本目录存放 **逐条变更记录** 与 **版本汇总**。根索引见 [../CHANGELOG.md](../CHANGELOG.md)。

## 目录结构

```
docs/changelog/
├── README.md           ← 本说明
├── v0.1.0.md           ← 已发布大版本汇总
├── v0.2.0.md           ← 当前开发版本汇总（未打 tag 前为 unreleased）
└── entries/            ← 每次修改一条（按日期 + 序号）
    ├── 2026-06-18-001-feat-symbol-attribution-phase1.md
    └── ...
```

## 何时写条目

**每次**完成以下任一操作，在 `entries/` 新增一条 Markdown：

- 新功能（`feat`）
- Bug 修复（`fix`）
- 性能优化（`perf`）
- 数据/行情变更（`data`）
- 文档/规划（`docs` / `plan`）
- 前端交互改进（`ui`）

## 条目命名

```
YYYY-MM-DD-NNN-<type>-<短描述>.md
```

- `NNN`：当日序号，从 `001` 起
- `type`：`feat` | `fix` | `perf` | `data` | `docs` | `plan` | `ui`

## 何时开新版本

在 `docs/changelog/` 新建 `vX.Y.Z.md`，并将 [../CHANGELOG.md](../CHANGELOG.md) 中对应区块从「未发布」迁入该版本。

| 版本位 | 建议场景 |
|--------|----------|
| **主版本 X** | 架构重构、API 不兼容、回测引擎语义变更、数据 schema 破坏性升级 |
| **次版本 Y** | 新模块/新 API/完整功能里程碑（如组合回测 MVP、归因 Phase 1、流式进度） |
| **修订 Z** | 单点修复、文案、小 UI、脚本补丁 |

当前约定：**v0.1.0** = 全市场组合回测 MVP 时代；**v0.2.0** = 归因 + 美股修复 + 回测 UX（进度/取消/缓存）。

## 条目模板

复制到 `entries/YYYY-MM-DD-NNN-type-slug.md`：

```markdown
# [type] 标题

- **日期**：YYYY-MM-DD
- **版本**：v0.x.x（未发布 / 已发布）
- **类型**：feat | fix | perf | data | docs | ui

## 背景

（可选）为什么要改

## 变更

- 要点 1
- 要点 2

## 相关文件

- `path/to/file`

## 验证

- 命令或手工步骤
```

## 维护流程

1. 改代码 → 在 `entries/` 新增一条
2. 在同版本的 `vX.Y.Z.md` 追加一行摘要（带链接到 entry）
3. 更新 [../CHANGELOG.md](../CHANGELOG.md) 的「未发布」或版本节
4. 打 Git tag 时：将「未发布」改为正式版本日期，并清空 unreleased 区块

## 与其它文档的关系

| 文档 | 用途 |
|------|------|
| [CHANGELOG.md](../CHANGELOG.md) | 对外索引，按版本/日期浏览 |
| `entries/*.md` | 单次改动的完整说明 |
| [FULL_MARKET_BACKTEST_IMPLEMENTATION_LOG.md](../FULL_MARKET_BACKTEST_IMPLEMENTATION_LOG.md) | 2026-05 前后组合引擎细粒度实现流水（历史） |
| `*_PLAN.md` | 功能规划，完成后在 entry 中链接 |
