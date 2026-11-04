# [docs] 建立 changelog 文件夹与版本规范

- **日期**：2026-06-18
- **版本**：v0.2.0（未发布）
- **类型**：docs

## 变更

- 新增 `docs/changelog/` 目录：`README.md`、版本汇总 `v0.1.0.md` / `v0.2.0.md`、`entries/` 逐条记录
- 更新根 `docs/CHANGELOG.md` 为索引，指向文件夹内条目
- 约定：每次修改写 `entries/`；大改动开新版本文件

## 相关文件

- `docs/changelog/README.md`
- `docs/changelog/v0.1.0.md`
- `docs/changelog/v0.2.0.md`
- `docs/changelog/entries/*.md`
- `docs/CHANGELOG.md`

## 验证

浏览 `docs/changelog/v0.2.0.md` 应能链接到全部 2026-06-18 条目。
