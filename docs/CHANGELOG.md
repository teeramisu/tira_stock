# 更新日志（CHANGELOG）

本文件为**版本索引**；逐条变更见 **[changelog/](./changelog/)** 目录。

| 版本 | 状态 | 说明 |
|------|------|------|
| [v0.3.0](./changelog/v0.3.0.md) | **开发中** | 数据泄露修复、Meta-Allocator 多方法、策略置信度加权、Evolution 真实数据 |
| [v0.2.0](./changelog/v0.2.0.md) | 冻结 | 归因 Phase 1、美股修复、回测进度/取消、前端缓存 |
| [v0.1.0](./changelog/v0.1.0.md) | 已发布（基线） | 全市场组合回测 MVP |

维护规范：[changelog/README.md](./changelog/README.md)

类型标签：`feat` · `fix` · `perf` · `data` · `docs` · `plan` · `ui`

---

## 未发布 — [v0.3.0](./changelog/v0.3.0.md)

> 修复关键数据泄露 + 架构断裂 + 策略改进，为产生正收益打下基础

| # | 类型 | 摘要 |
|---|------|------|
| 001 | fix | [Transformer 因果掩码修复](./changelog/entries/2026-07-19-001-fix-transformer-causal-mask.md) |
| 002 | fix | [特征归一化未来数据泄露修复](./changelog/entries/2026-07-19-002-fix-feature-normalization-leakage.md) |
| 003 | fix | [_judge_more_average MA 窗口错误修复](./changelog/entries/2026-07-19-003-fix-ma-window-bug.md) |
| 004 | fix | [滑点报告重复计算修复](./changelog/entries/2026-07-19-004-fix-slippage-double-count.md) |
| 005 | feat | [Meta-Allocator 多方法支持](./changelog/entries/2026-07-19-005-feat-meta-allocator-multi-method.md) |
| 006 | fix | [Evolution Loop 伪数据修复](./changelog/entries/2026-07-19-006-fix-evolution-loop-fake-data.md) |
| 007 | feat | [策略改进：置信度加权 + 现金回退](./changelog/entries/2026-07-19-007-feat-score-weighted-position-sizing.md) |
| 008 | fix | [Regime/Features vol 窗口泄露修复](./changelog/entries/2026-07-19-008-fix-vol-window-leakage.md) |
| 009 | fix | [交易成本默认值修复](./changelog/entries/2026-07-19-009-fix-cost-defaults.md) |
| 010 | feat | [行业分散约束](./changelog/entries/2026-07-19-010-feat-sector-diversification.md) |
| 011 | test | [v0.3.0 针对性测试覆盖（15项）](./changelog/entries/2026-07-19-011-test-coverage.md) |
| 012 | feat | [Walk-Forward 输出自动应用到策略参数](./changelog/entries/2026-07-19-012-feat-walk-forward-to-engine.md) |
| 013 | feat | [AI Agent 产出自动因子回测](./changelog/entries/2026-07-19-013-feat-ai-agent-auto-backtest.md) |
| 014 | feat | [策略生命周期自动化](./changelog/entries/2026-07-19-014-feat-lifecycle-automation.md) |
| 015 | feat | [因子扫描 Harness + 月度全因子扫描](./changelog/entries/2026-07-19-015-feat-factor-scan-harness.md) |

---

## v0.1.0 — 2026-05-06 ~ 2026-05-30

全市场组合回测 MVP：行情库、Universe、组合引擎、治理 artifacts、多策略分配、前端工作台。

- 版本汇总：[changelog/v0.1.0.md](./changelog/v0.1.0.md)
- 细粒度流水：[FULL_MARKET_BACKTEST_IMPLEMENTATION_LOG.md](./FULL_MARKET_BACKTEST_IMPLEMENTATION_LOG.md)

---

## 2026-06-18（已并入 v0.2.0）

以下条目已迁移至 `changelog/entries/`，此处保留日期锚点便于检索。

- 美股 exchange 修复、A 股/美股刷新、归因 MVP、前端 UX — 见 v0.2.0 上表

---

## 如何追加一条修改

1. 在 `docs/changelog/entries/` 新建 `YYYY-MM-DD-NNN-<type>-<slug>.md`（模板见 [changelog/README.md](./changelog/README.md)）
2. 在 [changelog/v0.x.x.md](./changelog/) 对应版本表加一行
3. 更新本文件「未发布」表格
4. **大改动**：新建 `vX.Y.Z.md` 并在上表增加版本行

### 何时升主版本

API 不兼容、回测语义变更、数据 schema 破坏性升级 → 递增 **X**（如 v1.0.0）。

功能里程碑、新模块 → 递增 **Y**（如 v0.2.0 → v0.3.0）。

小修复 → 递增 **Z** 或仅追加 entry。

---

*超长实现细节（2026-05 组合引擎优化）仍见 [FULL_MARKET_BACKTEST_IMPLEMENTATION_LOG.md](./FULL_MARKET_BACKTEST_IMPLEMENTATION_LOG.md)。*
