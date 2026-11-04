---
name: tira-mode
description: >-
  tira_stock 的工程流程路由器：把任务归类到 feature / bugfix / refactor / ci-change
  四个 playbook 之一，把步骤原样抄进待办，并要求每条结论附带证据（命令 + 输出 + 未覆盖部分）。
  适用于任何非平凡改动；开始新任务、或不确定该走哪个流程时调用。
license: MIT
compatibility: opencode
metadata:
  version: v1.0.0
  repo: tira_stock
---

# tira-mode

## 何时使用

任何**非平凡**改动都用它开场：

- 「加一个功能」「修一个 bug」「重构某模块」「改 CI」
- 不确定该怎么做、想避免「改着改着跑偏」时

纯问答、看代码、写文档草稿不必进 mode。

## 工作方式（严格按顺序）

1. **读原则**：`references/principles.md` 全文，不要跳。
2. **归类**：从下表选**一个** playbook。选不出来 → 先问一个澄清问题，不要猜。
3. **抄步骤**：打开对应 playbook，把它的步骤**原样**复制进你的待办列表。
   - 不允许「概括成更短的流程」；
   - 跳过的步骤必须保留并写明原因（例如「跳过：本次不涉及前端」）。
4. **执行**：逐步做，每步结束记录证据。
5. **收口**：汇报时必须包含「证据」与「未覆盖」两节。

## Playbook 选择表

| 任务形态 | playbook |
|---|---|
| 新增行为 / 新接口 / 新指标 | `playbooks/feature.md` |
| 现有行为错了（报错、结果不对、崩溃） | `playbooks/bugfix.md` |
| 行为不变，只改结构（拆分、改名、抽函数） | `playbooks/refactor.md` |
| 改 CI、脚本、门禁、依赖、仓库配置 | `playbooks/ci-change.md` |

模糊时的判断顺序：**先问「行为变不变」→ 变则 feature/bugfix，不变则 refactor；碰 `.github/`、`scripts/ci/`、`requirements*` 则一律 ci-change。**

## 硬性约束（来自 CI，绕不过去）

| 约束 | 触发者 |
|---|---|
| 一个 PR 只碰一个功能区，代码与 CI 必须分开 | `pr-scope` |
| 软上限 40 文件 / 1500 行，硬上限 120 / 4000 | `pr-size` |
| 描述必须含 背景 / 改动 / 验证（feat、refactor、perf 建议加 回退） | `pr-meta` |
| 新增代码必须符合 Google 风格（只判新增行） | `style` |
| 覆盖率不得下降 | `coverage` ratchet |
| 改测试/CI/依赖需人工放行（纯新增测试除外） | `ci-guard` |
| 提交信息与分支名规范 | `commits` |

**不要在 PR 里顺带修另一个问题。** 发现了就记下来，另开任务。

## 证据格式（每次汇报都要有）

```markdown
## 证据
- `scripts/ci/verify.sh --full` → 全绿（附关键输出 3~5 行）
- CI：<run 链接> → CI 汇总 success
- 手动：<命令> → <结果>

## 未覆盖
- 没测的部分、没跑的路径、已知限制
```

没有第三行「未覆盖」，就等于没做完 —— 但这不代表任务失败，而是**诚实边界**。

## 反模式（出现即停下重来）

| 反模式 | 后果 |
|---|---|
| 把 playbook 压缩成三步 | 静默跳过验证 |
| 「顺手」重构无关代码 | PR 范围超限，回滚困难 |
| 为了让 CI 变绿去改测试/放宽规则 | 守卫拦截，且是质量问题 |
| 只说「已修复」不给命令与输出 | 无法复核，等同没做 |
| 反复重试同一个失败命令 | 见 `tira-verify` 的失败分类 |
