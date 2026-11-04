# Agent 系统设计：Pstack / 本仓库现状 / 最优解

> 目标不是「让 agent 写更多代码」，而是**让它能证明自己写对了，且错了改不动判分标准**。

## 一、三方对照

| 维度 | Lauren Tan 的 Pstack | 本仓库现状（P0/P1 之后） | 本方案 |
|---|---|---|---|
| 入口 | `/poteto-mode` 路由器 | 人读 `CONTRIBUTING.md` | `/tira-mode` 路由 skill |
| 任务拆解 | 23 个 playbook，步骤原样抄进 todo | 无（靠人/agent 自由发挥） | 4 个 playbook（feature/bugfix/refactor/ci-change） |
| 行为准则 | 24 条单规则 principle skill | 散落在 CONTRIBUTING | 6 条原则，作为 mode 的 references |
| 强制层 | `/correct`：把重复错误变成 lint/类型/测试 | **已有**：CI 10 个 job + 守卫 + ratchet | 保留，作为唯一的「不可绕过」层 |
| 验证证据 | `/create-verification-skill` 生成项目级验证 skill | `scripts/ci/verify.sh`（10 步）+ CI | 保留 + 结构化失败摘要 |
| 并发 | `/arena` `/swarm` 多候选 | `subagent` 可用但未编排 | **暂不做**（理由见第五节） |
| PR 运维 | `/babysit` `/fix-ci` `/make-pr-easy-to-review` | `scripts/agent/auto_merge.py` | `scripts/agent/loop.py` 一条命令跑完一个 PR 周期 |
| 决策留痕 | `/show-me-your-work` 写 TSV | 无 | 写 `data/agent-log/*.md`（不入库） |
| 形式化 | TLA+/Lean/Kani | 无 | 暂不做 |

## 二、Pstack 真正值得抄的三件事

### 1. 路由器 + playbook，而不是一个巨型 prompt

一个 prompt 想覆盖「修 bug / 加功能 / 重构 / 改 CI」，必然每件事都写得很浅。
Pstack 的做法是：路由器只做**分类**，然后**把选中 playbook 的步骤原样复制到 todo**，
跳过的步骤必须写明原因。

我们采纳，但只做 4 条 playbook —— 这个仓库 90% 的改动落在这四类里。

### 2. 「证据」是动词，不是形容词

Pstack 的每条 principle 都要求 agent 说清「我跑了什么、输出是什么、还有什么没验」。
我们把它变成可执行的：`scripts/ci/verify.sh` + CI 链接 + 失败摘要。

### 3. `/correct`：同一个错误出现两次，就变成检查

这条我们已经在做，而且做得比提示词更硬：

| 重复出现的问题 | 我们落地成了什么 |
|---|---|
| 改了测试让 CI 变绿 | 受保护路径守卫（`ci-guard`） |
| 一次塞五件事 | `pr-scope`（功能区跨度） |
| 写不断言的测试 | 覆盖率 ratchet |
| 新代码不合风格 | Google 风格增量检查 |
| 提交信息随便写 | commit-msg hook + `commits` job |
| 偶发失败让 agent 空转 | 失败自动重跑一次，区分 flaky |

**这是本方案与 Pstack 最大的差异**：Pstack 的约束主要靠「模型读到并遵守」，
我们的约束主要靠「CI 拒绝」。两者互补，但**不可绕过的部分必须是机械的**。

## 三、本方案的分层

```
┌─ 知识层（skill，可被 agent 读，也可被人读）──────────────┐
│  /tira-mode   路由：选 playbook，抄步骤到 todo           │
│  playbooks/   feature / bugfix / refactor / ci-change    │
│  references/  6 条原则（证据、一件事、小步、可回退…）      │
│  /tira-verify 怎么跑、怎么读失败、什么算证据              │
│  /tira-pr     描述标准、标签、auto-merge、babysit         │
└───────────────────────┬────────────────────────────────┘
                        ↓ 执行
┌─ 执行层（脚本，幂等、可重试）───────────────────────────┐
│  scripts/ci/verify.sh      本地 10 步                     │
│  scripts/agent/loop.py     一个 PR 周期：验→提交→推→PR→合 │
│  scripts/agent/auto_merge.py                                │
└───────────────────────┬────────────────────────────────┘
                        ↓ 判定
┌─ 强制层（CI + ruleset，agent 改不动）───────────────────┐
│  10 个 job + CI 汇总 + 守卫 + 保护路径 + merge queue      │
└─────────────────────────────────────────────────────────┘
```

关键约束：**知识层可以改，强制层不能由 agent 改**。
`scripts/ci/**`、`.github/**`、`backend/tests/**` 都在守卫范围内。

## 四、执行循环（`scripts/agent/loop.py`）

```bash
python scripts/agent/loop.py --task "feat(macro): 增加省级排名变化" --type feat
```

```
1. 预检    分支名合规 / 工作区干净 / 无待提交敏感文件
2. 自检    scripts/ci/verify.sh --full          ← 本地先绿
3. 提交    Conventional Commits（hook 再校验一次）
4. 推送    失败自动重试（本机 SSH 间歇性拒连）
5. 开 PR   标题 + 四段描述（背景/改动/验证/回退）
6. 合并    auto_merge.py --wait，队列绿则自动合
7. 失败    拉失败 job 日志 → 生成「可修摘要」→ 退出码 2，交回 agent
8. 留痕    data/agent-log/<date>.md 记录本次决策与证据
```

与 Pstack 的 `/babysit` 对应，但只做**一个周期**：多轮修复由 agent 自己驱动，
脚本不做「猜代码怎么改」这种不可靠的事。

## 五、明确不做的部分（及理由）

| Pstack 有 | 为什么不做 |
|---|---|
| `/arena` 多候选并行 | 单仓库、单人维护，token 成本换不来收益；`subagent` 仍可按需手用 |
| `/swarm` 分片并行 | 同上，且并行写同一仓库会放大冲突 |
| TLA+/Lean 形式化 | 这套代码是数据管道 + 规则评分，没有复杂并发协议；投入产出比不成立 |
| 24 条 principle | 原则越多越没人读；6 条覆盖 90% 场景 |
| 23 个 playbook | 22 个用不上；4 个够，缺的用 `/figure-it-out` 式临时拆解 |
| 自动写代码修复 CI | 不可靠且危险；脚本只输出「可修摘要」，改代码由 agent 决策 |

## 六、采纳自 Pstack 但做了本地化的

| Pstack | 本地化后 | 原因 |
|---|---|---|
| playbook 步骤抄进 todo | 同左，且「跳过必须写原因」 | 防止模型悄悄简化流程 |
| 证据要求 | 变成 `verify.sh` + CI 链接 + 日志摘要 | 让证据可机器校验 |
| `/correct` | 已经是 CI 检查（见第二节表格） | 比提示词更硬 |
| `/show-me-your-work` | `data/agent-log/`（不入库） | 避免仓库被日志污染 |
| `/babysit` | `loop.py` 单周期 + 失败摘要 | 多轮循环交给 agent 判断 |

## 七、怎么用

```bash
# 人
scripts/ci/verify.sh --full

# agent（推荐）
/tira-mode            # 说明任务，路由器选 playbook
                      # 或直接说：用 tira-mode 做 <任务>

# 一条命令跑完一个 PR 周期
python scripts/agent/loop.py --task "fix(macro): 修复央行总资产类型" --type fix
```

## 八、演进路线

| 阶段 | 内容 | 触发条件 |
|---|---|---|
| 现在 | 4 playbook + 6 原则 + loop.py 单周期 | — |
| 下一步 | 失败摘要接入「重复失败计数」，第 3 次自动开 issue | 同一个 job 连续失败 ≥3 次 |
| 之后 | 覆盖率按变更行（diff-cover） | 整体覆盖率 > 60% |
| 远期 | 视觉 diff（截图基线对比） | 前端趋于稳定 |
