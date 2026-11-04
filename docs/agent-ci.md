# Agent 自主提交 / 自主合并：CI 怎么设计

> 目标：agent 能自己开分支、提交、开 PR、等 CI、合并，而**不会把质量搞坏**。
>
> 参考 Lauren Tan 的原则：先能验证，再谈并行；重复出现过两次的问题，
> 就变成一条会失败的检查，而不是写进提示词。

## 一、为什么「多加几个检查」不够

agent 和人不一样，它有三个特性：

| 特性 | 后果 | 必须的防线 |
|---|---|---|
| 会为了变绿而改「考卷」 | 删测试、放宽 lint、改 workflow | 检查不能被 PR 修改 |
| 只看机器可读输出 | 日志刷屏时它看不懂，只能乱猜 | 失败要有 file:line 与修复建议 |
| 并行提交 | 两个各自绿的 PR 合起来崩 | 合并要在组合结果上验证 |

对应三条机制：**受保护路径守卫**、**结构化失败输出**、**Merge Queue**。

## 二、三段式

```
Ring 0  本地（agent 自己跑）
        scripts/ci/verify.sh            8 步 / 40s
        commit-msg hook                 提交即校验
        ↓  自己先绿，再开 PR
Ring 1  PR fast lane（required）
        静态 · 4 分片 · 覆盖率 ratchet · 体量预算 · 前端 · 提交规范
        → 单一闸门：CI 汇总
        ↓  agent 开 auto-merge
Ring 2  Merge Queue（merge_group）
        在「base + 队列中前面的 PR」组合上重跑 required 检查
        绿 → 进 main；红 → 退回 PR 并给出冲突点
        ↓
Ring 3  事后
        Nightly 全量 / xfail 漂移 / 依赖审计 / 覆盖率趋势
```

## 三、六个机制

### 1. 受保护路径守卫（防改考卷）

`.github/workflows/ci-guard.yml` 用 **`pull_request_target`** 触发：

- checkout 的是 **base 分支**的脚本，PR 里的代码**一行都不执行**；
- 通过 API 拿到 PR 的变更文件，比对 `ci/protected-paths.json`；
- 命中受保护路径 → 失败，除非人工加了 `allow-guard-change` 标签。

```
受保护范围（ci/protected-paths.json）
  .github/workflows/ · scripts/ci/ · scripts/agent/
  backend/tests/ · requirements*.txt · pytest.ini · ruff.toml · .coveragerc · ci/
```

> 为什么用标签而不是 CODEOWNERS 强制评审：个人仓库里自己无法 approve 自己的 PR，
> 开「Require review from Code Owners」会让所有 PR 卡死。
> 标签方案的等效前提是：**agent 的 token 不含 `Issues: write`**（否则它能自己贴标签）。

### 2. 覆盖率 ratchet（防假测试）

```bash
pytest backend/tests --cov=backend --cov-report=json:coverage.json
python scripts/ci/check_coverage.py --report coverage.json
```

- 当前基线 37.5%，写入 `ci/coverage-baseline.json`；
- 下降超过 0.2pp → 失败；
- 同时输出覆盖率最低的 12 个文件，agent 知道该补哪里。

覆盖率不能证明测试是对的，但能挡住「一个断言都没有的测试」。

### 3. PR 体量预算（防一次性大改）

| 阈值 | 限制 | 处理 |
|---|---|---|
| 软上限 | 40 文件 / 1500 行 | 需人工加 `large-pr` 标签 |
| 硬上限 | 120 文件 / 4000 行 | 直接失败，必须拆 PR |

`data/`、`backend/rl/models/`、`*.min.js`、锁文件不计入。

### 4. Merge Queue（防组合冲突）

`ci.yml` 增加 `merge_group` 触发；`plan` job 用 `github.event.merge_group.base_sha` 作为 diff 基线。

在 ruleset 里加一条 `Require merge queue` 即可启用（需手动，见第五节）。

没有这一步，「PR 绿」只代表「单独看它是绿的」。

### 5. 自动合并

```bash
python scripts/agent/auto_merge.py --repo teeramisu/tira_stock --pr 12 --method squash --wait
```

走 GraphQL `enablePullRequestAutoMerge`，需要 token 有 `Pull requests: Read and write`。

### 6. 单一闸门

分支保护只勾 **`CI 汇总`**，它 depends on：`plan` `static` `tests` `coverage` `pr-size` `frontend` `commits` `pr-meta`。

新增 job 不用改仓库设置；按变更面跳过的算 `skipped`，不算失败。

## 四、agent 执行循环

```python
# scripts/agent/loop.py 的骨架（P1 再落地成脚本）
for attempt in range(3):
    if run("scripts/ci/verify.sh --full").ok:
        break
    apply_fixes(parse_errors())      # 只读 file:line 级的结构化输出

create_branch("feat/xxx")
commit_conventional()
push()
pr = open_pr()
auto_merge(pr)                       # 等 Ring1 + Ring2

if ci_failed(pr):
    label_via_ci("agent-fixing")     # agent 自己不能打标签
    attempt_fix(); push()
else:
    wait_merged(pr)
else:
    label_via_ci("needs-human")
    comment(failure_summary() + tried_fixes())
```

关键约束：

1. **三轮不过就交人**，避免无限重试烧 token；
2. agent 只读结构化输出（ruff 的 `file:line`、pytest 的 `FAILED` 行、覆盖率最低文件）；
3. agent 不能打标签 —— 标签由 CI 用 `GITHUB_TOKEN` 打，身份与权限分离。

## 五、仓库需要人工打开的两项设置

| 设置 | 位置 | 说明 |
|---|---|---|
| Allow auto-merge | Settings → General → Pull Requests | 不开的话 `enablePullRequestAutoMerge` 报错 |
| Require merge queue | Settings → Rules → protect-main → Edit | 加 `Require merge queue`，`CI 汇总` 会自动成为队列检查 |

> Merge Queue 对 public 仓库可用（本仓库是 public）。

## 六、agent token 的最小权限

用一个**独立**的 fine-grained token，只给：

| 权限 | 级别 | 为什么 |
|---|---|---|
| Contents | Read and write | 推分支 |
| Pull requests | Read and write | 开 PR / 开 auto-merge |
| Metadata | Read | 必需 |

**不要给**：

| 权限 | 原因 |
|---|---|
| Administration | 否则能改 ruleset，等于能绕过所有检查 |
| Issues | 否则能自己贴 `allow-guard-change` / `large-pr` 标签 |
| Actions / Workflows | 否则能改 CI 定义 |
| Secrets | 永远不给 |

## 七、这套机制的边界（诚实说明）

| 能挡住 | 挡不住 |
|---|---|
| 删测试 / 改 CI 变绿 | 写出「断言实现」的假测试（覆盖率只能部分缓解） |
| 覆盖率断崖式下跌 | 逻辑正确但业务不对（需要 spec 派生测试） |
| 一次改 3000 行 | 一百个 30 行的小 PR 各自合理、整体走偏 |
| 组合冲突 | 数据/外部依赖变化导致的行为漂移（靠 nightly + 监控） |
| 密钥与大文件入库 | 供应链投毒（需要 P1 的依赖审计 + lockfile 校验） |

所以「agent 自动合并」的正确姿势是：**小步、可回滚、每步有证据**，而不是「一次交付一个大功能」。

## 八、文件清单

```
.github/workflows/ci.yml          Ring 1 + merge_group 触发
.github/workflows/ci-guard.yml    受保护路径守卫（pull_request_target）
.github/CODEOWNERS                归属说明（暂未强制）
ci/protected-paths.json           受保护路径清单
ci/coverage-baseline.json         覆盖率基线
.coveragerc                       覆盖率统计口径
scripts/ci/check_guard.py         守卫逻辑
scripts/ci/check_coverage.py      覆盖率 ratchet
scripts/ci/check_pr_size.py       体量预算
scripts/agent/auto_merge.py       自动合并
scripts/ci/verify.sh              本地镜像（--full 含覆盖率与体量）
```
