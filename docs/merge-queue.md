# 自动合并链路：Merge Queue + auto-merge

> 目标：CI 绿了之后 agent 能自己合，且**合进去的结果也被验证过**。
>
> 这一篇只讲合并链路；受保护路径、覆盖率、体量预算见 [ci.md](ci.md)。

## 一、为什么不能「PR 绿了就合」

```
main:   A ── B
PR#12:        └── C     （基于 A，CI 绿）
PR#13:        └── D     （基于 A，CI 绿）

PR#12 先合 → main = A─B─C
PR#13 的绿，是在「A + D」上验证的，但实际合入的是「A + B + C + D」
```

两个各自绿的 PR，合起来可能编译失败、测试互踩、数据迁移顺序错乱。
PR 级别的 CI 永远看不到这种组合。

## 二、Merge Queue 怎么解决

```
agent 开 auto-merge
      ↓
GitHub 把 PR 放进队列，构造一个临时的 merge_group 提交：
      base(main) + 队列里排在前面的 PR + 本 PR
      ↓
在本仓库跑 required 检查（就是 ci.yml 的 CI 汇总）
      ↓
绿 → 合入 main，队列前进；红 → 把 PR 踢出队列并标红
```

所以 `ci.yml` 必须支持 `merge_group` 事件：

```yaml
on:
  merge_group:            # ← 队列里构造的组合提交
```

`plan` job 也要用组合提交的 base 做 diff 基线，否则变更面判断会失真：

```bash
elif [[ "${{ github.event_name }}" == "merge_group" ]]; then
  base="${{ github.event.merge_group.base_sha }}"
```

## 三、开启方式（仓库设置，需人工）

| 设置 | 位置 | 说明 |
|---|---|---|
| Allow auto-merge | Settings → General → Pull Requests | 不开则 GraphQL 报错 |
| Require merge queue | Settings → Rules → protect-main → Edit → 勾 `Require merge queue` | 队列才会跑 |

> Merge Queue 对 public 仓库可用（本仓库是 public）。
> 加进 ruleset 后，`CI 汇总` 会自动成为队列要求的检查，无需额外配置。

## 四、agent 侧怎么用

```bash
python scripts/agent/auto_merge.py --repo teeramisu/tira_stock --pr 12 --method squash --wait
```

| 参数 | 作用 |
|---|---|
| `--method squash` | 合并方式（默认 squash，`main` 保持一功能一提交） |
| `--wait --timeout 1800` | 阻塞等待合并完成，超时返回非 0 |
| `--disable` | 撤销 auto-merge |

脚本行为：

1. 读 PR 状态；draft 直接拒绝（否则会一直排队等不到）；
2. 已开 auto-merge 则跳过，幂等；
3. 失败时区分「仓库没开 auto-merge」「有冲突」「检查未配置」，给出对应提示；
4. `--wait` 模式每 20 秒轮询，状态变化才打印，避免刷屏。

## 五、token 权限

自动合并只需 `Pull requests: Read and write`，**不需要** `administration`。

建议给 agent 单独一个 fine-grained token：

| 权限 | 级别 | 原因 |
|---|---|---|
| Contents | Read and write | 推分支 |
| Pull requests | Read and write | 开 PR / 开 auto-merge |
| Metadata | Read | 必需 |

不要给：`Administration`（能改 ruleset，等于绕过所有检查）、`Issues`（能自己贴放行标签）、
`Actions/Workflows`（能改 CI 定义）、`Secrets`。

## 六、失败时会发生什么

| 现象 | 原因 | 处理 |
|---|---|---|
| `Resource not accessible` | token 缺 `pull_requests: write` | 加权限 |
| `Pull request is not mergeable` | 有冲突 / required check 未配置 | 先 rebase 或补 ruleset |
| 队列里一直 pending | 队列前面的 PR 挂了 | 看队列页，先修最前面的 |
| 合并后 CI 红（push 事件） | 组合验证漏了某条路径 | 立刻 revert，并补一条检查 |

## 七、和「一个 PR 一件事」的关系

队列会把多个 PR 串起来验证，但**这不能替代小 PR**：

- 一次 4000 行的 PR，队列里挂了很难定位是哪一段；
- 小 PR 出问题只需 revert 一条，`main` 立刻回到可用状态。

所以 `pr-size` 与 `pr-scope` 不是形式主义，是让合并链路在出错时**可回退**。

## 八、文件

```
.github/workflows/ci.yml      新增 merge_group 触发与 base_sha 支持
scripts/agent/auto_merge.py   自动合并脚本
docs/merge-queue.md           本篇
```
