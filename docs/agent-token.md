# agent 身份分离（AGENT_GITHUB_TOKEN）

> 受保护路径守卫能拦住 agent 改判分标准，**前提是 agent 和人不是同一个身份**。
>
> 用同一个 token 时，agent 可以自己贴 `allow-guard-change` 标签，
> 守卫就只剩提示作用。

## 一、现状（为什么必须做）

实测：当前 `~/.bashrc` 里的 `GITHUB_TOKEN` 具备

```
x-accepted-github-permissions: issues=write; pull_requests=write
```

也就是说自动化脚本可以：

- 给自己开的 PR 贴 `allow-guard-change` → 绕过守卫
- 贴 `large-pr` / `multi-area` → 绕过体量与范围限制
- （再加上 `administration` 就能改 ruleset，等于全套门禁失效）

## 二、目标结构

| 身份 | token | 权限 | 用来做什么 |
|---|---|---|---|
| **人** | `GITHUB_TOKEN` | 现有（含 issues / administration） | 贴放行标签、改仓库设置、紧急处置 |
| **agent** | `AGENT_GITHUB_TOKEN` | `Contents` RW、`Pull requests` RW、`Metadata` R | 推分支、开 PR、开 auto-merge、读 CI 状态 |

**agent 绝对不能有**：`Issues`（贴标签）、`Administration`（改 ruleset）、
`Actions`/`Workflows`（改 CI 定义）、`Secrets`。

## 三、创建步骤

```
GitHub → 头像 → Settings → Developer settings
  → Personal access tokens → Fine-grained tokens → Generate new token

Token name      : tira-agent
Expiration      : 90 days              （到期必须轮换，别选 No expiration）
Repository access: Only select repositories → teeramisu/tira_stock

Repository permissions
  Contents          Read and write
  Pull requests     Read and write
  Metadata          Read              （自动勾选，必需）

Account permissions
  （全部留空 —— 尤其是不要给 Issues）

→ Generate token → 复制 github_pat_...
```

## 四、配置

写进 `~/.bashrc`（**不要**提交进仓库）：

```bash
# 人的身份：贴标签、改设置
export GITHUB_TOKEN="github_pat_人的..."
# agent 身份：自动化专用
export AGENT_GITHUB_TOKEN="github_pat_agent_..."
```

让脚本优先使用 agent token —— `scripts/agent/auto_merge.py` 已经按
`AGENT_GITHUB_TOKEN → GITHUB_TOKEN → GH_TOKEN` 的顺序取，回退时会打印警告：

```
⚠️  未设置 AGENT_GITHUB_TOKEN，回退到 GITHUB_TOKEN（身份未分离）
```

## 五、验证分离是否成功

```bash
# 用 agent token 尝试贴标签，应该被拒（403 / issues=write）
AGENT_TOKEN="$AGENT_GITHUB_TOKEN"
curl -s -o /dev/null -w "%{http_code}\n" -X POST \
  -H "Authorization: Bearer $AGENT_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/teeramisu/tira_stock/issues/1/labels \
  -d '{"labels":["allow-guard-change"]}'
# 期望：403
```

再确认能开 PR：

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $AGENT_GITHUB_TOKEN" \
  https://api.github.com/repos/teeramisu/tira_stock/pulls?state=open
# 期望：200
```

## 六、日常操作分工

| 操作 | 用哪个身份 |
|---|---|
| `git push`（走 SSH key，不受 token 影响） | — |
| 开 PR / 关 PR / 开 auto-merge | agent token |
| 读 CI 状态、跑 `auto_merge.py --wait` | agent token |
| 贴 `allow-guard-change` / `large-pr` / `multi-area` | **人**（浏览器或人的 token） |
| 改 ruleset / 分支保护 / 仓库设置 | **人** |
| 紧急直推 main | 只能人（ruleset 挡 agent） |

## 七、轮换与失效

| 现象 | 原因 | 处理 |
|---|---|---|
| `403 Resource not accessible` | token 过期或权限被改 | 重新生成，更新 `~/.bashrc` |
| `⚠️ 回退到 GITHUB_TOKEN` 警告 | 没配 `AGENT_GITHUB_TOKEN` | 补上环境变量 |
| 守卫突然不拦了 | agent token 被误加了 `Issues` 权限 | 重新收敛权限 |

`~/.bashrc` 修改后记得 `source ~/.bashrc`，或在新的 shell 里执行。

## 八、为什么不在仓库里存 token

CI 里的 `secrets.GITHUB_TOKEN` 是 GitHub 自动注入的短期令牌，与本地 token 无关；
它只用于 workflow 内部，生命周期随任务结束。本地自动化 token 必须放在开发者机器上。

> 顺带一提：`scripts/ci/check_secrets.py` 会在每次 CI 里扫描 git 跟踪的文件，
> 一旦有人把 `github_pat_...` 写进代码，PR 会直接红。
