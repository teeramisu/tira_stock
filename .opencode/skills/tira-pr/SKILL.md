---
name: tira-pr
description: >-
  在 tira_stock 开 PR、写描述、处理受保护路径放行、开启自动合并、以及 CI 红了怎么定位。
  当代码已经通过本地验证、需要提交与合并，或需要修 CI 失败时调用。
license: MIT
compatibility: opencode
metadata:
  version: v1.0.0
  repo: tira_stock
---

# tira-pr

## 一个周期（推荐直接用脚本）

```bash
python scripts/agent/loop.py --task "feat(macro): 增加省级排名变化" --type feat
```

脚本会依次做：预检 → 本地验证 → 提交 → 推送（失败重试）→ 开 PR → 开启 auto-merge → 等待合并。
CI 红时它会拉失败 job 的日志，生成「可修摘要」并以退出码 2 返回，**不会猜着改代码**。

手动等价流程：

```bash
scripts/ci/verify.sh --full
git add -A && git commit -m "feat(macro): 增加省级排名变化"
git push -u origin feat/xxx
# 开 PR（标题同提交信息，描述四段）
python scripts/agent/auto_merge.py --repo teeramisu/tira_stock --pr <N> --method squash --wait
```

## PR 描述模板（CI 会校验）

```markdown
## 背景
为什么改：问题现象 / 需求来源。一段话，不写「优化了一下」。

## 改动
- 改了哪些文件、哪些行为
- 有意没做的部分也写出来

## 验证
命令 + 输出关键行（`verify.sh` 结果、CI 链接、手动步骤）

## 回退
（feat / refactor / perf 建议写）revert 是否安全？有无数据/配置副作用？
```

`pr-meta` 会检查三节是否齐全、长度 ≥120 字符；缺了直接红。

## 受保护路径：什么时候要人工放行

| 改动 | 需要 `allow-guard-change` |
|---|---|
| `.github/workflows/**`、`scripts/ci/**`、`scripts/agent/**` | ✅ |
| `requirements*.txt`、`pytest.ini`、`ruff.toml`、`.coveragerc`、`ci/**` | ✅ |
| `backend/tests/**` **只新增**（`deletions == 0`） | ❌ 自动放行 |
| `backend/tests/**` 有删改 | ✅ |

**不要自己贴标签。** agent 的 token 不应有 `Issues: write`；标签由人加。
贴上后守卫会在 `labeled` 事件上重跑并通过。

## CI 红了怎么定位

```bash
# 1. 找到失败的 run 与 job
curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
  "https://api.github.com/repos/teeramisu/tira_stock/actions/runs?per_page=5" | jq '.workflow_runs[] | {id,head_branch,conclusion}'

# 2. 拉某个 job 的日志，只留错误行
curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
  "https://api.github.com/repos/teeramisu/tira_stock/actions/jobs/<job_id>/logs" | grep -E "##\[error\]|FAILED|❌"
```

（`GH_TOKEN` 无 `jq` 时用 `python -c` 解析。）

常见红法对照：

| 红在哪 | 大概率原因 |
|---|---|
| `PR 范围（一件事）` | 同时改了业务代码和 CI，或跨了 2 个以上功能区 |
| `PR 体量预算` | 超 40 文件 / 1500 行 |
| `受保护路径守卫` | 改 CI/测试/依赖，等人工标签 |
| `覆盖率 ratchet` | 新增逻辑没测试，或删了测试 |
| `提交规范` | 分支名或提交信息格式不对 |
| `前端语法与视觉` | JS 语法错 / 截图 job 起不来服务 |

## 合并方式

| 场景 | 方式 |
|---|---|
| 多个零碎提交 | `squash`（默认，`main` 保持一功能一提交） |
| 提交本身已清晰 | `rebase` |
| 大功能 | merge commit（少用） |

## 合并后的收尾（容易漏）

1. **合并成功后再删分支** —— 先删分支会让 PR 自动关闭（`closed, merged=false`）。
2. 删分支用 API 更稳：
   ```bash
   curl -X DELETE -H "Authorization: Bearer $GITHUB_TOKEN" \
     https://api.github.com/repos/teeramisu/tira_stock/git/refs/heads/<branch>
   ```
3. 同步本地：`git checkout main && git pull --ff-only`。

## 本机已知问题：SSH 间歇性拒连

现象：`Permission denied (publickey)` 与 `Hi teeramisu!` 交替出现，同一时刻手动 `ssh -T` 可能成功。

处理：

```bash
# 1. 配置走 443（更稳）
cat ~/.ssh/config
# Host github.com
#   HostName ssh.github.com
#   Port 443
#   IdentityFile ~/.ssh/id_ed25519_tira
#   IdentitiesOnly yes

# 2. 推送失败就重试，脚本里已带重试
for i in 1 2 3 4 5; do
  git push -q origin <branch> && break
  sleep 15
done

# 3. 仍不行：用 API 校验远端是否已更新（比 ls-remote 可靠）
```

> 全局配置里有 `url.git@github.com:.insteadof=https://github.com/`，
> 会把 HTTPS 地址改写成 SSH —— 要用 token 走 HTTPS 时需 `GIT_CONFIG_GLOBAL=/dev/null`。
