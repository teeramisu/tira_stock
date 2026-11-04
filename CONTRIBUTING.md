# 贡献指南

> 目标：每次改动都能被追溯、被验证、能回滚。
> 规则尽量少，但每条都**由 CI 或 hook 强制执行**，不靠记性。

## 一、一任务一分支

任何改动都从最新 `main` 开新分支，**不要直接推 `main`**（`main` 已开分支保护，直推会被拒）。

```bash
git checkout main && git pull
git checkout -b feat/macro-provinces    # ← 新分支
# ... 改代码 ...
scripts/ci/verify.sh                   # 本地自检
git push -u origin feat/macro-provinces
# 开 PR → CI 绿 → 合并 → 删除分支
```

### 分支命名

```
<type>/<短横线描述>
```

| type | 用途 | 例 |
|---|---|---|
| `feat` | 新功能 | `feat/macro-provinces` |
| `fix` | 修 bug | `fix/portfolio-logger` |
| `refactor` | 重构（不改行为） | `refactor/marketdata-loader` |
| `perf` | 性能 | `perf/us-refresh-parallel` |
| `docs` | 文档 | `docs/ci-guide` |
| `test` | 只改测试 | `test/discovery-gates` |
| `ci` | CI / 脚本 | `ci/commit-conventions` |
| `chore` | 杂项 | `chore/bump-deps` |

描述用小写英文 + 短横线（`feat/macro-provinces`），**不要** `feat/增加省数据`、`feat/xxx`（少于 3 字符）。

> 由 `scripts/ci/check_commits.py --branch` 在 CI 中校验。

## 一之二、代码风格（Google Python Style Guide）

新代码必须符合 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)：

- 命名：`snake_case` 函数/变量、`CapWords` 类、`UPPER_CASE` 常量
- 导入：stdlib → 第三方 → 本地，分组之间空行（`ruff --select I`）
- docstring：`convention = "google"`，公开函数写 Args / Returns / Raises
- 不用可变默认参数（`def f(x=[])`）、不用 `l`/`O` 这类歧义单字母名
- 行宽 100

**只对新增行生效**，老代码不要求重排：

```bash
python scripts/ci/check_style.py --base origin/main --head HEAD
ruff check <file> --config ci/style-ruff.toml --fix   # 能自动修的
```

## 二、提交信息：Conventional Commits

```
<type>(<scope>)!: <描述>
```

```bash
git commit -m "feat(macro): 增加省级变化对比"
git commit -m "fix(portfolio): 修复策略注册表 logger 未定义"
git commit -m "ci: 去掉 pull_request 分支过滤"
git commit -m "refactor(marketdata)!: 统一复权口径"      # ! 表示破坏性变更
```

### 规则

1. **type 必填**，取值见上表；
2. **scope 可选**，常用：`macro` `discovery` `backtest` `portfolio` `marketdata` `factors` `frontend` `api` `ci` `docs` `deps` `rl` `research` `memory` `tests`；
3. **描述至少 4 个字符**，不要写 `update` / `修改` / `wip`；
4. **标题 ≤ 100 字符**，结尾不加句号；
5. 正文（可选）说明 **为什么** 这么改，而不是重复改了什么；
6. 一个提交做一件事，不要「顺手重构 + 顺手改格式」。

### 本地自动校验

```bash
scripts/git/install_hooks.sh                 # 装 commit-msg hook
scripts/git/install_hooks.sh --with-pre-push # 额外在 push 前跑 quick verify
```

装完后格式不对的提交信息会被直接拒绝，不用等 CI。

> 已经在用的仓库也可以手动检查：`python scripts/ci/check_commits.py --range origin/main..HEAD`

## 三、一个 PR 只做一件事

这条不是洁癖，是为了**回滚**和**定位**：

- 混在一起的 PR 出问题只能整包 revert，会连带回滚掉没问题的那部分；
- CI 红了分不清是「代码坏了」还是「判分标准变了」；
- review 只能整体点头，等于没 review。

### 怎么算「一件事」

一个分区。`check_pr_scope.py` 把仓库分成 6 个功能区：

| 功能区 | 路径 |
|---|---|
| `backend` | `backend/**`（除 tests） |
| `tests` | `backend/tests/**`（跟随业务代码，不计入跨度） |
| `frontend` | `frontend/**` |
| `ci` | `.github/**`、`scripts/ci/**`、`scripts/agent/**`、`ci/**`、`requirements*`、`pytest.ini`、`ruff.toml`、`.coveragerc` |
| `scripts` | `scripts/**`（数据管线脚本） |
| `docs` | `docs/**`、`*.md` |

规则：

1. **最多跨 2 个功能区**（超过需人工加 `multi-area` 标签）；
2. **业务代码和 CI 改动永远分开**（`ci` 与 `backend`/`frontend` 不能同框）——
   否则无法判断是代码坏了还是标准变了；
3. 测试跟随业务代码，不算独立分区。

### PR 体量预算

| 阈值 | 限制 | 处理 |
|---|---|---|
| 软上限 | 40 文件 / 1500 行 | 需人工 `large-pr` 标签 |
| 硬上限 | 120 文件 / 4000 行 | 直接失败，必须拆 |

`data/`、`backend/rl/models/`、`*.min.js`、锁文件不计入。

## 四、PR 描述 = 回滚说明书

CI 会检查三件事：标题格式、描述长度 ≥ 120 字符、三个小节齐全。

```markdown
## 背景
为什么改（问题现象、需求来源）。一段话，不写「优化了一下」。

## 改动
- 改了哪些文件、哪些行为
- 有意没做的部分也写出来

## 验证
命令 + 结果（`scripts/ci/verify.sh` 输出、CI 链接、手动验证步骤）

## 回退
（feat / refactor / perf 类建议写）revert 这条 PR 是否安全？
有没有数据迁移、配置变更、外部依赖需要同步回滚？
```

> 为什么强制「验证」一节：没有验证记录的 PR，出问题时无法判断它是
> 「本来就没验」还是「验了但环境不同」，排查成本翻倍。

## 五、PR 标题与合并方式

### 标题

与提交信息同格式（CI 用同一个正则校验）：

```
feat(macro): 增加省级变化对比
```

草稿请用 PR 的 **Draft** 状态，不要在标题写 `WIP`。

### 描述

CI 要求 ≥ 80 字符且有结构。建议四段：

```markdown
## 背景
为什么要做这个改动（问题 / 需求）

## 改动
- 做了什么（按模块列）

## 验证
- 本地跑过的命令与结果
- CI 链接

## 已知限制
- 没做的、以后再说的
```

### 合并方式

| 场景 | 方式 |
|---|---|
| 多个零碎提交 | **Squash and merge**（推荐，`main` 保持一条一功能） |
| 提交本身已清晰 | **Rebase and merge** |
| 大功能需要保留分支结构 | Merge commit |

## 五之一、身份分离（agent 与人不共用 token）

门禁要成立，前提是「被考核方」不能自己给自己放行：

| 身份 | 环境变量 | 权限 | 用途 |
|---|---|---|---|
| 人 | `GITHUB_TOKEN` | 含 `Issues` / `Administration` | 贴放行标签、改仓库设置 |
| agent | `AGENT_GITHUB_TOKEN` | 仅 `Contents` + `Pull requests` + `Metadata` | 推分支、开 PR、开 auto-merge |

给自动化脚本加 token 时，按 `AGENT_GITHUB_TOKEN → GITHUB_TOKEN` 顺序取，
回退到人的 token 会打印警告。创建与验证步骤见 [docs/agent-token.md](docs/agent-token.md)。

## 五之二、受保护路径（CI 脚本 / 测试 / 依赖）

以下路径的改动会被 `受保护路径守卫` 拦下，需要人工给 PR 加 `allow-guard-change` 标签：

```
.github/workflows/  scripts/ci/  scripts/agent/
backend/tests/  requirements*.txt  pytest.ini  ruff.toml  .coveragerc  ci/
```

原因：这些文件决定「什么算通过」。让被考核方自己改判分标准，门禁就没有意义了。

流程：正常开 PR → 守卫红 → 人在页面上加标签 → 守卫重跑通过 → 合并。
详见 [docs/protected-paths.md](docs/protected-paths.md)。

## 六、合并前必须满足

- [ ] `CI 汇总` 绿（分支保护强制）
- [ ] 只碰了一个功能区（`pr-scope` 强制）
- [ ] 体量在预算内（`pr-size` 强制）
- [ ] 没有把 `data/`、`backend/rl/models/`、`.env` 加进索引（`check_hygiene.py` 强制）
- [ ] 新增测试文件已登记到 `ci/shards.json`（`shard_tests.py` 强制）
- [ ] 没有新增 lint 债务（`lint_ratchet.py` 强制）
- [ ] 改了行为就补测试；如果暂时修不了，用 `@pytest.mark.xfail(strict=False, reason=...)` 显式记账

## 七、本地自检

```bash
scripts/ci/verify.sh          # 与 CI 完全相同的 9 步，约 45 秒
scripts/ci/verify.sh --quick  # 静态层 + macro 分片，约 10 秒
```

CI 里没有本地跑不了的检查。详细分层见 [docs/ci.md](docs/ci.md)。

## 八、紧急情况

生产事故需要绕过 PR 时：

1. 走 PR，但只跑必需检查（不要关规则）；
2. 真的必须直推，**先临时**把 ruleset 的 enforcement 改成 `Evaluate`（只记录不拦截），推完立刻改回 `Active`；
3. 事后补一个 PR 把改动走一遍流程，让 CI 跑一次。

> 不要删除 ruleset 来绕过——删了之后很容易忘记恢复。
