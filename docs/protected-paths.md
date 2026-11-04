# 受保护路径守卫（防「改考卷」）

> 允许 agent 自主合并的前提之一：**它改不动判分标准**。
>
> 普通 `ci.yml` 跑的是 PR 里的脚本 —— agent 想变绿，只要把检查删掉就行。
> 这篇讲怎么堵住这条路。

## 一、威胁模型

| 行为 | 后果 |
|---|---|
| 删掉失败的测试 | CI 绿了，功能其实坏了 |
| 放宽 `ruff.toml` / `pytest.ini` | 风格与门禁形同虚设 |
| 改 `.github/workflows/*.yml` | 直接删掉检查步骤 |
| 改 `ci/coverage-baseline.json` | 覆盖率 ratchet 失效 |
| 改 `requirements-ci.txt` | 换掉 lint / 测试工具版本 |

这些文件本身就该由人审，不该由「被考核方」自己改。

## 二、机制

`.github/workflows/ci-guard.yml`：

```yaml
on:
  pull_request_target:          # 注意：不是 pull_request
    types: [opened, synchronize, reopened, labeled, unlabeled]

steps:
  - uses: actions/checkout@v4
    with:
      ref: ${{ github.event.pull_request.base.sha }}   # ← 取 base，不取 PR
```

两个关键点：

1. **`pull_request_target`**：在默认分支的上下文运行，PR 里的代码一行都不执行；
2. **checkout base 的脚本**：判分逻辑用的是「已合并的版本」，agent 改 PR 里的副本不影响判定。

然后通过 API 拿 PR 的变更文件，比对 `ci/protected-paths.json`：

```json
{
  "patterns": [
    ".github/workflows/", "scripts/ci/", "backend/tests/",
    "requirements-ci.txt", "pytest.ini", "ruff.toml", ".coveragerc", "ci/"
  ]
}
```

命中 → 失败，除非 PR 有 `allow-guard-change` 标签。

### 例外：纯新增的测试改动自动放行

```json
{ "allow_pure_additions": ["backend/tests/"] }
```

判定条件（三者同时满足）：

| 条件 | 含义 |
|---|---|
| 路径命中 `allow_pure_additions` | 目前只有 `backend/tests/` |
| `deletions == 0` | 没有删除行 |
| `additions > 0` | 确实有新内容 |

理由：**新增测试无法削弱既有断言**。要作弊必须删掉或改写断言，
而那一定伴随 `deletions > 0`，仍然会被拦。

效果对照：

| 改动 | 结果 |
|---|---|
| `test_macro.py` 追加 30 行用例 | ✅ 自动放行 |
| `test_macro.py` 删 5 行改 3 行 | ❌ 需要标签 |
| `scripts/ci/verify.sh` 即使只加不清 | ❌ 需要标签（改的是判分逻辑本身） |
| 新增测试 + 改 `requirements-ci.txt` | ❌ 需要标签（依赖变更） |

## 三、为什么用标签而不是 CODEOWNERS

GitHub 的 `Require review from Code Owners` 在有协作者时是最优解。但本仓库是个人仓库：

- GitHub 不允许 approve 自己的 PR；
- 开了这条规则，所有 PR 都会永久卡住。

所以改用「人工标签」：**标签只能由人加**，前提是 agent 的 token 不含 `Issues: write`
（PR 标签属于 issues 权限）。

`.github/CODEOWNERS` 仍然保留，作为归属说明；将来有多人协作时，把 ruleset 里的
`Require review from Code Owners` 打开、`required approvals` 设为 1 即可切换过去。

## 四、正常改 CI 的流程

```bash
# 人来做
1. 开 PR（改 scripts/ci/** 或 .github/**）
2. CI 里 `受保护路径守卫` 会红 —— 这是预期
3. 人工在 GitHub 页面给 PR 加标签 allow-guard-change
4. 守卫重跑（labeled 事件触发）→ 绿
5. 合并
```

标签加上后如果又推了新提交，守卫会重新判断（`synchronize` 事件），
但标签仍在，所以不会反复要标签。

## 五、边界（诚实说明）

| 能挡住 | 挡不住 |
|---|---|
| PR 里删测试、放宽规则、改 workflow | 合并后直接 `git push` 到 main（由 ruleset 挡，两套机制互补） |
| agent 用 PR 里的脚本「自我放行」 | token 有 `Issues: write` 时 agent 自己贴标签（所以权限要收） |
| 悄悄改 `coverage-baseline.json` | 用另一个 PR 先降低基线、再改代码（需要人看 PR 描述） |

第三行是真实存在的绕过路径：**基线类文件最好成对审查**（基线与代码在同一个 PR 里改）。
如果以后接入 agent 自主合并，建议再加一条：`ci/*.json` 的改动必须与 `backend/**` 同 PR。

## 六、文件

```
.github/workflows/ci-guard.yml   守卫 workflow（pull_request_target）
.github/CODEOWNERS               归属说明（暂未强制）
ci/protected-paths.json          受保护路径清单
scripts/ci/check_guard.py        比对逻辑
```

## 七、本地自测

```bash
# 未触碰受保护路径
python scripts/ci/check_guard.py --files "backend/macro/provinces.py,README.md"
✅ CI 守卫通过（2 个变更文件，未触碰受保护路径）

# 触碰了
python scripts/ci/check_guard.py --files "scripts/ci/verify.sh"
❌ 受保护路径只允许人工改动

# 带人工标签
python scripts/ci/check_guard.py --files "scripts/ci/verify.sh" --labels allow-guard-change
✅ 已有人工打上 `allow-guard-change` 标签，放行
```
