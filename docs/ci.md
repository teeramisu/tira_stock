# CI 设计说明

> 参考 Lauren Tan 的验证优先级：**先能验证，再谈并行**。
> 检查要尽量往前压（静态 → 单元 → 集成 → 端到端），
> 每一次「同一个问题说过两遍」都要变成一条会失败的检查，
> 而不是靠记住它。

## 一、分层结构

```
                    ┌──────────────────────────────┐
   最贵、最慢  →    │  前端语法 + 资源引用（nose）  │  3 个 js + 4 段内联脚本
                    ├──────────────────────────────┤
                    │  测试主体（4 个并行分片）     │  385 passed / 27s
                    ├──────────────────────────────┤
                    │  静态层（秒级，先失败）       │  ruff / 卫生 / 密钥 / 分层
                    └──────────────────────────────┘
```

| 层 | 检查项 | 典型耗时 | 阻断 |
|---|---|---|---|
| 静态 | `ruff`（E9/F63/F7/F82）、字节码编译、`app.main` 可导入 | ~10s | ✅ |
| 静态 | lint 债务 ratchet | ~5s | ✅ |
| 静态 | 仓库卫生（大文件/本地数据/密钥文件） | ~2s | ✅ |
| 静态 | 密钥泄露扫描 | ~3s | ✅ |
| 静态 | 分层边界（API 层 ↔ 核心库） | ~2s | ✅ |
| 测试 | 4 个分片并行（macro-discovery / engine / portfolio / marketdata） | ~30s | ✅ |
| 前端 | `node --check` + 内联脚本 + 资源引用存在性 | ~3s | ✅ |
| 元信息 | 分支命名 + 提交信息（Conventional Commits） | ~1s | ✅ |
| 元信息 | PR 标题 Conventional Commits、描述长度与结构 | ~1s | ✅ |
| 闸门 | `CI 汇总` 单一 required check | — | ✅ |
| Nightly | 全量（含 network/integration）、xfail 漂移、依赖审计 | ~15min | ❌ 仅可见 |

## 二、为什么要「单一闸门」

分支保护只勾一个 **`CI 汇总`**。

- 以后新增 job 不用改仓库设置；
- 被 `if:` 主动跳过的 job（比如只改 README 不跑后端测试）算 `skipped`，不算失败；
- 真正的失败会顺着 `needs` 冒泡到闸门，红得明确。

## 二之二、分支与提交规范

「每次都开新分支」和「提交信息规范」不靠自觉，靠两道锁：

| 位置 | 检查 | 失败后果 |
|---|---|---|
| 本地 `commit-msg` hook | `check_commits.py --msg-file` | 提交直接被拒 |
| CI `commits` job | `--branch` + `--range base..head` | PR 红，无法合并 |

```
分支：<type>/<短横线描述>        feat/macro-provinces
提交：<type>(<scope>): <描述>    feat(macro): 增加省级变化对比
```

安装本地 hook（与 CI 同一套脚本）：

```bash
scripts/git/install_hooks.sh                 # commit-msg
scripts/git/install_hooks.sh --with-pre-push # 追加 push 前 quick verify
```

完整规则见 [CONTRIBUTING.md](../CONTRIBUTING.md)。

## 三、变更面过滤

`scripts/ci/changed_areas.py` 按路径判断这次动了什么：

| 区域 | 命中路径 | 触发的 job |
|---|---|---|
| backend | `backend/`、`scripts/`、`requirements*`、`pytest.ini`、`ruff.toml` | static + tests |
| frontend | `frontend/` | frontend |
| ci | `.github/`、`ci/`、`scripts/ci/` | **全部**（改 CI 本身必须验证整条链路） |
| docs | `docs/`、`README*` | 都不触发（只跑 pr-meta） |

## 四、债务 ratchet：不要求一次清零，但不许变多

仓库历史债务：`E` 225、`W` 104、`F` 181、`I` 173、`B` 91、`UP` 518、`SIM` 48。

`scripts/ci/lint_ratchet.py` 把当前数量记在 `ci/lint-baseline.json`：

```bash
python scripts/ci/lint_ratchet.py           # 校验：任何一类上涨 → 失败
python scripts/ci/lint_ratchet.py --write   # 债务下降后刷新基线
```

新代码不能新增任何一类问题；老代码可以慢慢还。
这比「一次格式化整个仓库」更安全（不会把 review 淹没在 289 个文件的 diff 里）。

## 五、已知失败用 xfail，不用删除

`backend/tests` 里有 7 个 `@pytest.mark.xfail(strict=False, reason=...)`，
都是「实现已经变了，断言还没跟上」的历史遗留：

```python
@pytest.mark.xfail(strict=False, reason="endpoint 已改为进程内刷新，ENABLE_MARKET_REFRESH_API 特性开关被移除，待重写")
def test_market_refresh_endpoint_rejects_when_disabled(monkeypatch):
    ...
```

- CI 里算 `xfailed`，不阻断；
- **Nightly 用 `--runxfail` 还原真实状态**，每天会告诉你还剩几个；
- 修好后 Nightly 会提示「可以移除 xfail」，避免 xfail 变成垃圾场。

## 六、本地必须能复现

```bash
scripts/ci/verify.sh          # 全量，约 40s
scripts/ci/verify.sh --quick  # 静态层 + macro 分片
```

CI 里的每一步都是这脚本里的一条命令，**没有只在 CI 里能跑的东西**。
本地绿了再推，推到 CI 就不会有惊喜。

## 七、这套 CI 上线当天抓到的真 bug

| 文件 | 问题 | 后果 |
|---|---|---|
| `backend/app/routers/backtest.py` | `json` 未导入 | 研究汇总接口必崩 `NameError` |
| `backend/portfolio/strategy_registry.py` | `logger` 未定义 | 有 walk-forward 参数覆盖时必崩 |
| `scripts/refresh_us_incremental.py` | `pd` 只在函数内导入却被用于注解 | 类型检查/运行期 `NameError` |

同时发现 `backend/rl/tb_logs/`（19 个 TensorBoard 事件文件）被误提交进 git，
已从索引移除并加入 `.gitignore`。

## 八、如何新增一条检查

1. 把逻辑写成 `scripts/ci/check_xxx.py`，**失败返回非 0，并打印可执行的修复建议**；
2. 在 `.github/workflows/ci.yml` 的 `static`（或对应 job）加一步；
3. 在 `scripts/ci/verify.sh` 里加同一行，保证本地可复现；
4. 如果有噪音风险，先设 `continue-on-error: true`，观察一周再转阻断。

## 九、CI 依赖与本地依赖的差异

`requirements-ci.txt` 剔除了约 2GB 的可选依赖：`torch`、`stable-baselines3`、
`gymnasium`、`optuna`、`tensorboard`、`transformers`、`accelerate`。

对应保护：

```python
pytest.importorskip("torch", reason="RL 测试需要可选依赖 torch")
```

新测试若依赖这些包，请用同样的方式跳过，而不是把它们塞回 CI。

## 十、文件清单

```
.github/workflows/ci.yml      主流水线（分层 + 闸门）
.github/workflows/nightly.yml 夜间全量
ci/shards.json               测试分片表
ci/lint-baseline.json        lint 债务基线
requirements-ci.txt          CI 依赖
ruff.toml                    静态规则
scripts/ci/verify.sh         本地镜像
scripts/ci/changed_areas.py  变更面探测
scripts/ci/shard_tests.py    分片校验/列举
scripts/ci/lint_ratchet.py   债务 ratchet
scripts/ci/check_hygiene.py  仓库卫生
scripts/ci/check_secrets.py  密钥扫描
scripts/ci/check_layers.py   分层边界
scripts/ci/check_frontend.py 前端语法与资源
```

## 十一、仓库设置（已实测）

`main` 使用 **Ruleset**（不是传统的 Branch protection rule），实测有效的配置：

```
name        : protect-main
enforcement : active
target      : branch
include     : ["~DEFAULT_BRANCH"]      ← 关键！空数组 = 规则不作用任何分支
rules       : deletion / non_fast_forward / pull_request(0 approvals)
              / required_status_checks(strict=true, checks=["CI 汇总"])
```

> 踩过的坑：UI 里选了 `Include default branch` 但没点「加入」，
> API 里 `include` 会是 `[]`，规则静默失效——直推 `main` 照样成功。
> 验证方法：`git push origin main` 应返回 `GH013 Repository rule violations`。

注意：ruleset 生效时 `GET /repos/{owner}/{repo}/branches/main` 的 `protected`
字段仍然是 `false`（那是传统 branch protection 的字段），不能用来判断。

分支保护无法用当前 CI token 设置（需要 `administration: write`），属于一次性手动操作。
