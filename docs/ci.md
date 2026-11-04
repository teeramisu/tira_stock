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
| 静态 | **Google 风格（仅新增行）** | ~5s | ✅ |
| 测试 | 4 个分片并行（macro-discovery / engine / portfolio / marketdata） | ~30s | ✅ |
| 测试 | 分片内失败自动重跑一次，区分偶发与真失败 | 仅失败时 | ✅ |
| 测试 | **覆盖率 ratchet**（整体只许升） | ~25s | ✅ |
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

## 二之三、Google 风格：新账不欠，旧债不动

仓库有近两千条历史风格问题，一次性 `ruff format` 全仓会让 review 失效，也容易引入回归。
所以采用**增量判定**：

```bash
python scripts/ci/check_style.py --base origin/main --head HEAD
```

1. 用 `git diff -U0` 解析出本次**新增的行区间**；
2. 用 `ci/style-ruff.toml`（Google 风格规则集）跑 ruff；
3. 只保留落在新增行上的问题 → 有则失败。

| 规则 | 来源 | 说明 |
|---|---|---|
| `E` `W` | pycodestyle | 基本格式 |
| `F` | pyflakes | 未使用导入/变量、未定义名 |
| `I` | isort | 导入分组与顺序 |
| `N` | pep8-naming | Google 用 `snake_case` / `CapWords` |
| `D` | pydocstyle（`convention = "google"`） | docstring 结构，Args/Returns/Raises |
| `B` | bugbear | 可变默认参数、循环变量捕获 |
| `C4` `PIE` `RET` `SIM` | 简化与反模式 | 同义反复、可合并的返回 |

行宽取 100（Google 是 80，本仓库表格/中文串较多，100 是可执行折中）。

> 好处：agent 新增的代码从第一行起就合规，而不用先还清历史债。

## 二之四、前端视觉证据

静态检查能发现 JS 语法错，但发现不了「按钮跑到屏幕外」「深色主题下文字看不见」
「移动端布局塌了」—— 这些只有渲染出来才知道。

```bash
# 本地（需先起服务）
python -m uvicorn main:app --app-dir backend/app --port 8000 &
python scripts/ci/frontend_screenshots.py --out artifacts/screenshots
```

产物：首页 / 宏观罗盘 / 回测 × 桌面(1440×900) / 移动端(390×844) = 6 张图 + `summary.json`，
CI 里作为 `frontend-screenshots` artifact 保留 14 天。

**只报告不阻断**：CI 是干净 checkout，没有 `data/`，部分面板本来就是空状态；
截图用于 review 时肉眼比对，而不是当断言。未捕获的 JS 异常会写进 Step Summary。

## 三、变更面过滤

`scripts/ci/changed_areas.py` 按路径判断这次动了什么：

| 区域 | 命中路径 | 触发的 job |
|---|---|---|
| backend | `backend/`、`scripts/`、`requirements*`、`pytest.ini`、`ruff.toml` | static + tests |
| frontend | `frontend/` | frontend |
| ci | `.github/`、`ci/`、`scripts/ci/` | **全部**（改 CI 本身必须验证整条链路） |
| docs | `docs/`、`README*` | 都不触发（只跑 pr-meta） |

## 三之一、偶发失败（flaky）处理

```bash
python scripts/ci/run_tests.py backend/tests/test_macro.py
```

| 第 1 次 | 重跑 | 判定 | 是否阻断 |
|---|---|---|---|
| 通过 | — | 正常 | 否 |
| 失败 | 通过 | **FLAKY** | 否，但明细写进 Step Summary |
| 失败 | 失败 | 真失败 | 是 |

为什么 **整片重跑**而不是只跑失败用例：只重跑失败用例会掩盖「测试间互相污染」，
而那种问题恰恰是改动引入回归的早期信号。

判定为 flaky 后要做的不是反复重试，而是**消除不确定性**：
固定随机种子、mock 时间/网络、或调整断言粒度。

## 三之二、覆盖率 ratchet

```bash
pytest backend/tests --cov=backend --cov-report=json:coverage.json
python scripts/ci/check_coverage.py --report coverage.json
```

- 基线 **34.7%**，按「CI 口径」记录（干净 checkout 无 `data/`，32 个 `requires_data` 测试被跳过）；
  本地有数据时约 37.5%，只会更高，不会误报；
- 下降超过 0.2pp → 失败（容忍跨环境抖动）；
- 失败时打印覆盖率最低的 12 个文件，agent 知道该补哪里。

它挡不住「断言等于实现」的假测试，但能挡住「一个断言都没有」。
真正的一致性要靠 spec 派生的测试，见 [agent-ci.md](agent-ci.md) 第七节。

## 三之三、依赖漏洞审计

agent 引依赖最容易出事：typosquat 包、已知 RCE 版本的工具库，都会直接进生产路径。

```bash
python scripts/ci/check_dependencies.py --requirements requirements-ci.txt
```

| 设计 | 原因 |
|---|---|
| 只在 `requirements*` 变更时跑 | 漏洞库每天更新，否则每周都有 PR 因上游新披露而红 |
| 豁免要有到期日 | `ci/dependency-audit.json` 的 `expires` 过期即失效，避免「加白名单就忘」 |
| 当前基线 0 条 | 71 个依赖无已知漏洞，任何新增都会被视为回归 |

豁免条目格式：

```json
{ "id": "GHSA-xxxx", "package": "foo", "reason": "仅本地脚本使用，不暴露网络", "expires": "2026-12-31" }
```

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

## 十之三、受保护路径守卫

允许 agent 自主合并的前提是「它改不动判分标准」：
`pull_request_target` + base 分支脚本 + 人工标签。
详见 [protected-paths.md](protected-paths.md)。

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
