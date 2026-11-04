---
name: tira-verify
description: >-
  在 tira_stock 里跑本地验证、读懂失败、把结果整理成可复核的证据。
  当需要确认改动没坏、CI 红了要定位、或要判断「这次失败是偶发还是真错」时调用。
license: MIT
compatibility: opencode
metadata:
  version: v1.0.0
  repo: tira_stock
---

# tira-verify

## 一条命令

```bash
scripts/ci/verify.sh --full     # 10 步 + 覆盖率，约 60~90s
scripts/ci/verify.sh            # 快速版（不含覆盖率），约 40s
```

**本地跑的东西必须和 CI 一样。** 如果 CI 上有一项本地跑不了，那说明设计有问题。

## 十步分别是什么、红了怎么办

| # | 步骤 | 典型失败原因 | 修法 |
|---|---|---|---|
| 1 | ruff 阻断规则（E9/F63/F7/F82） | 未定义名、语法错 | 按提示补导入/改名；`--fix` 能修一部分 |
| 2 | lint 债务 ratchet | 新增了未使用变量/导入等 | 删掉多余代码；**不要**去改基线 |
| 3 | 仓库卫生 | 大文件、`data/`、`.env` 被加入索引 | `git rm --cached <path>`，补 `.gitignore` |
| 4 | 密钥扫描 | 疑似 `sk-*` / `ghp_*` / 带密码连接串 | 删掉，必要时轮换 key |
| 5 | 分层边界 | API 层导了脚本、核心库导了 app | 调整依赖方向 |
| 6 | 前端语法/资源 | JS 语法错、引用了不存在的 js/css | 修路径或补文件 |
| 7 | 测试分片 | 测试失败 | 见下方「失败分类」 |
| 8 | 分支名 + 提交信息 | 分支不是 `<type>/<slug>`、提交不是 Conventional | `git branch -m` / `git commit --amend` |
| 9 | PR 范围与体量 | 跨了 2 个以上功能区、超过软上限 | 拆 PR |
| 10 | Google 风格（新增行） | 新行的命名/导入/docstring 不合规 | `ruff check <file> --config ci/style-ruff.toml --fix` |

## 失败分类（先分类，再动手）

| 现象 | 分类 | 处理 |
|---|---|---|
| 同一个用例每次本地都红 | **真失败** | 修代码或修测试 |
| 本地绿、CI 红，且是网络/外部接口 | **环境差异** | 检查是否需要 `requires_data` / `network` 标记 |
| 第 1 次红、重跑绿 | **FLAKY** | 消除不确定性（固定种子、mock 时间/网络），不要反复重试 |
| 只有某个分片红 | **范围问题** | 看该分片的测试文件是否互相污染 |
| 报 `未登记到任何分片` | **命名问题** | 测试文件要叫 `test_<模块>*.py` |

## 判断「偶发」的正确姿势

CI 已经实现了「失败自动重跑一次」：

- 重跑通过 → 标记 FLAKY，**不阻断**，明细写进 Step Summary
- 重跑仍失败 → 真失败

本地没有这个机制。如果怀疑偶发：

```bash
for i in 1 2 3; do pytest <用例> -q || break; done
```

三次都过 → 大概率是环境；三次内有失败 → 真问题。

## 证据怎么记

```markdown
## 证据
- `scripts/ci/verify.sh --full` → 全绿
  ```
  覆盖率 ratchet 通过  整体 34.7%（+0.0pp）
  385 passed, 2 skipped, 7 xfailed
  ```
- CI：https://github.com/teeramisu/tira_stock/actions/runs/<id> → CI 汇总 success
```

**贴输出，不要贴结论。** 「测试通过」不是证据，「385 passed」才是。

## 什么时候**不要**跑全量

| 场景 | 建议 |
|---|---|
| 只改文档 | `verify.sh` 会跳过大部分（变更面过滤），可以直接跑 |
| 只改 `.opencode/skills/` | 同上，无代码影响 |
| 正在调试单个函数 | 先 `pytest <单个文件> -q`，收敛后再跑全量 |

## 常见陷阱

| 陷阱 | 说明 |
|---|---|
| 用 `.venv/bin/pytest` 但不设 `PYTHONPATH` | 仓库已在 `pytest.ini` 配了 `pythonpath = backend`，直接 `pytest` 即可 |
| 覆盖率本地 37.5% 但 CI 34.7% | 正常：CI 无 `data/`，32 个 `requires_data` 测试被跳过 |
| 改了 `ci/coverage-baseline.json` 来「修复」红 | 这是改判分标准，需要单独 PR + 理由 |
