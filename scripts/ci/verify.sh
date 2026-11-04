#!/usr/bin/env bash
# 本地 CI 镜像：CI 里跑的每一步，都能在这里一条命令复现。
#
# 原则（Lauren Tan 的 "trust first"）：CI 里不允许有本地跑不了的检查。
# 本地红了就不要推，推了 CI 也不会给你惊喜。
#
# 用法：
#   scripts/ci/verify.sh            # 全量（约 40s）
#   scripts/ci/verify.sh --quick    # 只跑静态层 + macro 分片
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="$(pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
QUICK=0
FULL=0
[[ "${1:-}" == "--quick" ]] && QUICK=1
[[ "${1:-}" == "--full" ]] && FULL=1

have() { command -v "$1" >/dev/null 2>&1; }
step() { printf '\n\033[1;36m▶ %s\033[0m\n' "$1"; }
ok()   { printf '\033[1;32m✔ %s\033[0m\n' "$1"; }

step "静态分析（阻断规则 E9/F63/F7/F82）"
"$PY" -m ruff check backend scripts
ok "ruff 阻断规则通过"

step "Lint 债务 ratchet"
"$PY" scripts/ci/lint_ratchet.py

step "仓库卫生（大文件 / 本地数据 / 密钥文件）"
"$PY" scripts/ci/check_hygiene.py

step "密钥泄露扫描"
"$PY" scripts/ci/check_secrets.py

step "分层边界"
"$PY" scripts/ci/check_layers.py

step "前端语法与资源引用"
"$PY" scripts/ci/check_frontend.py

step "测试分片"
"$PY" scripts/ci/shard_tests.py --validate
if [[ $QUICK -eq 1 ]]; then
  "$PY" -m pytest $(scripts/ci/shard_tests.py --list macro-discovery | tr '\n' ' ') -q
else
  "$PY" -m pytest backend/tests -q
fi

step "分支命名与提交信息"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
"$PY" scripts/ci/check_commits.py --branch "$BRANCH"
if git rev-parse --verify --quiet origin/main >/dev/null; then
  "$PY" scripts/ci/check_commits.py --range "origin/main..HEAD" || {
    echo "提交信息不符合 Conventional Commits，见 CONTRIBUTING.md"; exit 1; }
else
  echo "（未找到 origin/main，跳过提交范围检查）"
fi

step "PR 范围与体量（对比 origin/main）"
if git rev-parse --verify --quiet origin/main >/dev/null; then
  "$PY" scripts/ci/check_pr_scope.py --base origin/main --head HEAD
  "$PY" scripts/ci/check_pr_size.py --base origin/main --head HEAD
else
  echo "（未找到 origin/main，跳过）"
fi

step "Google 风格（仅新增行）"
"$PY" scripts/ci/check_style.py --base origin/main --head HEAD

if [[ $FULL -eq 1 ]]; then
  step "覆盖率 ratchet（--full 才跑，约 30s）"
  "$PY" -m pytest backend/tests -q -p no:cacheprovider \
    --cov=backend --cov-report=json:coverage.json >/dev/null
  "$PY" scripts/ci/check_coverage.py --report coverage.json | tail -3
fi

printf '\n\033[1;32m全部通过 —— 可以推送\033[0m\n'
