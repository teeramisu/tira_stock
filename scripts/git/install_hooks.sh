#!/usr/bin/env bash
# 安装本地 git hooks（不会覆盖已有 hook，会先备份）
#
#   scripts/git/install_hooks.sh                 # 只装 commit-msg
#   scripts/git/install_hooks.sh --with-pre-push # 额外在 push 前跑一遍 quick verify
#
# hook 内容与 CI 使用同一套脚本，本地绿 = CI 绿。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="$(pwd)"
HOOKS="$ROOT/.git/hooks"
WITH_PRE_PUSH=0
[[ "${1:-}" == "--with-pre-push" ]] && WITH_PRE_PUSH=1

[[ -d "$HOOKS" ]] || { echo "❌ 当前目录不是 git 仓库：$ROOT"; exit 1; }

backup() {
  if [[ -f "$1" ]] && ! grep -q "tira_stock managed hook" "$1" 2>/dev/null; then
    mv "$1" "$1.bak.$(date +%Y%m%d%H%M%S)"
    echo "  已备份原 hook → $(basename "$1").bak.*"
  fi
}

# ---------------------------------------------------------------- commit-msg
backup "$HOOKS/commit-msg"
cat > "$HOOKS/commit-msg" <<'HOOK'
#!/usr/bin/env bash
# tira_stock managed hook —— 校验提交信息格式（Conventional Commits）
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
PY="python3"
[[ -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"
exec "$PY" scripts/ci/check_commits.py --msg-file "$1"
HOOK
chmod +x "$HOOKS/commit-msg"
echo "✅ commit-msg hook 已安装（提交信息格式校验）"

# ---------------------------------------------------------------- pre-push
if [[ $WITH_PRE_PUSH -eq 1 ]]; then
  backup "$HOOKS/pre-push"
  cat > "$HOOKS/pre-push" <<'HOOK'
#!/usr/bin/env bash
# tira_stock managed hook —— push 前跑 quick verify
# 跳过：SKIP_PRE_PUSH=1 git push
set -euo pipefail
[[ "${SKIP_PRE_PUSH:-0}" == "1" ]] && exit 0
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
echo "▶ pre-push: 运行 quick verify（跳过用 SKIP_PRE_PUSH=1）"
exec bash scripts/ci/verify.sh --quick
HOOK
  chmod +x "$HOOKS/pre-push"
  echo "✅ pre-push hook 已安装（push 前 quick verify）"
fi

echo
echo "分支命名：<type>/<短横线描述>   例：feat/macro-provinces"
echo "提交信息：<type>(<scope>): <描述> 例：feat(macro): 增加省级变化对比"
echo "完整规范：CONTRIBUTING.md"
