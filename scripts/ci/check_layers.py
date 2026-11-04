"""分层边界检查：把「架构约定」变成会失败的检查。

比代码评审更可靠：同一个架构意见只需说一次，之后由 CI 记住。

规则（当前仓库实际分层）：
  R1  backend/app/**（API 层）不得导入 scripts/**，也不得反向依赖 tests；
  R2  backend/macro、backend/discovery、backend/portfolio 等核心库
      不得导入 app.*（核心逻辑必须能在无 FastAPI 环境下单独跑）；
  R3  frontend 不得出现对本地绝对路径 / file:// 的硬编码引用。
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"

CORE_PACKAGES = ("macro", "discovery", "portfolio", "marketdata", "factors", "backtest", "research", "ml")


def iter_py(base: Path):
    for path in base.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def imports_of(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []  # 语法错交给 ruff E9
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # 相对导入，模块名不可靠，跳过
                continue
            if node.module:
                found.append((node.lineno, node.module))
    return found


def main() -> int:
    problems: list[str] = []

    # R1: API 层不得依赖 scripts/tests
    for path in iter_py(BACKEND / "app"):
        for lineno, mod in imports_of(path):
            if mod.split(".")[0] in {"scripts", "tests"}:
                problems.append(f"{path.relative_to(ROOT)}:{lineno} API 层禁止导入 {mod}")

    # R2: 核心库不得依赖 API 层
    for pkg in CORE_PACKAGES:
        base = BACKEND / pkg
        if not base.exists():
            continue
        for path in iter_py(base):
            for lineno, mod in imports_of(path):
                if mod.split(".")[0] == "app":
                    problems.append(f"{path.relative_to(ROOT)}:{lineno} 核心库禁止导入 API 层 {mod}")

    # R3: 前端不得硬编码本地路径
    local_ref = re.compile(r'(?:src|href)="(?:file://|/home/|/Users/|C:\\\\)')
    for path in (ROOT / "frontend").rglob("*"):
        if path.suffix not in {".html", ".js", ".css"} or not path.is_file():
            continue
        if path.name == "echarts.min.js":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if local_ref.search(line):
                problems.append(f"{path.relative_to(ROOT)}:{lineno} 前端禁止硬编码本地路径")

    if problems:
        print("❌ 分层边界检查未通过：")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("✅ 分层边界检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
