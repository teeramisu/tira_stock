"""变更面探测：决定这次 PR 需要跑哪些重活。

输出 GitHub Actions 的 key=value（写进 $GITHUB_OUTPUT），
本地也能直接跑来看结论：

    python scripts/ci/changed_areas.py --base origin/main
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RULES: dict[str, tuple[str, ...]] = {
    "backend": (
        "backend/", "scripts/", "requirements", "pytest.ini", "ruff.toml", "run.sh", "Dockerfile",
    ),
    "frontend": ("frontend/",),
    "ci": (".github/", "ci/", "scripts/ci/"),
    "docs": ("docs/", "README", "DEPLOY_"),
}

# 任一命中都视为「全量重跑」（保守策略：改 CI 本身必须验证整条链路）
FORCE_ALL = ("ci",)


def changed_files(base: str) -> list[str]:
    for rev in (f"{base}...HEAD", "HEAD~1", ""):
        cmd = ["git", "diff", "--name-only"] + ([rev] if rev else ["--cached"])
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            return [line for line in proc.stdout.splitlines() if line.strip()]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--output", default=None, help="写入文件（默认打印）")
    args = parser.parse_args()

    files = changed_files(args.base)
    areas = {name: any(f.startswith(prefix) for f in files for prefix in prefixes)
             for name, prefixes in RULES.items()}

    if any(areas[name] for name in FORCE_ALL):
        areas = {name: True for name in RULES}

    lines = [f"{name}={str(value).lower()}" for name, value in areas.items()]
    lines.append(f"count={len(files)}")

    text = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(f"变更文件 {len(files)} 个")
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
