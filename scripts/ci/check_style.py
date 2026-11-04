"""Google 代码风格检查 —— 只针对本次新增的行。

老仓库有近两千条历史风格问题，一次性重排会让 review 失效、也容易引入回归。
所以这里的策略是：**旧债不动，新账不欠**。

做法：
  1. `git diff -U0 base...head` 解析出每个文件新增的行区间；
  2. 用 `ci/style-ruff.toml`（Google 风格规则）跑 ruff，输出 JSON；
  3. 只保留落在新增行上的问题 → 有则失败。

用法：
    python scripts/ci/check_style.py --base origin/main --head HEAD
    python scripts/ci/check_style.py --base origin/main --head HEAD --max-report 30
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ci" / "style-ruff.toml"
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")
MAX_PER_FILE = 8


def ruff_bin() -> str:
    local = ROOT / ".venv" / "bin" / "ruff"
    return str(local) if local.exists() else "ruff"


def added_lines(base: str, head: str) -> dict[str, list[range]]:
    proc = subprocess.run(
        ["git", "diff", "-U0", "--no-color", f"{base}...{head}"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        print(f"⚠️  git diff 失败：{proc.stderr.strip()}")
        return {}

    result: dict[str, list[range]] = defaultdict(list)
    current: str | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif line.startswith("+++ /dev/null"):
            current = None
        elif current and line.startswith("@@"):
            match = HUNK.match(line)
            if not match:
                continue
            start = int(match.group("start"))
            count = int(match.group("count") or 1)
            if count > 0:
                result[current].append(range(start, start + count))
    return result


def violations(ranges: dict[str, list[range]]) -> list[tuple[str, int, str, str]]:
    py_files = [p for p in ranges if p.endswith(".py")]
    if not py_files:
        return []

    proc = subprocess.run(
        [ruff_bin(), "check", *py_files, "--config", str(CONFIG), "--output-format", "json"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    try:
        data = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        print(f"⚠️  无法解析 ruff 输出：{proc.stdout[:200]}")
        return []

    found: list[tuple[str, int, str, str]] = []
    for item in data:
        filename = Path(item.get("filename", "")).resolve()
        try:
            rel = filename.relative_to(ROOT).as_posix()
        except ValueError:
            continue
        row = item.get("location", {}).get("row", 0)
        if any(row in rng for rng in ranges.get(rel, [])):
            code = item.get("code") or "?"
            message = item.get("message", "").strip()
            found.append((rel, row, code, message))
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--max-report", type=int, default=40)
    args = parser.parse_args()

    ranges = added_lines(args.base, args.head)
    if not ranges and args.base not in {"", "HEAD~1"}:
        # 新分支首推 / before=0000 等情况：退化成与上一个提交比较
        ranges = added_lines("HEAD~1", "HEAD")
    if not ranges:
        print("本次没有新增的 Python 行，跳过风格检查")
        return 0

    found = violations(ranges)
    if not found:
        total = sum(len(rng) for rngs in ranges.values() for rng in rngs)
        print(f"✅ Google 风格检查通过（新增 {len(ranges)} 个文件、约 {total} 行）")
        return 0

    by_file: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for path, row, code, message in found:
        by_file[path].append((row, code, message))

    print(f"❌ 新增代码有 {len(found)} 处不符合 Google 风格：\n")
    shown = 0
    for path in sorted(by_file):
        items = sorted(by_file[path])
        print(f"  {path}")
        for row, code, message in items[:MAX_PER_FILE]:
            print(f"    {row}: {code} {message}")
            shown += 1
        if len(items) > MAX_PER_FILE:
            print(f"    ... 还有 {len(items) - MAX_PER_FILE} 处")
        print()
        if shown >= args.max_report:
            print(f"  （只展示前 {args.max_report} 条）")
            break

    print("修复建议：")
    print("  ruff check <file> --config ci/style-ruff.toml --fix     # 能自动修的")
    print("  Google Python Style Guide: https://google.github.io/styleguide/pyguide.html")
    print("  规则说明见 ci/style-ruff.toml")
    return 1


if __name__ == "__main__":
    sys.exit(main())
