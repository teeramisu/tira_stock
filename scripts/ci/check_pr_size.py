"""PR 体量预算：agent 容易一口气改 3000 行，让 review 失效。

软上限：超过后要求 `large-pr` 标签（人工确认「这是有意的」）
硬上限：直接失败，必须拆 PR

用法：
    python scripts/ci/check_pr_size.py --base origin/main --head HEAD
    python scripts/ci/check_pr_size.py --files 120,900 --labels large-pr   # 自测
"""
from __future__ import annotations

import argparse
import subprocess
import sys

SOFT_FILES = 40
SOFT_LINES = 1500
HARD_FILES = 120
HARD_LINES = 4000
SOFT_LABEL = "large-pr"

# 这些目录的改动不计入体量（锁文件、快照、二进制格式）
IGNORE_PREFIXES = ("data/", "backend/rl/models/")
IGNORE_SUFFIXES = (".min.js", ".lock", ".svg", ".png", ".jpg", ".snap")


def diff_stat(base: str, head: str) -> tuple[list[str], int, int]:
    proc = subprocess.run(
        ["git", "diff", "--numstat", f"{base}...{head}"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        print(f"⚠️  git diff 失败：{proc.stderr.strip()}")
        return [], 0, 0

    files: list[str] = []
    added = deleted = 0
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        a, d, path = parts
        if path.startswith(IGNORE_PREFIXES) or path.endswith(IGNORE_SUFFIXES):
            continue
        files.append(path)
        added += int(a) if a.isdigit() else 0
        deleted += int(d) if d.isdigit() else 0
    return files, added, deleted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--labels", default="", help="逗号分隔的 PR 标签")
    parser.add_argument("--files", default="", help="自测：逗号分隔的文件列表")
    parser.add_argument("--lines", type=int, default=-1, help="自测：改动行数")
    args = parser.parse_args()

    if args.files:
        files = [f for f in args.files.split(",") if f]
        added = args.lines if args.lines >= 0 else 0
        deleted = 0
    else:
        files, added, deleted = diff_stat(args.base, args.head)

    lines = added + deleted
    labels = {label.strip() for label in args.labels.split(",") if label.strip()}

    print(f"变更：{len(files)} 个文件，+{added} / -{deleted}（共 {lines} 行）")
    print(f"软上限：{SOFT_FILES} 文件 / {SOFT_LINES} 行   硬上限：{HARD_FILES} 文件 / {HARD_LINES} 行")

    hard_hit = len(files) > HARD_FILES or lines > HARD_LINES
    soft_hit = len(files) > SOFT_FILES or lines > SOFT_LINES

    if hard_hit:
        print("\n❌ 超过硬上限，必须拆分成多个 PR：")
        print("   建议按模块或「重构 / 新功能 / 测试」拆分，每个 PR 只做一件事")
        top = sorted(files)[:15]
        for f in top:
            print(f"   - {f}")
        if len(files) > len(top):
            print(f"   ... 共 {len(files)} 个文件")
        return 1

    if soft_hit:
        if SOFT_LABEL in labels:
            print(f"\n✅ 已有人工打上 `{SOFT_LABEL}` 标签，放行")
            return 0
        print(f"\n❌ 超过软上限：请补 `{SOFT_LABEL}` 标签（人工确认），或拆分 PR")
        return 1

    print("\n✅ 体量在预算内")
    return 0


if __name__ == "__main__":
    sys.exit(main())
