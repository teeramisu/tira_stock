"""PR 范围检查：一个 PR 只做一件事。

「一件事」在这里被定义成「只碰一个功能区」。混在一起改的 PR：
  - review 只能整体点头或整体拒绝
  - 出问题只能整包 revert
  - CI 失败时分不清是哪一半的锅

规则：
  R1  最多跨 2 个功能区（超过需人工 `multi-area` 标签）
  R2  「业务代码 + CI」永远拆开（ci 与 backend/frontend 不能同框）
  R3  测试可以跟随业务代码（tests 不计入功能区的独立判断）

用法：
    python scripts/ci/check_pr_scope.py --base origin/main --head HEAD
    python scripts/ci/check_pr_scope.py --files a.py,.github/x.yml --labels multi-area
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict

MAX_AREAS = 2
OVERRIDE_LABEL = "multi-area"

# 前缀 → 功能区
AREA_RULES: list[tuple[tuple[str, ...], str]] = [
    (("backend/tests/",), "tests"),
    (("backend/",), "backend"),
    (("frontend/",), "frontend"),
    ((".github/", "scripts/ci/", "scripts/agent/", "ci/"), "ci"),
    (("requirements", "pytest.ini", "ruff.toml", ".coveragerc"), "ci"),
    (("scripts/",), "scripts"),
    (("docs/",), "docs"),
    (("alembic/", "deploy/"), "backend"),
]
DOCS_SUFFIXES = (".md",)


def area_of(path: str) -> str:
    for prefixes, area in AREA_RULES:
        if path.startswith(prefixes):
            return area
    if path.endswith(DOCS_SUFFIXES):
        return "docs"
    return "other"


def changed_files(base: str, head: str) -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        print(f"⚠️  git diff 失败：{proc.stderr.strip()}")
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--labels", default="")
    parser.add_argument("--files", default="", help="自测：逗号分隔的文件列表")
    args = parser.parse_args()

    files = [f for f in args.files.split(",") if f] if args.files else changed_files(args.base, args.head)
    labels = {label.strip() for label in args.labels.split(",") if label.strip()}

    grouped: dict[str, list[str]] = defaultdict(list)
    for f in files:
        grouped[area_of(f)].append(f)

    areas = sorted(a for a in grouped if a not in {"tests", "other"})
    print(f"变更 {len(files)} 个文件，涉及功能区：{', '.join(areas) if areas else '（无）'}")
    for area in areas:
        sample = grouped[area][:4]
        more = f" ... +{len(grouped[area]) - 4}" if len(grouped[area]) > 4 else ""
        print(f"  {area:<10} {len(grouped[area]):>3} 个文件   {', '.join(sample)}{more}")

    problems: list[str] = []

    if "ci" in areas and ({"backend", "frontend", "scripts"} & set(areas)):
        problems.append(
            "CI/门禁改动与业务代码混在同一个 PR（规则 R2）——请拆成两个 PR，"
            "否则无法判断是「代码坏了」还是「判分标准变了」"
        )

    if len(areas) > MAX_AREAS and OVERRIDE_LABEL not in labels:
        problems.append(
            f"跨了 {len(areas)} 个功能区（上限 {MAX_AREAS}）：{', '.join(areas)}；"
            f"如果确实必须一起改，请人工加 `{OVERRIDE_LABEL}` 标签"
        )

    if problems:
        print()
        for p in problems:
            print(f"❌ {p}")
        print("\n拆分建议：按功能区各开一个 PR，从 main 切分支，互不依赖")
        return 1

    print("\n✅ 范围检查通过（一个 PR 一件事）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
