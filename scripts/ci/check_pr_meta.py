"""PR 元信息检查：标题格式 + 描述结构 + 回退说明。

遵循 Google CL 的描述习惯（第一行短摘要 / 正文说明 what & why / 测试记录），
理由很实际：**PR 描述就是回滚时的说明书**。没有「为什么」和「怎么验」，
出问题时只能靠猜。

用法：
    python scripts/ci/check_pr_meta.py --title "feat(macro): ..." --body-file body.md
    python scripts/ci/check_pr_meta.py --title ... --body "$BODY" --labels feat
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TITLE = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[a-z0-9._/-]+)\))?!?: (?P<subject>.+)$")
MIN_BODY = 120

# 每组至少命中一个（中英文都收）
SECTIONS: dict[str, tuple[str, ...]] = {
    "背景/动机（为什么改）": ("背景", "动机", "why", "context", "problem"),
    "改动内容（改了什么）": ("改动", "变更", "what", "changes", "实现"),
    "验证方式（怎么证明是对的）": ("验证", "测试", "tested", "test plan", "检查"),
}

# 这些类型的改动更需要回退说明
NEEDS_ROLLBACK = {"feat", "refactor", "perf"}
ROLLBACK_HINTS = ("回退", "回滚", "revert", "rollback")


def headings(body: str) -> list[str]:
    return [line.lstrip("#").strip().lower() for line in body.splitlines() if line.startswith("#")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", default="")
    parser.add_argument("--body-file", default="")
    parser.add_argument("--labels", default="")
    args = parser.parse_args()

    body = Path(args.body_file).read_text(encoding="utf-8") if args.body_file else args.body
    body = body or ""
    labels = {label.strip() for label in args.labels.split(",") if label.strip()}

    problems: list[str] = []
    warnings: list[str] = []

    # ---------------- 标题
    match = TITLE.match(args.title)
    if not match:
        problems.append(
            f"标题不符合 Conventional Commits：`{args.title}`\n"
            "     期望：<type>(<scope>): <描述>，例：feat(macro): 增加省级变化对比"
        )
    else:
        if len(args.title) > 100:
            problems.append(f"标题过长（{len(args.title)} > 100 字符）")
        if args.title.lower().startswith(("wip", "fixup")):
            problems.append("草稿请用 Draft 状态，不要在标题写 WIP")

    # ---------------- 描述长度
    if len(body.strip()) < MIN_BODY:
        problems.append(f"描述太短（{len(body.strip())} < {MIN_BODY} 字符），说明改了什么、怎么验证的")

    # ---------------- 结构化小节
    present = headings(body)
    missing = [name for name, keys in SECTIONS.items() if not any(k in h for h in present for k in keys)]
    for name in missing:
        problems.append(f"缺少小节约 `## {name.split('（')[0]}`")

    # ---------------- 回退说明
    ctype = match.group("type") if match else ""
    has_rollback = any(k in line for line in present for k in ROLLBACK_HINTS) or any(
        k in body.lower() for k in ROLLBACK_HINTS
    )
    if ctype in NEEDS_ROLLBACK and not has_rollback:
        warnings.append(
            f"`{ctype}` 类改动建议加一节 `## 回退`（revert 这条 PR 是否安全？有没有数据/迁移副作用？）"
        )

    # ---------------- 验证证据
    if not re.search(r"(passed|通过|✅|verify\.sh|actions/runs|pytest)", body, re.IGNORECASE):
        warnings.append("描述里没有验证证据（测试结果 / CI 链接 / 手动验证步骤）")

    for w in warnings:
        print(f"⚠️  {w}")
    if problems:
        print()
        for p in problems:
            print(f"❌ {p}")
        print("\n模板见 CONTRIBUTING.md 第三节")
        return 1

    print("✅ PR 元信息检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
