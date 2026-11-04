"""分支名 + 提交信息规范校验（Conventional Commits）。

三种用法：
    python scripts/ci/check_commits.py --msg-file .git/COMMIT_EDITMSG    # 本地 commit-msg hook
    python scripts/ci/check_commits.py --range origin/main..HEAD         # CI：校验本次提交
    python scripts/ci/check_commits.py --branch feat/macro-provinces     # CI：校验分支名

只校验「本次范围」内的提交，不动历史；合并提交自动跳过。
规范说明见 CONTRIBUTING.md。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TYPES = {
    "feat": "新功能",
    "fix": "修复 bug",
    "docs": "文档",
    "style": "格式（不影响逻辑）",
    "refactor": "重构（非新功能、非修复）",
    "perf": "性能优化",
    "test": "测试",
    "build": "构建 / 依赖",
    "ci": "CI / 脚本",
    "chore": "杂项（版本号、配置）",
    "revert": "回滚",
}

# 常用的模块名；写别的 scope 不阻断（只是提示）
KNOWN_SCOPES = {
    "macro", "discovery", "backtest", "portfolio", "marketdata", "factors",
    "frontend", "api", "ci", "docs", "deps", "rl", "research", "memory", "tests",
}

HEADER = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[a-z0-9._/-]+)\))?(?P<breaking>!)?: (?P<subject>.+)$"
)
MERGE = re.compile(r"^Merge (pull request|branch|remote)", re.IGNORECASE)
REVERT = re.compile(r'^Revert "', re.IGNORECASE)
BRANCH = re.compile(r"^(?P<type>[a-z]+)/(?P<slug>[a-z0-9][a-z0-9._-]{2,})$")
MEANINGLESS = {
    "update", "updates", "fix", "fixes", "change", "changes", "wip",
    "修改", "更新", "提交", "改动",
}
PROTECTED_BRANCHES = {"main", "master"}


def validate_header(header: str) -> tuple[list[str], list[str]]:
    """返回 (errors, warnings)。errors 阻断，warnings 只提示。"""
    match = HEADER.match(header)
    if not match:
        return [
            f"格式不符：`{header}`",
            "  期望：<type>(<scope>): <描述>，例：feat(macro): 增加省级变化对比",
            f"  可用 type：{', '.join(TYPES)}",
        ], []

    errors: list[str] = []
    warnings: list[str] = []
    ctype = match.group("type")
    scope = match.group("scope")
    subject = match.group("subject").strip()

    if ctype not in TYPES:
        errors.append(f"未知 type `{ctype}`，可用：{', '.join(TYPES)}")
    if scope and scope not in KNOWN_SCOPES:
        warnings.append(f"scope `{scope}` 不在常用列表（{', '.join(sorted(KNOWN_SCOPES))}）")
    if len(subject) < 4:
        errors.append(f"描述太短（`{subject}`），至少 4 个字符")
    if len(header) > 100:
        errors.append(f"标题过长（{len(header)} > 100 字符），细节放到正文")
    if subject.endswith((".", "。")):
        errors.append("描述结尾不要加句号")
    if subject.lower() in MEANINGLESS:
        errors.append(f"描述太笼统（`{subject}`），说明具体改了什么")
    return errors, warnings


def validate_branch(name: str) -> list[str]:
    if name in PROTECTED_BRANCHES:
        return []
    match = BRANCH.match(name)
    if not match:
        return [
            f"分支名不合规：`{name}`",
            "  期望：<type>/<短横线描述>，例：feat/macro-provinces、fix/portfolio-logger",
            f"  可用 type：{', '.join(TYPES)}",
        ]
    problems = []
    if match.group("type") not in TYPES:
        problems.append(f"未知分支前缀 `{match.group('type')}`，可用：{', '.join(TYPES)}")
    return problems


def read_message_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    body = [ln for ln in lines if not ln.startswith("#")]
    while body and not body[-1].strip():
        body.pop()
    return [body[0]] if body else []


def read_range(rev_range: str) -> list[str]:
    proc = subprocess.run(
        ["git", "log", "--no-merges", "--format=%s", rev_range],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        print(f"⚠️  无法读取提交范围 {rev_range}：{proc.stderr.strip()}")
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--msg-file", help="单个提交信息文件（commit-msg hook）")
    group.add_argument("--range", help="提交范围，如 origin/main..HEAD")
    group.add_argument("--branch", help="分支名（CI 校验 head 分支）")
    args = parser.parse_args()

    failures = 0

    if args.branch is not None:
        problems = validate_branch(args.branch)
        if problems:
            failures += 1
            print(f"❌ {args.branch}")
            for p in problems:
                print(f"   {p}")
        headers: list[str] = []
    elif args.msg_file:
        headers = read_message_file(Path(args.msg_file))
    else:
        headers = read_range(args.range)

    for header in headers:
        if MERGE.match(header) or REVERT.match(header):
            continue
        errors, warnings = validate_header(header)
        for w in warnings:
            print(f"⚠️  {header}\n   {w}（不阻断）")
        if errors:
            failures += 1
            print(f"❌ {header}")
            for p in errors:
                print(f"   {p}")

    if failures:
        print(f"\n共 {failures} 处不合规。规则见 CONTRIBUTING.md")
        print("示例：")
        print("  分支：feat/macro-provinces")
        print("  提交：feat(macro): 增加省级变化对比")
        print("  提交：fix(portfolio): 修复策略注册表 logger 未定义")
        return 1

    checked = len(headers) + (1 if args.branch is not None else 0)
    print(f"✅ 规范检查通过（{checked} 项）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
