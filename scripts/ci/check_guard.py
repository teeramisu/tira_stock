"""CI 完整性守卫：agent 不能靠「改考卷」让 CI 变绿。

为什么单独一个 workflow：

    普通 ci.yml 用的是 PR 里的脚本，agent 改了脚本就等于改了判分标准。
    本 workflow 用 pull_request_target 触发，checkout 的是 **base 分支**的
    check_guard.py，再通过 API 比对 PR 里的文件是否动了受保护路径。
    整个过程中不执行 PR 里的任何代码。

放行方式：
  1. `allow_pure_additions` 里的路径（默认 backend/tests/）只要「只增不删」就自动放行
     —— 新增测试无法削弱既有断言，删改才算；
  2. 其余情况需人工给 PR 加 `allow-guard-change` 标签（agent token 不应有 issues:write，
     所以它自己加不了）。详见 docs/protected-paths.md。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
PROTECTED_FILE = ROOT / "ci" / "protected-paths.json"
LABEL = "allow-guard-change"


def api(session: requests.Session, url: str) -> dict | list:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def pr_files(session: requests.Session, repo: str, pr: int) -> list[dict]:
    """返回 [{filename, additions, deletions}]，用于区分「新增」与「删改」。"""
    files: list[dict] = []
    page = 1
    while page <= 10:
        url = f"https://api.github.com/repos/{repo}/pulls/{pr}/files?per_page=100&page={page}"
        batch = api(session, url)
        if not batch:
            break
        files += [
            {
                "filename": f["filename"],
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
            }
            for f in batch
        ]
        page += 1
    return files


def matches(path: str, patterns: list[str]) -> bool:
    return any(path == p or path.startswith(p) for p in patterns)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("REPO", ""))
    parser.add_argument("--pr", type=int, default=0)
    parser.add_argument("--labels", default=os.environ.get("PR_LABELS", ""), help="逗号分隔的标签")
    parser.add_argument("--files", default="", help="逗号分隔的变更文件（本地调试，按有删改处理）")
    parser.add_argument(
        "--files-json",
        default="",
        help='本地调试：[{"filename":..,"additions":..,"deletions":..}]',
    )
    args = parser.parse_args()

    config = json.loads(PROTECTED_FILE.read_text(encoding="utf-8"))
    patterns = config["patterns"]
    pure_addition_prefixes = config.get("allow_pure_additions", [])

    if args.files:
        files = [
            {"filename": f, "additions": 0, "deletions": 1}
            for f in args.files.split(",")
            if f
        ]
    elif args.files_json:
        files = json.loads(args.files_json)
    else:
        if not (args.repo and args.pr):
            print("❌ 需要 --repo 与 --pr，或用 --files 本地调试")
            return 2
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not token:
            print("❌ 缺少 GITHUB_TOKEN")
            return 2
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
        files = pr_files(session, args.repo, args.pr)

    touched = [f for f in files if matches(f["filename"], patterns)]
    if not touched:
        print(f"✅ CI 守卫通过（{len(files)} 个变更文件，未触碰受保护路径）")
        return 0

    def is_pure_addition(entry: dict) -> bool:
        return (
            entry.get("deletions", 0) == 0
            and entry.get("additions", 0) > 0
            and any(entry["filename"].startswith(p) for p in pure_addition_prefixes)
        )

    allowed_additions = [f for f in touched if is_pure_addition(f)]
    blocking = [f for f in touched if not is_pure_addition(f)]

    if allowed_additions:
        print(f"纯新增放行（{len(allowed_additions)} 个，deletions=0）：")
        for f in allowed_additions[:10]:
            print(f"  + {f['filename']} (+{f['additions']})")
        if len(allowed_additions) > 10:
            print(f"  ... 还有 {len(allowed_additions) - 10} 个")

    if not blocking:
        print("\n✅ 受保护路径只有新增内容，未削弱既有断言，放行")
        return 0

    touched = blocking

    labels = {label.strip() for label in args.labels.split(",") if label.strip()}
    print(f"受保护路径被修改或删除（{len(touched)} 个）：")
    for f in touched[:20]:
        print(f"  - {f['filename']} (+{f['additions']} / -{f['deletions']})")
    if len(touched) > 20:
        print(f"  ... 还有 {len(touched) - 20} 个")

    if LABEL in labels:
        print(f"\n✅ 已有人工打上 `{LABEL}` 标签，放行")
        return 0

    print("\n❌ 受保护路径只允许人工改动")
    print(f"   如果是合理改动，请人工（非 agent 身份）给 PR 加标签：`{LABEL}`")
    print("   受保护范围见 ci/protected-paths.json（CI 脚本、测试、依赖、工作流）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
