"""自动合并（GitHub Auto-merge / Merge Queue）。

agent 的最后一环：CI 绿了就自己合，不用人点。前提是仓库开了 Merge Queue，
否则 auto-merge 会在「分支落后于 main」时直接合，跳过组合验证。

用法：
    python scripts/agent/auto_merge.py --repo teeramisu/tira_stock --pr 12 --method squash
    python scripts/agent/auto_merge.py --repo ... --pr 12 --disable
    python scripts/agent/auto_merge.py --repo ... --pr 12 --wait --timeout 1800

需要 token 权限：Pull requests: Read and write（不需要 administration）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import requests

API = "https://api.github.com"
GRAPHQL = "https://api.github.com/graphql"

ENABLE = """
mutation($id: ID!, $method: PullRequestMergeMethod!) {
  enablePullRequestAutoMerge(input: {pullRequestId: $id, mergeMethod: $method}) {
    pullRequest { number isDraft autoMergeRequest { enabledAt mergeMethod } }
  }
}
"""

DISABLE = """
mutation($id: ID!) {
  disablePullRequestAutoMerge(input: {pullRequestId: $id}) {
    pullRequest { number autoMergeRequest { enabledAt } }
  }
}
"""


def token() -> str:
    value = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not value:
        print("❌ 缺少 GITHUB_TOKEN")
        raise SystemExit(2)
    return value


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token()}", "Accept": "application/vnd.github+json"})
    return s


def get_pr(s: requests.Session, repo: str, pr: int) -> dict:
    resp = s.get(f"{API}/repos/{repo}/pulls/{pr}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def graphql(s: requests.Session, query: str, variables: dict) -> dict:
    resp = s.post(GRAPHQL, json={"query": query, "variables": variables}, timeout=30)
    body = resp.json()
    if body.get("errors"):
        raise RuntimeError(body["errors"][0].get("message", "graphql error"))
    return body["data"]


def describe(pr: dict) -> str:
    return (
        f"#{pr['number']} {pr['state']} draft={pr['draft']} "
        f"mergeable={pr.get('mergeable')} state={pr.get('mergeable_state')} "
        f"checks_ok={pr.get('mergeable_state') in {'clean', 'unstable', 'has_hooks', None}}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--method", default="squash", choices=["squash", "merge", "rebase"])
    parser.add_argument("--disable", action="store_true")
    parser.add_argument("--wait", action="store_true", help="等到合并完成")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    s = session()
    pr = get_pr(s, args.repo, args.pr)
    print("当前状态：", describe(pr))

    if pr["state"] != "open":
        print(f"PR 状态是 {pr['state']}，无需操作")
        return 0
    if pr["draft"]:
        print("❌ PR 还是 draft，先标记 ready for review")
        return 1

    if args.disable:
        graphql(s, DISABLE, {"id": pr["node_id"]})
        print("✅ 已关闭 auto-merge")
        return 0

    if (pr.get("auto_merge") or {}).get("enabled_at"):
        print(f"auto-merge 已开启（{pr['auto_merge'].get('merge_method')}）")
    else:
        try:
            data = graphql(s, ENABLE, {"id": pr["node_id"], "method": args.method.upper()})
        except RuntimeError as exc:
            message = str(exc)
            print(f"❌ 开启 auto-merge 失败：{message}")
            if "not enabled" in message.lower() or "auto-merge" in message.lower():
                print("   提示：仓库需开启 Settings → General → Allow auto-merge")
                print("        且启用 Merge Queue 时本仓库需是 public 或组织仓库")
            if "not mergeable" in message.lower():
                print("   提示：可能有冲突或 required check 未配置，先修 PR")
            return 1
        pr_node = data["enablePullRequestAutoMerge"]["pullRequest"]
        print(f"✅ 已开启 auto-merge：#{pr_node['number']} method={args.method}")

    if not args.wait:
        return 0

    print(f"等待合并（最多 {args.timeout}s）...")
    deadline = time.time() + args.timeout
    last = ""
    while time.time() < deadline:
        pr = get_pr(s, args.repo, args.pr)
        state = f"{pr['state']} merged={pr['merged']} {pr.get('mergeable_state')}"
        if state != last:
            print("  ", state)
            last = state
        if pr["merged"]:
            print(f"✅ 已合并：{pr['html_url']}")
            return 0
        if pr["state"] == "closed":
            print("⚠️ PR 被关闭但未合并")
            return 1
        time.sleep(20)

    print("⚠️ 等待超时，PR 仍未合并（检查 CI 与合并队列）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
