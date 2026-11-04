"""一个 PR 周期的执行器：验 → 提交 → 推 → 开 PR → 自动合并。

设计边界（重要）：
    脚本**不猜代码怎么改**。CI 红了就拉日志、生成「可修摘要」、以退出码 2 返回，
    由调用它的 agent 决定怎么修。自动改代码在多文件仓库里不可靠，也会掩盖真问题。

用法：
    python scripts/agent/loop.py --task "feat(macro): 增加省级排名变化" --type feat
    python scripts/agent/loop.py --task "..." --dry-run          # 只跑到本地验证
    python scripts/agent/loop.py --task "..." --no-merge         # 开 PR 但不自动合并

前置：AGENT_GITHUB_TOKEN（或 GITHUB_TOKEN）已配置；见 docs/agent-token.md。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
API = "https://api.github.com"
VALID_TYPES = ("feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert")
HEADER = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[a-z0-9._/-]+)\))?!?: (?P<subject>.+)$")


def run(cmd: list[str], *, check: bool = False, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=check, timeout=timeout)


def token() -> str:
    for key in ("AGENT_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(key)
        if value:
            if key != "AGENT_GITHUB_TOKEN":
                print(f"⚠️  未设置 AGENT_GITHUB_TOKEN，回退到 {key}（身份未分离）")
            return value
    print("❌ 缺少 AGENT_GITHUB_TOKEN / GITHUB_TOKEN")
    raise SystemExit(2)


def session(tok: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"})
    return s


def check_branch(name: str) -> list[str]:
    proc = run([sys.executable, "scripts/ci/check_commits.py", "--branch", name])
    return [] if proc.returncode == 0 else [proc.stdout.strip()]


def check_message(msg: str) -> list[str]:
    match = HEADER.match(msg)
    if not match:
        return [f"提交信息不符合 Conventional Commits：{msg}"]
    if match.group("type") not in VALID_TYPES:
        return [f"未知 type：{match.group('type')}"]
    return []


def working_tree_dirty() -> bool:
    return bool(run(["git", "status", "--porcelain"]).stdout.strip())


def verify_full() -> tuple[int, str]:
    proc = run(["bash", "scripts/ci/verify.sh", "--full"], timeout=1200)
    return proc.returncode, (proc.stdout + proc.stderr)


def commit_all(message: str) -> None:
    run(["git", "add", "-A"], check=True)
    run(["git", "commit", "-m", message], check=True)


def push(branch: str, s: requests.Session, repo: str, attempts: int = 5) -> bool:
    """推送并重试。本机 SSH 会间歇性拒连，ls-remote 不可靠，用 API 校验。"""
    local = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    for attempt in range(1, attempts + 1):
        proc = run(["git", "push", "-q", "-u", "origin", branch], timeout=120)
        ref = s.get(f"{API}/repos/{repo}/git/ref/heads/{branch}", timeout=30)
        remote = (ref.json().get("object", {}) or {}).get("sha", "") if ref.status_code == 200 else ""
        if remote == local:
            print(f"✅ 已推送（{local[:8]}）")
            return True
        detail = proc.stderr.strip().splitlines()[-1][:100] if proc.stderr.strip() else "无输出"
        print(f"  推送第 {attempt} 次失败：{detail}")
        time.sleep(15)
    return False


def pr_body(task: str, verify_tail: str, stat: str) -> str:
    return f"""## 背景

{task}

## 改动

```
{stat}
```

## 验证

```
{verify_tail}
```

## 回退

`git revert` 本 PR 即可；无数据迁移与配置变更。
"""


def create_pr(s: requests.Session, repo: str, head: str, base: str, title: str, body: str) -> int:
    existing = s.get(f"{API}/repos/{repo}/pulls?head={repo.split('/')[0]}:{head}&state=open", timeout=30).json()
    if existing:
        print(f"ℹ️  PR 已存在：#{existing[0]['number']}")
        return existing[0]["number"]
    resp = s.post(
        f"{API}/repos/{repo}/pulls",
        json={"title": title, "head": head, "base": base, "body": body},
        timeout=60,
    )
    if resp.status_code >= 300:
        print(f"❌ 开 PR 失败：{resp.status_code} {resp.text[:200]}")
        raise SystemExit(2)
    data = resp.json()
    print(f"✅ PR #{data['number']} {data['html_url']}")
    return data["number"]


def failing_digest(s: requests.Session, repo: str, pr: int) -> str:
    """拉 PR head 的失败 job，抽错误行，生成可修摘要。"""
    pr_data = s.get(f"{API}/repos/{repo}/pulls/{pr}", timeout=30).json()
    sha = pr_data["head"]["sha"]
    runs = s.get(f"{API}/repos/{repo}/actions/runs?per_page=20", timeout=30).json().get("workflow_runs", [])
    target = next((r for r in runs if r["head_sha"] == sha and r["name"] == "CI"), None)
    if not target:
        return "（没找到对应的 CI run，稍后重试或手动查看 PR 页面）"

    jobs = s.get(f"{API}/repos/{repo}/actions/runs/{target['id']}/jobs", timeout=30).json().get("jobs", [])
    bad = [j for j in jobs if j.get("conclusion") == "failure"]
    parts = [f"CI run: {target['html_url']}", f"失败 job：{', '.join(j['name'] for j in bad) or '（无）'}"]
    for job in bad[:2]:
        log = s.get(f"{API}/repos/{repo}/actions/jobs/{job['id']}/logs", timeout=60)
        if log.status_code != 200:
            continue
        lines = [ln.split("Z ", 1)[-1] for ln in log.text.splitlines()]
        picked = [ln for ln in lines if ("##[error]" in ln or ln.startswith("❌") or "FAILED" in ln)][:8]
        parts.append(f"\n--- {job['name']} ---\n" + "\n".join(picked))
    return "\n".join(parts)


def write_log(task: str, branch: str, pr: int | None, evidence: str) -> None:
    log_dir = ROOT / "data" / "agent-log"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{date.today().isoformat()}.md"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {time.strftime('%H:%M:%S')} {task}\n- 分支：`{branch}`\n- PR：{pr or '—'}\n")
        fh.write("```\n" + evidence.strip()[-1500:] + "\n```\n")
    print(f"📝 决策留痕：{path.relative_to(ROOT)}（未入库）")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="PR 标题，需符合 Conventional Commits")
    parser.add_argument("--type", default="", help="仅为兼容旧调用，标题里已含 type 时忽略")
    parser.add_argument("--repo", default="teeramisu/tira_stock")
    parser.add_argument("--base", default="main")
    parser.add_argument("--method", default="squash", choices=["squash", "merge", "rebase"])
    parser.add_argument("--dry-run", action="store_true", help="只跑到本地验证")
    parser.add_argument("--no-merge", action="store_true", help="开 PR 但不自动合并")
    parser.add_argument("--wait", type=int, default=1800, help="等待合并的秒数上限")
    args = parser.parse_args()

    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    print(f"▶ 分支：{branch}")

    # --type 作为兜底：标题没带 type 前缀时自动补上
    task = args.task
    if args.type and not HEADER.match(task):
        task = f"{args.type}: {task}"

    problems = check_branch(branch) + check_message(task)
    if problems:
        for p in problems:
            print(f"❌ {p}")
        return 1

    print("▶ 1/5 本地验证")
    code, output = verify_full()
    print(output[-1200:])
    if code != 0:
        print("❌ 本地验证未通过，先修再提交（不推送）")
        return 2

    if args.dry_run:
        print("✅ dry-run 结束（未提交/未推送）")
        return 0

    print("▶ 2/5 提交")
    if working_tree_dirty():
        commit_all(task)
        print(f"  已提交：{task}")
    else:
        print("  工作区干净，跳过提交")

    tok = token()
    s = session(tok)

    print("▶ 3/5 推送")
    if not push(branch, s, args.repo):
        print("❌ 推送失败（SSH 间歇性拒连，可重跑本命令）")
        return 2

    print("▶ 4/5 开 PR")
    stat = run(["git", "diff", "--stat", f"origin/{args.base}...HEAD"]).stdout
    body = pr_body(task, output.strip()[-800:], stat.strip()[-800:])
    pr = create_pr(s, args.repo, branch, args.base, task, body)

    if args.no_merge:
        write_log(task, branch, pr, "已开 PR，未开启自动合并（--no-merge）")
        return 0

    print("▶ 5/5 开启自动合并并等待")
    merge = run([
        sys.executable, "scripts/agent/auto_merge.py",
        "--repo", args.repo, "--pr", str(pr),
        "--method", args.method, "--wait", "--timeout", str(args.wait),
    ], timeout=args.wait + 120)
    print(merge.stdout[-800:])
    if merge.returncode == 0:
        write_log(task, branch, pr, "已合并\n" + output.strip()[-600:])
        return 0

    print("\n❌ 未能自动合并，下面是可修摘要：\n")
    digest = failing_digest(s, args.repo, pr)
    print(digest)
    write_log(task, branch, pr, "合并失败\n" + digest)
    print("\n修复后重跑：python scripts/agent/loop.py --task \"...\"")
    return 2


if __name__ == "__main__":
    sys.exit(main())
