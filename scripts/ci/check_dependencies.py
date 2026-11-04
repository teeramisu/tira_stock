"""依赖漏洞审计（pip-audit）。

agent 引入新依赖时最容易忽略供应链风险：一个 typosquat 包、一个已知 RCE 版本的
工具库，都能直接进入生产路径。本检查在「依赖文件变更」时运行。

设计取舍：
  - **不阻断所有 PR**：只有 requirements*.txt 变化时才跑（漏洞库每天变，
    否则每周都会有 PR 因为上游新披露而红）；
  - **允许有期限的豁免**：`ci/dependency-audit.json` 里写清原因与到期日，
    到期后自动失效，避免「加进白名单就永远忘了」。

用法：
    python scripts/ci/check_dependencies.py
    python scripts/ci/check_dependencies.py --requirements requirements.txt
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = ROOT / "ci" / "dependency-audit.json"


def pip_audit_bin() -> str | None:
    local = ROOT / ".venv" / "bin" / "pip-audit"
    if local.exists():
        return str(local)
    return shutil.which("pip-audit")


def run_audit(requirements: Path) -> list[dict]:
    binary = pip_audit_bin()
    if not binary:
        print("❌ 未找到 pip-audit，安装：pip install pip-audit")
        raise SystemExit(2)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        report = Path(tmp.name)
    proc = subprocess.run(
        [binary, "-r", str(requirements), "--format", "json",
         "--progress-spinner", "off", "--output", str(report)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if proc.returncode not in (0, 1):
        print(f"❌ pip-audit 执行失败（rc={proc.returncode}）")
        print((proc.stderr or proc.stdout or "")[:600])
        raise SystemExit(2)
    try:
        return json.loads(report.read_text(encoding="utf-8")).get("dependencies", [])
    except (json.JSONDecodeError, OSError) as exc:
        print(f"❌ 无法解析 pip-audit 输出：{exc}")
        raise SystemExit(2) from exc
    finally:
        report.unlink(missing_ok=True)


def load_allowlist() -> list[dict]:
    if not ALLOWLIST.exists():
        return []
    return json.loads(ALLOWLIST.read_text(encoding="utf-8")).get("allow", [])


def allowlist_status(entries: list[dict]) -> tuple[dict[tuple[str, str], dict], list[str]]:
    """把豁免整理成 (package, id) → 条目，并返回已过期的条目说明。"""
    active: dict[tuple[str, str], dict] = {}
    expired: list[str] = []
    today = date.today()
    for entry in entries:
        key = (entry.get("package", ""), entry.get("id", ""))
        expires = entry.get("expires", "")
        try:
            deadline = date.fromisoformat(expires)
        except ValueError:
            expired.append(f"{key[0]} {key[1]}：expires 格式不对（{expires!r}），应为 YYYY-MM-DD")
            continue
        if deadline < today:
            expired.append(f"{key[0]} {key[1]}：豁免已于 {expires} 到期，请升级或续期并说明原因")
            continue
        active[key] = entry
    return active, expired


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", default="requirements-ci.txt")
    args = parser.parse_args()

    requirements = ROOT / args.requirements
    if not requirements.exists():
        print(f"❌ 找不到 {args.requirements}")
        return 2

    deps = run_audit(requirements)
    active, expired = allowlist_status(load_allowlist())

    findings: list[str] = []
    waived: list[str] = []
    total_vulns = 0
    for dep in deps:
        for vuln in dep.get("vulns") or []:
            total_vulns += 1
            key = (dep.get("name", ""), vuln.get("id", ""))
            line = f"{key[0]} {dep.get('version')}: {key[1]}"
            fixes = ", ".join(vuln.get("fix_versions") or [])
            if key in active:
                waived.append(f"{line}（豁免至 {active[key]['expires']}：{active[key].get('reason', '')}）")
            else:
                findings.append(f"{line}{'  修复版本: ' + fixes if fixes else ''}")

    print(f"审计依赖：{len(deps)} 个包（{args.requirements}）")
    print(f"发现漏洞：{total_vulns} 条，其中豁免 {len(waived)} 条，待处理 {len(findings)} 条")

    for item in waived:
        print(f"  ⚪ {item}")

    if expired:
        print("\n❌ 豁免清单存在问题：")
        for item in expired:
            print(f"  - {item}")

    if findings:
        print("\n❌ 存在未豁免的已知漏洞：")
        for item in findings:
            print(f"  - {item}")
        print("\n处理方式：")
        print("  1. 优先升级到修复版本（改 requirements*.txt）")
        print("  2. 暂时无法升级时，在 ci/dependency-audit.json 里登记原因与到期日（≤90 天）")

    if findings or expired:
        return 1

    print("\n✅ 依赖审计通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
