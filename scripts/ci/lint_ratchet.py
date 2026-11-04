"""Lint 债务 ratchet：只允许变少，不允许变多。

背景：这个仓库有历史 lint 债务（E/W/I/UP 上千条），一次性清零不现实，
但也不能放任增加。做法是记录当前基线，任何一类规则数量上涨就失败，
数量下降则自动更新基线（在本地跑 --write 时）。

用法：
    python scripts/ci/lint_ratchet.py            # 校验（CI 用）
    python scripts/ci/lint_ratchet.py --write    # 债务减少后刷新基线
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "ci" / "lint-baseline.json"

# 计入债务统计的规则族（阻断规则 E9/F63/F7/F82 不在其中，它们必须为 0）
DEBT_FAMILIES = {
    "F": "Pyflakes（未使用导入/变量等）",
    "I": "isort（导入顺序）",
    "E": "pycodestyle 错误",
    "W": "pycodestyle 警告",
    "B": "flake8-bugbear",
    "UP": "pyupgrade",
    "SIM": "flake8-simplify",
}
TARGETS = ["backend", "scripts"]


def ruff_bin() -> str:
    local = ROOT / ".venv" / "bin" / "ruff"
    if local.exists():
        return str(local)
    found = shutil.which("ruff")
    if not found:
        print("❌ 未找到 ruff，请 pip install ruff==0.17.3")
        raise SystemExit(2)
    return found


def count(select: str) -> int:
    proc = subprocess.run(
        [ruff_bin(), "check", *TARGETS, "--select", select, "--statistics"],
        cwd=ROOT, capture_output=True, text=True,
    )
    total = 0
    for line in proc.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].replace(",", "").isdigit():
            total += int(parts[0].replace(",", ""))
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="刷新基线（仅允许在债务下降时）")
    args = parser.parse_args()

    current = {fam: count(fam) for fam in DEBT_FAMILIES}
    baseline: dict[str, int] = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}

    regressions = {f: (baseline.get(f, 0), n) for f, n in current.items() if n > baseline.get(f, n)}
    improvements = {f: (baseline.get(f, n), n) for f, n in current.items() if n < baseline.get(f, n)}

    print("规则族    基线    当前    变化")
    for fam, desc in DEBT_FAMILIES.items():
        base = baseline.get(fam, current[fam])
        delta = current[fam] - base
        flag = "❌" if delta > 0 else ("✅" if delta < 0 else "  ")
        print(f"{fam:<9} {base:>6} {current[fam]:>7} {delta:>+7}  {flag} {desc}")

    if args.write:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\n✅ 基线已更新 → {BASELINE.relative_to(ROOT)}")
        return 0

    if not BASELINE.exists():
        print("\n⚠️  缺少基线文件，请先运行：python scripts/ci/lint_ratchet.py --write")
        return 1

    if regressions:
        print("\n❌ Lint 债务增加：")
        for fam, (base, now) in regressions.items():
            print(f"  - {fam}: {base} → {now} (+{now - base})")
        print("  修复：清理新增问题；若确属误报，在 ruff.toml 里按文件 ignore 并说明理由")
        return 1

    if improvements:
        print(f"\n💡 债务已下降 {len(improvements)} 项，可运行 --write 刷新基线")

    print("\n✅ Lint ratchet 通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
