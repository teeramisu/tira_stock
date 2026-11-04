"""覆盖率 ratchet：整体覆盖率只许升、不许降。

agent 最容易做的事是「写一个什么都不断言的测试让 CI 变绿」。
覆盖率不能证明测试是对的，但能挡住「完全没测到」这一类。

用法：
    python scripts/ci/check_coverage.py                 # 校验（读 coverage.json）
    python scripts/ci/check_coverage.py --write         # 刷新基线
    python scripts/ci/check_coverage.py --report report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "ci" / "coverage-baseline.json"
TOLERANCE = 0.2      # 允许 0.2pp 的抖动（不同环境浮点/跳过差异）
LOW_FILE_THRESHOLD = 40.0
LOW_FILE_LIMIT = 12  # 只展示最差的若干个文件，避免刷屏


def load_total(path: Path) -> tuple[float, dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    total = float(data.get("totals", {}).get("percent_covered", 0.0))
    per_file: dict[str, float] = {}
    for name, info in (data.get("files") or {}).items():
        summary = info.get("summary") or {}
        if "percent_covered" in summary:
            per_file[name.replace("\\", "/")] = float(summary["percent_covered"])
    return total, per_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="coverage.json")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    report = ROOT / args.report
    if not report.exists():
        print(f"❌ 找不到覆盖率报告 {args.report}")
        print("   生成方式：pytest backend/tests --cov=backend --cov-report=json:coverage.json")
        return 1

    total, per_file = load_total(report)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    base_total = float(baseline.get("total", total))

    print(f"整体覆盖率：{total:.1f}%   基线：{base_total:.1f}%   变化：{total - base_total:+.1f}pp")

    if args.write:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(
            json.dumps({"total": round(total, 2)}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"✅ 基线已更新 → {BASELINE.relative_to(ROOT)}")
        return 0

    if total + TOLERANCE < base_total:
        print(f"\n❌ 覆盖率下降超过容忍度（{TOLERANCE}pp）")
        print("   修复方向：给新增逻辑补测试；确实无法测的用 `# pragma: no cover` 并写原因")
        low = sorted(per_file.items(), key=lambda kv: kv[1])[:LOW_FILE_LIMIT]
        if low:
            print("\n   当前覆盖率最低的文件：")
            for name, pct in low:
                print(f"     {pct:5.1f}%  {name}")
        return 1

    low = [(n, p) for n, p in sorted(per_file.items(), key=lambda kv: kv[1]) if p < LOW_FILE_THRESHOLD]
    if low:
        print(f"\n💡 有 {len(low)} 个文件低于 {LOW_FILE_THRESHOLD:.0f}%（不阻断，作为补测试的参考）：")
        for name, pct in low[:LOW_FILE_LIMIT]:
            print(f"   {pct:5.1f}%  {name}")

    print("\n✅ 覆盖率 ratchet 通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
