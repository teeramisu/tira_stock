"""测试分片执行器：失败自动重跑一次，区分「真失败」与「偶发」。

问题：偶发失败（网络抖动、时序、随机数）会让 agent 陷入「重试 → 再失败 → 再重试」，
既烧 token 又污染判断。人工 review 时也分不清「这次红」是不是噪音。

做法：
  1. 正常跑一次，写 JUnit XML；
  2. 失败则**整个分片重跑一次**（而不是只跑失败用例——避免掩盖依赖顺序问题）；
  3. 第二次全绿 → 判定为 flaky：**不失败**，但把第一次挂掉的用例写进 summary；
  4. 第二次仍失败 → 真失败，用第二次的输出报错。

为什么整片重跑：只重跑失败用例会掩盖「测试间互相污染」这类问题，
而那种问题恰恰是 agent 改坏了的早期信号。

用法：
    python scripts/ci/run_tests.py backend/tests/test_macro.py ...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def failed_cases(junit: Path) -> list[str]:
    if not junit.exists():
        return []
    try:
        root = ET.parse(junit).getroot()
    except ET.ParseError:
        return []
    names: list[str] = []
    for case in root.iter("testcase"):
        if case.find("failure") is None and case.find("error") is None:
            continue
        classname = case.get("classname", "")
        name = case.get("name", "")
        names.append(f"{classname}::{name}" if classname else name)
    return names


def run_pytest(targets: list[str], junit: Path) -> int:
    cmd = [
        sys.executable, "-m", "pytest", *targets, "-q",
        "-p", "no:cacheprovider",
        f"--junitxml={junit}",
    ]
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="+", help="测试文件或目录")
    parser.add_argument("--retries", type=int, default=1, help="失败后重跑次数（默认 1）")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        first = Path(tmp) / "first.xml"
        print(f"▶ 第 1 次运行：{' '.join(args.targets)}")
        rc = run_pytest(args.targets, first)

        if rc == 0:
            print("✅ 全部通过")
            return 0

        flaky = failed_cases(first)
        print(f"\n⚠️  第 1 次失败（{len(flaky)} 个用例），重跑一次以区分偶发与真失败：")
        for name in flaky[:20]:
            print(f"   - {name}")

        for attempt in range(1, args.retries + 1):
            retry_xml = Path(tmp) / f"retry{attempt}.xml"
            print(f"\n▶ 重跑 {attempt}/{args.retries}")
            rc = run_pytest(args.targets, retry_xml)
            if rc == 0:
                print("\n🟡 判定为 FLAKY：第 1 次失败、重跑通过，本次不阻断")
                print("   处理建议：把下面用例改成确定性（固定随机种子/时间/网络 mock），")
                print("   或在 PR 里说明原因；连续两次出现应考虑隔离。")
                summary = Path("/tmp/flaky-summary.md")
                summary.write_text(
                    "# 偶发失败（第 2 次通过）\n\n"
                    + "\n".join(f"- `{n}`" for n in flaky)
                    + "\n",
                    encoding="utf-8",
                )
                print(f"   明细已写入 {summary}")
                return 0

        print("\n❌ 重跑仍然失败：判定为真实失败")
        return rc


if __name__ == "__main__":
    sys.exit(main())
