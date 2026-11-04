"""测试分片：把 28 个测试文件按领域切成 4 片，CI 并行跑。

分片表在 ci/shards.json；本脚本负责：
  1. 校验每个测试文件恰好属于一个分片（新增测试文件忘记登记 → 直接失败，
     避免「测试写了但从没跑过」这种静默失效）；
  2. 输出指定分片的文件列表（供 CI 调 pytest）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARDS_FILE = ROOT / "ci" / "shards.json"
TESTS_DIR = ROOT / "backend" / "tests"


def load_shards() -> dict[str, list[str]]:
    return json.loads(SHARDS_FILE.read_text(encoding="utf-8"))


def discover() -> set[str]:
    return {p.name for p in TESTS_DIR.glob("test_*.py")}


def validate(shards: dict[str, list[str]]) -> list[str]:
    problems: list[str] = []
    seen: dict[str, str] = {}
    for shard, files in shards.items():
        for name in files:
            if name in seen:
                problems.append(f"{name} 同时出现在 {seen[name]} 和 {shard}")
            seen[name] = shard
            if not (TESTS_DIR / name).exists():
                problems.append(f"{shard} 引用了不存在的测试文件 {name}")
    for name in sorted(discover() - set(seen)):
        problems.append(f"{name} 未登记到任何分片（不会被 CI 执行）")
    for name in sorted(set(seen) - discover()):
        problems.append(f"{name} 已登记但文件不存在")
    return problems


def main(argv: list[str]) -> int:
    shards = load_shards()

    if argv and argv[0] == "--validate":
        problems = validate(shards)
        if problems:
            print("❌ 分片表校验失败：")
            for p in problems:
                print(f"  - {p}")
            print("\n修复：在 ci/shards.json 中登记该测试文件")
            return 1
        total = sum(len(v) for v in shards.values())
        print(f"✅ 分片表校验通过（{len(shards)} 片 / {total} 个测试文件）")
        return 0

    if len(argv) == 2 and argv[0] == "--list":
        shard = argv[1]
        if shard not in shards:
            print(f"未知分片: {shard}（可选：{', '.join(shards)}）")
            return 2
        for name in shards[shard]:
            print(f"backend/tests/{name}")
        return 0

    if argv and argv[0] == "--names":
        for name in shards:
            print(name)
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
