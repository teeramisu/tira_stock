"""测试分片：按 glob 把测试文件分配到 4 个并行 job。

分片表在 ci/shards.json，值是 glob（支持 `*`、`?`、`[seq]`）。

两条硬约束：
  1. 每个测试文件必须**恰好**匹配一个分片 —— 漏配会让测试永远不被执行，
     重配会让同一文件在两个 job 里跑两遍；
  2. glob 必须至少匹配到一个文件 —— 防止改名后留下死规则。

新增测试只要按命名约定（test_macro_xxx.py / test_discovery_xxx.py ...）就自动归位。
"""
from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARDS_FILE = ROOT / "ci" / "shards.json"
TESTS_DIR = ROOT / "backend" / "tests"


def load_shards() -> dict[str, list[str]]:
    raw = json.loads(SHARDS_FILE.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def discover() -> list[str]:
    return sorted(p.name for p in TESTS_DIR.glob("test_*.py"))


def match(pattern: str, name: str) -> bool:
    return fnmatch.fnmatchcase(name, pattern)


def members(shards: dict[str, list[str]], shard: str) -> list[str]:
    return [n for n in discover() if any(match(p, n) for p in shards[shard])]


def validate(shards: dict[str, list[str]]) -> list[str]:
    problems: list[str] = []
    files = discover()

    for shard, patterns in shards.items():
        for pattern in patterns:
            if not any(match(pattern, n) for n in files):
                problems.append(f"{shard}: 规则 `{pattern}` 没有匹配到任何测试文件（改名后遗留？）")

    for name in files:
        hits = [shard for shard in shards if any(match(p, name) for p in shards[shard])]
        if not hits:
            problems.append(f"{name} 没有匹配任何分片（会被 CI 跳过）")
        elif len(hits) > 1:
            problems.append(f"{name} 同时匹配多个分片：{', '.join(hits)}")

    return problems


def main(argv: list[str]) -> int:
    shards = load_shards()

    if argv and argv[0] == "--validate":
        problems = validate(shards)
        if problems:
            print("❌ 分片表校验失败：")
            for p in problems:
                print(f"  - {p}")
            print("\n修复：调整 ci/shards.json 的 glob，或按命名约定重命名测试文件")
            return 1
        total = sum(len(members(shards, s)) for s in shards)
        print(f"✅ 分片表校验通过（{len(shards)} 片 / {total} 个测试文件，glob 匹配）")
        return 0

    if len(argv) == 2 and argv[0] == "--list":
        shard = argv[1]
        if shard not in shards:
            print(f"未知分片: {shard}（可选：{', '.join(shards)}）")
            return 2
        for name in members(shards, shard):
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
