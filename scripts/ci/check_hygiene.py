"""仓库卫生检查：防止大文件、本地数据、密钥文件被误提交。

CI 里是阻断项（public 仓库，泄露不可逆）；本地 verify.sh 同样会跑。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 路径级禁止（前缀匹配）
FORBIDDEN_PREFIXES = (
    "data/",
    "backend/data/",
    "backend/rl/models/",
    "backend/rl/tb_logs/",
    "backend/.ruff_cache/",
    ".venv/",
    "node_modules/",
)

# 路径级禁止（后缀/文件名匹配）
FORBIDDEN_SUFFIXES = (
    ".duckdb", ".duckdb.wal", ".parquet", ".zip", ".pt", ".pth", ".onnx",
    ".pyc", ".log",
)
FORBIDDEN_NAMES = {".env", ".env.local", ".DS_Store"}
FORBIDDEN_NAME_PREFIXES = (".env.",)
# 模板文件必须放行，否则新人无法知道要配哪些变量
ALLOWED_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}

# 单文件大小上限（字节）。frontend/js/echarts.min.js 是随仓库分发的本地依赖，单独放行。
MAX_FILE_BYTES = 1_500_000
SIZE_ALLOWLIST = {"frontend/js/echarts.min.js"}


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def main() -> int:
    problems: list[str] = []
    files = tracked_files()

    for path in files:
        name = Path(path).name
        if any(path.startswith(p) for p in FORBIDDEN_PREFIXES):
            problems.append(f"禁止入库的目录: {path}")
            continue
        if name in ALLOWED_ENV_TEMPLATES:
            pass
        elif name in FORBIDDEN_NAMES or name.startswith(FORBIDDEN_NAME_PREFIXES):
            problems.append(f"禁止入库的配置文件: {path}")
            continue
        if any(path.endswith(s) for s in FORBIDDEN_SUFFIXES):
            problems.append(f"禁止入库的文件类型: {path}")
            continue
        if path in SIZE_ALLOWLIST:
            continue
        abs_path = ROOT / path
        if abs_path.is_file() and abs_path.stat().st_size > MAX_FILE_BYTES:
            size_mb = abs_path.stat().st_size / 1048576
            problems.append(f"文件过大 {size_mb:.1f}MB > 1.5MB: {path}")

    if problems:
        print("❌ 仓库卫生检查未通过：")
        for p in problems[:40]:
            print(f"  - {p}")
        if len(problems) > 40:
            print(f"  ... 还有 {len(problems) - 40} 条")
        print("\n修复：从索引移除并补进 .gitignore —— git rm --cached <path>")
        return 1

    print(f"✅ 仓库卫生检查通过（{len(files)} 个跟踪文件）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
