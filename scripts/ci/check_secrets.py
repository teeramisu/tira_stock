"""密钥泄露扫描（自包含，不依赖第三方 action）。

在 public 仓库里这是最后一道闸门：一旦推上去就不可逆。
扫描的是 git 跟踪的文件，而不是工作区。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 只扫文本类文件；这些后缀跳过
SKIP_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg",
    ".duckdb", ".parquet", ".zip", ".pt", ".pth", ".onnx", ".woff", ".woff2",
)
# 压缩/内联的大文件跳过（echarts 内含大量随机字符串，误报源）
SKIP_PATHS = {"frontend/js/echarts.min.js"}
MAX_SCAN_BYTES = 2_000_000

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI/DeepSeek key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("GitHub classic token", re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("私钥块", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |PGP )?PRIVATE KEY-----")),
    ("带密码的连接串", re.compile(r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?)://[^\s:@/]+:[^\s@/]{6,}@")),
]

# 允许出现的占位符（文档/示例）
ALLOWLIST_LINE = re.compile(
    r"(xxxx|<your|YOUR_|example|placeholder|\$\{|\*\*\*|://[^\s:@/]+:(?:password|pass|pwd|secret)@)",
    re.IGNORECASE,
)


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def main() -> int:
    findings: list[str] = []

    for rel in tracked_files():
        if rel in SKIP_PATHS or rel.endswith(SKIP_SUFFIXES):
            continue
        path = ROOT / rel
        if not path.is_file() or path.stat().st_size > MAX_SCAN_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if ALLOWLIST_LINE.search(line):
                continue
            for label, pattern in PATTERNS:
                match = pattern.search(line)
                if match:
                    token = match.group(0)
                    masked = token[:6] + "…" + token[-4:] if len(token) > 12 else "***"
                    findings.append(f"{rel}:{lineno}  {label}  ({masked})")

    if findings:
        print("❌ 疑似密钥泄露：")
        for f in findings:
            print(f"  - {f}")
        print("\n修复：删除该行 → git add → 轮换该 key（历史里仍然存在）")
        return 1

    print("✅ 密钥扫描通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
