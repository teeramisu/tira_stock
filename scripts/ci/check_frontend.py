"""前端语法与资源引用检查（轻量前端闸门，无需构建）。

覆盖三件事：
  1. frontend/js/*.js 逐个 node --check；
  2. index.html 里的内联 <script> 抽出来逐个 node --check
     （之前手工排过 JS 语法错，这里固化成检查）；
  3. index.html / css 里引用的本地 js/css 资源必须真实存在（防 404）。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"

SCRIPT_SRC = re.compile(r'<script[^>]*\ssrc="([^"]+)"', re.IGNORECASE)
STYLE_HREF = re.compile(r'<link[^>]*\shref="([^"]+)"', re.IGNORECASE)
INLINE_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.DOTALL | re.IGNORECASE)


def node_check(code: str, label: str, tmpdir: Path) -> str | None:
    tmp = Path(tmpdir) / (re.sub(r"[^A-Za-z0-9_.-]", "_", label) + ".js")
    tmp.write_text(code, encoding="utf-8")
    proc = subprocess.run(["node", "--check", str(tmp)], capture_output=True, text=True)
    if proc.returncode != 0:
        return f"{label}: {proc.stderr.strip().splitlines()[0] if proc.stderr else 'syntax error'}"
    return None


def main() -> int:
    if shutil.which("node") is None:
        print("⚠️  未安装 node，跳过前端语法检查")
        return 0

    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        for js in sorted(FRONTEND.glob("js/*.js")):
            problems += filter(None, [node_check(js.read_text(encoding="utf-8"), js.name, tmpdir)])

        index = FRONTEND / "index.html"
        html = index.read_text(encoding="utf-8")
        blocks = INLINE_SCRIPT.findall(html)
        for i, block in enumerate(blocks):
            if not block.strip():
                continue
            problems += filter(None, [node_check(block, f"index.html#inline{i}", tmpdir)])

        # 资源引用存在性
        for ref in SCRIPT_SRC.findall(html) + STYLE_HREF.findall(html):
            if ref.startswith(("http://", "https://", "//", "data:")):
                continue
            clean = ref.split("?", 1)[0]
            # FastAPI 把 /assets 挂到 frontend/ 根目录（见 backend/app/main.py）
            if clean.startswith("/assets/"):
                clean = clean[len("/assets/"):]
            target = (FRONTEND / clean.lstrip("./")).resolve()
            if not target.exists():
                problems.append(f"index.html 引用不存在的资源: {ref}")

    if problems:
        print("❌ 前端检查未通过：")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"✅ 前端检查通过（{len(list(FRONTEND.glob('js/*.js')))} 个 js + {len(blocks)} 段内联脚本）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
