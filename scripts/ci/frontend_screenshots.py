"""前端视觉证据：给改动前后留一张图，而不是只看 CI 有没有红。

为什么需要：
    静态检查能发现 JS 语法错，但发现不了「按钮跑到屏幕外」「深色主题下文字看不见」
    「移动端布局塌了」。这些只有渲染出来才知道。

做法：
    1. 起一个本地服务（CI 里由 workflow 启动 uvicorn）；
    2. Playwright 打开首页，按导航切换主要页面；
    3. 每个页面截桌面（1440×900）与移动端（390×844）两张；
    4. 收集未捕获的 JS 异常，写进 summary（默认只报告，不阻断 ——
       CI 没有本地 data/，部分面板本来就是空状态）。

用法：
    python scripts/ci/frontend_screenshots.py --base-url http://127.0.0.1:8000 --out artifacts/screenshots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VIEWPORTS = {"desktop": (1440, 900), "mobile": (390, 844)}

# 首页顶部导航的 data-page 值 → 可读名字
PAGES = {
    "home": "首页",
    "macro": "宏观罗盘",
    "backtest": "回测",
    "portfolio": "组合",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default="artifacts/screenshots")
    parser.add_argument("--timeout", type=int, default=20000)
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ 未安装 playwright：pip install playwright && playwright install chromium")
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    errors: list[dict] = []
    shots: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for label, (width, height) in VIEWPORTS.items():
            page = browser.new_page(viewport={"width": width, "height": height})
            page.on("pageerror", lambda exc, l=label: errors.append({"viewport": l, "error": str(exc)}))

            try:
                page.goto(args.base_url, wait_until="networkidle", timeout=args.timeout)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  {label} 打开首页失败：{exc}")
                page.close()
                continue

            for key, name in PAGES.items():
                if key != "home":
                    selector = f'.top-nav-btn[data-page="{key}"]'
                    if page.locator(selector).count() == 0:
                        continue
                    try:
                        page.click(selector, timeout=5000)
                        page.wait_for_timeout(1200)
                    except Exception:  # noqa: BLE001
                        continue
                target = out / f"{key}-{label}.png"
                page.screenshot(path=str(target), full_page=False)
                shots.append(str(target))

            page.close()
        browser.close()

    summary = {
        "screenshots": shots,
        "js_errors": errors,
        "base_url": args.base_url,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"📸 截图 {len(shots)} 张 → {out}")
    for shot in shots:
        print(f"   - {shot}")

    if errors:
        print(f"\n⚠️  捕获 {len(errors)} 个未捕获 JS 异常（不阻断，供参考）：")
        for item in errors[:10]:
            print(f"   [{item['viewport']}] {item['error'][:120]}")
    else:
        print("\n✅ 没有未捕获的 JS 异常")

    return 0


if __name__ == "__main__":
    sys.exit(main())
