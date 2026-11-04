"""Build local cache of board constituent stocks.

Fetches all industry and concept board constituents from THS and saves
to data/board_constituents.json for offline/fast lookup.

Usage:
    python scripts/build_board_constituents.py [--force] [--type industry|concept]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.marketdata.sources.industry_board import (
    _get_ths_board_code_map,
    _fetch_ths_board_constituents,
)

CACHE_FILE = PROJECT_ROOT / "data" / "board_constituents.json"


def build_cache(board_type: str = "industry", force: bool = False) -> None:
    """Scrape all board constituents and save to JSON cache."""
    # Load existing cache if not forcing rebuild
    existing: dict = {}
    if not force and CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)

    board_map = _get_ths_board_code_map(board_type)
    if not board_map:
        print(f"[ERROR] No {board_type} boards found")
        return

    total = len(board_map)
    print(f"Building {board_type} constituent cache: {total} boards")

    type_data = existing.get(board_type, {})
    success = 0
    failed = 0

    for idx, (name, ths_code) in enumerate(sorted(board_map.items()), 1):
        # Skip if already cached (unless force)
        if not force and name in type_data and type_data[name].get("stocks"):
            print(f"  [{idx}/{total}] {name}: cached ({len(type_data[name]['stocks'])} stocks)")
            continue

        try:
            stocks = _fetch_ths_board_constituents(ths_code, board_type)
            type_data[name] = {
                "ths_code": ths_code,
                "stocks": stocks,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            status = f"{len(stocks)} stocks"
            if stocks:
                success += 1
            else:
                failed += 1
                status = "EMPTY"
            print(f"  [{idx}/{total}] {name}: {status}")

            # Save incrementally every 10 boards
            if idx % 10 == 0:
                _save_cache(existing, board_type, type_data)

            # Rate limiting: be polite to THS
            time.sleep(0.5)

        except Exception as e:
            failed += 1
            print(f"  [{idx}/{total}] {name}: FAILED - {e}")
            time.sleep(1)

    # Final save
    _save_cache(existing, board_type, type_data)
    print(f"\nDone: {success} success, {failed} failed, {total} total")
    print(f"Cache saved to: {CACHE_FILE}")


def _save_cache(existing: dict, board_type: str, type_data: dict) -> None:
    existing[board_type] = type_data
    existing["_meta"] = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force rebuild all boards")
    parser.add_argument("--type", choices=["industry", "concept", "all"], default="all")
    args = parser.parse_args()

    types = ["industry", "concept"] if args.type == "all" else [args.type]
    for bt in types:
        build_cache(board_type=bt, force=args.force)
