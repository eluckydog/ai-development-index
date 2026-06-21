"""
AIDI Batch Period Search
批量运行所有86期的搜索，逐期调用 period_search.py
用法: python engine/batch_search.py --start 2022-12-01 --end 2026-06-16
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta


def generate_periods(start_date, end_date):
    """Generate bi-monthly periods from start to end."""
    periods = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        # Period 1st ~ 15th
        start = current.replace(day=1)
        mid = start.replace(day=15) if start.day <= 15 else start
        periods.append((start.strftime("%Y-%m-%d"), mid.strftime("%Y-%m-%d"), start.strftime("%Y-%m-%d")))

        # Period 16th ~ end of month
        p_start = current.replace(day=16)
        if current.month == 12:
            p_end = current.replace(year=current.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            p_end = current.replace(month=current.month + 1, day=1) - timedelta(days=1)

        if p_start <= end:
            periods.append((p_start.strftime("%Y-%m-%d"), p_end.strftime("%Y-%m-%d"), p_start.strftime("%Y-%m-%d")))

        current = current.replace(month=current.month + 1) if current.month < 12 else current.replace(year=current.year + 1, month=1)

    return periods


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-12-01")
    parser.add_argument("--end", default="2026-06-16")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    periods = generate_periods(args.start, args.end)
    print(f"[AIDI Batch Search] {len(periods)} periods: {args.start} ~ {args.end}")
    print(f"  Skip existing: {args.skip_existing}")

    total = len(periods)
    success = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    for i, (p_start, p_end, p_name) in enumerate(periods):
        out_path = f"data/raw/{p_name}/_search.json"

        if args.skip_existing and os.path.exists(out_path):
            with open(out_path) as f:
                existing = json.load(f)
            # Check if it has real data or just placeholder
            if "arxiv" in existing and len(existing.get("arxiv", [])) > 0:
                skipped += 1
                continue

        print(f"  [{i+1}/{total}] {p_name} ({p_start}~{p_end})...", end=" ")

        result = subprocess.run(
            [sys.executable, "engine/period_search.py",
             "--start", p_start, "--end", p_end, "--period", p_name,
             "--output", out_path],
            capture_output=True, text=True, timeout=60,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) or ".",
            env=os.environ.copy()
        )

        if result.returncode == 0:
            success += 1
            print("OK")
        else:
            errors += 1
            print(f"FAIL: {result.stderr[:100]}")

        # Rate limiting: be nice to APIs
        time.sleep(1)

    elapsed = time.time() - start_time
    print(f"\n[AIDI Batch] Done: {success} OK, {skipped} skipped, {errors} errors")
    print(f"  Time: {elapsed:.0f}s ({elapsed/total:.1f}s/period avg)")
    print(f"  Rate: {total/(elapsed/60):.0f} periods/min")


if __name__ == "__main__":
    main()
