"""Summarize actual displayed intervals from a PresentMon CSV, not FPS counters.

Example: python tools/win/analyze_presentmon.py capture.csv --after-ms 20000
Use a gameplay-only time window; loading screens and captures cause long gaps.
This measures display cadence, not whether interpolated image content is correct.
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file")
    parser.add_argument("--after-ms", type=float, default=0)
    parser.add_argument("--before-ms", type=float, default=float("inf"))
    parser.add_argument("--refresh-hz", type=float, help="physical display refresh rate, for interpreting tearing-enabled captures")
    args = parser.parse_args()
    chains = defaultdict(list)
    with open(args.csv_file, newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            timestamp = number(row.get("TimeInMs", row.get("CPUStartTime")))
            if timestamp is None or not args.after_ms <= timestamp < args.before_ms:
                continue
            key = (row.get("Application"), row.get("ProcessID"), row.get("SwapChainAddress"))
            chains[key].append(row)
    if not chains:
        parser.error("no frames in this time window")
    for key, rows in sorted(chains.items(), key=lambda item: len(item[1]), reverse=True):
        displayed = [number(row.get("MsBetweenDisplayChange")) for row in rows]
        intervals = [value for value in displayed if value is not None and value > 0]
        presents = [number(row.get("MsBetweenPresents")) for row in rows]
        presents = [value for value in presents if value is not None and value > 0]
        print(f"{key[0]} PID {key[1]} swapchain {key[2]}: {len(rows)} submitted rows")
        if presents:
            print(f"  submitted cadence: {1000 / statistics.mean(presents):.2f} FPS")
            print("  submission interval ms: " + ", ".join(
                f"p{int(p*100)}={percentile(presents,p):.3f}" for p in (.1,.5,.9,.99)))
        if not intervals:
            print("  actual display timing unavailable; presentation rate alone does not prove smoothness")
            continue
        unavailable = sum(value is None for value in displayed)
        print(f"  actual display cadence: {1000 / statistics.mean(intervals):.2f} FPS, "
              f"{len(intervals)} intervals, {unavailable} rows without display timing")
        if args.refresh_hz and 1000 / statistics.mean(intervals) > args.refresh_hz * 1.01:
            print(f"  above {args.refresh_hz:g} Hz: display changes can include torn scanouts; "
                  "this is not evidence of that many complete frames per second")
        print("  display interval ms: " + ", ".join(
            f"p{int(p*100)}={percentile(intervals,p):.3f}" for p in (.1,.5,.9,.99)))
        median = statistics.median(intervals)
        short = sum(value < median * .6 for value in intervals)
        long = sum(value > median * 1.5 for value in intervals)
        print(f"  intervals below 0.6x median: {short}; above 1.5x median: {long}")
        print("  present modes: " + ", ".join(sorted({row.get('PresentMode', '?') for row in rows})))


if __name__ == "__main__":
    main()
