"""Summarize VK_NV_low_latency2 driver timestamps, not display/photon latency."""
import argparse
import json
import re
from pathlib import Path


PATTERN = re.compile(
    r"reflex timing: id=(\d+) input=(\d+) sim=(\d+)\.\.(\d+) "
    r"render=(\d+)\.\.(\d+) present=(\d+)\.\.(\d+) gpu=(\d+)\.\.(\d+)"
)


def summarize(text, tail_seconds=39):
    # Timings can be queried repeatedly. Use each completed real frame once.
    reports = {int(m[0]): tuple(map(int, m)) for m in PATTERN.findall(text)}
    rows = sorted(reports.values())
    if rows and tail_seconds:
        threshold = max(r[2] for r in rows) - tail_seconds * 1_000_000
        rows = [r for r in rows if r[2] >= threshold]
    invalid = 0
    metrics = {k: [] for k in ("simulation_to_gpu_end_ms", "input_to_gpu_end_ms",
                              "simulation_to_present_ms", "input_to_present_ms", "gpu_work_ms")}
    missing_input = 0
    for r in rows:
        _, inp, sim0, sim1, render0, render1, present0, present1, gpu0, gpu1 = r
        if (not all(r[2:]) or sim1 < sim0 or render1 < render0 or
                present1 < present0 or gpu1 < gpu0 or gpu1 < sim0 or present0 < sim0):
            invalid += 1
            continue
        metrics["simulation_to_gpu_end_ms"].append((gpu1 - sim0) / 1000)
        metrics["simulation_to_present_ms"].append((present0 - sim0) / 1000)
        metrics["gpu_work_ms"].append((gpu1 - gpu0) / 1000)
        if sim0 <= inp <= min(gpu1, present0):
            metrics["input_to_gpu_end_ms"].append((gpu1 - inp) / 1000)
            metrics["input_to_present_ms"].append((present0 - inp) / 1000)
        else:
            missing_input += 1

    def percentiles(values):
        if not values:
            return None
        values = sorted(values)
        return {f"p{p}": round(values[round((len(values) - 1) * p / 100)], 3)
                for p in (50, 95, 99)}

    return {"frames": len(rows), "invalid_reports": invalid,
            "missing_or_invalid_input": missing_input,
            **{k: percentiles(v) for k, v in metrics.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--tail-seconds", type=float, default=39)
    args = parser.parse_args()
    print(json.dumps(summarize(args.log.read_text(errors="replace"), args.tail_seconds), indent=2))
    print("Driver timestamps exclude monitor scanout and photon latency.")
