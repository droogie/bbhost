#!/usr/bin/env python3
"""Check every lift against its translation, one captured draw at a time.

For each draw capture under CAPTURES (a directory of capture directories, as
`BBHOST_CAPTURE_DRAW='*'` writes them), runs

    gcnlift <capture> <out>/<name>/lift
    drawreplay <capture> --ps <out>/<name>/lift/lifted.spv --no-live --out <out>/<name>/replay

gcnlift rebuilds the pixel stage's translation options from the manifest,
requires the retranslation to equal the captured SPIR-V, and lifts. drawreplay
renders the draw with the captured modules and again with the lifted pixel
shader, and compares the outputs byte for byte. A draw captured with its lift
bound (BBHOST_DECOMP, on by default) is replayed against the translation
instead (--ps <out>/<name>/lift/reference.spv): either way one run has the
lift and the other the translation.

Writes <out>/summary.json and prints one line per capture plus the totals:
lifted and byte-identical, lifted but different, rejected (with the reasons),
not comparable (the rebuilt options did not reproduce the captured shader, or
the capture could not be replayed). --delete-verified removes a capture once
its lift has been shown identical, since captures are ~50 MiB each.
"""
import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def verify(capture, out, gcnlift, drawreplay, timeout, keep_replay, stage="ps"):
    name = capture.name
    row = {"capture": str(capture), "name": name, "stage": stage}
    lift_dir = out / name / "lift"
    lift = subprocess.run([str(gcnlift), str(capture), str(lift_dir)] + (["--vs"] if stage == "vs" else []),
                          capture_output=True, text=True, timeout=timeout)
    report = load(lift_dir / "report.json")
    row["gcnlift_exit"] = lift.returncode
    if report is None:
        row["status"] = "error"
        row["detail"] = (lift.stderr or lift.stdout).strip().splitlines()[-1:] or ["no report"]
        return row
    row["pipeline"] = report.get("pipeline")
    row["reference_words"] = report.get("reference_words")
    row["lifted_words"] = report.get("lifted_words")
    if not report.get("reference_matches_capture"):
        row["status"] = "not comparable"
        row["detail"] = ["the rebuilt options do not reproduce the captured shader"]
        return row
    if not report.get("lifted"):
        row["status"] = "rejected"
        row["detail"] = report.get("rejections", [])
        return row
    if not report.get("spirv_valid"):
        row["status"] = "invalid"
        row["detail"] = [report.get("spirv_error", "invalid SPIR-V")]
        return row
    replay_dir = out / name / "replay"
    # The other compiler's module: the lift against a captured translation,
    # the translation against a captured lift.
    captured_lift = report.get("captured_module") == "lifted"
    other = lift_dir / ("reference.spv" if captured_lift else "lifted.spv")
    row["compared"] = "captured lift, translation replayed" if captured_lift else "captured translation, lift replayed"
    if not other.is_file():
        row["status"] = "error"
        row["detail"] = ["gcnlift wrote no %s" % other.name]
        return row
    replay = subprocess.run([str(drawreplay), str(capture), f"--{stage}", str(other), "--no-live", "--out", str(replay_dir)],
                            capture_output=True, text=True, timeout=timeout)
    rep = load(replay_dir / "report.json")
    row["drawreplay_exit"] = replay.returncode
    variant = ((rep or {}).get("comparisons") or {}).get("A_vs_variant")
    if rep is None or variant is None:
        row["status"] = "not comparable"
        row["detail"] = (replay.stderr or replay.stdout).strip().splitlines()[-2:] or ["no replay report"]
        return row
    row["replay_pass"] = rep.get("pass")
    row["bytes_identical"] = variant.get("bytes_identical")
    row["variant_pass"] = variant.get("pass")
    if not keep_replay:
        # Each replay writes every target four times (~300 MiB); report.json holds the verdict.
        for f in replay_dir.iterdir():
            if f.suffix in (".bin", ".ppm"):
                f.unlink()
    if replay.returncode != 0:
        # The captured modules failed their own checks: nothing to compare against.
        failed = sorted(k for k, v in ((rep or {}).get("verdict") or {}).items() if v is False)
        row["status"] = "not comparable"
        if failed == ["pass", "self_tests_detect_perturbations"]:
            row["detail"] = ["the draw does not respond to the replay's perturbation self-tests (no covered texel to compare)"]
        else:
            row["detail"] = ["the capture's own replay failed (drawreplay exit %d: %s)" % (replay.returncode, ", ".join(failed) or "no verdict")]
    elif variant.get("bytes_identical"):
        row["status"] = "identical"
    else:
        row["status"] = "different"
        row["detail"] = [json.dumps({k: v for k, v in variant.items() if k not in ("note",)})[:400]]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("captures", type=Path, help="directory holding the capture directories")
    parser.add_argument("--out", type=Path, required=True, help="output directory under tmp/ or build/")
    parser.add_argument("--gcnlift", type=Path, default=ROOT / "build/gcnlift")
    parser.add_argument("--drawreplay", type=Path, default=ROOT / "build/drawreplay")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--delete-verified", action="store_true", help="remove a capture once its lift is byte-identical")
    parser.add_argument("--keep-replay", action="store_true", help="keep drawreplay's target dumps and heatmaps (default: report.json only)")
    parser.add_argument("--stage", choices=("ps", "vs"), default="ps", help="lift the pixel (default) or the vertex shader")
    args = parser.parse_args()
    captures = sorted(p for p in args.captures.iterdir() if (p / "manifest.json").is_file())
    if not captures:
        print(f"no captures under {args.captures}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for capture in captures:
        try:
            row = verify(capture, args.out, args.gcnlift, args.drawreplay, args.timeout, args.keep_replay, args.stage)
        except subprocess.TimeoutExpired:
            row = {"capture": str(capture), "name": capture.name, "status": "error", "detail": ["timed out"]}
        rows.append(row)
        detail = "; ".join(row.get("detail", []))[:160]
        print(f"{row['status']:15s} {capture.name}  {detail}", flush=True)
        if args.delete_verified and row["status"] == "identical":
            shutil.rmtree(capture, ignore_errors=True)
    totals = Counter(r["status"] for r in rows)
    reasons = Counter()
    for r in rows:
        if r["status"] == "rejected":
            for why in r.get("detail", []):
                reasons[why.split(": ", 1)[-1] if why[:1].isdigit() or why[:1] == "0" else why] += 1
    summary = {
        "schema": "bbhost-lift-verify",
        "stage": args.stage,
        "captures": len(rows),
        "totals": dict(totals),
        "rejection_reasons": reasons.most_common(),
        "rows": rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("totals: " + ", ".join(f"{k} {v}" for k, v in totals.most_common()))
    for why, n in reasons.most_common(12):
        print(f"  {n:3d} rejected: {why}")
    return 1 if totals.get("different") or totals.get("invalid") else 0


if __name__ == "__main__":
    sys.exit(main())
