#!/usr/bin/env python3
"""Check the HLSL path against the direct one, one captured draw at a time.

For each draw capture under CAPTURES (a directory of capture directories, as
`BBHOST_CAPTURE_DRAW='*ps'` or `'*'` writes them, or of folders of them, as
tools/area_check.py keeps a place's) whose bound pixel or vertex
shader is a lift (the manifest's translate.ps_lifted / vs_lifted), and each
dispatch capture (BBHOST_CAPTURE_DISPATCH) whose compute shader is one
(translate.lifted; stage cs), runs

    gcn2hlsl --spv <capture>/blobs/<stage>-spirv.bin --out <out>/<name>/<stage>
    drawreplay <capture> --<stage> <out>/<name>/<stage>.dxc.spv --no-live --out <out>/<name>/replay-<stage>

gcn2hlsl prints the captured lift as HLSL and compiles it with DXC (the module
the game would bind with BBHOST_HLSL=1); drawreplay renders the draw with the
captured modules (A, the lift) and again with DXC's stage (the variant), and
compares the outputs byte for byte (a dispatch's counters exactly, the
records it appends as a multiset: "identical, records reordered" when only
their order differs). --lavapipe <manifest> replays on Mesa's CPU Vulkan
driver, named by its ICD manifest (lvp_icd.*.json): repeatable, independent
of the GPU's scheduling, and with one thread (LP_NUM_THREADS=1 unless set)
so a dispatch's appends land in one order. Without it the replay runs on the
system's device.

Prints one line per capture and stage and the totals: identical (with the
texels the draw changed from the target's captured contents, and how many
distinct values it left: "identical, nothing drawn" when it changed none),
different (with the differing texels), HLSL not made (the reason), not
comparable (the capture's own replay failed its self-checks), not a lift.
Writes <out>/summary.json. Replays keep report.json only unless
--keep-replay.
"""
import argparse
import json
import os
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


def drawn(capture, replay_dir, targets):
    """Per target: the texels the replayed draw (A) left different from the
    capture's initial contents, and how many distinct texel values A holds -
    so an identical verdict on a draw that wrote nothing shows as such."""
    out = {}
    for name, t in (targets or {}).items():
        a, init = replay_dir / f"A-{name}.bin", capture / "blobs" / f"{name}-initial.bin"
        w, h = t.get("width") or 0, t.get("height") or 0
        if not a.is_file() or not init.is_file() or not w or not h:
            continue
        ab, ib = a.read_bytes(), init.read_bytes()
        if len(ab) != len(ib) or len(ab) % (w * h):
            continue
        size = len(ab) // (w * h)
        try:
            import numpy
            dt = {1: numpy.uint8, 2: numpy.uint16, 4: numpy.uint32, 8: numpy.uint64}.get(size)
            if dt is None:
                raise ImportError
            av, iv = numpy.frombuffer(ab, dt), numpy.frombuffer(ib, dt)
            out[name] = {"texels": w * h, "drawn": int((av != iv).sum()), "distinct": int(len(numpy.unique(av)))}
        except ImportError:
            texels = [ab[k:k + size] for k in range(0, len(ab), size)]
            out[name] = {"texels": w * h, "drawn": sum(1 for k, tx in enumerate(texels) if tx != ib[k * size:(k + 1) * size]),
                         "distinct": len(set(texels))}
    return out


def verify(capture, stage, out, args, env):
    name = capture.name
    row = {"capture": str(capture), "name": name, "stage": stage}
    m = load(capture / "manifest.json")
    if m is None:
        row["status"] = "error"
        row["detail"] = ["no manifest"]
        return row
    translate = m.get("translate") or {}
    if stage == "cs":
        lifted = "dispatch" in m and translate.get("lifted")
    else:
        lifted = "dispatch" not in m and translate.get("ps_lifted" if stage == "ps" else "vs_lifted")
    blob = capture / "blobs" / f"{stage}-spirv.bin"
    if not lifted or not blob.is_file():
        row["status"] = "not a lift"
        return row
    work = out / name
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / stage
    cmd = [str(args.gcn2hlsl), "--spv", str(blob), "--out", str(prefix)] + (["--O0"] if args.O0 else [])
    made = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    row["hlsl"] = (made.stdout or made.stderr).strip()
    if made.returncode != 0:
        row["status"] = "HLSL not made"
        row["detail"] = [row["hlsl"]]
        return row
    replay_dir = work / f"replay-{stage}"
    replay = subprocess.run([str(args.drawreplay), str(capture), f"--{stage}", f"{prefix}.dxc.spv", "--no-live", "--out", str(replay_dir)],
                            capture_output=True, text=True, timeout=args.timeout, env=env)
    rep = load(replay_dir / "report.json")
    row["drawreplay_exit"] = replay.returncode
    variant = ((rep or {}).get("comparisons") or {}).get("A_vs_variant")
    if variant:
        row["drawn"] = drawn(capture, replay_dir, variant.get("targets"))
    if not args.keep_replay and replay_dir.is_dir():
        for f in replay_dir.iterdir():
            if f.suffix in (".bin", ".ppm"):
                f.unlink()
    if rep is None or variant is None:
        row["status"] = "not comparable"
        row["detail"] = (replay.stderr or replay.stdout).strip().splitlines()[-2:] or ["no replay report"]
        return row
    if replay.returncode != 0 and variant.get("bytes_identical") is None:
        failed = sorted(k for k, v in (rep.get("verdict") or {}).items() if v is False)
        row["status"] = "not comparable"
        row["detail"] = ["the capture's own replay failed: " + (", ".join(failed) or "no verdict")]
        return row
    if stage == "cs" and variant.get("pass") and not variant.get("bytes_identical"):
        row["status"] = "identical, records reordered"
        row["detail"] = [f"{variant.get('reordered_ranges', 0)} written ranges in another order"]
    elif variant.get("bytes_identical"):
        # The capture's self-checks can fail independently of the variant
        # (a draw that covers nothing); an identical variant is still identical.
        row["status"] = "identical"
        cover = row.get("drawn") or {}
        if cover and not any(c["drawn"] for c in cover.values()):
            row["status"] = "identical, nothing drawn"
        if replay.returncode != 0:
            row["detail"] = ["the capture's own self-checks failed: " +
                             ", ".join(sorted(k for k, v in (rep.get("verdict") or {}).items() if v is False))]
    else:
        row["status"] = "different"
        row["detail"] = [json.dumps({k: v for k, v in variant.items() if k not in ("note",)})[:500]]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("captures", type=Path, help="directory holding the capture directories")
    parser.add_argument("--out", type=Path, required=True, help="output directory (under tmp/ or build/: it holds HLSL made from the game's shaders)")
    parser.add_argument("--stage", choices=("ps", "vs", "cs", "all"), default="all")
    parser.add_argument("--gcn2hlsl", type=Path, default=ROOT / "build/gcn2hlsl")
    parser.add_argument("--drawreplay", type=Path, default=ROOT / "build/drawreplay")
    parser.add_argument("--lavapipe", type=Path, metavar="MANIFEST", help="replay on lavapipe (CPU Vulkan), by its ICD manifest")
    parser.add_argument("--O0", action="store_true", help="compile with DXC -O0")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--keep-replay", action="store_true")
    args = parser.parse_args()

    env = dict(os.environ)
    if args.lavapipe:
        if not args.lavapipe.is_file():
            sys.exit(f"no lavapipe manifest at {args.lavapipe}")
        env["VK_DRIVER_FILES"] = str(args.lavapipe)
        env["VK_ICD_FILENAMES"] = str(args.lavapipe)
        env.setdefault("LP_NUM_THREADS", "1")
    args.out.mkdir(parents=True, exist_ok=True)
    # Capture directories here or one level down (area_check keeps a place's in a folder of its own).
    captures = sorted(p for p in list(args.captures.iterdir()) + [q for d in args.captures.iterdir() if d.is_dir() for q in d.iterdir()]
                      if p.is_dir() and (p / "manifest.json").is_file())
    stages = ("ps", "vs", "cs") if args.stage == "all" else (args.stage,)
    rows = []
    for capture in captures:
        for stage in stages:
            row = verify(capture, stage, args.out, args, env)
            rows.append(row)
            if row["status"] != "not a lift":
                cover = ", ".join(f"{n} {c['drawn']}/{c['texels']} drawn, {c['distinct']} values" for n, c in (row.get("drawn") or {}).items())
                print(f"{row['status']:16} {stage} {row['name']}" + (f"  [{cover}]" if cover else "") +
                      (f"  {row['detail'][0]}" if row.get("detail") else ""), flush=True)
    totals = Counter(r["status"] for r in rows)
    (args.out / "summary.json").write_text(json.dumps({"totals": totals, "rows": rows}, indent=1))
    print("totals: " + ", ".join(f"{k} {v}" for k, v in sorted(totals.items())))
    return 0 if totals.get("different", 0) == 0 and totals.get("HLSL not made", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
