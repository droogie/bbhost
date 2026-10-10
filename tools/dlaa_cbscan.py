#!/usr/bin/env python3
"""Finds the camera in an F12 constant-buffer capture (cbs-<flip>.txt/.bin, src/host/dlaa.cpp).

Looks in every bound buffer, at every 16-byte offset, for a 4x4 float matrix
shaped like a perspective projection (row- or column-major): m00 and m11
non-zero with m11/m00 near the aspect ratio, a +-1 in the w column / row, m33
zero and the other off-diagonal terms near zero. Prints where each one is, by
pipeline, with the matrix and the near/far it implies.

  python3 tools/dlaa_cbscan.py <f12 folder or cbs-N.txt> [--aspect 1.7778] [--draws main|all]
"""
import argparse
import collections
import math
import os
import struct
import sys


def load(path):
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path) if n.startswith("cbs-") and n.endswith(".txt"))
        if not names:
            sys.exit(f"no cbs-*.txt in {path}")
        path = os.path.join(path, names[0])
    blob = open(path[:-4] + ".bin", "rb").read()
    draws = []
    for line in open(path):
        if line.startswith("draw "):
            parts = line.split()
            d = {"seq": int(parts[1]), "name": parts[2], "cbs": []}
            for p in parts[3:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    d[k] = v
            draws.append(d)
        elif line.startswith("  cb "):
            _, st, slot, base, size, at, ln = line.split()
            draws[-1]["cbs"].append((int(st), int(slot), int(base, 16), int(size), blob[int(at):int(at) + int(ln)]))
    return path, draws


def projection(m, aspect):
    """m: 16 floats row-major as stored. Returns (kind, info) or None."""
    for kind, g in (("rows", lambda r, c: m[r * 4 + c]), ("cols", lambda r, c: m[c * 4 + r])):
        a, b = g(0, 0), g(1, 1)
        if not (0.05 < abs(a) < 20 and 0.05 < abs(b) < 20):
            continue
        if abs(abs(b / a) - aspect) > 0.02 * aspect:
            continue
        if abs(g(3, 3)) > 1e-4:
            continue
        w = g(3, 2)
        if abs(abs(w) - 1) > 1e-3:
            continue
        off = [g(0, 1), g(0, 3), g(1, 0), g(1, 3), g(3, 0), g(3, 1), g(2, 0), g(2, 1)]
        if max(abs(x) for x in off) > 1e-3:
            continue
        c, d = g(2, 2), g(2, 3)  # z' = c z + d, w' = w z
        info = f"x={a:.5f} y={b:.5f} c={c:.6g} d={d:.6g} w={w:+.0f} jitter=({g(0, 2):.3g},{g(1, 2):.3g})"
        # D3D-style 0..1 depth: near = -d/c, far = d/(w - c) for w = +-1 forms.
        if abs(c) > 1e-9:
            near = -d / c
            far = d / (w - c) if abs(w - c) > 1e-9 else float("inf")
            info += f" near~{near:.4g} far~{far:.4g} fovY~{math.degrees(2 * math.atan(1 / abs(b))):.2f}deg"
        return kind, info
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--aspect", type=float, default=16 / 9)
    ap.add_argument("--draws", default="all")
    args = ap.parse_args()
    path, draws = load(args.path)
    print(f"{path}: {len(draws)} draws")
    hits = collections.OrderedDict()
    for d in draws:
        if args.draws == "main" and not d.get("vp", "").startswith("1920"):
            continue
        for st, slot, base, size, data in d["cbs"]:
            for off in range(0, len(data) - 63, 16):
                m = struct.unpack_from("<16f", data, off)
                if any(math.isnan(x) or math.isinf(x) for x in m):
                    continue
                p = projection(m, args.aspect)
                if p:
                    key = (d["name"], st, slot, off, p[0])
                    hits.setdefault(key, []).append((d["seq"], base, size, p[1], d))
    for (name, st, slot, off, kind), lst in hits.items():
        seq, base, size, info, d = lst[0]
        print(f"{name} {'VS' if st == 0 else 'PS'} slot {slot} +0x{off:x} ({kind}) cb size {size}: {len(lst)} draws, "
              f"first draw {seq} vp={d.get('vp')} depth={d.get('depth')} base=0x{base:x}\n    {info}")


if __name__ == "__main__":
    main()
