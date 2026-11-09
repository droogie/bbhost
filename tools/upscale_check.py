#!/usr/bin/env python3
"""The upscalers' reprojection check, read from a run's log.

Run the game with BBHOST_UPSCALE_CHECK=4 BBHOST_UPSCALE_JITTER=0 (the check in
place of the upscaler's picture, host/upscale_pass.cpp) and an upscaler on; every
30 frames the log says which share of the pixels that changed the motion did
not explain: red / (red + green) over the picture and in sixteen bands of its
width from the left. This prints those windows and sums up the ones where the
picture moved (the camera turning, the hunter walking): the share over them, the
bands' (the left and right edges against the middle), and whether the log has
the camera the motion comes from.

    tools/upscale_check.py RUN.log [--moving 5] [--all]

--moving PCT: a window counts as moving when red + green cover at least PCT
percent of its pixels (default 5); --all prints every window, not only those.
Exit status 0 when there is a moving window, 1 when there is none, 2 when the
log has no check line at all.
"""
import argparse
import re
import sys

LINE = re.compile(
    r"upscaler: the reprojection check over 30 frames to flip (\d+) \((\d+)x(\d+)\): red/\(red\+green\) (\S+); "
    r"red ([\d.]+)%, green ([\d.]+)%, blue ([\d.]+)% of the pixels; red/\(red\+green\) by sixteenth of the width from the left:(.*)$")
CAMERA = re.compile(r"upscaler: the camera's view block at \S+ \(tangents ([\d.]+) ([\d.]+)")


def number(s):
    return None if s == "-" else float(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log")
    ap.add_argument("--moving", type=float, default=5.0)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    windows, camera = [], None
    with open(a.log, errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if m:
                bands = [number(x) for x in m.group(8).split()]
                windows.append({"flip": int(m.group(1)), "size": "%sx%s" % (m.group(2), m.group(3)), "ratio": number(m.group(4)),
                                "red": float(m.group(5)), "green": float(m.group(6)), "blue": float(m.group(7)), "bands": bands})
                continue
            m = CAMERA.search(line)
            if m and camera is None:
                camera = (float(m.group(1)), float(m.group(2)))
    if not windows:
        print("no reprojection check line (BBHOST_UPSCALE_CHECK and an upscaler on?)")
        return 2
    print("camera: %s" % ("tangents %.3f %.3f (aspect %.3f)" % (camera[0], camera[1], camera[0] / camera[1]) if camera else
                          "NONE - the motion has no camera (zero everywhere)"))
    moving = [w for w in windows if w["red"] + w["green"] >= a.moving]
    for w in windows if a.all else moving:
        r = "%.3f" % w["ratio"] if w["ratio"] is not None else "  -  "
        print("flip %7d %s  r/(r+g) %s  red %6.2f%%  green %6.2f%%  blue %5.2f%%  bands %s" % (
            w["flip"], w["size"], r, w["red"], w["green"], w["blue"],
            " ".join("%.2f" % b if b is not None else " -- " for b in w["bands"])))
    if not moving:
        print("%d windows, none moving (red + green under %g%% of the pixels)" % (len(windows), a.moving))
        return 1
    red, green = sum(w["red"] for w in moving), sum(w["green"] for w in moving)
    band = []
    for k in range(16):
        v = [w["bands"][k] for w in moving if len(w["bands"]) == 16 and w["bands"][k] is not None]
        band.append(sum(v) / len(v) if v else None)
    edge = [b for b in band[:2] + band[14:] if b is not None]
    middle = [b for b in band[6:10] if b is not None]
    print("%d of %d windows moving: r/(r+g) %.3f; by band %s; edges %s, middle %s" % (
        len(moving), len(windows), red / (red + green) if red + green else float("nan"),
        " ".join("%.2f" % b if b is not None else "--" for b in band),
        "%.3f" % (sum(edge) / len(edge)) if edge else "-", "%.3f" % (sum(middle) / len(middle)) if middle else "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
