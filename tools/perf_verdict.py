#!/usr/bin/env python3
"""Which stage held the frame rate in a run: the bottleneck meter's verdicts
(host/perf_meter.h) read from a log, by place and over the run.

    tools/perf_verdict.py LOG [LOG...] [--all] [--json]

A log carries one of three kinds of line, read in this order of preference:

  perf:          each second's verdict (BBHOST_FRAME_STATS=1, which the test
                 harnesses set);
  perf summary:  about a minute's verdicts, always on (a player's log);
  frames:        a log from before the meter (BBHOST_FRAME_STATS=1): each
                 second's counters are put through the meter's rules again
                 (classify below, host/bottleneck.h's copy) as far as the line
                 has them - no GX barrier spin, no back-pressure (the command
                 processor counts as the limit from 92% of a core), the
                 recorder's CPU in place of its replay time.

Places: a harness log's measure windows (tools/area_check.py's probe marks
`mark <place>:measure` ... `:measured`) - the seconds outside them go to
"(between)" and are left out of "all places" unless --all; otherwise the map
the meter names (`at m24_01`), or the probe's map block.
"""
import argparse
import json
import re
import sys

STAGES = ["none", "capped", "stall", "main", "gx", "render", "cp", "record", "gpu", "present"]
LOADS = ["main", "gx", "render", "cp", "record", "gpu", "present"]
MAPS = {
    "m21_00": "Hunter's Dream", "m22_00": "Forbidden Woods", "m23_00": "Old Yharnam", "m24_00": "Cathedral Ward",
    "m24_01": "Central Yharnam", "m24_02": "Upper Cathedral Ward", "m25_00": "Cainhurst", "m26_00": "Nightmare of Mensis",
    "m27_00": "Forbidden Woods", "m28_00": "Hemwick", "m29_00": "chalice dungeon", "m32_00": "Byrgenwerth",
    "m33_00": "Nightmare Frontier", "m34_00": "Hunter's Nightmare", "m35_00": "Research Hall", "m36_00": "Fishing Hamlet",
}

PERF = re.compile(r"\] perf: ([\d.]+) s(?: at (m\d+_\d+))?: limit (\w+)(?: (\d+)%)?(?:, next (\w+) (\d+)%)? \((.*?)\); "
                  r"([\d.]+) fps ([\d.]+) ms; loads (.*?); main loop a frame: (.*?) ms; held back by the command processor "
                  r"([\d.]+)%; long frames ([\d.]+) ms")
SUMMARY = re.compile(r"\] perf summary: ([\d.]+) s(?: at (m\d+_\d+))?: ([\d.]+) fps \(1% low ([\d.]+)\); limit (\w+)(?: (\d+)%)?"
                     r"(?:, next (\w+) (\d+)%)? \((.*?)\); seconds by limit: (.*?); loads (.*?); main loop a frame ([\d.]+) ms: "
                     r"(.*?) ms; held back by the command processor ([\d.]+)%; long frames ([\d.]+) ms")
LOAD = re.compile(r"(\w+) (-|\d+)%?")
FRAME = re.compile(r"own (\S+) gx (\S+) waits (\S+) file (\S+) limiter (\S+)")
SECONDS = re.compile(r"(\w+) ([\d.]+) s")
FRAMES = re.compile(r"\] frames: ([\d.]+) s: (\d+) flips \(([\d.]+)/s\), displayed ([\d.]+)/s, interval avg ([\d.]+) p95 ([\d.]+) "
                    r"max ([\d.]+) ms, \d+ over 16.7, \d+ over 33.3; cpu \d+%:(.*?); main loop waited (\d+) ms, slept (\d+) ms, "
                    r"file I/O (\d+) ms")
WORK = re.compile(r"main loop work avg ([\d.]+) p95")
GPU_BUSY = re.compile(r"gpu busy ([\d.]+) ms/s")
IMPORTS = re.compile(r"imports waited for \d+ in ([\d.]+) ms")
THREAD = re.compile(r" (\S+):(\d+) (\d+)%")
INTERVALS = re.compile(r"\] frame intervals ms:((?: [\d.]+)+)\s*$")
MARK = re.compile(r"mapval mark (\S+?):(measure|measured) ")
POS = re.compile(r"mapval pos [\d.]+ ([0-9a-f]+) ")
MODE = re.compile(r"frame source: SprjFlipper::Update is ours \(0x[0-9a-f]+, (\w+)")
UNCAPPED_AT = re.compile(r"frame rate: uncapped, at most ([\d.]+) fps")


def share(ms, win):
    return None if ms is None or win <= 0 else ms / win


def classify(w):
    """host/bottleneck.h's classify, for a window dict (ms; None = not
    measured). One extension for logs without back-pressure (backlog None):
    the command processor counts as the limit from 92% of a core."""
    v = {"limit": "none", "next": "none", "limit_load": None, "next_load": None, "why": ""}
    secs = w["secs"]
    if secs <= 0:
        return v
    win = secs * 1000.0
    flips = w["flips"]
    v["fps"] = flips / secs
    v["frame_ms"] = win / flips if flips > 0 else 0
    own = w.get("main_cpu_ms")
    spin = w.get("gx_spin_ms")
    if own is not None and spin is not None:
        own = max(0.0, own - spin)
    loads = {"main": share(own, win), "gx": share(spin, win), "render": share(w.get("render_cpu_ms"), win),
             "cp": share(w.get("cp_cpu_ms"), win), "record": share(w.get("rec_busy_ms"), win),
             "gpu": share(w.get("gpu_busy_ms"), win), "present": share(w.get("present_cpu_ms"), win)}
    v["load"] = loads
    lim_ms = w.get("limiter_ms")
    main_loop = None if lim_ms is None else min(1.0, max(0.0, (win - lim_ms) / win))
    v["main_loop"] = main_loop
    backlog = w.get("backlog_ms")
    v["backlog"] = None if backlog is None else backlog / win

    def ld(s):
        x = loads.get(s)
        return -1 if x is None else x

    busy = win - lim_ms if lim_ms is not None else win
    gx_leads = spin is not None and own is not None and (spin >= own or spin * 3 >= busy)

    def decide(s, why):
        v["limit"], v["why"] = s, why
        ml = s in ("main", "gx")
        v["limit_load"] = main_loop if ml and main_loop is not None else (loads.get(s) if s in loads else None)
        best, bl = "none", -1
        by_share = s in ("capped", "stall", "none") and main_loop is not None
        for t in LOADS:
            if t == s or (ml and t in ("main", "gx")):
                continue
            x = ld(t)
            if t in ("main", "gx") and by_share:
                if t != ("gx" if gx_leads else "main"):
                    continue
                x = main_loop
            if x > bl:
                best, bl = t, x
        v["next"], v["next_load"] = best, (bl if bl >= 0 else None)
        return v

    loading = w.get("loading_ms", 0) or 0
    if flips < 2 * secs:
        return decide("stall", "loading screen" if loading * 2 >= win else "no frames")
    if loading * 2 >= win:
        return decide("stall", "loading screen")
    long_ms = w.get("long_ms", 0) or 0
    if long_ms >= 0.2 * win:
        why, most = "long frames", 0.25 * long_ms
        for key, name in (("main_file_ms", "file I/O"), ("pipeline_ms", "pipeline builds"), ("import_ms", "imports")):
            if (w.get(key) or 0) > most:
                most, why = w.get(key), "long frames: " + name
        return decide("stall", why)
    lim = None if lim_ms is None else lim_ms / win
    cap = w.get("cap_fps") or 0
    at_cap = cap > 0 and v["fps"] >= 0.985 * cap
    if lim is not None and lim >= 0.02 and at_cap:
        return decide("capped", "at the frame cap")
    if ld("gpu") >= 0.90:
        return decide("gpu", "the GPU is busy")
    if backlog is None:
        if ld("cp") >= 0.92:
            return decide("cp", "the command processor is busy (no back-pressure in this log)")
    elif backlog / win >= 0.05:
        cp, rec, gpu = ld("cp"), ld("record"), ld("gpu")
        if rec >= 0.85 and rec >= cp:
            return decide("record", "the game waits for the command processor, the recorder busiest")
        if gpu >= 0.80 and (w.get("cp_gpu_wait_ms") or 0) >= 0.10 * win and gpu > cp:
            return decide("gpu", "the game waits for the command processor, which waits for the GPU")
        return decide("cp", "the game waits for the command processor")
    if ld("render") >= 0.85:
        return decide("render", "the render thread is busy")
    if main_loop is None or main_loop >= 0.90:
        if gx_leads:
            return decide("gx", "the main loop waits for the GX workers")
        return decide("main", "the main loop is busy (the GX barrier not measured)" if spin is None else "the main loop is busy")
    disp = w.get("displayed")
    if disp is not None and disp < 0.9 * flips and ld("present") >= 0.80:
        return decide("present", "the display shows fewer flips than the game makes")
    if lim is not None and lim >= 0.10:
        return decide("capped", "the limiter waits, the frames uneven")
    return decide("none", "no stage is near full")


def num(s):
    return None if s in ("-", None) else float(s)


def parse_loads(text):
    return {k: (None if p == "-" else int(p) / 100.0) for k, p in LOAD.findall(text) if k in LOADS}


def parse_frame(text):
    m = FRAME.search(text)
    return {k: num(m.group(i + 1)) for i, k in enumerate(("own", "gx", "waits", "file", "limiter"))} if m else {}


class Run:
    """One log's seconds, each a dict: place, secs, limit, fps, frame_ms, loads, frame (ms a frame), why."""

    def __init__(self, path):
        self.path = path
        self.seconds = []
        self.summaries = []
        self.kind = None
        self.read()

    def read(self):
        lines = open(self.path, errors="replace").read().splitlines()
        have_perf = any("] perf: " in l for l in lines)
        have_summary = any("] perf summary: " in l for l in lines)
        self.kind = "perf" if have_perf else "summary" if have_summary else "frames"
        windowed = any(MARK.search(l) for l in lines)
        place, block = None, None
        cap, pending = 0.0, None
        for i, line in enumerate(lines):
            m = MARK.search(line)
            if m:
                place = m.group(1) if m.group(2) == "measure" else None
                continue
            m = POS.search(line)
            if m:
                b = int(m.group(1), 16)
                block = "m%02d_%02d" % ((b >> 24) & 0xff, (b >> 16) & 0xff)
            m = MODE.search(line)
            if m:  # the frame cap, for old logs (the meter's lines need none)
                mode = m.group(1)
                cap = (cap if cap > 0 else 240.0) if mode == "uncapped" else float(mode) if mode.isdigit() else 30.0
            m = UNCAPPED_AT.search(line)
            if m:
                cap = float(m.group(1))
            where = place if windowed else None
            if self.kind == "perf":
                m = PERF.search(line)
                if not m:
                    continue
                self.seconds.append({
                    "place": where if windowed else (m.group(2) or block or "(unknown)"),
                    "windowed": where is not None, "secs": float(m.group(1)), "limit": m.group(3),
                    "next": m.group(5), "why": m.group(7), "fps": float(m.group(8)), "frame_ms": float(m.group(9)),
                    "loads": parse_loads(m.group(10)), "frame": parse_frame(m.group(11)), "backlog": float(m.group(12)) / 100,
                    "long_ms": float(m.group(13))})
            elif self.kind == "summary":
                m = SUMMARY.search(line)
                if not m:
                    continue
                secs = float(m.group(1))
                by = {k: float(s) for k, s in SECONDS.findall(m.group(10))}
                self.summaries.append({
                    "place": where if windowed else (m.group(2) or block or "(unknown)"), "windowed": where is not None,
                    "secs": secs, "fps": float(m.group(3)), "low": float(m.group(4)), "limit": m.group(5), "why": m.group(9),
                    "by": by, "loads": parse_loads(m.group(11)), "frame_ms": float(m.group(12)), "frame": parse_frame(m.group(13)),
                    "backlog": float(m.group(14)) / 100})
            else:
                m = FRAMES.search(line)
                if m:
                    if pending:
                        self.add_frames(*pending)
                    pending = (m, line, where if windowed else (block or "(unknown)"), where is not None, cap, None)
                    continue
                m = INTERVALS.search(line)
                if m and pending:
                    iv = (pending[5] or []) + [float(x) for x in m.group(1).split()]
                    pending = pending[:5] + (iv,)
        if pending:
            self.add_frames(*pending)

    def add_frames(self, m, line, place, windowed, cap, intervals):
        secs, flips = float(m.group(1)), int(m.group(2))
        threads = THREAD.findall(m.group(8))
        cpu = {}
        for name, _tid, p in threads:
            ms = int(p) * 10.0 * secs
            # The main thread: Linux leaves it unnamed (the busiest "bbhost"), Windows names it "main".
            key = {"bbhost": "main", "main": "main", "GXRenderThread": "render", "bb-cp0": "cp", "bb-record": "record",
                   "bb-present": "present"}.get(name)
            if key and (key not in cpu or ms > cpu[key]):
                cpu[key] = ms
        work = WORK.search(line)
        gpu = GPU_BUSY.search(line)
        imp = IMPORTS.search(line)
        w = {"secs": secs, "flips": flips, "displayed": float(m.group(4)) * secs, "cap_fps": cap,
             "main_cpu_ms": cpu.get("main"), "gx_spin_ms": None, "render_cpu_ms": cpu.get("render"), "cp_cpu_ms": cpu.get("cp"),
             "rec_busy_ms": cpu.get("record"), "present_cpu_ms": cpu.get("present"),
             "gpu_busy_ms": float(gpu.group(1)) * secs if gpu else None, "backlog_ms": None,
             "main_wait_ms": float(m.group(9)) + float(m.group(10)), "main_file_ms": float(m.group(11)),
             "import_ms": float(imp.group(1)) if imp else 0.0, "pipeline_ms": 0.0, "loading_ms": 0.0}
        w["limiter_ms"] = max(0.0, secs * 1000.0 - float(work.group(1)) * flips) if work else None
        if intervals:
            s = sorted(intervals)
            med = s[len(s) // 2]
            over = max(2 * med, 25.0)
            w["long_ms"] = sum(x - med for x in intervals if x > over)
        else:
            avg, worst = float(m.group(5)), float(m.group(7))
            w["long_ms"] = max(0.0, worst - 2 * avg) if worst > max(2 * avg, 25.0) else 0.0
        v = classify(w)
        n = max(flips, 1)
        frame = {"own": None if w["main_cpu_ms"] is None else w["main_cpu_ms"] / n, "gx": None, "waits": w["main_wait_ms"] / n,
                 "file": w["main_file_ms"] / n, "limiter": None if w["limiter_ms"] is None else w["limiter_ms"] / n}
        self.seconds.append({"place": place, "windowed": windowed, "secs": secs, "limit": v["limit"], "next": v["next"],
                             "why": v["why"], "fps": v["fps"], "frame_ms": v["frame_ms"], "loads": v["load"], "frame": frame,
                             "backlog": None, "long_ms": w["long_ms"]})


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def wmean(pairs):
    pairs = [(x, wt) for x, wt in pairs if x is not None]
    tot = sum(wt for _, wt in pairs)
    return sum(x * wt for x, wt in pairs) / tot if tot else None


def group(rows, summaries):
    """By place: seconds, verdict seconds, fps, loads, the main loop's frame."""
    out = {}
    for r in rows:
        g = out.setdefault(r["place"], {"secs": 0.0, "by": {}, "fps": [], "frame_ms": [], "loads": {k: [] for k in LOADS},
                                        "frame": {k: [] for k in ("own", "gx", "waits", "file", "limiter")}, "backlog": [],
                                        "why": {}})
        g["secs"] += r["secs"]
        g["by"][r["limit"]] = g["by"].get(r["limit"], 0.0) + r["secs"]
        g["why"].setdefault(r["limit"], {})
        g["why"][r["limit"]][r["why"]] = g["why"][r["limit"]].get(r["why"], 0) + 1
        g["fps"].append((r["fps"], r["secs"]))
        g["frame_ms"].append((r["frame_ms"], r["secs"]))
        for k in LOADS:
            g["loads"][k].append(((r["loads"] or {}).get(k), r["secs"]))
        for k in g["frame"]:
            g["frame"][k].append(((r["frame"] or {}).get(k), r["secs"]))
        g["backlog"].append((r["backlog"], r["secs"]))
    for s in summaries:
        g = out.setdefault(s["place"], {"secs": 0.0, "by": {}, "fps": [], "frame_ms": [], "loads": {k: [] for k in LOADS},
                                        "frame": {k: [] for k in ("own", "gx", "waits", "file", "limiter")}, "backlog": [],
                                        "why": {}})
        g["secs"] += s["secs"]
        for k, secs in s["by"].items():
            g["by"][k] = g["by"].get(k, 0.0) + secs
        g["why"].setdefault(s["limit"], {})
        g["why"][s["limit"]][s["why"]] = g["why"][s["limit"]].get(s["why"], 0) + 1
        g["fps"].append((s["fps"], s["secs"]))
        g["frame_ms"].append((s["frame_ms"], s["secs"]))
        for k in LOADS:
            g["loads"][k].append((s["loads"].get(k), s["secs"]))
        for k in g["frame"]:
            g["frame"][k].append((s["frame"].get(k), s["secs"]))
        g["backlog"].append((s["backlog"], s["secs"]))
    res = {}
    for place, g in out.items():
        res[place] = {
            "secs": g["secs"], "by": dict(sorted(g["by"].items(), key=lambda kv: -kv[1])),
            "fps": wmean(g["fps"]), "frame_ms": wmean(g["frame_ms"]),
            "loads": {k: wmean(v) for k, v in g["loads"].items()},
            "frame": {k: wmean(v) for k, v in g["frame"].items()}, "backlog": wmean(g["backlog"]),
            "why": {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in g["why"].items()},
        }
    return res


def pct(x):
    return "   -" if x is None else "%3.0f%%" % (x * 100)


def ms(x):
    return "   -" if x is None else "%5.2f" % x


def show(name, g, width):
    by = ", ".join("%s %.0f s (%.0f%%)" % (k, s, 100 * s / g["secs"]) for k, s in list(g["by"].items())[:3])
    top = next(iter(g["by"]), "none")
    loads = " ".join(pct(g["loads"][k]) for k in LOADS)
    fr = g["frame"]
    label = MAPS.get(name, "") if re.match(r"m\d+_\d+$", name or "") else ""
    print("%-*s %5.0f %6.1f %6.2f  %-44s %s  %s %s %s %s  %s" % (
        width, (name or "") + (" (%s)" % label if label else ""), g["secs"], g["fps"] or 0, g["frame_ms"] or 0, by, loads,
        ms(fr["own"]), ms(fr["gx"]), ms(fr["waits"]), ms(fr["limiter"]),
        "" if g["backlog"] is None else "%.1f%%" % (100 * g["backlog"])))
    why = g["why"].get(top)
    if why:
        print("%-*s   %s: %s" % (width, "", top, why))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--all", action="store_true", help="count the seconds outside a harness's measure windows in the total")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    report = {}
    for path in a.logs:
        run = Run(path)
        places = group(run.seconds, run.summaries)
        counted = [r for r in run.seconds if a.all or r["windowed"] or not any(x["windowed"] for x in run.seconds)]
        csum = [s for s in run.summaries if a.all or s["windowed"] or not any(x["windowed"] for x in run.summaries)]
        overall = group([dict(r, place="all") for r in counted], [dict(s, place="all") for s in csum]).get("all")
        report[path] = {"kind": run.kind, "places": places, "all": overall}
        if a.json:
            continue
        print("%s (%s lines)" % (path, {"perf": "perf:", "summary": "perf summary:", "frames": "old frames:"}[run.kind]))
        names = [p for p in places if p is not None]
        width = max([len(n) + len(MAPS.get(n, "")) + 3 for n in names] + [12])
        print("%-*s %5s %6s %6s  %-44s %s  %s" % (width, "place", "secs", "fps", "ms", "limit (seconds)",
                                                  " ".join("%4s" % k[:4] for k in LOADS), " own    gx waits  limit  CP-held"))
        for name in names:
            show(name, places[name], width)
        if None in places:
            show("(between)", places[None], width)
        if overall:
            show("all places" if any(r["windowed"] for r in run.seconds + run.summaries) else "all", overall, width)
        print()
    if a.json:
        json.dump(report, sys.stdout, indent=1, default=str)
        print()


if __name__ == "__main__":
    main()
