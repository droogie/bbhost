#!/usr/bin/env python3
"""The full-game tour: every map entered, the player put down at stops spread
over its walkable ground with the camera turned through 360 degrees at each,
every cutscene played. Two uses: the pipeline list a first start can compile
everything the game draws from, and a full-game regression check (crashes,
hangs, missing files, draw failures, stalls, per map and stop).

    tools/full_tour.py inventory --app0 DIR [--out FILE]
    tools/full_tour.py plan --app0 DIR --navmesh-tools DIR --out PLAN [--radius 60] ...
    tools/full_tour.py run --plan PLAN --chunk N --bbhost BIN --config TOML --seed SAVE --runs RUNS [--live DIR]
    tools/full_tour.py report --plan PLAN --runs RUNS [--out DIR]
    tools/full_tour.py list --runs RUNS --app0 DIR --eboot EBOOT --out LIST

`inventory` lists what the tour visits, from the game's files: the map
blocks (map/mapstudio, the world list), each block's lamps (ReturnPointParam),
its light states by the world's time (GameMapParam: four time slots, world
time = event flags 9800-9802), the chalice dungeons' fixed layouts (the
dungeon view list's 固定 entries), and every cutscene - the remo files and the
event scripts' 2002 instructions that play them, with their play mode, the
region a player is moved to and the world time they set.

`plan` turns it into stops and chunks:
  - a block: the lamp travel (api->lamp_warp, the headstones' way), then stops
    on its navmesh (the worldmap navmesh module, as tools/mapval_blocks.py
    takes it, given by --navmesh-tools: Havok packfile faces placed by the
    block's collision parts), farthest first from a seeded start until every
    face's centroid is within --radius of a stop (height counts twice) or
    --max-stops; faces with nothing the layout places within --anchor m (the
    empty slabs under the streets) are left out; then the block's cutscenes;
  - a block's other light states (the time slots whose GameMapParam setting
    differs from the seed's): the world time's flags set, the travel, fewer
    stops (--state-radius);
  - the Hunter's Dream's event states (m21_00's event 12100300 switches its
    objects and fire on 12101852 and 9462 as well as the time);
  - a fixed chalice dungeon: the altar slot holds its map id, the travel to
    2902950 enters it, stops spread over what its layout places;
  - the twelve places of tools/area_check.py first, so what the tour records
    after them is what those places miss;
  - the movies (RequestPlayMovie ids) at the end.
Chunks hold at most --chunk-minutes of estimated tour each (a game.lock hold:
the start, the tour, the exit).

`run` plays one chunk: a copy of the seed's saves with the caches of the
chunk before (the chain, on --live: tmpfs keeps the frame dumps and the
caches' writes off a network file system), the pipeline manifest the union of
every earlier chunk's (bbhost drops recipes no start used for 16 starts; the
union keeps them), the probe (tools/mapval_plugin.c) and the chunk's tour.
The probe ends the run the way BBHOST_EXIT_SECONDS does (exit reports, caches
saved). The chunk's folder in --runs keeps run.log, the tour, its pipeline
manifest, JPEG shots (1280x720) of every stop, and done.json.

`report` reads every chunk: per item and stop the new pipeline recipes
(BBHOST_LOG_NEW_RECIPES), the stages the command processor translated
itself, its pipeline builds over 5 ms, flip stalls, loads, draw failures by
reason, the exit reports' fallbacks, crashes, timed-out waits, missing files;
per item the convergence (new recipes by stop) and what it adds to the twelve
places. `list` merges the chunks' manifests (tools/pipeline_list.py) into a
list.

Test-only: the probe moves the player and needs BBHOST_TEST_WARP=1; it turns
on the debug menu's Player No Dead and Player Hide. Never point --seed at a
player's data: the run plays a copy, and only the copy.
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import area_check  # noqa: E402  (run_toml, launch, game_env, AREAS)
import bbparam  # noqa: E402
import msb as msbmod  # noqa: E402
import pipeline_list  # noqa: E402
import world_baseline  # noqa: E402

# A test box can gate runs on a file holding the end of its GPU window (epoch seconds):
# BBHOST_GPU_WINDOW names it. Without it every chunk may run.
GATE = Path(os.environ["BBHOST_GPU_WINDOW"]) if os.environ.get("BBHOST_GPU_WINDOW") else None
TIME_FLAGS = (9800, 9801, 9802)
TIME_BITS = (0, 4, 6, 7)  # world time 0..3 as flags 9800-9802, the first the top bit (0x4d25010)
MAIN_BLOCKS = ["m24_01_00_00", "m24_00_00_00", "m24_02_00_00", "m22_00_00_00", "m23_00_00_00", "m25_00_00_00", "m27_00_00_00",
               "m28_00_00_00", "m26_00_00_00", "m32_00_00_00", "m33_00_00_00", "m21_00_00_00", "m21_01_00_00", "m34_00_00_00",
               "m35_00_00_00", "m36_00_00_00"]
DLC_BLOCKS = {"m34_00_00_00", "m35_00_00_00", "m36_00_00_00"}

# Per-step seconds the plan budgets (measured on the first chunks).
# Measured on the first chunks (2026-10-16, RTX 4070, warm caches): a stop
# 6.1 s, a travel 5-19 s (a chalice dungeon's load 16 s), a cutscene its skip
# time + 1 s, the start ~25 s to the tour (more when the union is large).
EST = {"start": 45.0, "travel": 14.0, "stop": 6.6, "remo": 33.0, "movie": 78.0, "view": 3.0, "twelve_hold": 30.0}


# --- the game's files -----------------------------------------------------------

def dcx(path):
    d = Path(path).read_bytes()
    return zlib.decompress(d[0x4c:]) if d[:4] == b"DCX\0" else d


def block_hex(name):
    a, b, c, d = (int(x) for x in name[1:].split("_"))
    return a << 24 | b << 16 | c << 8 | d


def lamps(app0):
    """block -> [(headstone slot, ReturnPointParam row)] (the Dream's have no slot)."""
    out = {}
    for rid, r in bbparam.load_table("ReturnPointParam", app0):
        if r["areaNo"] <= 0:
            continue
        m = "m%02d_%02d_00_00" % (r["areaNo"], r["blockNo"])
        out.setdefault(m, []).append((r["warpChairNo"] if r["warpChairNo"] >= 0 else 99, rid))
    return {m: sorted(v) for m, v in out.items()}


def light_states(app0):
    """block -> [setting for time 0..3] (GameMapParam.settingNo_timeSlotA..D)."""
    out = {}
    for rid, r in bbparam.load_table("GameMapParam", app0):
        if rid:
            m = "m%02d_%02d_00_00" % (rid // 1000000, rid // 10000 % 100)
            out[m] = [r["settingNo_timeSlot" + s] for s in "ABCD"]
    return out


def emevd_events(path):
    """[(event id, [(bank, id, args bytes)], [(instruction, dst, src, n)])] of a 64-bit EMEVD."""
    b = dcx(path)
    if b[:4] != b"EVD\0":
        raise ValueError("not an EMEVD")
    h = struct.unpack_from("<16q", b, 0x10)
    ev_n, ev_off, _in_n, in_off, pa_off, ar_off = h[0], h[1], h[2], h[3], h[9], h[13]
    out = []
    for i in range(ev_n):
        eid, n, ioff, pn, poff, _rest, _ = struct.unpack_from("<qqqqqIi", b, ev_off + i * 0x30)
        ins = []
        for k in range(n):
            bank, iid, alen, aoff, _lay = struct.unpack_from("<iiqqq", b, in_off + ioff + k * 0x20)
            ins.append((bank, iid, b[ar_off + aoff:ar_off + aoff + alen]))
        params = [struct.unpack_from("<qqqq", b, pa_off + poff + k * 0x20) for k in range(pn)]
        out.append((eid, ins, params))
    return out


def ints(a):
    return list(struct.unpack_from("<%di" % (len(a) // 4), a)) if len(a) >= 4 else []


def cutscenes(app0):
    """Every remo file, and the event instructions that play one (bank 2002):
    [{"id", "file", "map", "plays": [{"script", "event", "instruction", "args", "mode", "time"}]}]."""
    root = Path(app0) / "dvdroot_ps4"
    files = {}
    for p in sorted((root / "remo").glob("s*.remobnd.dcx")):
        m = re.match(r"s(\d\d)_(\d\d)_(\d{4})", p.name)
        if m:
            files[int(m.group(1)) * 1000000 + int(m.group(2)) * 10000 + int(m.group(3))] = p.name
    plays = {}
    scripts = sorted((root / "event").glob("*.emevd.dcx")) + sorted((root / "event").glob("m29_*/*.emevd.dcx"))
    for p in scripts:
        try:
            events = emevd_events(p)
        except (ValueError, struct.error, zlib.error):
            continue
        for eid, ins, params in events:
            for k, (bank, iid, a) in enumerate(ins):
                if bank != 2002:
                    continue
                v = ints(a)
                if not v or any(ii == k and dst == 0 for ii, dst, _s, _n in params):
                    continue  # the cutscene is an event parameter (2002[08]: a character's)
                # 2002[03] (id, mode, player); [04] (id, mode, region, area|block, player);
                # [06] as [04] + the world time; [07] (id, mode, player, time)
                t = v[5] & 0xff if iid == 6 and len(v) > 5 else v[3] & 0xff if iid == 7 and len(v) > 3 else None
                plays.setdefault(v[0], []).append({"script": p.name.split(".")[0], "event": eid, "instruction": "2002[%02d]" % iid,
                                                   "args": v, "mode": v[1] if len(v) > 1 else None, "time": t})
    out = []
    for cid in sorted(set(files) | set(plays)):
        if cid <= 0:
            continue
        out.append({"id": cid, "file": files.get(cid), "map": "m%02d_%02d_00_00" % (cid // 1000000, cid // 10000 % 100),
                    "plays": plays.get(cid, [])})
    return out


def fixed_dungeons(app0):
    """The chalice dungeons' fixed layouts (固定 in the dungeon view list): [(map name, description)]."""
    text = dcx(Path(app0) / "dvdroot_ps4" / "map" / "mapviewlist_fordungeon.loadlistlist").decode("cp932", "replace")
    out = []
    for line in text.splitlines():
        m = re.match(r"map:/MapStudio/(m29_\d\d_90_00)\.msb\s+#(.*)", line.strip())
        if m and "固定" in m.group(2):
            out.append((m.group(1), m.group(2).strip()))
    return out


def movies(app0):
    return sorted(p.name for p in (Path(app0) / "dvdroot_ps4" / "movie").glob("*.mp4"))


def msb_parts(app0, name):
    p = Path(app0) / "dvdroot_ps4" / "map" / "mapstudio" / (name + ".msb.dcx")
    if not p.is_file():
        p = Path(app0) / "dvdroot_ps4" / "map" / "mapstudio" / name / (name + ".msb.dcx")
    return msbmod.parts(msbmod.load(str(p))) if p.is_file() else []


def msb_variants(app0):
    root = Path(app0) / "dvdroot_ps4" / "map" / "mapstudio"
    out = {}
    for p in sorted(root.glob("m*_*_*_*.msb.dcx")):
        n = p.name.split(".")[0]
        out.setdefault(n[:-2] + "00", []).append(n)
    return out


def do_inventory(args):
    app0 = args.app0
    inv = {"blocks": [], "chalice": [], "cutscenes": cutscenes(app0), "movies": movies(app0)}
    lm, ls, var = lamps(app0), light_states(app0), msb_variants(app0)
    for b in MAIN_BLOCKS:
        inv["blocks"].append({"block": b, "lamps": [rid for _slot, rid in lm.get(b, [])], "light_states": ls.get(b, [0, 0, 0, 0]),
                              "msb_variants": var.get(b, [b]), "dlc": b in DLC_BLOCKS,
                              "cutscenes": [c["id"] for c in inv["cutscenes"] if c["map"] == b]})
    for name, desc in fixed_dungeons(app0):
        inv["chalice"].append({"map": name, "map_id": "%08x" % block_hex(name), "description": desc})
    inv["chalice_tile_sets"] = sorted(p.name for p in (Path(app0) / "dvdroot_ps4" / "map").glob("m29_*") if p.is_dir())
    text = json.dumps(inv, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print("%d blocks, %d fixed chalice dungeons (%d m29 map folders), %d cutscenes (%d with a remo file, %d played by an event script), %d movies" % (
        len(inv["blocks"]), len(inv["chalice"]), len(inv["chalice_tile_sets"]), len(inv["cutscenes"]),
        sum(1 for c in inv["cutscenes"] if c["file"]), sum(1 for c in inv["cutscenes"] if c["plays"]), len(inv["movies"])))
    return 0


# --- stops -----------------------------------------------------------------------

NAVMESH_NEEDS = ("the worldmap navmesh module: a folder holding a worldmap package with navmesh.py (faces(data): the polygons of a "
                 "Havok 2014 navmesh packfile) and msb.py (MSB, part_matrix), as tools/mapval_blocks.py takes it (MAPVAL_NAVMESH_TOOLS); "
                 "give it with --navmesh-tools")


def navmesh_module(tools):
    """(navmesh, MSB, part_matrix) from the worldmap package under tools, or an
    error naming what is missing. Only `plan` needs it, for the blocks' stops."""
    if not tools:
        return None, "plan needs " + NAVMESH_NEEDS
    sys.path.insert(0, str(tools))
    try:
        import numpy  # noqa: F401
        from worldmap import navmesh  # noqa: E402
        from worldmap.msb import MSB, part_matrix  # noqa: E402
    except ImportError as e:
        return None, "plan needs %s - %s under %s: %s" % (NAVMESH_NEEDS, "numpy" if "numpy" in str(e) else "worldmap", tools, e)
    return (navmesh, MSB, part_matrix), None


def navmesh_faces(app0, block, nav):
    """The block's navmesh face centroids in the MSB's frame (the warps' frame),
    via the worldmap module (its Havok reader and MSB transform): nav is
    navmesh_module's (navmesh, MSB, part_matrix)."""
    import numpy as np
    navmesh, MSB, part_matrix = nav
    root = Path(app0) / "dvdroot_ps4" / "map"
    msb = MSB((root / "mapstudio" / (block + ".msb.dcx")).read_bytes())
    path = root / block / (block + ".nvmhktbnd.dcx")
    if not path.is_file():
        return np.zeros((0, 3))
    cols = {}
    for part in msb.parts:
        if part["type"] == 5 and part["model"]:
            cols.setdefault(part["model"], []).append(part)
    cents = []
    for name, data in bbparam.bnd4(bbparam.dcx(str(path))).items():
        num = name.rsplit(".", 1)[0].rsplit("_", 1)[-1]
        polys = None
        for part in cols.get("h" + num, []):
            if polys is None:
                polys = navmesh.faces(data)
            M = part_matrix(part["pos"], part["rot"], part["scale"])
            for poly in polys:
                cents.append((poly.astype(np.float64) @ M[:3, :3] + M[3, :3]).mean(axis=0))
    return np.array(cents) if cents else np.zeros((0, 3))


def spread(points, radius, max_n, rng, first=None, min_n=1):
    """Farthest-first picks (height weighted twice) until every point is within
    radius of one (and at least min_n): [(index, coverage radius when it was picked)]."""
    import numpy as np
    if not len(points):
        return []
    w = points * np.array([1.0, 2.0, 1.0])
    i = first if first is not None else rng.randrange(len(points))
    picks = [(i, math.inf)]
    d = ((w - w[i]) ** 2).sum(1)
    while len(picks) < max_n:
        i = int(np.argmax(d))
        r = math.sqrt(float(d[i]))
        if r < radius and len(picks) >= min_n:
            break
        if r <= 0.5:
            break
        picks.append((i, r))
        d = np.minimum(d, ((w - w[i]) ** 2).sum(1))
    return picks


def anchored(points, parts, reach):
    """The points with a layout object, character or player start within reach (3D)."""
    import numpy as np
    anchors = np.array([p["pos"] for p in parts if p["type"] in (1, 2, 4)], np.float64)
    if not len(points) or not len(anchors):
        return points
    keep = np.zeros(len(points), bool)
    for k in range(0, len(points), 2048):
        chunk = points[k:k + 2048]
        d = ((chunk[:, None, :] - anchors[None, :, :]) ** 2).sum(2).min(1)
        keep[k:k + 2048] = d <= reach * reach
    return points[keep]


# --- the plan ----------------------------------------------------------------------

def item_estimate(it):
    s = EST["travel"] + EST["stop"] * len(it.get("stops", [])) + EST["remo"] * len(it.get("remos", []))
    s += EST["movie"] * len(it.get("movies", [])) + (EST["view"] if it.get("view") else 0)
    s += EST["twelve_hold"] if it.get("kind") == "twelve" else 0
    s += it.get("title_idle_s", 0)
    s += it.get("wait_s", 0) if it.get("kind") == "bosses" else 0
    return round(s, 1)


def do_plan(args):
    nav, why = navmesh_module(args.navmesh_tools)
    if not nav:
        print("full_tour: " + why, file=sys.stderr)
        return 2
    import numpy as np
    app0 = args.app0
    rng = random.Random(args.seed)
    lm, ls = lamps(app0), light_states(app0)
    cuts = cutscenes(app0)
    items, state_items = [], []

    # The twelve places first: what the tour records after them is what they miss.
    if not args.no_twelve:
        for a in area_check.AREAS:
            v = a.view()
            items.append({"id": "twelve:" + a.id, "kind": "twelve", "block": a.block, "return_point": a.return_point, "dlc": a.dlc,
                          "door_flags": list(a.flags), "view": [round(c, 4) for c in v] if v else None, "stops": [], "remos": []})

    def lamp_of(block):
        return lm[block][0][1]

    for b in MAIN_BLOCKS:
        if args.blocks and b not in args.blocks.split(","):
            continue
        pts = navmesh_faces(app0, b, nav)
        parts = msb_parts(app0, b)
        pts = anchored(pts, parts, args.anchor)
        picks = spread(pts, args.radius, args.max_stops, rng, min_n=args.min_stops)
        stops = [{"name": "s%02d" % k, "pos": [round(float(pts[i][0]), 3), round(float(pts[i][1]) + args.lift, 3), round(float(pts[i][2]), 3)],
                  "radius": None if math.isinf(r) else round(r, 1)} for k, (i, r) in enumerate(picks)]
        remos = [c["id"] for c in cuts if c["map"] == b and c["file"]][:args.remo_limit]
        items.append({"id": b, "kind": "block", "block": b, "return_point": lamp_of(b), "dlc": b in DLC_BLOCKS, "stops": stops,
                      "remos": remos, "faces": int(len(pts))})
        # Its other light states, by the world's time (the seed's is --seed-time:
        # flags 9800-9802 all off on the DLC seed, the evening).
        states = ls.get(b, [0, 0, 0, 0])
        seen = {states[args.seed_time]}
        for t in (0, 1, 2, 3):
            if states[t] in seen or args.no_states:
                continue
            seen.add(states[t])
            sp = spread(pts, args.state_radius, args.state_max_stops, rng, min_n=args.state_min_stops)
            bits = TIME_BITS[t]
            state_items.append({"id": "%s@t%d" % (b, t), "kind": "state", "block": b, "return_point": lamp_of(b), "time": t,
                          "light_state": states[t], "dlc": False,
                          "flags": [[f, (bits >> (2 - k)) & 1] for k, f in enumerate(TIME_FLAGS)],
                          "stops": [{"name": "s%02d" % k, "pos": [round(float(pts[i][0]), 3), round(float(pts[i][1]) + args.lift, 3),
                                                                  round(float(pts[i][2]), 3)], "radius": None if math.isinf(r) else round(r, 1)}
                                    for k, (i, r) in enumerate(sp)], "remos": []})
        if b == "m21_00_00_00" and not args.no_states:
            sp = spread(pts, args.state_radius, args.state_max_stops, rng, min_n=args.state_min_stops)
            for flag in (9462, 12101852):  # at the Blood Moon, with that flag on
                state_items.append({"id": "%s@%d" % (b, flag), "kind": "state", "block": b, "return_point": lamp_of(b), "time": 3,
                              "dlc": False, "flags": [[f, 1] for f in TIME_FLAGS] + [[flag, 1]], "restore": [[flag, 0]],
                              "stops": [{"name": "s%02d" % k, "pos": [round(float(pts[i][0]), 3), round(float(pts[i][1]) + args.lift, 3),
                                                                      round(float(pts[i][2]), 3)], "radius": None} for k, (i, _r) in enumerate(sp)],
                              "remos": []})

    if not args.no_chalice:
        for name, desc in fixed_dungeons(app0):
            parts = msb_parts(app0, name)
            pts = np.array([p["pos"] for p in parts if p["type"] in (2, 4)], np.float64)
            picks = spread(pts, args.chalice_radius, args.chalice_max_stops, rng)
            items.append({"id": name, "kind": "chalice", "block": "m29_00_00_00", "map_id": "%08x" % block_hex(name), "slot": 6,
                          "return_point": 2902950, "description": desc, "dlc": False,
                          "stops": [{"name": "s%02d" % k, "pos": [round(float(pts[i][0]), 3), round(float(pts[i][1]) + args.lift, 3),
                                                                  round(float(pts[i][2]), 3)], "radius": None if math.isinf(r) else round(r, 1)}
                                    for k, (i, r) in enumerate(picks)], "remos": []})

    if args.bosses:
        n = len(BOSS_BLOCK)
        for first in range(1, n + 1, args.boss_rounds):
            rounds = min(args.boss_rounds, n - first + 1)
            items.append({"id": "bosses:%d-%d" % (first, first + rounds - 1), "kind": "bosses", "block": None, "first": first, "rounds": rounds,
                          "test_kill": args.boss_seconds, "wait_s": rounds * (args.boss_seconds + 45), "stops": [], "remos": []})
    if not args.no_movies:
        # The title's attract movie (sprj_advertise) plays when the title sits
        # idle (within 240 s; not within 80 s): the chunk holding this item
        # waits there before pressing anything. In the world RequestPlayMovie
        # played nothing for ids 0-5; the ids here are tried at the item's place.
        items.append({"id": "movies", "kind": "movies", "block": None, "movies": [int(x) for x in args.movie_ids.split(",") if x],
                      "title_idle_s": args.title_idle, "stops": [], "remos": []})
    items += state_items  # after the seed's own states: a chunk of them sets the world's time as it goes

    if args.only:
        keep = args.only.split(",")
        items = [it for it in items if it["id"] in keep]
    for it in items:
        it["estimate_s"] = item_estimate(it)
    # Chunks: in order, each at most --chunk-minutes of tour (the start counted).
    chunks, cur, cur_s = [], [], EST["start"]
    limit = args.chunk_minutes * 60
    for it in items:
        # the time states and the Dream's need their flags: a chunk of their own kind
        prev = items_by_id(items, cur[-1])["kind"] if cur else None
        kind_change = cur and ((it["kind"] == "state") != (prev == "state") or (it["kind"] == "twelve") != (prev == "twelve"))
        if cur and (cur_s + it["estimate_s"] > limit or (kind_change and it["kind"] != "movies")):
            chunks.append(cur)
            cur, cur_s = [], EST["start"]
        cur.append(it["id"])
        cur_s += it["estimate_s"]
    if cur:
        chunks.append(cur)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("chunk*.txt"):
        old.unlink()
    plan = {"schema": "bbhost-full-tour-plan", "made": time.strftime("%Y-%m-%d %H:%M:%S"), "options": {k: v for k, v in vars(args).items()
                                                                                                     if k not in ("func",)},
            "items": items, "chunks": [{"n": k, "items": c, "estimate_s": round(EST["start"] + sum(items_by_id(items, i)["estimate_s"] for i in c)),
                                        "title_idle_s": max(items_by_id(items, i).get("title_idle_s", 0) for i in c)}
                                       for k, c in enumerate(chunks)]}
    plan["options"]["navmesh_tools"] = str(args.navmesh_tools)
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n")
    for ch in plan["chunks"]:
        (out / ("chunk%02d.txt" % ch["n"])).write_text(chunk_tour(plan, ch, args))
    total_stops = sum(len(i["stops"]) for i in items)
    print("%d items, %d stops, %d cutscenes, %d chunks, %.0f min estimated:" % (
        len(items), total_stops, sum(len(i["remos"]) for i in items), len(chunks), sum(c["estimate_s"] for c in plan["chunks"]) / 60))
    for ch in plan["chunks"]:
        print("  chunk %02d  %4.1f min  %s" % (ch["n"], ch["estimate_s"] / 60, ", ".join(ch["items"])))
    return 0


def items_by_id(items, ident):
    for it in items:
        if it["id"] == ident:
            return it
    raise KeyError(ident)


def chunk_tour(plan, ch, args):
    o = plan["options"]
    hold, headings, heading_s = o.get("stop_hold", 1.0), o.get("headings", 6), o.get("heading_s", 0.5)
    bosses = any(items_by_id(plan["items"], i)["kind"] == "bosses" for i in ch["items"])
    # A boss chunk lets the bosses see the player (their attacks are what it is
    # for); Player No Dead still holds.
    lines = ["# tools/full_tour.py chunk %d: %s" % (ch["n"], ", ".join(ch["items"])), "god" if bosses else "god hide", "flaglog 9800 3",
             "flaglog 6899 1"]
    for ident in ch["items"]:
        it = items_by_id(plan["items"], ident)
        tag = it["id"]
        lines.append("# %s (%s)" % (tag, it["kind"]))
        if it["kind"] == "bosses":
            # plugins/boss_rush runs these rounds by itself (auto_start; the run's
            # config, do_run); the probe only waits, then ends the run.
            lines.append("mark %s:start" % tag)
            lines.append("wait %.0f" % it["wait_s"])
            lines.append("mark %s:end" % tag)
            continue
        if it["kind"] == "movies":
            for mid in it["movies"]:
                lines.append("movie %d %.0f %.0f" % (mid, o.get("movie_skip", 20), o.get("movie_max", 30)))
                lines.append("mark %s:movie%d" % (tag, mid))
            continue
        for f, v in it.get("flags", []):
            lines.append("flags %d=%d" % (f, v))
        if it.get("flags"):
            lines.append("flaglog 9800 3")
        if it["kind"] == "chalice":
            lines.append("chalice %s %d %s" % (it["map_id"], it["slot"], tag))
        else:
            lines.append(" ".join(["travel %d %08x %s" % (it["return_point"], block_hex(it["block"]), tag)] + ["%d" % f for f in it.get("door_flags", [])]))
        lines.append("mark %s:arrived" % tag)
        lines.append("hold 2.0")
        if it["kind"] == "twelve":
            # as tools/area_check.py stands there: its view (or the lamp), the
            # camera behind, the settle and the window, no turning
            if it.get("view"):
                lines.append("warp %s:view %.3f %.3f %.3f %.4f 1.0" % (tag, *it["view"]))
            lines.append("camera")
            lines.append("mark %s:view" % tag)
            lines.append("hold %.1f" % EST["twelve_hold"])
            lines.append("dump %s-view" % safe(tag))
        for s in it["stops"]:
            name = "%s:%s" % (tag, s["name"])
            lines.append("warp %s %.3f %.3f %.3f 0 %.1f" % (name, s["pos"][0], s["pos"][1], s["pos"][2], hold))
            lines.append("camera")
            lines.append("hold 0.5")
            lines.append("mark %s" % name)
            lines.append("dump %s-%s" % (safe(tag), s["name"]))
            lines.append("turn %d %.2f" % (headings, heading_s))
        for rid in it["remos"]:
            lines.append("remo %d 0 %.0f %.0f" % (rid, o.get("remo_skip", 40), o.get("remo_max", 120)))
            lines.append("mark %s:remo%d" % (tag, rid))
        for f, v in it.get("restore", []):
            lines.append("flags %d=%d" % (f, v))
        lines.append("mark %s:end" % tag)
    lines.append("exit")
    lines.append("wait 30")
    return "\n".join(lines) + "\n"


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


# --- running one chunk ------------------------------------------------------------------

def gate_open():
    if GATE is None:
        return True
    try:
        return time.time() < float(GATE.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return False


def refresh_union(src, dst):
    """The union manifest as the next chunk's own: every recipe's last start set
    to the file's, so none is dropped as idle (kRecipeIdleRuns)."""
    run, origin, recipes = pipeline_list.read_file(src)
    for rc in recipes:
        rc.last_run = run
    pipeline_list.write_file(dst, run, origin, recipes)
    return len(recipes)


def merge_union(union, manifest):
    files = [str(f) for f in (union, manifest) if f and Path(f).is_file()]
    if not files:
        return 0
    merged, origin, run = {}, None, 0
    for f in files:
        r, o, recipes = pipeline_list.read_file(f)
        origin = origin or o
        run = max(run, r)
        for rc in recipes:
            have = merged.get((rc.compute, rc.key))
            if have is None or rc.last_run > have.last_run:
                merged[(rc.compute, rc.key)] = rc
    tmp = str(union) + ".new"
    pipeline_list.write_file(tmp, run, origin, list(merged.values()))
    os.replace(tmp, union)
    return len(merged)


def shots(frames, dest, w=1920, h=1080):
    """Every PPM dump as a 1280x720 JPEG (the top-left w x h of the display
    buffer: what the game rendered), the PPM removed."""
    from PIL import Image
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(frames.glob("*.ppm")):
        try:
            with Image.open(p) as im:
                im.crop((0, 0, min(w, im.width), min(h, im.height))).resize((1280, 720), Image.BILINEAR).save(dest / (p.stem + ".jpg"), quality=85)
            n += 1
        except OSError:
            pass
        p.unlink(missing_ok=True)
    return n


R_FRAMES_LINE = re.compile(r"\] frames: ")
R_WATCHDOG = re.compile(r"\] watchdog: no flip for")


def tour_launch(pass_dir, plan, say, hang_s=90.0, title_idle=0.0):
    """area_check.launch's run (confirm pressed through the title until the
    world, the log teed to run.log), plus: a hang - no per-second frames line
    for hang_s once in the world - stops the run (bbhost's watchdog has logged
    its threads by then); the probe's exit request ends it, the game exiting by
    itself."""
    import selectors
    import signal
    env = area_check.game_env(plan["settings"])
    requests = Path(plan["settings"]["BBHOST_TEST_REQUESTS"])
    started = time.monotonic()
    deadline = started + plan["budget_s"]
    world_deadline = started + 300 + title_idle
    next_tap, taps = started + max(area_check.TITLE_TAP_FIRST_S, title_idle), 0
    stop_reason, term_at, pending = None, None, b""
    world = read_one = exiting = False
    last_frames = started
    with (pass_dir / "run.log").open("wb") as log:
        with subprocess.Popen(plan["argv"], cwd=pass_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              start_new_session=True) as proc:
            sel = selectors.DefaultSelector()
            sel.register(proc.stdout, selectors.EVENT_READ)
            try:
                while sel.get_map():
                    now = time.monotonic()
                    if stop_reason is None and now > deadline:
                        stop_reason = "out of time"
                    if stop_reason is None and not world and now > world_deadline:
                        stop_reason = "the world was not reached"
                    if stop_reason is None and world and not exiting and now - last_frames > hang_s:
                        stop_reason = "hang: no frames for %.0f s" % (now - last_frames)
                    if stop_reason is None and exiting and now - exiting > 40:
                        stop_reason = "the exit did not end the game in 40 s"
                    if not world and stop_reason is None and now >= next_tap:
                        taps += 1
                        area_check.title_tap(requests, taps, None if read_one else area_check.TITLE_TAP_SEARCH_MS)
                        next_tap = now + area_check.TITLE_TAP_EVERY_S
                    if stop_reason and term_at is None:
                        say("stopping: " + stop_reason)
                        proc.send_signal(signal.SIGTERM)
                        term_at = now
                    if term_at is not None and now - term_at > 15 and proc.poll() is None:
                        proc.kill()
                    if term_at is not None and now - term_at > 30:
                        break
                    for key, _ in sel.select(timeout=0.25):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            sel.unregister(key.fileobj)
                            continue
                        log.write(chunk)
                        pending += chunk
                        while b"\n" in pending:
                            raw, pending = pending.split(b"\n", 1)
                            line = raw.decode(errors="replace")
                            if R_FRAMES_LINE.search(line):
                                last_frames = time.monotonic()
                            if not world and area_check.WORLD.search(line):
                                world = True
                                last_frames = time.monotonic()
                                say("world reached")
                            if not read_one:
                                m = area_check.PAD_READ.search(line)
                                if m and int(m.group(2), 16):
                                    read_one = True
                            if "test: exit asked for" in line and not exiting:
                                exiting = time.monotonic()
                                say("the probe asked for the exit")
                            m = R_STEP.search(line)
                            if m and m.group(1) == "travel":
                                say("travel " + m.group(2))
                            m = R_TFAIL.search(line)
                            if m:
                                say("%s: travel failed" % m.group(1))
                            m = R_REMO.search(line)
                            if m and m.group(1) in ("start", "end", "none", "timeout"):
                                say("remo %s %s" % (m.group(2), m.group(1)))
                            if R_WATCHDOG.search(line):
                                say("watchdog: " + line.split("watchdog: ", 1)[1][:120])
                            if stop_reason is None and R_FATAL.search(line):
                                stop_reason = "the game died: " + line.strip()[:160]
                    log.flush()
                rc = proc.wait(timeout=30)
            except BaseException:
                proc.kill()
                proc.wait()
                raise
            finally:
                sel.close()
    return {"returncode": rc, "stop_reason": stop_reason or "the game exited", "elapsed_s": round(time.monotonic() - started, 1)}


def do_run(args):
    plan = json.loads((Path(args.plan) / "plan.json").read_text())
    ch = plan["chunks"][args.chunk]
    runs = Path(args.runs)
    out = runs / ("c%02d" % args.chunk)
    if (out / "done.json").is_file() and not args.again:
        print("chunk %d already done (%s)" % (args.chunk, out / "done.json"))
        return 0
    if not gate_open() and not args.no_gate:
        print("the GPU window is closed (%s)" % GATE)
        return 3
    app0, eboot = area_check.read_paths(args.config)
    live = Path(args.live)
    chain = live / "chain"
    pass_dir = live / ("c%02d" % args.chunk)
    if pass_dir.exists():
        shutil.rmtree(pass_dir)
    for sub in ("cfg", "requests", "frames", "build"):
        (pass_dir / sub).mkdir(parents=True)
    seed = Path(args.seed)
    data = pass_dir / "data"
    shutil.copytree(seed, data, ignore=shutil.ignore_patterns("bbhost"))
    # The caches: the chain's (the chunk before), else the seed's.
    if (chain / "bbhost").is_dir() and not args.cold:
        shutil.copytree(chain / "bbhost", data / "bbhost")
    elif (seed / "bbhost").is_dir() and not args.cold:
        shutil.copytree(seed / "bbhost", data / "bbhost")
    else:
        (data / "bbhost").mkdir()
    union = runs / "union-manifest.bin"
    had = 0
    if union.is_file() and not args.cold:
        had = refresh_union(union, data / "bbhost" / "pipeline-manifest.bin")
    if args.list:
        pass
    (data / "plugins").mkdir(exist_ok=True)
    probe = pass_dir / "mapval.so"
    area_check.build_probe(probe)
    shutil.copy2(probe, data / "plugins" / "mapval.so")
    toml = area_check.run_toml(app0, eboot, data, args.p2p_port, 1920, 1080, old_hunters=True)
    boss = next((items_by_id(plan["items"], i) for i in ch["items"] if items_by_id(plan["items"], i)["kind"] == "bosses"), None)
    if boss:
        # plugins/boss_rush runs the rounds: by itself from the world's first
        # load, as the first time (each boss woken by a warp next to it), every
        # boss killed test_kill seconds into its round (its test switch).
        toml = toml.replace("[plugins]\n", "[plugins]\nboss_rush = true\n", 1)
        toml += "\n".join(["[boss_rush]", "auto_start = true", "start_delay = 5", "order = %s" % area_check.toml_str("story"), "dlc = true",
                           "chalice = true", "awake = false", "first = %d" % boss["first"], "rounds = %d" % boss["rounds"],
                           "test_kill = %d" % boss["test_kill"], ""])
    (pass_dir / "cfg" / "bbhost.toml").write_text(toml)
    tour = (Path(args.plan) / ("chunk%02d.txt" % args.chunk)).read_text()
    (pass_dir / "tour.txt").write_text(tour)
    settings = {
        "BBHOST_HEADLESS": "1", "BBHOST_NP_SIGNED_OUT": "1", "BBHOST_SETUP_WINDOW": "0", "BBHOST_SKIP_INTRO": "1",
        "BBHOST_CONFIG_DIR": str(pass_dir / "cfg"), "BBHOST_OPTIONS_PATH": str(pass_dir / "cfg" / "bbhost-options.toml"),
        "BBHOST_GAME_FPS": "60", "BBHOST_FRAME_STATS": "1", "BBHOST_FRAME_DETAIL": "1", "BBHOST_STALL_MS": "40",
        "BBHOST_TEST_WARP": "1", "BBHOST_TEST_REQUESTS": str(pass_dir / "requests"),
        "BBHOST_MAPVAL_TOUR": str(pass_dir / "tour.txt"), "BBHOST_MAPVAL_START": "8", "BBHOST_MAPVAL_DUMPS": str(pass_dir / "frames"),
        "BBHOST_MAPVAL_WARP_TRIES": "3",
        "BBHOST_MAPVAL_FLOAT": "1",  # gravity held at 0 through a map's stops: no fall through unloaded collision
        # the probe skips to the exit this long after it loads: the run ends
        # cleanly (reports, caches saved) inside the lock hold's budget
        "BBHOST_MAPVAL_DEADLINE": "%.0f" % max(120.0, args.budget - 75.0),
        "BBHOST_LOG_NEW_RECIPES": "1", "BBHOST_LOG_STAGE_TRANSLATIONS": "1", "BBHOST_LOG_PIPELINE_BUILDS": "5",
        "BBHOST_FS_TRACE": ".msb",
    }
    if args.list:
        settings["BBHOST_PIPELINE_LIST"] = str(Path(args.list).resolve())
    for item in args.env or []:
        k, _, v = item.partition("=")
        settings[k] = v
    budget = args.budget
    argv = world_baseline.launch_argv([str(Path(args.bbhost).resolve()), "--config", str(pass_dir / "cfg" / "bbhost.toml")], args.numa_node)
    if args.cpus:
        # A smaller machine's stand-in (the Steam Deck: 8 threads): those CPUs only.
        argv = ["numactl", "--physcpubind=" + args.cpus, "--membind=%d" % (args.numa_node or 0)] + argv[argv.index("--membind=%d" % args.numa_node) + 1:] \
            if args.numa_node is not None else ["numactl", "--physcpubind=" + args.cpus] + argv
    lplan = {"kind": "tour", "argv": argv, "cwd": str(pass_dir), "settings": settings, "budget_s": budget}
    out.mkdir(parents=True, exist_ok=True)
    (out / "pass-plan.json").write_text(json.dumps(lplan, indent=1) + "\n")
    (out / "tour.txt").write_text(tour)
    env_extra = {}
    if args.glcache:
        Path(args.glcache).mkdir(parents=True, exist_ok=True)
        env_extra = {"__GL_SHADER_DISK_CACHE": "1", "__GL_SHADER_DISK_CACHE_PATH": str(Path(args.glcache).resolve()),
                     "__GL_SHADER_DISK_CACHE_SIZE": "8589934592", "__GL_SHADER_DISK_CACHE_SKIP_CLEANUP": "1"}
    os.environ.update(env_extra)
    started = time.time()
    print("[%s] chunk %d: %s (%.1f min estimated, budget %.0f s, %d recipes from earlier chunks)" % (
        time.strftime("%H:%M:%S"), args.chunk, ", ".join(ch["items"]), ch["estimate_s"] / 60, budget, had), flush=True)
    launched = tour_launch(pass_dir, lplan, lambda s: print("[%s]   %s" % (time.strftime("%H:%M:%S"), s), flush=True),
                           title_idle=float(ch.get("title_idle_s", 0)))
    elapsed = time.time() - started
    shutil.copy2(pass_dir / "run.log", out / "run.log")
    man = data / "bbhost" / "pipeline-manifest.bin"
    if man.is_file():
        shutil.copy2(man, out / "manifest.bin")
    n_shots = shots(pass_dir / "frames", out / "shots")
    log = (out / "run.log").read_text(errors="replace")
    clean_exit = "test: exit asked for" in log
    total = merge_union(union, out / "manifest.bin") if man.is_file() and not args.no_chain else None
    # The chain moves on only from a run that ended as planned.
    if clean_exit and not args.no_chain and not args.cold:
        if (chain / "bbhost").exists():
            shutil.rmtree(chain / "bbhost")
        chain.mkdir(parents=True, exist_ok=True)
        shutil.move(str(data / "bbhost"), str(chain / "bbhost"))
    done = {"chunk": args.chunk, "items": ch["items"], "launched": launched, "elapsed_s": round(elapsed, 1), "clean_exit": clean_exit,
            "shots": n_shots, "union_recipes": total, "recipes_before": had, "bbhost": str(Path(args.bbhost).resolve()),
            "bbhost_sha256": world_baseline.digest(Path(args.bbhost)), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / ("done.json" if clean_exit or args.mark_done else "failed.json")).write_text(json.dumps(done, indent=1) + "\n")
    if not args.keep_live:
        shutil.rmtree(pass_dir, ignore_errors=True)
    print("[%s] chunk %d: %s in %.0f s, %d shots, union %s recipes%s" % (time.strftime("%H:%M:%S"), args.chunk, launched["stop_reason"], elapsed,
                                                                     n_shots, total, "" if clean_exit else " - NOT a clean exit"), flush=True)
    return 0 if clean_exit else 1


# --- the report -----------------------------------------------------------------------

PROBE = r"\] (?:plugin: )?mapval "
R_STEP = re.compile(PROBE + r"step \d+ (warp|travel) (\S+) ([\d.]+)")
R_MARK = re.compile(PROBE + r"mark (\S+) ([\d.]+) ([0-9a-f]+) (\S+) (\S+) (\S+) yaw")
R_REMO = re.compile(PROBE + r"remo-(request|start|end|none|timeout|skip) (\d+) ([\d.]+)(.*)")
R_MOVIE = re.compile(PROBE + r"movie-(request|end|skip) (\d+) ([\d.]+)(.*)")
R_TDONE = re.compile(PROBE + r"travel-done (\S+) ([\d.]+) ([0-9a-f]+) \S+ \S+ \S+ after ([\d.]+) s, ([\d.]+) s of it loading")
R_TFAIL = re.compile(PROBE + r"travel-failed (\S+) ([\d.]+) in ([0-9a-f]+) after ([\d.]+) s")
R_DEATH = re.compile(PROBE + r"death ([\d.]+) ([0-9a-f]+)")
R_RETRAVEL = re.compile(PROBE + r"retravel (\S+)")
R_REWARP = re.compile(PROBE + r"rewarp (\S+)")
R_SKIP = re.compile(PROBE + r"skip (\S+)")
R_DEADLINE = re.compile(PROBE + r"deadline [\d.]+ after (\d+) s: steps (\d+)-(\d+) skipped \(from (\S+)\)")
R_FLAGS = re.compile(PROBE + r"flags (\d+)\+(\d+) [\d.]+ = (-?\d+)")
R_TURN = re.compile(PROBE + r"turn-to (\d+) [\d.]+ yaw (\S+) \(was (\S+)\)")
R_NEW = re.compile(r"\] render: pipeline manifest: new recipe ([0-9a-f]+) (\S+)\+(\S*?)( tessellated)? \(")
R_XLATE = re.compile(r"\] render: CP translates (VS|PS) ([0-9a-f]+)")
R_BUILD = re.compile(r"\] render: pipeline build (\S+) .*? at flip \d+: ([\d.]+) ms")
R_STALL = re.compile(r"\] stall: flip #(\d+) came (\d+) ms after")
R_DUMP = re.compile(r"\] test: frame dump at flip (\d+)")
R_STALL_XLATE = re.compile(r"pipeline ms: translate (\d+)")
R_STALL_WORK = re.compile(r"pipeline work ms: shader libraries (\d+) \(\d+\), interface libraries (\d+) \(\d+\), links (\d+) \(\d+\), "
                          r"whole pipelines (\d+) \(\d+\), layouts (\d+) \(\d+\), waiting for a library (\d+), compute pipelines (\d+)")
R_DETAIL = re.compile(r"\] frame detail: draws (\d+): .*?draw failures (\d+)(?: \((.*)\))?")
# The host tessellator's LS pass built by the command processor at a draw's
# first sight (render.cpp tess_ls_pass_locked); a list's recipes carry them.
R_LS_CP = re.compile(r"\] render: tessellation LS (\S+) (ready|failed) \(")
R_LS_EXIT = re.compile(r"the tessellator's LS passes: (\d+) compute pipelines built from recipes at the start \((\d+) failed\), (\d+) of them taken by draws, "
                       r"(\d+) built by the command processor")
R_FS = re.compile(r"\] FS: (\S+) (.*?) \(")
R_MSB = re.compile(r"\] FS (?:trace|overlay): \S*?mapstudio/(\S+?\.msb)")
R_FATAL = area_check.FATAL
R_PROBLEM = area_check.PROBLEM
R_BYREASON = re.compile(r"\] .*(by reason|fallback|refused)", re.I)
R_LOADING = re.compile(r"\] loading: ([\d.]+) s \(flips")
R_AV = re.compile(r"\] (?:HLE fake: )?sceAvPlayerAddSource (\S+)(.*)")
R_BOSS_ROUND = re.compile(r"\] (?:plugin: )?boss_rush: round (\d+) - (.+?)\s*$")
R_BOSS_DONE = re.compile(r"\] (?:plugin: )?boss_rush: (.+?) defeated in (\d+:\d+)")


def new_seg(item, stop, chunk):
    return {"item": item, "stop": stop, "chunk": chunk, "new_keys": [], "new_recipes": 0, "new_tess": 0, "translations": 0, "builds_over_5ms": 0,
            "build_ms": 0.0, "stalls": 0, "stall_ms": 0, "draw_failures": 0, "failure_reasons": {}, "problems": [], "fatal": [],
            "missing": [], "msb": [], "movie_files": [], "dump_stalls": 0, "stall_pipeline_ms": 0, "hangs": [], "ps_first": 0, "ps_not_lifted": {}, "deaths": 0, "rewarps": 0, "retravels": 0, "skipped": 0, "frames_s": 0,
            "ls_cp": 0, "stall_compute_ms": 0, "notes": []}


def read_chunk(path, chunk):
    """[segment] of a chunk's log: the start, then one per item arrival, stop, cutscene, movie."""
    segs = [new_seg("(start)", "(start)", chunk)]
    items = {}
    exitlines = []
    flags = {}
    turns = []
    cur = segs[0]
    at_exit = False
    dump_flip = -100
    for line in Path(path).read_text(errors="replace").splitlines():
        if "] exit: " in line or "test: exit asked for" in line:
            at_exit = True
        if at_exit and R_BYREASON.search(line):
            exitlines.append(line.split("] ", 1)[-1][:600])
        m = R_STEP.search(line)
        if m:
            kind, name = m.group(1), m.group(2)
            item, _, stop = name.rpartition(":") if kind == "warp" else (name, "", "")
            if kind == "travel":
                cur = new_seg(name, "(travel)", chunk)
                segs.append(cur)
                items.setdefault(name, {"item": name, "chunk": chunk, "travel_s": None, "loading_s": None, "travel_failed": False,
                                        "remos": {}, "movies": {}})
            else:
                cur = new_seg(item, stop or "(warp)", chunk)
                segs.append(cur)
            continue
        m = R_REMO.search(line)
        if m:
            what, rid = m.group(1), int(m.group(2))
            if what == "request":
                cur = new_seg(cur["item"], "remo%d" % rid, chunk)
                segs.append(cur)
            it = items.get(cur["item"])
            if it is not None:
                r = it["remos"].setdefault(rid, {"started": False, "ended": None, "skips": 0})
                if what == "start":
                    r["started"] = True
                elif what in ("end", "none", "timeout"):
                    r["ended"] = what
                    mm = re.search(r"played ([\d.]+) s", m.group(4))
                    if mm:
                        r["played_s"] = float(mm.group(1))
                elif what == "skip":
                    r["skips"] += 1
            continue
        m = R_BOSS_ROUND.search(line)
        if m:
            name = m.group(2)
            cur = new_seg("boss:%s:%s" % (BOSS_MAP.get(name, "m00_00_00_00"), name), "fight", chunk)
            segs.append(cur)
            continue
        m = R_BOSS_DONE.search(line)
        if m:
            cur["notes"].append("defeated in %s" % m.group(2))
            continue
        m = R_MOVIE.search(line)
        if m:
            if m.group(1) == "request":
                cur = new_seg("movies", "movie%s" % m.group(2), chunk)
                segs.append(cur)
            continue
        m = R_TDONE.search(line)
        if m and m.group(1) in items:
            items[m.group(1)]["travel_s"] = float(m.group(4))
            items[m.group(1)]["loading_s"] = float(m.group(5))
            continue
        m = R_TFAIL.search(line)
        if m and m.group(1) in items:
            items[m.group(1)]["travel_failed"] = True
            cur["problems"].append("travel failed after %s s" % m.group(4))
            continue
        m = R_FLAGS.search(line)
        if m:
            flags.setdefault(cur["item"], []).append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
            continue
        m = R_DEADLINE.search(line)
        if m:
            cur["problems"].append("deadline after %s s: the rest of the chunk skipped from %s" % (m.group(1), m.group(4)))
            continue
        m = R_TURN.search(line)
        if m:
            # heading k > 0: the camera read just before the write still holds heading k-1's yaw
            k, want, was = int(m.group(1)), float(m.group(2)), float(m.group(3))
            if k > 0 and turns and turns[-1][0] == k - 1:
                turns.append((k, want, turns[-1][1], was))
            else:
                turns.append((k, want, None, None))
            continue
        if R_DEATH.search(line):
            cur["deaths"] += 1
        if R_REWARP.search(line):
            cur["rewarps"] += 1
        if R_RETRAVEL.search(line):
            cur["retravels"] += 1
        if R_SKIP.search(line):
            cur["skipped"] += 1
        m = R_NEW.search(line)
        if m:
            cur["new_keys"].append(m.group(1))
            cur["new_recipes"] += 1
            if m.group(4):
                cur["new_tess"] += 1
            continue
        if R_XLATE.search(line):
            cur["translations"] += 1
            continue
        if area_check.PS_LIFTED.search(line):
            cur["ps_first"] += 1
            continue
        m = area_check.PS_NOT_LIFTED.search(line)
        if m:
            cur["ps_first"] += 1
            why = area_check.reason_key(m.group(2))
            cur["ps_not_lifted"][why] = cur["ps_not_lifted"].get(why, 0) + 1
            continue
        m = R_BUILD.search(line)
        if m:
            ms = float(m.group(2))
            if ms >= 5:
                cur["builds_over_5ms"] += 1
                cur["build_ms"] += ms
            continue
        m = R_DUMP.search(line)
        if m:
            dump_flip = int(m.group(1))
            continue
        m = R_STALL.search(line)
        if m:
            if 0 <= int(m.group(1)) - dump_flip <= 3:
                cur["dump_stalls"] += 1  # the frame dump's own readback
            else:
                cur["stalls"] += 1
                cur["stall_ms"] += int(m.group(2))
                x, w = R_STALL_XLATE.search(line), R_STALL_WORK.search(line)
                cur["stall_pipeline_ms"] += (int(x.group(1)) if x else 0) + (sum(int(v) for v in w.groups()) if w else 0)
                cur["stall_compute_ms"] += int(w.group(7)) if w else 0
            continue
        if R_LS_CP.search(line):
            cur["ls_cp"] += 1
            continue
        m = R_DETAIL.search(line)
        if m:
            cur["frames_s"] += 1
            n = int(m.group(2))
            if n:
                cur["draw_failures"] += n
                for k, v in area_check.parse_reasons(m.group(3)).items():
                    cur["failure_reasons"][k] = cur["failure_reasons"].get(k, 0) + v
            continue
        m = R_FS.search(line)
        if m:
            cur["missing"].append("%s %s" % (m.group(1), m.group(2)))
            continue
        m = R_AV.search(line)
        if m:
            cur["movie_files"].append(m.group(1).rsplit("/", 1)[-1] + (" (skipped)" if "skipped" in m.group(2) else ""))
            continue
        m = R_MSB.search(line)
        if m:
            if m.group(1) not in cur["msb"]:
                cur["msb"].append(m.group(1))
            continue
        if R_WATCHDOG.search(line):
            cur["hangs"].append(line.split("] ", 1)[-1][:200])
        elif R_FATAL.search(line):
            cur["fatal"].append(line.split("] ", 1)[-1][:300])
        elif R_PROBLEM.search(line):
            cur["problems"].append(line.split("] ", 1)[-1][:200])
    return segs, items, exitlines, flags, turns


def do_report(args):
    plan = json.loads((Path(args.plan) / "plan.json").read_text())
    runs = Path(args.runs)
    out = Path(args.out or runs)
    out.mkdir(parents=True, exist_ok=True)
    all_segs, all_items, chunks = [], {}, []
    for ch in plan["chunks"]:
        d = runs / ("c%02d" % ch["n"])
        if not (d / "run.log").is_file():
            continue
        status = json.loads((d / "done.json").read_text()) if (d / "done.json").is_file() else (
            json.loads((d / "failed.json").read_text()) if (d / "failed.json").is_file() else {})
        segs, items, exitlines, flags, turns = read_chunk(d / "run.log", ch["n"])
        all_segs += segs
        all_items.update(items)
        chunks.append({"chunk": ch["n"], "items": ch["items"], "clean_exit": status.get("clean_exit"), "elapsed_s": status.get("elapsed_s"),
                       "stop_reason": (status.get("launched") or {}).get("stop_reason"), "exit_report": exitlines, "flags": flags,
                       "turns_checked": sum(1 for t in turns if t[2] is not None),
                       "turns_off": sum(1 for t in turns if t[2] is not None and abs(math.remainder(t[2] - t[3], 2 * math.pi)) > 0.05),
                       "new_recipes": sum(s["new_recipes"] for s in segs)})
    # Per item; a bosses item as a row per boss round (its segments are named
    # boss:<map>:<name> by the round's line).
    rows = []
    for it in plan["items"]:
        if it.get("kind") == "bosses":
            chunk_n = next((ch["n"] for ch in plan["chunks"] if it["id"] in ch["items"]), None)
            names = []
            for s in all_segs:
                if s["chunk"] == chunk_n and s["item"].startswith("boss:") and s["item"] not in names:
                    names.append(s["item"])
            rows += [({"id": n, "kind": "boss", "stops": []}, [s for s in all_segs if s["chunk"] == chunk_n and s["item"] == n]) for n in names]
        else:
            rows.append((it, [s for s in all_segs if s["item"] == it["id"]]))
    report_items = []
    for it, segs in rows:
        if not segs:
            continue
        stops = [s for s in segs if re.match(r"^[a-z]\d+$", s["stop"])]  # s00 (a plan's), d00 (a denser pass's), p00/q00 (a revisit's)
        info = all_items.get(it["id"], {})
        conv = [s["new_recipes"] for s in stops]
        tail = conv[len(conv) // 2:] if conv else []
        report_items.append({
            "id": it["id"], "kind": it["kind"], "chunk": segs[0]["chunk"], "stops_planned": len(it["stops"]), "stops_visited": len(stops),
            "travel_s": info.get("travel_s"), "loading_s": info.get("loading_s"), "travel_failed": info.get("travel_failed"),
            "new_recipes": sum(s["new_recipes"] for s in segs), "new_on_arrival": sum(s["new_recipes"] for s in segs if s["stop"] in ("(travel)", "arrived")),
            "new_at_stops": sum(conv), "new_in_cutscenes": sum(s["new_recipes"] for s in segs if s["stop"].startswith("remo")),
            "new_by_stop": conv, "new_in_second_half": sum(tail), "translations": sum(s["translations"] for s in segs),
            "ps_first": sum(s["ps_first"] for s in segs), "ps_not_lifted": merge_counts(s["ps_not_lifted"] for s in segs),
            "builds_over_5ms": sum(s["builds_over_5ms"] for s in segs), "build_ms": round(sum(s["build_ms"] for s in segs)),
            "stalls": sum(s["stalls"] for s in segs), "stall_ms": sum(s["stall_ms"] for s in segs),
            "stall_pipeline_ms": sum(s["stall_pipeline_ms"] for s in segs),
            "draw_failures": sum(s["draw_failures"] for s in segs),
            "failure_reasons": merge_counts(s["failure_reasons"] for s in segs),
            "deaths": sum(s["deaths"] for s in segs), "rewarps": sum(s["rewarps"] for s in segs), "retravels": sum(s["retravels"] for s in segs),
            "fatal": [f for s in segs for f in s["fatal"]], "hangs": [h for s in segs for h in s["hangs"]],
            "dump_stalls": sum(s["dump_stalls"] for s in segs), "problems": sorted(set(p for s in segs for p in s["problems"]))[:20],
            "missing": sorted(set(p for s in segs for p in s["missing"]))[:20], "msb": sorted(set(m for s in segs for m in s["msb"])),
            "remos": info.get("remos", {}), "movie_files": {s["stop"]: s["movie_files"] for s in segs if s["movie_files"]},
            "notes": [n for s in segs for n in s["notes"]]})
    missing = {}
    for sg in all_segs:
        for m in sg["missing"]:
            if not KNOWN_PROBES.search(m):
                missing.setdefault(m, set()).add(sg["item"])
    by_kind = {}
    for it in report_items:
        k = by_kind.setdefault(it["kind"], {"items": 0, "stops": 0, "new_recipes": 0, "new_at_stops": 0, "new_in_cutscenes": 0})
        k["items"] += 1
        k["stops"] += it["stops_visited"]
        k["new_recipes"] += it["new_recipes"]
        k["new_at_stops"] += it["new_at_stops"]
        k["new_in_cutscenes"] += it["new_in_cutscenes"]
    for sg in all_segs:
        sg.pop("new_keys", None)
    rep = {"schema": "bbhost-full-tour-report", "made": time.strftime("%Y-%m-%d %H:%M:%S"), "chunks": chunks, "items": report_items,
           "by_kind": by_kind, "missing_files": {m: sorted(v) for m, v in sorted(missing.items())}, "segments": all_segs}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n")
    if args.shots:
        write_shots(plan, runs, Path(args.shots), report_items, chunks)
    text = summary(rep)
    (out / "summary.txt").write_text(text)
    print(text)
    return 0


# Files the game asks for and does without on every run: a dummy tile set per
# area, the logo, two parts the equipment tables name, a .patch beside each
# archive (the update's, where there is one), area-wide event scripts the DLC
# areas do not have, per-lamp entry file lists, onav files. Not misses.
KNOWN_PROBES = re.compile(r"_9999\.tpf\.dcx|menu/logo\.tpf\.dcx|wp_a_6010|lg_a_9700|\.patch is not|/event/m\d\d\.eme(vd|ld)\.dcx|"
                          r"/entryfilelist/e\d+_00\.entryfilelist|/onav/")


def write_shots(plan, runs, dest, report_items, chunks):
    """Every stop's JPEG under dest/<item>/<stop>.jpg, and dest/index.txt: per
    item its chunk, the build, its stops and what each found."""
    dest.mkdir(parents=True, exist_ok=True)
    by_item = {it["id"]: it for it in report_items}
    lines = ["full tour shots: 1280x720 JPEGs of each stop's first heading (the camera behind the player, before the turn)", ""]
    for ch in plan["chunks"]:
        d = runs / ("c%02d" % ch["n"])
        done = d / "done.json"
        build = json.loads(done.read_text()).get("bbhost_sha256", "")[:12] if done.is_file() else "(not run)"
        for ident in ch["items"]:
            src = sorted((d / "shots").glob(safe(ident) + "-*.jpg")) if (d / "shots").is_dir() else []
            if not src:
                continue
            (dest / safe(ident)).mkdir(exist_ok=True)
            for f in src:
                shutil.copy2(f, dest / safe(ident) / (f.stem[len(safe(ident)) + 1:] + ".jpg"))
            it = by_item.get(ident, {})
            lines.append("%s: chunk %02d, bbhost %s, %d shots; new recipes %s (by stop %s), draw failures %s%s" % (
                ident, ch["n"], build, len(src), it.get("new_recipes"), it.get("new_by_stop"), it.get("draw_failures"),
                ", FATAL" if it.get("fatal") else ""))
    (dest / "index.txt").write_text("\n".join(lines) + "\n")


def merge_counts(dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def summary(rep):
    L = []
    L.append("by kind: " + "; ".join("%s %d items, %d stops, %d new recipes (%d at stops, %d in cutscenes)" % (
        k, v["items"], v["stops"], v["new_recipes"], v["new_at_stops"], v["new_in_cutscenes"]) for k, v in rep.get("by_kind", {}).items()))
    if rep.get("missing_files"):
        L.append("files the game could not open (not its usual probes): %d" % len(rep["missing_files"]))
        for m, where in list(rep["missing_files"].items())[:30]:
            L.append("  %s  (%s)" % (m, ", ".join(where)[:80]))
    L.append("full tour: %d chunks, %d items" % (len(rep["chunks"]), len(rep["items"])))
    for c in rep["chunks"]:
        L.append("  chunk %02d: %s, %s s, %d new recipes; %s" % (c["chunk"], "clean exit" if c["clean_exit"] else "NOT CLEAN (%s)" % c["stop_reason"],
                                                               c["elapsed_s"], c["new_recipes"], ", ".join(c["items"])[:200]))
    L.append("")
    L.append("%-30s %5s %6s %5s %5s %5s %5s %7s %6s %5s %6s %5s  %s" % ("item", "stops", "travel", "new", "arriv", "stops", "2nd/2", "remos",
                                                                      "xlate", ">5ms", "stalls", "fail", "notes"))
    for it in rep["items"]:
        notes = []
        if it["travel_failed"]:
            notes.append("TRAVEL FAILED")
        if it["fatal"]:
            notes.append("FATAL: " + it["fatal"][0][:80])
        if it.get("hangs"):
            notes.append("HANG: " + it["hangs"][0][:60])
        if it["deaths"]:
            notes.append("%d deaths" % it["deaths"])
        if it["missing"]:
            notes.append("missing %d" % len(it["missing"]))
        if it["problems"]:
            notes.append(it["problems"][0][:60])
        notes += it.get("notes", [])
        remos = it["remos"]
        rtxt = "%d/%d" % (sum(1 for r in remos.values() if r.get("started")), len(remos)) if remos else "-"
        L.append("%-30s %2d/%-2d %6s %5d %5d %5d %5d %7s %6d %5d %6d %5d  %s" % (
            it["id"][:30], it["stops_visited"], it["stops_planned"], "%.0f" % it["travel_s"] if it["travel_s"] else "-", it["new_recipes"],
            it["new_on_arrival"], it["new_at_stops"], it["new_in_second_half"], rtxt, it["translations"], it["builds_over_5ms"],
            it["stalls"], it["draw_failures"], "; ".join(notes)))
    return "\n".join(L) + "\n"


def item_totals(plan, runs):
    """item -> totals over its segments, from every chunk of runs."""
    out = {}
    for ch in plan["chunks"]:
        log = Path(runs) / ("c%02d" % ch["n"]) / "run.log"
        if not log.is_file():
            continue
        segs = read_chunk(log, ch["n"])[0]
        for sg in segs:
            t = out.setdefault(sg["item"], {"new_recipes": 0, "new_tess": 0, "translations": 0, "builds_over_5ms": 0, "build_ms": 0.0, "stalls": 0,
                                            "stall_ms": 0, "stall_pipeline_ms": 0, "ps_first": 0, "draw_failures": 0, "segments": 0, "ls_cp": 0,
                                            "stall_compute_ms": 0})
            t["segments"] += 1
            for k in ("new_recipes", "new_tess", "translations", "builds_over_5ms", "build_ms", "stalls", "stall_ms", "stall_pipeline_ms", "ps_first",
                      "draw_failures", "ls_cp", "stall_compute_ms"):
                t[k] += sg[k]
        text = log.read_text(errors="replace")
        m = re.search(r"pipeline manifest: (\d+) pipelines built from it \((\d+) relinked\) and (\d+) failed, ([\d.]+) s after it was read", text)
        if m:
            out.setdefault("(list)", {})["built"] = "%s built (%s relinked, %s failed) %s s after it was read" % m.groups()
        m = R_LS_EXIT.search(text)
        if m:
            out.setdefault("(list)", {})["ls"] = ("%s LS-pass compute pipelines built from recipes at the start (%s failed), %s taken by draws, "
                                                  "%s built by the command processor" % m.groups())
    return out


def do_compare(args):
    """Two runs of one plan side by side per item: the CP's pipeline work on
    first sight (builds over 5 ms, their ms, translations), new recipes, stalls."""
    plan = json.loads((Path(args.plan) / "plan.json").read_text())
    a, b = item_totals(plan, args.a), item_totals(plan, args.b)
    rows = ["%-26s %22s %22s %16s %16s %12s %20s" % ("item", "CP builds >5ms (ms) A|B", "translations A|B", "new recipes A|B", "stalls A|B",
                                                    "1st PS A|B", "pipeline ms in stalls A|B")]
    keys = [k for k in list(a) + [k for k in b if k not in a] if k != "(list)"]
    tot = {"a": [0, 0.0, 0, 0, 0], "b": [0, 0.0, 0, 0, 0]}
    for k in keys:
        x, y = a.get(k, {}), b.get(k, {})
        rows.append("%-26s %10s | %-10s %9s | %-10s %7s | %-7s %7s | %-7s %5s | %-5s %9s | %-9s" % (
            k[:26], "%d (%.0f)" % (x.get("builds_over_5ms", 0), x.get("build_ms", 0)), "%d (%.0f)" % (y.get("builds_over_5ms", 0), y.get("build_ms", 0)),
            x.get("translations", 0), y.get("translations", 0), x.get("new_recipes", 0), y.get("new_recipes", 0),
            "%d" % x.get("stalls", 0), "%d" % y.get("stalls", 0), x.get("ps_first", 0), y.get("ps_first", 0),
            x.get("stall_pipeline_ms", 0), y.get("stall_pipeline_ms", 0)))
        for side, d in (("a", x), ("b", y)):
            tot[side][0] += d.get("builds_over_5ms", 0)
            tot[side][1] += d.get("build_ms", 0)
            tot[side][2] += d.get("translations", 0)
            tot[side][3] += d.get("new_recipes", 0)
            tot[side][4] += d.get("stalls", 0)
    rows.append("%-26s %10s | %-10s %9d | %-10d %7d | %-7d %7d | %-7d" % ("total", "%d (%.0f)" % (tot["a"][0], tot["a"][1]), "%d (%.0f)" % (tot["b"][0], tot["b"][1]),
                                                                    tot["a"][2], tot["b"][2], tot["a"][3], tot["b"][3], tot["a"][4], tot["b"][4]))
    # The tessellated draws: new tessellated recipes, the host tessellator's
    # LS passes the command processor built at first sight, and the compute
    # pipelines' share of the stalls' pipeline ms (the LS passes are in it).
    rows.append("")
    rows.append("%-26s %18s %22s %26s" % ("tessellated", "new tess recipes A|B", "LS passes CP-built A|B", "compute ms in stalls A|B"))
    tt = {"a": [0, 0, 0], "b": [0, 0, 0]}
    for k in keys:
        x, y = a.get(k, {}), b.get(k, {})
        for side, d in (("a", x), ("b", y)):
            tt[side][0] += d.get("new_tess", 0)
            tt[side][1] += d.get("ls_cp", 0)
            tt[side][2] += d.get("stall_compute_ms", 0)
        if any(d.get(f, 0) for d in (x, y) for f in ("new_tess", "ls_cp", "stall_compute_ms")):
            rows.append("%-26s %8d | %-8d %10d | %-10d %12d | %-12d" % (k[:26], x.get("new_tess", 0), y.get("new_tess", 0), x.get("ls_cp", 0),
                                                                      y.get("ls_cp", 0), x.get("stall_compute_ms", 0), y.get("stall_compute_ms", 0)))
    rows.append("%-26s %8d | %-8d %10d | %-10d %12d | %-12d" % ("total", tt["a"][0], tt["b"][0], tt["a"][1], tt["b"][1], tt["a"][2], tt["b"][2]))
    for side, d in (("A", a), ("B", b)):
        if "(list)" in d:
            if "built" in d["(list)"]:
                rows.append("%s: the list: %s" % (side, d["(list)"]["built"]))
            if "ls" in d["(list)"]:
                rows.append("%s: %s" % (side, d["(list)"]["ls"]))
    text = "\n".join(rows) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


R_FELL = re.compile(PROBE + r"fell (\S+) [\d.]+ at")


def do_refall(args):
    """A plan of the stops that fell through the world in earlier chunks (the
    probe's "fell" lines), each map's again with its own travel: run with the
    gravity hold, they are seen where they stand."""
    plan = json.loads((Path(args.plan) / "plan.json").read_text())
    fell = {}
    for log in sorted(Path(args.runs).glob("c*/run.log")):
        for m in R_FELL.finditer(log.read_text(errors="replace")):
            item, _, stop = m.group(1).rpartition(":")
            fell.setdefault(item, set()).add(stop)
    items = []
    for it in plan["items"]:
        if it["id"] not in fell:
            continue
        stops = [st for st in it["stops"] if st["name"] in fell[it["id"]]]
        if not stops:
            continue
        r = dict(it)
        r["id"] = it["id"] + "~refall"
        r["stops"] = stops
        r["remos"] = []
        r.pop("restore", None)
        r["estimate_s"] = item_estimate(r)
        items.append(r)
    chunks, cur, cur_s = [], [], EST["start"]
    for it in items:
        if cur and cur_s + it["estimate_s"] > args.chunk_minutes * 60:
            chunks.append(cur)
            cur, cur_s = [], EST["start"]
        cur.append(it["id"])
        cur_s += it["estimate_s"]
    if cur:
        chunks.append(cur)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("chunk*.txt"):
        old.unlink()
    new = {"schema": "bbhost-full-tour-plan", "made": time.strftime("%Y-%m-%d %H:%M:%S"), "options": plan["options"], "items": items,
           "chunks": [{"n": k, "items": c, "estimate_s": round(EST["start"] + sum(items_by_id(items, i)["estimate_s"] for i in c)), "title_idle_s": 0}
                      for k, c in enumerate(chunks)]}
    (out / "plan.json").write_text(json.dumps(new, ensure_ascii=False, indent=1) + "\n")
    for ch in new["chunks"]:
        (out / ("chunk%02d.txt" % ch["n"])).write_text(chunk_tour(new, ch, None))
    print("%d stops that fell in %d maps, %d chunks, %.1f min estimated" % (sum(len(i["stops"]) for i in items), len(items), len(chunks),
                                                                         sum(c["estimate_s"] for c in new["chunks"]) / 60))
    return 0


# The order a player reaches things (owner, 2026-10-17: what a player reaches
# first builds first): the title and the start, Central Yharnam and the
# Hunter's Dream at the evening, then the story's maps with the world's time
# moving where the story moves it (night after Cathedral Ward's event 12400750,
# time 2 after Vicar Amelia, the Blood Moon after Rom), the late optional
# areas, The Old Hunters, then the chalice dungeons by depth.
STORY = ["(start)", "movies", "m24_01_00_00", "m21_00_00_00", "m24_00_00_00", "m23_00_00_00", "m22_00_00_00", "m24_01_00_00@t1",
         "m27_00_00_00", "m24_01_00_00@t2", "m24_00_00_00@t2", "m22_00_00_00@t2", "m23_00_00_00@t2", "m28_00_00_00@t2", "m21_01_00_00@t2",
         "m32_00_00_00", "m24_01_00_00@t3", "m24_00_00_00@t3", "m21_00_00_00@t3", "m28_00_00_00", "m28_00_00_00@t3", "m33_00_00_00",
         "m24_02_00_00", "m25_00_00_00", "m21_01_00_00", "m21_01_00_00@t3", "m26_00_00_00", "m21_00_00_00@9462", "m21_00_00_00@12101852",
         "m34_00_00_00", "m35_00_00_00", "m36_00_00_00"] + ["m29_%02d_90_00" % n for n in (10, 20, 21, 30, 31, 40, 42, 50, 52, 53)]
TWELVE_BLOCK = {"twelve:" + a.id: a.block for a in area_check.AREAS}
# plugins/boss_rush's bosses in its story order, by the map each fights in (its
# lamp's block; a chalice boss's fixed dungeon).
BOSS_BLOCK = [("Cleric Beast", "m24_01_00_00"), ("Father Gascoigne", "m24_01_00_00"), ("Blood-starved Beast", "m23_00_00_00"),
              ("Vicar Amelia", "m24_00_00_00"), ("The Witch of Hemwick", "m22_00_00_00"), ("Darkbeast Paarl", "m23_00_00_00"),
              ("Shadow of Yharnam", "m27_00_00_00"), ("Rom, the Vacuous Spider", "m32_00_00_00"), ("The One Reborn", "m28_00_00_00"),
              ("Martyr Logarius", "m25_00_00_00"), ("Amygdala", "m33_00_00_00"), ("Celestial Emissary", "m24_02_00_00"),
              ("Ebrietas, Daughter of the Cosmos", "m24_02_00_00"), ("Mergo's Wet Nurse", "m26_00_00_00"), ("Ludwig, the Holy Blade", "m34_00_00_00"),
              ("Living Failures", "m35_00_00_00"), ("Lady Maria of the Astral Clocktower", "m35_00_00_00"),
              ("Laurence, the First Vicar", "m34_00_00_00"), ("Orphan of Kos", "m36_00_00_00"), ("Undead Giant", "m29_10_90_00"),
              ("Merciless Watchers", "m29_10_90_00"), ("Watchdog of the Old Lords", "m29_10_90_00"), ("Beast-possessed Soul", "m29_20_90_00"),
              ("Keeper of the Old Lords", "m29_20_90_00"), ("Pthumerian Descendant", "m29_20_90_00"), ("Bloodletting Beast", "m29_30_90_00"),
              ("Pthumerian Elder", "m29_31_90_00"), ("Abhorrent Beast", "m29_42_90_00"), ("Yharnam, Pthumerian Queen", "m29_50_90_00")]
BOSS_MAP = dict(BOSS_BLOCK)


def story_rank(item):
    """Where an item's place stands in STORY (a map's refall, boss or denser
    items as the map; the twelve places as their maps)."""
    base = item.split("~")[0]
    base = TWELVE_BLOCK.get(base, base)
    if base.startswith("boss:"):
        base = base.split(":", 2)[1]
    for k, v in enumerate(STORY):
        if base == v:
            return k
    m = re.match(r"(m\d\d_\d\d_00_00)", base)
    if m and m.group(1) in STORY:
        return STORY.index(m.group(1))
    return len(STORY)


def recipe_ranks(run_dirs):
    """content key -> (story rank, position in the logs) of the item each
    recipe was first recorded in, over the run folders' chunk logs in order."""
    ranks, seq = {}, 0
    for runs in run_dirs:
        for log in sorted(Path(runs).glob("c*/run.log")):
            for sg in read_chunk(log, 0)[0]:
                for key in sg["new_keys"]:
                    seq += 1
                    k = int(key, 16)
                    if k not in ranks:
                        ranks[k] = (story_rank(sg["item"]), seq)
    return ranks


R_CS_HERE = re.compile(r"\] gpu: compute pipelines built at the dispatch instead: .*?;((?: \S+ \([^)]*\);)*)")


def built_at_dispatch(run_dirs):
    """The compute programs the chunks' dispatches had to build themselves
    (gpu.cpp's exit line, its notes): the ones nothing else precompiles."""
    names = set()
    for runs in run_dirs:
        for log in sorted(Path(runs).glob("c*/run.log")):
            for m in R_CS_HERE.finditer(log.read_text(errors="replace")):
                names.update(re.findall(r" (\S+) \(", m.group(1)))
    return names


def do_list(args):
    runs = [Path(r) for r in args.runs]
    union = runs[-1] / "union-manifest.bin"
    run, origin, recipes = pipeline_list.read_file(union)
    # The compute records the dispatches had to build (a union recorded before
    # recipe_note_compute kept to those holds the ones GX's creation compiles
    # too, which only crowd the start).
    here = built_at_dispatch(runs)
    computes = [rc for rc in recipes if rc.compute]
    if here and computes:
        recipes = [rc for rc in recipes if not rc.compute or rc.name in here]
    progs = pipeline_list.bundle_programs(args.app0)
    if not progs:
        print("full_tour: no shader bundles under %s" % args.app0, file=sys.stderr)
        return 2
    for name, code in pipeline_list.eboot_programs(args.eboot).items():
        progs.setdefault(name, code)
    ranks = recipe_ranks(runs) if args.order == "story" else {}
    # The same inputs under two keys (a start re-keys a recipe whose build now
    # makes another key, and the union keeps both): one record each, the one
    # a log names first.
    seen, unique, dupes = {}, [], 0
    for rc in sorted(recipes, key=lambda r: (r.key not in ranks, ranks.get(r.key, (0, 0)))):
        content = rc.body()[24:]  # past flags, key, uses, last run, order
        if content in seen:
            dupes += 1
            continue
        seen[content] = rc
        unique.append(rc)
    # A geometry-shader record names its copy shader by its GS's container
    # (pipeline_list.copy_programs); without them every such record is left out.
    out, left, plain, patched = pipeline_list.make_list(unique, progs, bool(args.eboot), not args.no_fetch_words,
                                                        pipeline_list.copy_programs(args.app0))
    if dupes:
        left["the same inputs as another record (re-keyed at a start)"] += dupes
    unknown = 0
    if args.order == "story":
        far = (len(STORY) + 1, 0)
        def where(rc):  # the compute records first: few, and mostly the title's
            return (-1, rc.order) if rc.compute else ranks.get(rc.key, far)
        unknown = sum(1 for rc in out if not rc.compute and rc.key not in ranks)
        out.sort(key=where)
        for k, rc in enumerate(out):
            rc.order = k  # bbhost builds a list's records in this order (pipeline_manifest_load)
    pipeline_list.write_file(args.out, 0, origin, out)
    size = Path(args.out).stat().st_size
    fetch_bytes = sum(len(rc.fetch or b"") + len(rc.ls_fetch or b"") + sum(len(e["fetch"] or b"") for e in rc.lsps) for rc in out if not rc.compute)
    computes = sum(1 for rc in out if rc.compute)
    print("%s: %d of %d pipelines, %d of them compute (%.1f KiB; programs named, %d as the game's files have them, %d as bbhost patches them; "
          "%.1f KiB of fetch shaders' words); order %s%s" % (args.out, len(out), len(recipes), computes, size / 1024.0, plain, patched,
                                                           fetch_bytes / 1024.0, args.order,
                                                           " (%d not found in the logs, last)" % unknown if unknown else ""))
    for why, n in left.most_common():
        print("  left out: %d with %s" % (n, why))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inventory")
    p.add_argument("--app0", required=True)
    p.add_argument("--out")
    p = sub.add_parser("plan")
    p.add_argument("--app0", required=True)
    p.add_argument("--navmesh-tools", type=Path, default=os.environ.get("MAPVAL_NAVMESH_TOOLS") or None,
                   help="the folder holding the worldmap navmesh module (navmesh.py, msb.py); default $MAPVAL_NAVMESH_TOOLS. "
                        "Only plan reads it, for the blocks' stops")
    p.add_argument("--out", required=True)
    p.add_argument("--blocks", help="only these blocks (comma list)")
    p.add_argument("--radius", type=float, default=60.0)
    p.add_argument("--max-stops", type=int, default=40)
    p.add_argument("--min-stops", type=int, default=6)
    p.add_argument("--state-min-stops", type=int, default=3)
    p.add_argument("--remo-limit", type=int, default=99)
    p.add_argument("--only", help="keep only these item ids (a test chunk)")
    p.add_argument("--state-radius", type=float, default=120.0)
    p.add_argument("--state-max-stops", type=int, default=12)
    p.add_argument("--chalice-radius", type=float, default=40.0)
    p.add_argument("--chalice-max-stops", type=int, default=24)
    p.add_argument("--anchor", type=float, default=25.0)
    p.add_argument("--lift", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seed-time", type=int, default=0, choices=(0, 1, 2, 3),
                   help="the test save's world time (flags 9800-9802; the probe logs it as 'flags 9800+3')")
    p.add_argument("--chunk-minutes", type=float, default=14.0)
    p.add_argument("--stop-hold", type=float, default=1.0)
    p.add_argument("--headings", type=int, default=6)
    p.add_argument("--heading-s", type=float, default=0.5)
    p.add_argument("--remo-skip", type=float, default=40.0)
    p.add_argument("--remo-max", type=float, default=120.0)
    p.add_argument("--movie-ids", default="", help="RequestPlayMovie ids to try in the world (0-5 played nothing)")
    p.add_argument("--movie-skip", type=float, default=60.0)
    p.add_argument("--movie-max", type=float, default=75.0)
    p.add_argument("--title-idle", type=float, default=240.0,
                   help="seconds the movies chunk waits at the title: its attract movie came within 240 s, not within 80 s")
    p.add_argument("--no-twelve", action="store_true")
    p.add_argument("--bosses", action="store_true", help="plugins/boss_rush's 29 bosses, each fought boss_seconds before its test kill")
    p.add_argument("--boss-rounds", type=int, default=10)
    p.add_argument("--boss-seconds", type=int, default=45)
    p.add_argument("--no-states", action="store_true")
    p.add_argument("--no-chalice", action="store_true")
    p.add_argument("--no-movies", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--bbhost", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--live", required=True, help="where the run's data and frames live (tmpfs)")
    p.add_argument("--p2p-port", type=int, default=9801)
    p.add_argument("--numa-node", type=int, default=0)
    p.add_argument("--cpus", help="run on these CPUs only (numactl --physcpubind), a smaller machine's stand-in")
    p.add_argument("--glcache", help="the NVIDIA shader disk cache folder for the chain")
    p.add_argument("--budget", type=float, default=1140.0, help="seconds the run may take at most (a game.lock hold)")
    p.add_argument("--list", help="a pipeline list for the run (BBHOST_PIPELINE_LIST)")
    p.add_argument("--cold", action="store_true", help="no caches at all (a first-ever start)")
    p.add_argument("--no-chain", action="store_true", help="leave the chain and the union as they are")
    p.add_argument("--again", action="store_true")
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--mark-done", action="store_true")
    p.add_argument("--keep-live", action="store_true")
    p.add_argument("--env", action="append")
    p = sub.add_parser("report")
    p.add_argument("--plan", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--out")
    p.add_argument("--shots", help="collect the stops' JPEGs here with an index.txt")
    p = sub.add_parser("compare")
    p.add_argument("--plan", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out")
    p = sub.add_parser("refall")
    p.add_argument("--plan", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chunk-minutes", type=float, default=17.5)
    p = sub.add_parser("list")
    p.add_argument("--runs", required=True, nargs="+", help="run folders in tour order; the last one's union is listed")
    p.add_argument("--app0", required=True)
    p.add_argument("--eboot")
    p.add_argument("--out", required=True)
    p.add_argument("--order", choices=("story", "recorded"), default="story")
    p.add_argument("--no-fetch-words", action="store_true")
    args = ap.parse_args(argv)
    return {"inventory": do_inventory, "plan": do_plan, "run": do_run, "report": do_report, "compare": do_compare, "refall": do_refall, "list": do_list}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
