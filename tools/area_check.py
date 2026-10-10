#!/usr/bin/env python3
"""The area check: one build visits the same places in the game, stands still
in each and measures the same things there; two builds' visits are compared.

    tools/area_check.py run --bbhost BUILD/bbhost --config TEST.toml --seed SAVE --out DIR
    tools/area_check.py ab --a BUILD_A/bbhost --b BUILD_B/bbhost --config TEST.toml --seed SAVE --out DIR
    tools/area_check.py compare --a RUN [RUN...] --b RUN [RUN...] --out DIR
    tools/area_check.py report RUN          the report again from a run's logs
    tools/area_check.py areas --app0 DIR    the places, checked against the game's data

A run plays a copy of SAVE (the saves of a data folder: a character in the
world, whose Continue loads it) headless, pinned to one NUMA node, with the
map-validation probe (tools/mapval_plugin.c, BBHOST_TEST_WARP=1) as its only
driver. For each place in turn the probe travels to a lamp the way the
Hunter's Dream headstones do, moves the player to the place's view, switches
off the ordinary enemies within 40 m, puts the camera behind the player and
then holds still:

  settle      --settle seconds (15): streaming and the first compiles
  window      --measure seconds (20): what the place costs
  frames      the displayed frame twice, 2 s apart, for a visual A/B

The places (`areas`) are visited in the order given. An arena's view stands
just outside the boss's fog wall, facing in, as plugins/boss_rush starts its
rounds; two arenas are behind doors, which the travel there opens with the
boss rush's flags (they stay open for the rest of the run). Two places are
views with a standpoint of their own, two of the heaviest scenes the game
draws: Mergo's Loft from the rocks above Mensis's north path, and the
staircase up to the Nightmare Grand Cathedral. The Old Hunters' places need
its patch on (bbhost's default; the run's config says so when they are
listed).

That is the perf pass. The capture pass visits the same places and, after the
settle, lets BBHOST_CAPTURE_DRAW=*ps take the first draw of up to --captures
pixel shaders (6) not captured before; tools/lift_verify.py then replays each
with its lift and with its translation and compares the outputs byte for byte.
Captures stall the GPU, which is why they get a session of their own.

Per place the report gives, for the window: fps and the exact frame-time
percentiles (BBHOST_FRAME_DETAIL=1: every flip interval), the GPU busy meter
(ms of GPU work a second; not BBHOST_GPU_PROFILE, which records in place), the
main loop's work, the draws by how their pixel shader ran (lifted, translated,
the page-table fallback's translation, none) and the draw failures by reason;
for the arrival (the travel and the settle): the loads, pipeline compiles over
30 ms, flip stalls over 40 ms, main-loop frames over 33 ms, and the pixel
shaders met for the first time there, lifted or not and why; the frames; and
the lift results. report.json holds all of it, summary.txt the same to read.

`ab` runs A and B alternately for --rounds rounds (2: the second round of
each build is the noise floor) - perf passes every round, the capture pass in
the first - and compares: a difference counts only where it is larger than the
same build's own spread, and a frame only where it differs from the other
build's more than from its own repeat (lantern flicker, fog and cloth move
5-30% of the pixels between identical runs).

--dry-run does everything but start the game: it checks the configuration,
copies the seed (and checks the copy), checks the places' lamps and views
against the game's data, builds the probe and writes the plan, the tours, the
configs and the command lines. Nothing touches the GPU.

The run's own config: the game paths from --config (or --app0/--eboot), its
data the seed's copy, offline (NP signed out; online host 127.0.0.1) with its
own P2P port, its own config folder (BBHOST_CONFIG_DIR) and settings file, the
sample plugin and visitor plugins off. Each run starts from the same save.
Exit status: 0 when every place was measured and no lift differed, 1 when
something did not, 2 for bad arguments.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import world_baseline  # noqa: E402  (launch_argv, file_inventory, digest)

SCHEMA = "bbhost-area-check"
EBOOT_109_SHA256 = "941f887a562aae054fac35af8cc8f27cf075f3d4cc2e029fb5ae2a663aaa5ae7"
# The ordinary enemies' team (NpcParam teamType 23) within this many metres of
# the view are switched off at arrival and again before the window.
QUIET_RADIUS = 40.0
# How far behind a boss's fog wall the view stands: the boss rush plugin's
# stand-off, which it checked in each arena (plugins/boss_rush).
FOG_STANDOFF = 1.0
# A standpoint of a place's own is checked against the layout: something it
# puts on the ground (an object, a character, a player start) within this many
# metres. A lookout over a vista can be 40 m from the nearest; a standpoint in
# the block's own frame instead of the MSB's is hundreds of metres out.
STAND_REACH = 50.0
# The way through the title: the menus' confirm button, asked of the host
# (BBHOST_TEST_REQUESTS "menutap confirm") every TITLE_TAP_EVERY_S seconds from
# TITLE_TAP_FIRST_S until the world's first frame. The host presses it only
# while one of the game's menus has the input, so a request that arrives as the
# world loads presses nothing; and it reaches the title whenever the title
# comes - a new build's first start compiles its shader caches before it.
TITLE_TAP_FIRST_S = 5.0
TITLE_TAP_EVERY_S = 2.0


# --- the places ---------------------------------------------------------------

class Area:
    """A place: a lamp (ReturnPointParam row) the travel takes, and where to
    stand after it - where the travel puts the player, just outside a boss's
    fog wall facing the arena (`fog`: the wall's MSB position and rotation, as
    plugins/boss_rush keeps them), or a standpoint of its own (`stand`: an MSB
    position on the ground and the facing in degrees, for a view no wall
    gives). `flags`: event flags turned on before the travel's load reads
    them - the doors in front of two arenas, as the boss rush opens them."""

    def __init__(self, ident, name, block, return_point, fog=None, dlc=False, flags=(), stand=None):
        self.id, self.name, self.block, self.return_point, self.fog, self.dlc = ident, name, block, return_point, fog, dlc
        self.flags = tuple(flags)
        self.stand = tuple(stand) if stand else None

    def block_hex(self):
        a, b, c, d = (int(x) for x in self.block[1:].split("_"))
        return "%08x" % (a << 24 | b << 16 | c << 8 | d)

    def view(self):
        """(x, y, z, yaw in radians) in the MSB's frame, or None. A thing at yaw
        t faces (-sin t, -cos t): the wall faces into its arena, so the view
        stands behind it, facing as it does. A standpoint is the view as it is."""
        if self.stand:
            x, y, z, deg = self.stand
            return (x, y, z, math.radians(deg))
        if not self.fog:
            return None
        x, y, z, deg = self.fog
        t = math.radians(deg)
        return (x + math.sin(t) * FOG_STANDOFF, y + 0.05, z + math.cos(t) * FOG_STANDOFF, t)

    def as_dict(self):
        v = self.view()
        return {"id": self.id, "name": self.name, "block": self.block, "return_point": self.return_point, "dlc": self.dlc,
                "fog": list(self.fog) if self.fog else None, "stand": list(self.stand) if self.stand else None,
                "view": [round(c, 4) for c in v] if v else None, "flags": list(self.flags)}


AREAS = (
    Area("hunters-dream", "Hunter's Dream", "m21_00_00_00", 2102950),
    Area("central-yharnam", "Central Yharnam", "m24_01_00_00", 2412951),
    Area("cathedral-ward", "Cathedral Ward", "m24_00_00_00", 2402950),
    Area("old-yharnam", "Old Yharnam", "m23_00_00_00", 2302950),
    Area("forbidden-woods", "Forbidden Woods (tessellated meshes)", "m27_00_00_00", 2702950),
    # The lake's door (m32_00's 13200040 and its opening's 13200120) open.
    Area("moonside-lake", "Byrgenwerth: Moonside Lake, Rom's arena", "m32_00_00_00", 3202950,
         fog=(-452.874, -175.03, 392.65, 70.0), flags=(13200040, 13200120)),
    # A lookout: the top of the rock slope off the north path (navmesh piece
    # h000003), facing Mergo's Loft's west front and the brain's lit window.
    # The brain's gaze (the evil eye, m26_00's 2600100/2600101) frenzies the
    # player here; the probe keeps the HP full.
    Area("nightmare-of-mensis", "Nightmare of Mensis: Mergo's Loft from the north path's rocks", "m26_00_00_00", 2602950,
         stand=(15.0, 1020.2, 95.0, -85.0)),
    Area("fishing-hamlet", "Fishing Hamlet: the coast, the Orphan of Kos", "m36_00_00_00", 3602950,
         fog=(-700.2, 1580.5, -906.2, 0.0), dlc=True),
    Area("astral-clocktower", "Astral Clocktower: Lady Maria's arena", "m35_00_00_00", 3502950,
         fog=(-452.0, 1595.51, -824.41, 90.0), dlc=True),
    # The balcony's door (m35_00's 13501200 and 13504220) open.
    Area("lumenwood-garden", "Research Hall: Lumenwood Garden, the Living Failures", "m35_00_00_00", 3502950,
         fog=(-378.54, 1592.9, -824.55, 90.0), dlc=True, flags=(13501200, 13504220)),
    Area("hunters-nightmare", "Hunter's Nightmare: Ludwig's arena", "m34_00_00_00", 3402950,
         fog=(-407.93, 1503.8, -716.45, 0.0), dlc=True),
    # The great staircase up to the Nightmare Grand Cathedral (Laurence's
    # arena), on its upper flight 24 m below the fog wall, facing up. The
    # executioner on the landing below (m34_00's event 13405103) wakes when
    # the player enters region 3402300 - from 26 m below the wall down - or
    # comes within 7 m of it.
    Area("laurence-staircase", "Hunter's Nightmare: the staircase up to the Nightmare Grand Cathedral", "m34_00_00_00", 3402950,
         stand=(-462.9, 1530.7, -313.47, -150.0), dlc=True),
)
AREA_BY_ID = {a.id: a for a in AREAS}


def pick_areas(spec):
    """'all', or a comma list of ids (or unambiguous prefixes), in the given order."""
    if not spec or spec == "all":
        return list(AREAS)
    out = []
    for word in (w.strip() for w in spec.split(",")):
        if not word:
            continue
        hits = [a for a in AREAS if a.id == word] or [a for a in AREAS if a.id.startswith(word)]
        if len(hits) != 1:
            raise ValueError("area %r: %s" % (word, "ambiguous" if hits else "unknown") + " (have %s)" % ", ".join(a.id for a in AREAS))
        if hits[0] in out:
            raise ValueError("area %s is listed twice" % hits[0].id)
        out.append(hits[0])
    return out


# --- the plan -------------------------------------------------------------------

DEFAULTS = {"settle": 15.0, "measure": 20.0, "dumps": 2, "dump_gap": 2.0, "quiet": QUIET_RADIUS, "start": 8.0, "fps": 60,
            "captures": 6, "capture_settle": 10.0, "capture_window": 30.0}


def tour(areas, opts, kind):
    """The probe's tour (tools/mapval_plugin.c steps) for one pass: 'perf' or 'capture'.

    Each place starts with its travel - the step a failed travel skips to the
    next of - so the log splits into places at the travel lines."""
    lines = ["# area_check %s pass: %d places" % (kind, len(areas))]
    for a in areas:
        lines.append("# %s: %s" % (a.id, a.name))
        lines.append(" ".join(["travel %d %s %s" % (a.return_point, a.block_hex(), a.id)] + ["%d" % f for f in a.flags]))
        v = a.view()
        if v:
            lines.append("warp %s:view %.3f %.3f %.3f %.4f 1.0" % (a.id, *v))
        if opts["quiet"] > 0:
            lines.append("quiet %.0f" % opts["quiet"])
        lines.append("camera")
        lines.append("mark %s:arrived" % a.id)
        if kind == "perf":
            lines.append("hold %.1f" % opts["settle"])
            if opts["quiet"] > 0:
                lines.append("quiet %.0f" % opts["quiet"])
            lines.append("mark %s:measure" % a.id)
            lines.append("hold %.1f" % opts["measure"])
            lines.append("mark %s:measured" % a.id)
            for k in range(opts["dumps"]):
                if k:
                    lines.append("hold %.1f" % opts["dump_gap"])
                lines.append("dump %s-%s" % (a.id, chr(ord("a") + k)))
            lines.append("hold 1.0")
        else:
            lines.append("hold %.1f" % opts["capture_settle"])
            lines.append("capture %d %s" % (opts["captures"], a.id))
            lines.append("mark %s:capture" % a.id)
            lines.append("hold %.1f" % opts["capture_window"])
            lines.append("capture 0")
        lines.append("mark %s:end" % a.id)
    lines.append("wait 2")
    return "\n".join(lines) + "\n"


def toml_str(s):
    """A quoted value as bbhost's config reader takes it: everything between
    the quotes as it is, no escapes - so no quote, backslash or line break."""
    s = str(s)
    if any(c in s for c in '"\\\n'):
        raise ValueError("a path the run's config cannot hold (a quote, backslash or line break): %r" % s)
    return '"%s"' % s


def run_toml(app0, eboot, data, p2p_port, width, height, old_hunters=False):
    lines = [
        "# tools/area_check.py: this run's own configuration - offline, on a copy of the seed.",
        "[paths]", "app0 = %s" % toml_str(app0), "data = %s" % toml_str(data), "eboot = %s" % toml_str(eboot), "",
        "[online]", 'host = "127.0.0.1"', "p2p_port = %d" % p2p_port, "",
        "# The window and what the game renders at (the Resolution setting).",
        "[video]", "width = %d" % width, "height = %d" % height, "resolution = %s" % toml_str("%dx%d" % (width, height)), "",
        "[startup]", "skip_intro = true", "",
        "# The developer sample and visitor plugins stay out of the measurement.",
        "[plugins]", "hello = false", "join_modded_worlds = false", ""]
    if old_hunters:
        lines += ["# The Old Hunters' places are on the list.", "[patches]", "old-hunters = true", ""]
    return "\n".join(lines)


def pass_settings(pass_dir, areas, opts, kind):
    """The game's BBHOST_* settings for one pass."""
    s = {
        "BBHOST_HEADLESS": "1", "BBHOST_NP_SIGNED_OUT": "1", "BBHOST_SETUP_WINDOW": "0", "BBHOST_SKIP_INTRO": "1",
        "BBHOST_CONFIG_DIR": str(pass_dir / "cfg"), "BBHOST_OPTIONS_PATH": str(pass_dir / "cfg" / "bbhost-options.toml"),
        "BBHOST_PIPELINE_CACHE": "1" if opts["pipeline_cache"] else "0",
        "BBHOST_GAME_FPS": str(opts["fps"]), "BBHOST_FRAME_STATS": "1", "BBHOST_FRAME_DETAIL": "1", "BBHOST_STALL_MS": "40",
        "BBHOST_TEST_WARP": "1", "BBHOST_TEST_REQUESTS": str(pass_dir / "requests"),
        "BBHOST_MAPVAL_TOUR": str(pass_dir / "tour.txt"), "BBHOST_MAPVAL_START": "%g" % opts["start"],
        "BBHOST_MAPVAL_DUMPS": str(pass_dir / "frames"), "BBHOST_MAPVAL_CAPTURES": str(pass_dir / "captures"),
    }
    if kind == "capture":
        # The set cache makes the params block a dynamic (or pushed) uniform
        # buffer, which changes the translated shader and is not in a
        # capture's manifest: gcnlift would rebuild the other variant and
        # every lift check read "not comparable". The perf pass keeps it.
        s.update({"BBHOST_CAPTURE_DRAW": "*ps", "BBHOST_CAPTURE_ARMED": "1",
                  "BBHOST_CAPTURE_COUNT": str(max(1, opts["captures"]) * len(areas)),
                  "BBHOST_CAPTURE_DIR": str(pass_dir / "captures"), "BBHOST_SET_CACHE": "0"})
    for item in opts.get("env", []):
        name, sep, value = item.partition("=")
        if not sep or not name.startswith("BBHOST_") or name in s:
            raise ValueError("--env takes BBHOST_NAME=VALUE and must not override the run's own settings: %s" % item)
        s[name] = value
    return s


def pass_budget(areas, opts, kind):
    """Seconds the pass may take: the start, then per place the probe's own
    limits (a travel gives up after 150 s, a warp after ten 3 s tries) and the
    holds."""
    per = 150 + 35 + 10
    per += (opts["settle"] + opts["measure"] + opts["dumps"] * (opts["dump_gap"] + 2) + 3 if kind == "perf"
            else opts["capture_settle"] + opts["capture_window"] + 60)
    return 300 + opts["start"] + per * len(areas) + 60


def estimate(areas, opts, kind):
    """Seconds a pass usually takes (a travel ~25 s)."""
    per = 25 + (3 if any(a.view() for a in areas) else 0)
    per += (opts["settle"] + opts["measure"] + opts["dumps"] * opts["dump_gap"] + 1 if kind == "perf"
            else opts["capture_settle"] + opts["capture_window"])
    return 60 + opts["start"] + per * len(areas)


# --- the log --------------------------------------------------------------------

FRAMES = re.compile(r"\] frames: ([\d.]+) s: (\d+) flips \(([\d.]+)/s\), displayed ([\d.]+)/s, interval avg ([\d.]+) "
                    r"p95 ([\d.]+) max ([\d.]+) ms, (\d+) over 16\.7, (\d+) over 33\.3; cpu (\d+)%")
WORK = re.compile(r"main loop work avg ([\d.]+) p95 ([\d.]+) max ([\d.]+) ms")
GPU_BUSY = re.compile(r"gpu busy ([\d.]+) ms/s")
DETAIL = re.compile(r"\] frame detail: draws (\d+): pixel shader lifted (\d+), translated (\d+), fallback (\d+), none (\d+); "
                    r"draw failures (\d+)(?: \((.*)\))?")
INTERVALS = re.compile(r"\] frame intervals ms:((?: [\d.]+)+)\s*$")
# The probe's lines, as bbhost logs a plugin's ("plugin: <line>", host/plugins.cpp).
PROBE = r"\] (?:plugin: )?mapval "
TRAVEL = re.compile(PROBE + r"step \d+ travel (\S+) ([\d.]+) return point (\d+) block ([0-9a-f]+) -> (\w+)")
TRAVEL_DONE = re.compile(PROBE + r"travel-done (\S+) ([\d.]+) ([0-9a-f]+) \S+ \S+ \S+ after ([\d.]+) s")
TRAVEL_FAILED = re.compile(PROBE + r"travel-failed (\S+) ([\d.]+) in ([0-9a-f]+) after ([\d.]+) s")
MARK = re.compile(PROBE + r"mark (\S+?):(\w+) ([\d.]+) ([0-9a-f]+) (\S+) (\S+) (\S+) yaw (\S+)")
AT = re.compile(PROBE + r"at (\S+?):view ([\d.]+) ([0-9a-f]+) (\S+) (\S+) (\S+)")
REWARP = re.compile(PROBE + r"rewarp ")
RETRAVEL = re.compile(PROBE + r"retravel ")
QUIET = re.compile(PROBE + r"quiet [\d.]+ (\d+) enemies within")
CAMERA = re.compile(PROBE + r"camera [\d.]+ (\w+)")
DUMP_DONE = re.compile(r"\] test: frame dump at flip (\d+) to (\S+?)( failed)?\s*$")
CAPTURE_WROTE = re.compile(r"\] capture: wrote (\S+)/manifest\.json \((\S+), flip (\d+)")
CAPTURE_REJECTED = re.compile(r"\] capture: (\S+) draw \d+ \(flip \d+\) not captured: (.*?)\s*$")
LOADING = re.compile(r"\] loading: ([\d.]+) s \(flips")
SLOW_PIPELINE = re.compile(r"\] render: slow pipeline (?!library )\S+.*vkCreateGraphicsPipelines (\d+) ms")
SLOW_LIBRARY = re.compile(r"\] render: slow pipeline library .*?: (\d+) ms")
STALL = re.compile(r"\] stall: flip #\d+ came (\d+) ms after")
STALL_PIPELINE = re.compile(r"pipeline ms: translate (\d+), modules (\d+), create (\d+)")
LONG_FRAME = re.compile(r"\] main loop: a ([\d.]+) ms frame")
PS_LIFTED = re.compile(r"\] render: PS (\w+) lifted \(")
PS_NOT_LIFTED = re.compile(r"\] render: PS (\w+) not lifted, translated shader kept: (.*?)\s*$")
PIPELINE_FAILED = re.compile(r"\] render: pipeline (\S+) failed to build")
WORLD = re.compile(r"\] world: the first in-game frame, flip (\d+), ([\d.]+) s after start")
TOUR_START = re.compile(PROBE + r"tour-start ")
PROBE_LINE = re.compile(PROBE)
TOUR_DONE = re.compile(PROBE + r"tour-done ")
VERSION = re.compile(r"\] bbhost (v\S+) \(")
RENDERING = re.compile(r"\] resolution: display buffers \d+x\d+, rendering (\d+)x(\d+)")
FATAL = re.compile(r"SIGSEGV pc=|dumped core|Segmentation fault|the device is lost|VK_ERROR_DEVICE_LOST|\] DL_PANIC ")
PROBLEM = re.compile(r"HLE stub #|descriptor set allocation failed|wait timed out|capture: .*abandoned")


def reason_key(why):
    """A lift rejection without the instruction offset it names."""
    return re.sub(r"^(?:0x[0-9a-fA-F]+|\d+): ", "", why.strip())[:120]


def parse_reasons(text):
    """'pixel shader 2, vertex shader 1' -> {'pixel shader': 2, 'vertex shader': 1}"""
    out = {}
    for part in (text or "").split(", "):
        m = re.match(r"(.+) (\d+)$", part.strip())
        if m:
            out[m.group(1)] = out.get(m.group(1), 0) + int(m.group(2))
    return out


class Log:
    """A pass's log as events by line index: the per-second records (`frames:`
    and the detail and interval lines after it) and everything a place is
    measured by."""

    def __init__(self, lines):
        self.lines = lines
        self.seconds = []     # (index, record)
        self.events = []      # (index, kind, data)
        self.version = None
        self.world = None
        self.rendering = None  # [w, h]: the last size the game rendered at
        self.tour_done = False
        self.fatal = []
        for i, line in enumerate(lines):
            self._feed(i, line.rstrip("\n"))

    def _ev(self, i, kind, **data):
        self.events.append((i, kind, data))

    def _feed(self, i, line):
        if "] frame" in line:
            m = FRAMES.search(line)
            if m:
                w = WORK.search(line)
                g = GPU_BUSY.search(line)
                self.seconds.append((i, {
                    "secs": float(m.group(1)), "flips": int(m.group(2)), "fps": float(m.group(3)),
                    "displayed": float(m.group(4)), "avg_ms": float(m.group(5)), "p95_ms": float(m.group(6)),
                    "max_ms": float(m.group(7)), "over_16_7": int(m.group(8)), "over_33_3": int(m.group(9)),
                    "cpu_pct": int(m.group(10)),
                    "work": [float(w.group(k)) for k in (1, 2, 3)] if w else None,
                    "gpu_busy": float(g.group(1)) if g else None, "detail": None, "intervals": []}))
                return
            m = DETAIL.search(line)
            if m and self.seconds:
                self.seconds[-1][1]["detail"] = {
                    "draws": int(m.group(1)), "lifted": int(m.group(2)), "translated": int(m.group(3)),
                    "fallback": int(m.group(4)), "none": int(m.group(5)), "failures": int(m.group(6)),
                    "reasons": parse_reasons(m.group(7))}
                return
            m = INTERVALS.search(line)
            if m and self.seconds:
                self.seconds[-1][1]["intervals"].extend(float(v) for v in m.group(1).split())
                return
        if "mapval " in line and PROBE_LINE.search(line):
            for kind, rx in (("travel", TRAVEL), ("travel_done", TRAVEL_DONE), ("travel_failed", TRAVEL_FAILED),
                             ("mark", MARK), ("at", AT), ("quiet", QUIET), ("camera", CAMERA)):
                m = rx.search(line)
                if m:
                    if kind == "travel":
                        self._ev(i, kind, area=m.group(1), t=float(m.group(2)), return_point=int(m.group(3)),
                                 block=m.group(4), queued=m.group(5) == "queued")
                    elif kind == "travel_done":
                        self._ev(i, kind, area=m.group(1), t=float(m.group(2)), block=m.group(3), seconds=float(m.group(4)))
                    elif kind == "travel_failed":
                        self._ev(i, kind, area=m.group(1), t=float(m.group(2)), block=m.group(3), seconds=float(m.group(4)))
                    elif kind == "mark":
                        self._ev(i, kind, area=m.group(1), name=m.group(2), t=float(m.group(3)), block=m.group(4),
                                 pos=[float(m.group(k)) for k in (5, 6, 7)], yaw=float(m.group(8)))
                    elif kind == "at":
                        self._ev(i, kind, area=m.group(1), t=float(m.group(2)), block=m.group(3),
                                 pos=[float(m.group(k)) for k in (4, 5, 6)])
                    elif kind == "quiet":
                        self._ev(i, kind, switched_off=int(m.group(1)))
                    else:
                        self._ev(i, kind, result=m.group(1))
                    return
            if REWARP.search(line):
                self._ev(i, "rewarp")
            elif RETRAVEL.search(line):
                self._ev(i, "retravel")
            elif TOUR_START.search(line):
                self._ev(i, "tour_start")
            elif TOUR_DONE.search(line):
                self.tour_done = True
                self._ev(i, "tour_done")
            return
        for kind, rx in (("dump", DUMP_DONE), ("capture", CAPTURE_WROTE), ("capture_rejected", CAPTURE_REJECTED),
                         ("loading", LOADING), ("slow_pipeline", SLOW_PIPELINE), ("slow_library", SLOW_LIBRARY),
                         ("stall", STALL), ("long_frame", LONG_FRAME), ("ps_lifted", PS_LIFTED),
                         ("ps_not_lifted", PS_NOT_LIFTED), ("pipeline_failed", PIPELINE_FAILED), ("world", WORLD)):
            m = rx.search(line)
            if not m:
                continue
            if kind == "dump":
                self._ev(i, kind, flip=int(m.group(1)), path=m.group(2), ok=not m.group(3))
            elif kind == "capture":
                self._ev(i, kind, dir=m.group(1), pipeline=m.group(2), flip=int(m.group(3)))
            elif kind == "capture_rejected":
                self._ev(i, kind, pipeline=m.group(1), why=m.group(2))
            elif kind in ("loading", "slow_pipeline", "slow_library", "long_frame"):
                self._ev(i, kind, value=float(m.group(1)))
            elif kind == "stall":
                p = STALL_PIPELINE.search(line)
                self._ev(i, kind, value=float(m.group(1)), pipeline_ms=sum(int(x) for x in p.groups()) if p else 0)
            elif kind == "ps_lifted":
                self._ev(i, kind, ps=m.group(1))
            elif kind == "ps_not_lifted":
                self._ev(i, kind, ps=m.group(1), why=reason_key(m.group(2)))
            elif kind == "pipeline_failed":
                self._ev(i, kind, pipeline=m.group(1))
            elif kind == "world":
                self.world = {"flip": int(m.group(1)), "seconds": float(m.group(2))}
            return
        m = VERSION.search(line)
        if m and not self.version:
            self.version = m.group(1)
        m = RENDERING.search(line)
        if m:
            self.rendering = [int(m.group(1)), int(m.group(2))]
        if FATAL.search(line):
            self.fatal.append(line.strip()[:300])
        elif PROBLEM.search(line):
            self._ev(i, "problem", text=line.strip()[:300])

    def between(self, start, end, kinds=None):
        return [(i, k, d) for i, k, d in self.events if start < i < end and (kinds is None or k in kinds)]

    def seconds_between(self, start, end):
        """The seconds wholly inside (start, end): the first `frames:` line after
        start covers a second that began before it, and is left out."""
        inside = [rec for i, rec in self.seconds if start < i < end]
        return inside[1:]


def percentile(sorted_values, p):
    """Nearest-rank percentile of an ascending list."""
    if not sorted_values:
        return None
    k = max(1, math.ceil(p / 100.0 * len(sorted_values)))
    return sorted_values[min(k, len(sorted_values)) - 1]


def summarize_values(values):
    if not values:
        return None
    s = sorted(values)
    return {"mean": round(sum(s) / len(s), 3), "p95": percentile(s, 95), "max": s[-1], "n": len(s)}


def window_stats(seconds):
    """What a window of per-second records says. Frame times are exact when
    every second carried its intervals (BBHOST_FRAME_DETAIL=1); otherwise fps
    and the worst second's p95 and max stand in, marked approximate."""
    if not seconds:
        return None
    secs = sum(r["secs"] for r in seconds)
    flips = sum(r["flips"] for r in seconds)
    out = {"seconds": round(secs, 2), "flips": flips, "fps": round(flips / secs, 2) if secs else None,
           "displayed_fps": round(sum(r["displayed"] * r["secs"] for r in seconds) / secs, 2) if secs else None,
           "over_16_7": sum(r["over_16_7"] for r in seconds), "over_33_3": sum(r["over_33_3"] for r in seconds),
           "cpu_pct": round(sum(r["cpu_pct"] for r in seconds) / len(seconds), 1)}
    intervals = [v for r in seconds for v in r["intervals"]]
    exact = all(r["intervals"] for r in seconds) and len(intervals) >= flips * 0.9
    if exact:
        s = sorted(intervals)
        out["frame_ms"] = {"mean": round(sum(s) / len(s), 3), "p50": percentile(s, 50), "p90": percentile(s, 90),
                           "p95": percentile(s, 95), "p99": percentile(s, 99), "max": s[-1], "n": len(s)}
        out["exact_percentiles"] = True
    else:
        out["frame_ms"] = {"mean": round(secs * 1000.0 / flips, 3) if flips else None,
                           "p95": max(r["p95_ms"] for r in seconds), "max": max(r["max_ms"] for r in seconds)}
        out["exact_percentiles"] = False
    busy = [r["gpu_busy"] for r in seconds if r["gpu_busy"] is not None]
    if busy:
        out["gpu_busy_ms_per_s"] = summarize_values(busy)
        out["gpu_ms_per_flip"] = round(sum(busy) / len(busy) / (flips / secs), 3) if flips and secs else None
    work = [r["work"] for r in seconds if r["work"]]
    if work:
        out["main_loop_work_ms"] = {"mean": round(sum(w[0] for w in work) / len(work), 3),
                                    "p95": round(sum(w[1] for w in work) / len(work), 3), "max": max(w[2] for w in work)}
    details = [r["detail"] for r in seconds if r["detail"]]
    if details:
        d = {k: sum(x[k] for x in details) for k in ("draws", "lifted", "translated", "fallback", "none", "failures")}
        shaded = d["lifted"] + d["translated"] + d["fallback"]
        d["lifted_share"] = round(d["lifted"] / shaded, 4) if shaded else None
        d["per_second"] = round(d["draws"] / secs, 1) if secs else None
        reasons = {}
        for x in details:
            for k, v in x["reasons"].items():
                reasons[k] = reasons.get(k, 0) + v
        d["failure_reasons"] = reasons
        out["draws"] = d
    return out


def hitches(events):
    """Compiles, stalls and long frames among (index, kind, data) events."""
    def pick(kind):
        return [d["value"] for _, k, d in events if k == kind]
    slow, lib, stalls, longf = pick("slow_pipeline"), pick("slow_library"), pick("stall"), pick("long_frame")
    return {
        "slow_pipelines": {"count": len(slow), "ms": round(sum(slow)), "max_ms": max(slow, default=0)},
        "slow_libraries": {"count": len(lib), "ms": round(sum(lib)), "max_ms": max(lib, default=0)},
        "stalls": {"count": len(stalls), "max_ms": max(stalls, default=0),
                   "pipeline_ms": sum(d.get("pipeline_ms", 0) for _, k, d in events if k == "stall")},
        "long_frames": {"count": len(longf), "over_50": sum(1 for v in longf if v >= 50), "max_ms": max(longf, default=0)},
    }


def area_segments(log, area_ids):
    """{area: (start index, end index)}: from its travel line to the next
    place's, or the tour's end."""
    travels = [(i, d["area"]) for i, k, d in log.events if k == "travel" and d["area"] in area_ids]
    end_of_log = len(log.lines)
    done = next((i for i, k, _ in log.events if k == "tour_done"), end_of_log)
    out = {}
    for n, (i, area) in enumerate(travels):
        if area in out:
            continue  # a retravel of the same place: the first visit is the place's
        nxt = next((j for j, a in travels[n + 1:] if a != area), done)
        out[area] = (i, nxt)
    return out


def area_report(log, area, seg, kind):
    """One place's numbers from one pass's log."""
    start, end = seg
    evs = log.between(start, end)
    marks = {d["name"]: i for i, k, d in evs if k == "mark" and d["area"] == area.id}
    rep = {"id": area.id, "name": area.name, "block": area.block}
    failed = next((d for _, k, d in evs if k == "travel_failed" and d["area"] == area.id), None)
    done = next((d for _, k, d in evs if k == "travel_done" and d["area"] == area.id), None)
    rep["travel"] = {"seconds": done["seconds"] if done else None, "block": (done or failed or {}).get("block"),
                     "failed": bool(failed), "retravels": sum(1 for _, k, _ in evs if k == "retravel")}
    at = next((d for _, k, d in evs if k == "at" and d["area"] == area.id), None)
    if area.view():
        rep["view"] = {"reached": bool(at), "pos": at["pos"] if at else None,
                       "rewarps": sum(1 for _, k, _ in evs if k == "rewarp")}
    rep["quiet"] = sum(d["switched_off"] for _, k, d in evs if k == "quiet")
    rep["camera"] = next((d["result"] for _, k, d in evs if k == "camera"), None)
    arrived = marks.get("arrived")
    if failed or arrived is None:
        rep["status"] = "travel failed" if failed else "not reached"
    elif done and done["block"] != area.block_hex():
        rep["status"] = "wrong block"
    elif area.view() and not at:
        rep["status"] = "view not reached"
    else:
        rep["status"] = "ok"
    # The arrival: the travel and the settle, up to the window (or, in the
    # capture pass, to the captures).
    arrival_end = marks.get("measure" if kind == "perf" else "capture", marks.get("end", end))
    arrival = log.between(start, arrival_end)
    lifted = sorted({d["ps"] for _, k, d in arrival if k == "ps_lifted"})
    not_lifted = {}
    for _, k, d in arrival:
        if k == "ps_not_lifted":
            not_lifted.setdefault(d["why"], set()).add(d["ps"])
    rep["arrival"] = dict(hitches(arrival), loads=[d["value"] for _, k, d in arrival if k == "loading"],
                          pipelines_failed=sorted({d["pipeline"] for _, k, d in arrival if k == "pipeline_failed"}),
                          new_pixel_shaders={"lifted": len(lifted),
                                             "translated": sum(len(v) for v in not_lifted.values()),
                                             "reasons": {w: len(v) for w, v in sorted(not_lifted.items(), key=lambda kv: -len(kv[1]))},
                                             "translated_shaders": {w: sorted(v) for w, v in not_lifted.items()}},
                          draw_failures=sum((r["detail"] or {}).get("failures", 0) for r in
                                            [rec for i, rec in log.seconds if start < i < arrival_end]),
                          problems=[d["text"] for _, k, d in arrival if k == "problem"][:20])
    if kind == "perf" and "measure" in marks and "measured" in marks:
        w0, w1 = marks["measure"], marks["measured"]
        rep["window"] = window_stats(log.seconds_between(w0, w1))
        inside = log.between(w0, w1)
        if rep["window"] is not None:
            rep["window"].update(hitches(inside))
            rep["window"]["problems"] = [d["text"] for _, k, d in inside if k == "problem"][:20]
        rep["frames"] = {}
        for _, k, d in log.between(w1, marks.get("end", end), {"dump"}):
            name = Path(d["path"]).stem
            prefix = area.id + "-"
            rep["frames"][name[len(prefix):] if name.startswith(prefix) else name] = {"path": d["path"], "flip": d["flip"], "ok": d["ok"]}
        if not rep["window"] and rep["status"] == "ok":
            rep["status"] = "no window"
    elif kind == "perf" and rep["status"] == "ok":
        rep["status"] = "no window"
    if kind == "capture":
        cap = log.between(start, end, {"capture", "capture_rejected"})
        rep["captures"] = {"written": [{"dir": d["dir"], "pipeline": d["pipeline"], "flip": d["flip"]} for _, k, d in cap if k == "capture"],
                           "rejected": [{"pipeline": d["pipeline"], "why": d["why"]} for _, k, d in cap if k == "capture_rejected"]}
    return rep


def pass_report(lines, areas, kind):
    """A pass's report from its log lines."""
    log = Log(lines)
    segs = area_segments(log, {a.id for a in areas})
    out = {"kind": kind, "version": log.version, "world": log.world, "tour_done": log.tour_done, "fatal": log.fatal[:20],
           "rendering": log.rendering, "areas": {}}
    for a in areas:
        if a.id in segs:
            out["areas"][a.id] = area_report(log, a, segs[a.id], kind)
        else:
            out["areas"][a.id] = {"id": a.id, "name": a.name, "block": a.block, "status": "not reached"}
    return out


# --- frames -----------------------------------------------------------------------

try:
    import numpy as np  # optional: the same numbers, faster
except ImportError:  # pragma: no cover - CI has no numpy
    np = None


class Image:
    def __init__(self, width, height, rgb):
        self.width, self.height, self.rgb = width, height, rgb  # bytes, RGB8 rows


def read_ppm(path):
    data = Path(path).read_bytes()
    fields, at = [], 0
    while len(fields) < 4:
        while data[at:at + 1].isspace():
            at += 1
        if data[at:at + 1] == b"#":
            at = data.index(b"\n", at) + 1
            continue
        end = at
        while end < len(data) and not data[end:end + 1].isspace():
            end += 1
        if end == at:
            raise ValueError("%s: not a PPM" % path)
        fields.append(data[at:end])
        at = end
    if fields[0] != b"P6" or int(fields[3]) != 255:
        raise ValueError("%s: not an 8-bit binary PPM" % path)
    w, h = int(fields[1]), int(fields[2])
    rgb = data[at + 1:at + 1 + w * h * 3]
    if len(rgb) != w * h * 3:
        raise ValueError("%s: short pixel data" % path)
    return Image(w, h, rgb)


def write_ppm(path, image):
    Path(path).write_bytes(b"P6\n%d %d\n255\n" % (image.width, image.height) + bytes(image.rgb))


def luma_list(img):
    d = img.rgb
    return [0.299 * d[k] + 0.587 * d[k + 1] + 0.114 * d[k + 2] for k in range(0, len(d), 3)]


def image_stats(img):
    """Mean of each channel and of the luma: enough to spot a black or white frame."""
    if np is not None:
        a = np.frombuffer(img.rgb, np.uint8).reshape(img.height, img.width, 3).astype(np.float64)
        mean = a.reshape(-1, 3).mean(axis=0).tolist()
    else:
        n = img.width * img.height
        mean = [sum(img.rgb[c::3]) / n for c in range(3)]
    return {"width": img.width, "height": img.height, "mean_rgb": [round(v, 2) for v in mean],
            "mean_luma": round(0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2], 2)}


GRID = 8  # the frame in GRID x GRID blocks for the block table
BLUR = 4  # box radius, pixels, of the blurred luma difference


def box_mean(values, w, h, r):
    """Mean over the (2r+1)^2 box around each pixel, clipped at the edges (a
    summed-area table)."""
    sat = [0.0] * ((w + 1) * (h + 1))
    for y in range(h):
        row = 0.0
        base, prev = (y + 1) * (w + 1), y * (w + 1)
        for x in range(w):
            row += values[y * w + x]
            sat[base + x + 1] = sat[prev + x + 1] + row
    out = [0.0] * (w * h)
    for y in range(h):
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        for x in range(w):
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            s = sat[y1 * (w + 1) + x1] - sat[y0 * (w + 1) + x1] - sat[y1 * (w + 1) + x0] + sat[y0 * (w + 1) + x0]
            out[y * w + x] = s / ((y1 - y0) * (x1 - x0))
    return out


def compare_images(a, b):
    """How B's frame differs from A's:
      changed_pct      pixels with a channel more than 8 levels apart
      mean_abs         mean absolute difference over every channel, in levels
      mean_signed      mean luma difference (B - A): a tint or exposure shift
      blocks           the luma difference's mean in each of GRID x GRID blocks
      max_block        the largest of those, in absolute value: a region gone or changed
      blurred_pct      pixels whose box-blurred luma moved more than 4 levels:
                       changes larger than flicker and noise"""
    if (a.width, a.height) != (b.width, b.height):
        return {"error": "sizes differ: %dx%d against %dx%d" % (a.width, a.height, b.width, b.height)}
    w, h = a.width, a.height
    gx_n, gy_n = min(GRID, w), min(GRID, h)  # a frame smaller than the grid: a block a pixel
    bw, bh = w // gx_n, h // gy_n
    if np is not None:
        A = np.frombuffer(a.rgb, np.uint8).reshape(h, w, 3).astype(np.float64)
        B = np.frombuffer(b.rgb, np.uint8).reshape(h, w, 3).astype(np.float64)
        d = B - A
        changed = float((np.abs(d).max(axis=2) > 8).mean() * 100.0)
        mean_abs = float(np.abs(d).mean())
        dl = d[..., 0] * 0.299 + d[..., 1] * 0.587 + d[..., 2] * 0.114
        blocks = [[float(dl[gy * bh:(gy + 1) * bh if gy < gy_n - 1 else h, gx * bw:(gx + 1) * bw if gx < gx_n - 1 else w].mean())
                   for gx in range(gx_n)] for gy in range(gy_n)]
        sat = np.zeros((h + 1, w + 1))
        sat[1:, 1:] = dl.cumsum(0).cumsum(1)
        ys, xs = np.arange(h), np.arange(w)
        y0, y1 = np.maximum(ys - BLUR, 0), np.minimum(ys + BLUR + 1, h)
        x0, x1 = np.maximum(xs - BLUR, 0), np.minimum(xs + BLUR + 1, w)
        s = sat[y1][:, x1] - sat[y0][:, x1] - sat[y1][:, x0] + sat[y0][:, x0]
        blurred = s / np.outer(y1 - y0, x1 - x0)
        blurred_pct = float((np.abs(blurred) > 4).mean() * 100.0)
        mean_signed = float(dl.mean())
    else:
        ra, rb = a.rgb, b.rgb
        n = w * h
        changed_n, abs_sum = 0, 0
        dl = [0.0] * n
        for p in range(n):
            k = p * 3
            d0, d1, d2 = rb[k] - ra[k], rb[k + 1] - ra[k + 1], rb[k + 2] - ra[k + 2]
            if max(abs(d0), abs(d1), abs(d2)) > 8:
                changed_n += 1
            abs_sum += abs(d0) + abs(d1) + abs(d2)
            dl[p] = 0.299 * d0 + 0.587 * d1 + 0.114 * d2
        changed = changed_n * 100.0 / n
        mean_abs = abs_sum / (3.0 * n)
        mean_signed = sum(dl) / n
        blocks = []
        for gy in range(gy_n):
            row = []
            for gx in range(gx_n):
                ya, yb = gy * bh, (gy + 1) * bh if gy < gy_n - 1 else h
                xa, xb = gx * bw, (gx + 1) * bw if gx < gx_n - 1 else w
                vals = [dl[y * w + x] for y in range(ya, yb) for x in range(xa, xb)]
                row.append(sum(vals) / len(vals) if vals else 0.0)
            blocks.append(row)
        blurred = box_mean(dl, w, h, BLUR)
        blurred_pct = sum(1 for v in blurred if abs(v) > 4) * 100.0 / n
    flat = [v for row in blocks for v in row]
    return {"changed_pct": round(changed, 3), "mean_abs": round(mean_abs, 3), "mean_signed": round(mean_signed, 3),
            "max_block": round(max((abs(v) for v in flat), default=0.0), 3), "blurred_pct": round(blurred_pct, 3),
            "blocks": [[round(v, 2) for v in row] for row in blocks]}


def diff_image(a, b, gain=4):
    """|B - A| per channel, times gain: where two frames differ."""
    if np is not None:
        A = np.frombuffer(a.rgb, np.uint8).astype(np.int16)
        B = np.frombuffer(b.rgb, np.uint8).astype(np.int16)
        return Image(a.width, a.height, np.clip(np.abs(B - A) * gain, 0, 255).astype(np.uint8).tobytes())
    return Image(a.width, a.height, bytes(min(255, abs(y - x) * gain) for x, y in zip(a.rgb, b.rgb)))


# Thresholds past the noise floor: a difference counts when it is more than
# NOISE_FACTOR times the largest same-build difference plus these margins.
NOISE_FACTOR = 1.5
FRAME_MARGINS = {"changed_pct": 1.0, "max_block": 2.0, "blurred_pct": 0.5, "mean_signed": 0.5}


def frame_verdict(ab_pairs, noise_pairs):
    """ab_pairs, noise_pairs: compare_images results (A against B; a build
    against its own repeat). A frame differs beyond noise when a metric of every
    A-B pair passes the floor the repeats set."""
    ab_pairs = [p for p in ab_pairs if "error" not in p]
    noise_pairs = [p for p in noise_pairs if "error" not in p]
    if not ab_pairs:
        return {"verdict": "not compared"}
    worst = {k: max(abs(p[k]) for p in ab_pairs) for k in FRAME_MARGINS}
    least = {k: min(abs(p[k]) for p in ab_pairs) for k in FRAME_MARGINS}
    if not noise_pairs:
        return {"verdict": "no noise floor", "ab": worst}
    floor = {k: max(abs(p[k]) for p in noise_pairs) for k in FRAME_MARGINS}
    limit = {k: floor[k] * NOISE_FACTOR + FRAME_MARGINS[k] for k in FRAME_MARGINS}
    beyond = sorted(k for k in FRAME_MARGINS if least[k] > limit[k])
    return {"verdict": "differs" if beyond else "within noise", "beyond": beyond, "ab": worst, "noise": floor,
            "limit": {k: round(v, 3) for k, v in limit.items()}}


# --- comparing runs -----------------------------------------------------------------

# Per-window metrics: (path in the window, better when higher, the smallest
# difference worth reporting).
PERF_METRICS = (
    ("fps", ("fps",), True, 0.3),
    ("frame p50 ms", ("frame_ms", "p50"), False, 0.2),
    ("frame p95 ms", ("frame_ms", "p95"), False, 0.3),
    ("frame p99 ms", ("frame_ms", "p99"), False, 0.5),
    ("frame max ms", ("frame_ms", "max"), False, 2.0),
    ("frames over 33.3 ms", ("over_33_3",), False, 1),
    ("GPU busy ms/s", ("gpu_busy_ms_per_s", "mean"), False, 5.0),
    ("GPU ms a flip", ("gpu_ms_per_flip",), False, 0.1),
    ("main loop work ms", ("main_loop_work_ms", "mean"), False, 0.2),
    ("CPU %", ("cpu_pct",), False, 5.0),
)
ARRIVAL_METRICS = (
    ("slow pipelines", ("slow_pipelines", "count"), False, 2),
    ("slow pipeline ms", ("slow_pipelines", "ms"), False, 100),
    ("stalls", ("stalls", "count"), False, 2),
    ("long frames", ("long_frames", "count"), False, 2),
    ("draw failures", ("draw_failures",), False, 1),
)


def dig(d, path):
    for k in path:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def metric_delta(a_vals, b_vals, higher_better, min_abs):
    """Mean of each arm, their difference, and whether it is beyond the noise:
    the larger of the two arms' own spreads (max - min), when each has two or
    more runs."""
    a_vals = [v for v in a_vals if v is not None]
    b_vals = [v for v in b_vals if v is not None]
    if not a_vals or not b_vals:
        return None
    ma, mb = sum(a_vals) / len(a_vals), sum(b_vals) / len(b_vals)
    delta = mb - ma
    out = {"a": [round(v, 3) for v in a_vals], "b": [round(v, 3) for v in b_vals], "a_mean": round(ma, 3), "b_mean": round(mb, 3),
           "delta": round(delta, 3), "rel_pct": round(delta * 100.0 / ma, 2) if ma else None}
    if len(a_vals) >= 2 and len(b_vals) >= 2:
        noise = max(max(a_vals) - min(a_vals), max(b_vals) - min(b_vals))
        out["noise"] = round(noise, 3)
        if abs(delta) > noise and abs(delta) >= min_abs:
            out["verdict"] = "better" if (delta > 0) == higher_better else "worse"
        else:
            out["verdict"] = "within noise"
    else:
        out["verdict"] = "no noise floor" if abs(delta) >= min_abs else "same"
    return out


def lift_rows(report_area):
    """{pixel shader: status} from a place's lift results."""
    rows = ((report_area.get("lift") or {}).get("rows")) or []
    out = {}
    for r in rows:
        ps = capture_ps(r.get("name", ""))
        if ps:
            out[ps] = r.get("status")
    return out


def capture_ps(name):
    """The pixel shader of a capture directory '<vs>+<ps>-f<flip>-d<draw>'."""
    m = re.match(r"[^+]*\+([0-9a-f]+)", name)
    return m.group(1) if m else None


def compare_reports(a_runs, b_runs, frame_pairs=None):
    """Two builds' run reports (one or more each) compared place by place.
    frame_pairs: {area: {frame: {"ab": [...], "noise": [...]}}} from
    compare_frames, when the frames were read."""
    ids = []
    for r in a_runs + b_runs:
        for k in r.get("areas", {}):
            if k not in ids:
                ids.append(k)
    out = {"schema": SCHEMA + "-ab", "a_runs": [r.get("dir") for r in a_runs], "b_runs": [r.get("dir") for r in b_runs],
           "a_version": sorted({str(r["run"]["version"]) for r in a_runs if (r.get("run") or {}).get("version")}),
           "b_version": sorted({str(r["run"]["version"]) for r in b_runs if (r.get("run") or {}).get("version")}), "areas": {}}
    for aid in ids:
        A = [r["areas"].get(aid, {}) for r in a_runs]
        B = [r["areas"].get(aid, {}) for r in b_runs]
        ent = {"status": {"a": [x.get("status") for x in A], "b": [x.get("status") for x in B]}, "perf": {}, "arrival": {}}
        for label, path, hb, mn in PERF_METRICS:
            d = metric_delta([dig(x.get("window"), path) for x in A], [dig(x.get("window"), path) for x in B], hb, mn)
            if d:
                ent["perf"][label] = d
        for label, path, hb, mn in ARRIVAL_METRICS:
            d = metric_delta([dig(x.get("arrival"), path) for x in A], [dig(x.get("arrival"), path) for x in B], hb, mn)
            if d:
                ent["arrival"][label] = d
        share = metric_delta([dig(x.get("window"), ("draws", "lifted_share")) for x in A],
                             [dig(x.get("window"), ("draws", "lifted_share")) for x in B], True, 0.001)
        if share:
            ent["lifted_share"] = share
        wf = metric_delta([dig(x.get("window"), ("draws", "failures")) for x in A],
                          [dig(x.get("window"), ("draws", "failures")) for x in B], False, 1)
        if wf:
            ent["window_draw_failures"] = wf
        la = next((x.get("lift") for x in A if x.get("lift")), None)
        lb = next((x.get("lift") for x in B if x.get("lift")), None)
        if la or lb:
            ra = next((lift_rows(x) for x in A if x.get("lift")), {})
            rb = next((lift_rows(x) for x in B if x.get("lift")), {})
            changed = {ps: {"a": ra.get(ps), "b": rb.get(ps)} for ps in sorted(set(ra) & set(rb)) if ra.get(ps) != rb.get(ps)}
            ent["lift"] = {"a": (la or {}).get("totals"), "b": (lb or {}).get("totals"), "changed": changed,
                           "b_different": sorted(ps for ps, s in rb.items() if s in ("different", "invalid"))}
        if frame_pairs and aid in frame_pairs:
            ent["frames"] = {name: dict(frame_verdict(p.get("ab", []), p.get("noise", [])),
                                        diff=p.get("diff"), blocks=p.get("blocks"))
                             for name, p in sorted(frame_pairs[aid].items())}
        out["areas"][aid] = ent
    flagged, improved = [], []
    for aid, ent in out["areas"].items():
        for group in ("perf", "arrival"):
            for label, d in ent[group].items():
                if d.get("verdict") in ("worse", "better"):
                    (flagged if d["verdict"] == "worse" else improved).append(
                        "%s: %s %s (%+.3g, noise %.3g)" % (aid, label, d["verdict"], d["delta"], d.get("noise", 0)))
        for name, f in (ent.get("frames") or {}).items():
            if f.get("verdict") == "differs":
                flagged.append("%s: frame %s differs beyond noise (%s)" % (aid, name, ", ".join(f["beyond"])))
        if (ent.get("lift") or {}).get("b_different"):
            flagged.append("%s: lifts that render differently in B: %s" % (aid, ", ".join(ent["lift"]["b_different"])))
        wf = ent.get("window_draw_failures")
        if wf and wf["b_mean"] > wf["a_mean"]:
            flagged.append("%s: more draw failures in B's window (%g against %g)" % (aid, wf["b_mean"], wf["a_mean"]))
        if any(s != "ok" for s in ent["status"]["a"] + ent["status"]["b"]):
            flagged.append("%s: not measured in every run (A %s, B %s)" % (aid, ent["status"]["a"], ent["status"]["b"]))
    out["flagged"] = flagged
    out["improved"] = improved
    return out


def compare_frames(a_dirs, b_dirs, area_ids, out_dir=None):
    """Reads the runs' frames: every A-B pair, and the repeats within each
    build as the noise floor. Writes a difference image of the first A-B pair
    of each frame into out_dir/diffs."""
    def frames_of(run_dir):
        try:
            rep = json.loads((Path(run_dir) / "report.json").read_text())
        except (OSError, ValueError):
            return {}
        return {aid: {n: f["path"] for n, f in (a.get("frames") or {}).items() if f.get("ok")}
                for aid, a in rep.get("areas", {}).items()}
    fa = [frames_of(d) for d in a_dirs]
    fb = [frames_of(d) for d in b_dirs]
    cache = {}

    def load(p):
        if p not in cache:
            try:
                cache[p] = read_ppm(p)
            except (OSError, ValueError):
                cache[p] = None
        return cache[p]
    out = {}
    for aid in area_ids:
        names = sorted({n for f in fa + fb for n in f.get(aid, {})})
        for name in names:
            pa = [f[aid][name] for f in fa if name in f.get(aid, {})]
            pb = [f[aid][name] for f in fb if name in f.get(aid, {})]
            ent = {"ab": [], "noise": []}
            for x in pa:
                for y in pb:
                    ia, ib = load(x), load(y)
                    if not (ia and ib):
                        continue
                    c = compare_images(ia, ib)
                    blocks = c.pop("blocks", None)
                    if "blocks" not in ent and blocks:
                        ent["blocks"] = blocks  # the first pair's block table
                    ent["ab"].append(c)
                    if out_dir and "diff" not in ent and "error" not in c:
                        d = Path(out_dir) / "diffs"
                        d.mkdir(parents=True, exist_ok=True)
                        target = d / ("%s-%s.ppm" % (aid, name))
                        write_ppm(target, diff_image(ia, ib))
                        ent["diff"] = str(target)
            for group in (pa, pb):
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        ia, ib = load(group[i]), load(group[j])
                        if ia and ib:
                            c = compare_images(ia, ib)
                            c.pop("blocks", None)
                            ent["noise"].append(c)
            out.setdefault(aid, {})[name] = ent
        cache.clear()  # a place's frames at a time: 6 MB each at 1080p
    return out


# --- text ----------------------------------------------------------------------------

def fmt(v, spec="%.1f"):
    return "-" if v is None else spec % v


def summary_text(report):
    """The run's report to read."""
    run = report.get("run", {})
    out = ["area check: %s" % report.get("dir", ""),
           "build %s (%s), sha256 %s" % (run.get("version") or "?", run.get("bbhost"), (run.get("bbhost_sha256") or "?")[:16]),
           "pinned to NUMA node %s; %s fps; pipeline cache %s; started %s" % (
               run.get("numa_node"), run.get("fps"), "seed's" if run.get("pipeline_cache") else "off", run.get("started_utc"))]
    for kind in ("perf", "capture"):
        p = report.get("passes", {}).get(kind)
        if p:
            out.append("%s pass: %s in %.0f s, world %s, tour %s%s%s" % (
                kind, p.get("stop_reason"), p.get("elapsed_s", 0), "reached" if p.get("world") else "NOT reached",
                "done" if p.get("tour_done") else "NOT done",
                (", rendering %dx%d" % tuple(p["rendering"])) if p.get("rendering") else "",
                ("; FATAL: " + p["fatal"][0]) if p.get("fatal") else ""))
    out.append("")
    for aid, a in report.get("areas", {}).items():
        out.append("%-19s %s (%s): %s" % (aid, a.get("name"), a.get("block"), a.get("status")))
        w = a.get("window")
        if w:
            fm = w.get("frame_ms", {})
            out.append("  window %s s: %s fps; frame ms p50 %s p95 %s p99 %s max %s%s; over 33.3 ms %d" % (
                fmt(w.get("seconds"), "%.0f"), fmt(w.get("fps"), "%.2f"), fmt(fm.get("p50"), "%.2f"), fmt(fm.get("p95"), "%.2f"),
                fmt(fm.get("p99"), "%.2f"), fmt(fm.get("max"), "%.1f"), "" if w.get("exact_percentiles") else " (approximate)",
                w.get("over_33_3", 0)))
            out.append("  GPU busy %s ms/s (%s ms a flip); main loop work %s ms; CPU %s%%" % (
                fmt(dig(w, ("gpu_busy_ms_per_s", "mean")), "%.0f"), fmt(w.get("gpu_ms_per_flip"), "%.2f"),
                fmt(dig(w, ("main_loop_work_ms", "mean")), "%.2f"), fmt(w.get("cpu_pct"), "%.0f")))
            d = w.get("draws")
            if d:
                out.append("  draws %s/s: pixel shader lifted %s (%s), translated %d, fallback %d, none %d; draw failures %d%s" % (
                    fmt(d.get("per_second"), "%.0f"), d["lifted"], fmt((d.get("lifted_share") or 0) * 100, "%.1f%%"), d["translated"],
                    d["fallback"], d["none"], d["failures"],
                    (" (" + ", ".join("%s %d" % kv for kv in d["failure_reasons"].items()) + ")") if d.get("failure_reasons") else ""))
            out.append("  in the window: %d slow pipelines, %d stalls (worst %d ms), %d main-loop frames over 33 ms" % (
                w["slow_pipelines"]["count"], w["stalls"]["count"], w["stalls"]["max_ms"], w["long_frames"]["count"]))
        ar = a.get("arrival")
        if ar:
            t = a.get("travel", {})
            nps = ar.get("new_pixel_shaders", {})
            out.append("  arrival: travel %s s, loads %s; %d slow pipelines (%d ms, worst %d), %d stalls (worst %d ms), "
                       "%d long frames; draw failures %d" % (
                           fmt(t.get("seconds")), ", ".join("%.1f s" % v for v in ar.get("loads", [])) or "-",
                           ar["slow_pipelines"]["count"], ar["slow_pipelines"]["ms"], ar["slow_pipelines"]["max_ms"],
                           ar["stalls"]["count"], ar["stalls"]["max_ms"], ar["long_frames"]["count"], ar.get("draw_failures", 0)))
            if nps.get("lifted") or nps.get("translated"):
                reasons = "; ".join("%s x%d" % kv for kv in list(nps.get("reasons", {}).items())[:4])
                out.append("  pixel shaders first met here: %d lifted, %d translated%s" % (
                    nps.get("lifted", 0), nps.get("translated", 0), (" (" + reasons + ")") if reasons else ""))
            if ar.get("pipelines_failed"):
                out.append("  pipelines that failed to build: " + ", ".join(ar["pipelines_failed"][:8]))
        fr = a.get("frames")
        if fr:
            out.append("  frames: " + ", ".join("%s %s" % (n, "luma %.1f" % f["stats"]["mean_luma"] if f.get("stats") else
                                                           ("missing" if not f.get("ok") else "-")) for n, f in sorted(fr.items())))
        lf = a.get("lift")
        if lf:
            tot = lf.get("totals", {})
            out.append("  lift checks on %d captures: %s" % (sum(tot.values()), ", ".join("%s %d" % kv for kv in sorted(tot.items())) or "none"))
            for why, n in (lf.get("rejection_reasons") or [])[:3]:
                out.append("    rejected x%d: %s" % (n, why[:100]))
        elif a.get("captures") is not None:
            c = a["captures"]
            out.append("  captures: %d written, %d rejected" % (len(c.get("written", [])), len(c.get("rejected", []))))
    if report.get("problems"):
        out.append("")
        out.extend("problem: " + p for p in report["problems"])
    return "\n".join(out) + "\n"


def ab_text(ab):
    out = ["area check A/B", "A: %s (%s)" % (", ".join(str(d) for d in ab["a_runs"]), ", ".join(ab["a_version"]) or "?"),
           "B: %s (%s)" % (", ".join(str(d) for d in ab["b_runs"]), ", ".join(ab["b_version"]) or "?"), ""]
    for aid, e in ab["areas"].items():
        out.append("%s: A %s, B %s" % (aid, "/".join(str(s) for s in e["status"]["a"]), "/".join(str(s) for s in e["status"]["b"])))
        for group in ("perf", "arrival"):
            for label, d in e[group].items():
                out.append("  %-22s A %-9s B %-9s %+8.3f%s  %s" % (
                    label, fmt(d["a_mean"], "%.3f"), fmt(d["b_mean"], "%.3f"), d["delta"],
                    (" (noise %.3f)" % d["noise"]) if "noise" in d else "", d.get("verdict", "")))
        if "lifted_share" in e:
            s = e["lifted_share"]
            out.append("  %-22s A %.1f%% B %.1f%%" % ("draws lifted", s["a_mean"] * 100, s["b_mean"] * 100))
        for name, f in (e.get("frames") or {}).items():
            ab_m = f.get("ab", {})
            out.append("  frame %-4s %-15s changed %s%% blurred %s%% max block %s levels%s" % (
                name, f["verdict"], fmt(ab_m.get("changed_pct")), fmt(ab_m.get("blurred_pct")), fmt(ab_m.get("max_block")),
                (" (noise %s%% / %s%% / %s)" % (fmt(f["noise"]["changed_pct"]), fmt(f["noise"]["blurred_pct"]), fmt(f["noise"]["max_block"])))
                if f.get("noise") else ""))
        if e.get("lift"):
            l = e["lift"]
            out.append("  lift A %s; B %s" % (l["a"], l["b"]))
            for ps, s in list(l["changed"].items())[:10]:
                out.append("    %s: %s -> %s" % (ps, s["a"], s["b"]))
    out.append("")
    out.append("flagged:" if ab["flagged"] else "nothing worse beyond the noise floor")
    out.extend("  " + f for f in ab["flagged"])
    if ab.get("improved"):
        out.append("better beyond the noise floor:")
        out.extend("  " + f for f in ab["improved"])
    return "\n".join(out) + "\n"


# --- checks (no GPU) ---------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_paths(config):
    """app0 and eboot from a test config's [paths] (relative paths from its folder)."""
    import tomllib
    data = tomllib.loads(Path(config).read_text())
    paths = data.get("paths", {})
    base = Path(config).resolve().parent

    def resolve(v):
        return str((base / v).resolve()) if v and not Path(v).is_absolute() else v
    return resolve(paths.get("app0")), resolve(paths.get("eboot"))


def check_areas(areas, app0):
    """Each place's lamp and view against the game's data: the ReturnPointParam
    row is there and in the place's block, the block's layout is there, a fog
    wall's view stands within 3 m of a map object (the wall it was taken from)
    and a standpoint within STAND_REACH of something the layout puts on the
    ground (an object, a character, a player start) - not out in the void, or
    in the block's own frame instead of the MSB's."""
    import bbparam
    import msb
    rows = dict(bbparam.load_table("ReturnPointParam", app0))
    results = []
    for a in areas:
        r = {"id": a.id, "problems": []}
        row = rows.get(a.return_point)
        if row is None:
            r["problems"].append("no ReturnPointParam row %d" % a.return_point)
        else:
            blk = "m%02d_%02d_00_00" % (row["areaNo"], row["blockNo"])
            r["lamp"] = {"row": a.return_point, "block": blk, "headstone_slot": row["warpChairNo"]}
            if blk != a.block:
                r["problems"].append("row %d is in %s, not %s" % (a.return_point, blk, a.block))
        path = Path(app0) / "dvdroot_ps4" / "map" / "mapstudio" / (a.block + ".msb.dcx")
        if not path.is_file():
            r["problems"].append("no layout %s" % path.name)
        elif a.fog or a.stand:
            parts = msb.parts(msb.load(str(path)))
            if a.fog:
                fx, fy, fz, _ = a.fog
                near = min(((p["pos"][0] - fx) ** 2 + (p["pos"][1] - fy) ** 2 + (p["pos"][2] - fz) ** 2, p["name"])
                           for p in parts if p["type"] == 1)
                r["fog_object"] = {"name": near[1], "distance_m": round(math.sqrt(near[0]), 3)}
                if near[0] > 9.0:
                    r["problems"].append("no map object within 3 m of the fog wall position (nearest %s at %.1f m)" % (near[1], math.sqrt(near[0])))
            if a.stand:
                sx, sy, sz, _ = a.stand
                near = min((((p["pos"][0] - sx) ** 2 + (p["pos"][1] - sy) ** 2 + (p["pos"][2] - sz) ** 2, p["name"])
                            for p in parts if p["type"] in (1, 2, 4)), default=(math.inf, None))
                r["stand_near"] = {"name": near[1], "distance_m": round(math.sqrt(near[0]), 3)}
                if near[0] > STAND_REACH ** 2:
                    r["problems"].append("nothing the layout places within %.0f m of the standpoint (nearest %s at %.1f m)" % (
                        STAND_REACH, near[1], math.sqrt(near[0])))
        results.append(r)
    return results


def check_note(c):
    """What a place's check found the view near: 'fog object o349000_0000 at 0.00 m'."""
    out = []
    if c.get("fog_object"):
        out.append("fog object %s at %.2f m" % (c["fog_object"]["name"], c["fog_object"]["distance_m"]))
    if c.get("stand_near"):
        out.append("standpoint %.2f m from %s" % (c["stand_near"]["distance_m"], c["stand_near"]["name"]))
    return "; ".join(out)


def tool_paths(bbhost):
    d = Path(bbhost).resolve().parent
    return {"gcnlift": d / "gcnlift", "drawreplay": d / "drawreplay"}


# --- running -------------------------------------------------------------------------

def build_probe(out_so):
    out_so.parent.mkdir(parents=True, exist_ok=True)
    cc = os.environ.get("CC", "cc")
    r = subprocess.run([cc, "-shared", "-fPIC", "-O2", "-Wall", "-I" + str(ROOT / "include"), "-o", str(out_so),
                        str(ROOT / "tools" / "mapval_plugin.c"), "-lm"], capture_output=True, text=True)
    if r.returncode != 0:
        raise ValueError("the probe did not build:\n" + r.stderr)


def prepare_pass(pass_dir, kind, areas, opts, paths):
    """Everything a pass needs, in its folder: the seed's copy with the probe,
    the config, the tour."""
    seed = Path(opts["seed"])
    pass_dir.mkdir(parents=True)
    for sub in ("cfg", "requests", "frames", "captures"):
        (pass_dir / sub).mkdir()
    shutil.copytree(seed, pass_dir / "data", symlinks=False)
    copy = world_baseline.file_inventory(pass_dir / "data")
    if copy != opts["seed_files"]:
        raise ValueError("the seed's copy in %s differs from the seed" % (pass_dir / "data"))
    (pass_dir / "data" / "plugins").mkdir(exist_ok=True)
    shutil.copy2(opts["probe"], pass_dir / "data" / "plugins" / "mapval.so")
    toml = run_toml(paths["app0"], paths["eboot"], pass_dir / "data", opts["p2p_port"], opts["width"], opts["height"],
                    old_hunters=any(a.dlc for a in areas))
    (pass_dir / "cfg" / "bbhost.toml").write_text(toml)
    (pass_dir / "tour.txt").write_text(tour(areas, opts, kind))
    settings = pass_settings(pass_dir, areas, opts, kind)
    argv = world_baseline.launch_argv([str(Path(opts["bbhost"]).resolve()), "--config", str(pass_dir / "cfg" / "bbhost.toml")],
                                      opts["numa_node"])
    plan = {"kind": kind, "argv": argv, "cwd": str(pass_dir), "settings": settings, "budget_s": pass_budget(areas, opts, kind),
            "estimate_s": estimate(areas, opts, kind)}
    (pass_dir / "pass-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def game_env(settings):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BBHOST_", "GCN2SPV_"))}
    env.update(settings)
    env["SDL_AUDIO_DRIVER"] = "dummy"  # a test run makes no sound on the machine it runs on
    return env


def title_tap(requests, n):
    """The n-th press of the menus' confirm button on the way through the
    title, as a request file the host carries out on a flip (after the probe's
    numbered ones, which sort first)."""
    (requests / ("title-%03d.req" % n)).write_text("menutap confirm\n")


def launch(pass_dir, plan, say):
    """Runs the game until the tour is done (or it dies, or the budget is out),
    with its log in run.log. Until the world's first frame it presses confirm
    in the title's menus (TITLE_TAP_EVERY_S)."""
    env = game_env(plan["settings"])
    requests = Path(plan["settings"]["BBHOST_TEST_REQUESTS"])
    started = time.monotonic()
    deadline = started + plan["budget_s"]
    world_deadline = started + 300
    next_tap, taps = started + TITLE_TAP_FIRST_S, 0
    stop_reason, term_at, pending = None, None, b""
    world = done = False
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
                    if not world and stop_reason is None and now >= next_tap:
                        taps += 1
                        title_tap(requests, taps)
                        next_tap = now + TITLE_TAP_EVERY_S
                    if stop_reason and term_at is None:
                        proc.send_signal(signal.SIGTERM)
                        term_at = now
                    if term_at is not None and now - term_at > 15 and proc.poll() is None:
                        proc.kill()
                    if term_at is not None and now - term_at > 30:
                        break  # something else holds the pipe open; the game is gone
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
                            if not world and WORLD.search(line):
                                world = True
                                say("world reached")
                            m = MARK.search(line)
                            if m:
                                say("%s: %s" % (m.group(1), m.group(2)))
                            m = TRAVEL_FAILED.search(line)
                            if m:
                                say("%s: travel failed" % m.group(1))
                            if TOUR_DONE.search(line) and not done:
                                done = True
                                stop_reason = stop_reason or "tour done"
                            if stop_reason is None and FATAL.search(line):
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


def verify_lifts(pass_dir, areas, opts, say):
    """tools/lift_verify.py over each place's captures: lifted against
    translated, byte for byte (drawreplay; this needs the GPU)."""
    tools = tool_paths(opts["bbhost"])
    out = {}
    for a in areas:
        cap = pass_dir / "captures" / a.id
        if not cap.is_dir() or not any((p / "manifest.json").is_file() for p in cap.iterdir()):
            continue
        dest = pass_dir / "lift" / a.id
        argv = [sys.executable, str(ROOT / "tools" / "lift_verify.py"), str(cap), "--out", str(dest),
                "--gcnlift", str(tools["gcnlift"]), "--drawreplay", str(tools["drawreplay"])]
        if not opts["keep_captures"]:
            argv.append("--delete-verified")
        say("%s: checking the lifts of its captures" % a.id)
        r = subprocess.run(argv, capture_output=True, text=True)
        (dest.parent).mkdir(parents=True, exist_ok=True)
        (pass_dir / "lift" / (a.id + ".txt")).write_text(r.stdout + r.stderr)
        try:
            s = json.loads((dest / "summary.json").read_text())
        except (OSError, ValueError):
            s = {"error": (r.stderr or r.stdout).strip().splitlines()[-1:] or ["no summary"]}
        out[a.id] = s
    return out


def finish_pass(pass_dir, areas, kind, launched, opts):
    lines = (pass_dir / "run.log").read_text(errors="replace").splitlines() if (pass_dir / "run.log").is_file() else []
    rep = pass_report(lines, areas, kind)
    rep.update(launched or {})
    if not opts.get("keep_data"):
        shutil.rmtree(pass_dir / "data", ignore_errors=True)
    return rep


def merge(perf, capture, lifts, areas):
    """The places' report from the two passes: the perf pass's numbers, the
    capture pass's captures and lift results."""
    out = {}
    for a in areas:
        ent = dict((perf or {}).get("areas", {}).get(a.id) or {"id": a.id, "name": a.name, "block": a.block, "status": "not run"})
        for name, f in (ent.get("frames") or {}).items():
            if f.get("ok") and Path(f["path"]).is_file():
                try:
                    f["stats"] = image_stats(read_ppm(f["path"]))
                except (OSError, ValueError) as exc:
                    f["error"] = str(exc)
        if capture:
            c = capture.get("areas", {}).get(a.id) or {}
            ent["capture_pass"] = {"status": c.get("status"), "arrival": c.get("arrival")}
            ent["captures"] = c.get("captures")
            if lifts and a.id in lifts:
                s = lifts[a.id]
                ent["lift"] = {"totals": s.get("totals", {}), "rejection_reasons": s.get("rejection_reasons", []),
                               "rows": [{k: r.get(k) for k in ("name", "status", "detail", "pipeline")} for r in s.get("rows", [])],
                               "error": s.get("error")}
        out[a.id] = ent
    return out


def problems_of(report):
    out = []
    for kind, p in report.get("passes", {}).items():
        if p.get("fatal"):
            out.append("%s pass: the game died (%s)" % (kind, p["fatal"][0][:120]))
        if not p.get("tour_done"):
            out.append("%s pass: the tour did not finish (%s)" % (kind, p.get("stop_reason")))
        size = report.get("run", {}).get("size")
        if size and p.get("rendering") and list(p["rendering"]) != list(size):
            out.append("%s pass: rendered at %dx%d, not the %dx%d asked for" % (kind, p["rendering"][0], p["rendering"][1], size[0], size[1]))
    for aid, a in report.get("areas", {}).items():
        if a.get("status") != "ok":
            out.append("%s: %s" % (aid, a.get("status")))
        cp = a.get("capture_pass")
        if cp and cp.get("status") != "ok":
            out.append("%s: %s in the capture pass" % (aid, cp.get("status")))
        tot = (a.get("lift") or {}).get("totals", {})
        if tot.get("different") or tot.get("invalid"):
            out.append("%s: %d lifted shaders render differently, %d invalid" % (aid, tot.get("different", 0), tot.get("invalid", 0)))
    return out


def resolve_options(args):
    """The run's options from the command line, checked; raises ValueError."""
    areas = pick_areas(args.areas)
    if args.config:
        app0, eboot = read_paths(args.config)
    else:
        app0, eboot = None, None
    app0 = args.app0 or app0
    eboot = args.eboot or eboot
    if not app0 or not (Path(app0) / "dvdroot_ps4").is_dir():
        raise ValueError("the game folder (--app0, or [paths] app0 in --config) has no dvdroot_ps4: %s" % app0)
    if not eboot or not Path(eboot).is_file():
        raise ValueError("no eboot (--eboot, or [paths] eboot in --config): %s" % eboot)
    seed = Path(args.seed).resolve()
    if not (seed / "saves").is_dir():
        raise ValueError("the seed %s has no saves/ folder (a data folder holding a character in the world)" % seed)
    numa = args.numa_node
    if numa == "auto":
        numa = 0 if shutil.which("numactl") else None
    elif numa == "none":
        numa = None
    else:
        numa = int(numa)
        if not shutil.which("numactl"):
            raise ValueError("--numa-node needs numactl")
    opts = dict(DEFAULTS)
    opts.update({"settle": args.settle, "measure": args.measure, "dumps": args.dumps, "quiet": args.quiet, "start": args.start,
                 "fps": args.fps, "captures": args.captures, "capture_window": args.capture_window,
                 "pipeline_cache": args.pipeline_cache == "seed", "p2p_port": args.p2p_port, "width": args.width,
                 "height": args.height, "env": args.env, "keep_data": args.keep_data, "keep_captures": args.keep_captures,
                 "numa_node": numa, "seed": str(seed)})
    if opts["measure"] < 5 or opts["settle"] < 0 or opts["dumps"] < 0 or opts["dumps"] > 8 or opts["captures"] < 0:
        raise ValueError("--measure must be 5 s or more, --settle and --captures not negative, --dumps 0-8")
    return areas, {"app0": str(Path(app0).resolve()), "eboot": str(Path(eboot).resolve())}, opts


def start_out(out):
    """A new output folder, or one a dry run left (which a real run replaces).
    No whitespace in it: the probe's requests to the host are space-separated."""
    out = Path(out).resolve()
    if any(c.isspace() for c in str(out)):
        raise ValueError("--out must not contain whitespace: %s" % out)
    if out.exists():
        if (out / "dry-run.json").is_file() and not (out / "report.json").exists():
            shutil.rmtree(out)
        elif any(out.iterdir()):
            raise ValueError("%s exists: give a new --out" % out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def do_run(args, bbhost=None, out=None, captures=True, say=print):
    """One build: the perf pass, the capture pass and its lift checks, the
    report. With --dry-run, everything up to starting the game."""
    areas, paths, opts = resolve_options(args)
    opts["bbhost"] = str(Path(bbhost or args.bbhost).resolve())
    out = start_out(out or args.out)
    seed = Path(opts["seed"])
    if out.is_relative_to(seed) or seed.is_relative_to(out):
        raise ValueError("--out and the seed must not contain each other")
    problems = []
    if not os.access(opts["bbhost"], os.X_OK):
        problems.append("%s is not an executable" % opts["bbhost"])
    eboot_sha = sha256_file(paths["eboot"])
    if eboot_sha != EBOOT_109_SHA256:
        problems.append("the eboot is not the 1.09 build (sha256 %s): the probe and the warps need it" % eboot_sha)
    do_captures = captures and opts["captures"] > 0
    if do_captures:
        for name, p in tool_paths(opts["bbhost"]).items():
            if not os.access(p, os.X_OK):
                problems.append("no %s beside the build (%s): the lift checks need it (or --captures 0)" % (name, p))
    if problems:
        raise ValueError("; ".join(problems))
    say("area check: %d places, %s" % (len(areas), ", ".join(a.id for a in areas)))
    opts["seed_files"] = world_baseline.file_inventory(seed)
    opts["probe"] = out / "probe" / "mapval.so"
    build_probe(opts["probe"])
    kinds = ["perf"] + (["capture"] if do_captures else [])
    plans = {k: prepare_pass(out / k, k, areas, opts, paths) for k in kinds}
    checks = check_areas(areas, paths["app0"])
    plan = {"schema": SCHEMA + "-plan", "areas": [a.as_dict() for a in areas], "area_checks": checks,
            "options": {k: v for k, v in opts.items() if k not in ("seed_files", "probe")}, "paths": paths,
            "bbhost_sha256": world_baseline.digest(Path(opts["bbhost"])), "passes": plans}
    (out / "plan.json").write_text(json.dumps(plan, indent=2, default=str) + "\n")
    bad = [c for c in checks if c["problems"]]
    for c in checks:
        extra = ("; " + check_note(c)) if check_note(c) else ""
        say("  %-19s lamp %s%s%s" % (c["id"], (c.get("lamp") or {}).get("row", "?"), extra,
                                      ("; PROBLEM: " + "; ".join(c["problems"])) if c["problems"] else ""))
    for k, p in plans.items():
        say("%s pass: about %.0f min (at most %.0f), %s" % (k, p["estimate_s"] / 60, p["budget_s"] / 60, " ".join(p["argv"])))
    if args.dry_run:
        for k in kinds:  # the copies were checked; a dry run leaves no 400 MB behind
            shutil.rmtree(out / k / "data", ignore_errors=True)
        (out / "dry-run.json").write_text(json.dumps({"ok": not bad, "area_problems": bad}, indent=2) + "\n")
        say("dry run: %s; plan in %s" % ("ready" if not bad else "PROBLEMS with %d places" % len(bad), out / "plan.json"))
        return 0 if not bad else 1
    if bad:
        raise ValueError("places with problems: " + "; ".join("%s: %s" % (c["id"], ", ".join(c["problems"])) for c in bad))
    report = {"schema": SCHEMA, "version": 1, "dir": str(out), "run": {
        "bbhost": opts["bbhost"], "bbhost_sha256": plan["bbhost_sha256"], "numa_node": opts["numa_node"], "fps": opts["fps"],
        "pipeline_cache": opts["pipeline_cache"], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(), "load_before": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "settle_s": opts["settle"], "measure_s": opts["measure"], "size": [opts["width"], opts["height"]]}, "passes": {}}
    passes, lifts = {}, None
    for k in kinds:
        t0 = time.monotonic()

        def say_t(msg, k=k, t0=t0):
            say("[%s %4.0f s] %s" % (k, time.monotonic() - t0, msg))
        say_t("starting (%s)" % (out / k))
        launched = launch(out / k, plans[k], say_t)
        passes[k] = finish_pass(out / k, areas, k, launched, opts)
        report["passes"][k] = {x: passes[k].get(x) for x in ("stop_reason", "returncode", "elapsed_s", "world", "tour_done", "fatal", "version",
                                                             "rendering")}
        say_t("%s after %.0f s" % (launched["stop_reason"], launched["elapsed_s"]))
        if k == "capture":
            lifts = verify_lifts(out / k, areas, opts, say_t)
    report["run"]["version"] = (passes.get("perf") or {}).get("version")
    report["run"]["load_after"] = os.getloadavg() if hasattr(os, "getloadavg") else None
    report["run"]["seed_unchanged"] = world_baseline.file_inventory(seed) == opts["seed_files"]
    report["areas"] = merge(passes.get("perf"), passes.get("capture"), lifts, areas)
    report["problems"] = problems_of(report)
    if not report["run"]["seed_unchanged"]:
        report["problems"].append("the seed changed during the run")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    text = summary_text(report)
    (out / "summary.txt").write_text(text)
    say(text)
    return 1 if report["problems"] else 0


def do_report(args):
    """The report again from a run's logs (after a change to this script)."""
    run = Path(args.run).resolve()
    plan = json.loads((run / "plan.json").read_text())
    areas = [AREA_BY_ID[a["id"]] for a in plan["areas"] if a["id"] in AREA_BY_ID]
    old = {}
    if (run / "report.json").is_file():
        old = json.loads((run / "report.json").read_text())
    passes = {}
    for k in ("perf", "capture"):
        if (run / k / "run.log").is_file():
            passes[k] = pass_report((run / k / "run.log").read_text(errors="replace").splitlines(), areas, k)
            passes[k].update({x: (old.get("passes", {}).get(k) or {}).get(x) for x in ("stop_reason", "returncode", "elapsed_s")})
    lifts = {}
    for a in areas:
        p = run / "capture" / "lift" / a.id / "summary.json"
        if p.is_file():
            lifts[a.id] = json.loads(p.read_text())
    report = {"schema": SCHEMA, "version": 1, "dir": str(run), "run": old.get("run", {}), "passes": {
        k: {x: v.get(x) for x in ("stop_reason", "returncode", "elapsed_s", "world", "tour_done", "fatal", "version", "rendering")}
        for k, v in passes.items()}}
    report["areas"] = merge(passes.get("perf"), passes.get("capture"), lifts, areas)
    report["problems"] = problems_of(report)
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    text = summary_text(report)
    (run / "summary.txt").write_text(text)
    print(text)
    return 1 if report["problems"] else 0


def load_report(run):
    rep = json.loads((Path(run) / "report.json").read_text())
    rep["dir"] = str(Path(run).resolve())
    return rep


def do_compare(a_dirs, b_dirs, out, say=print):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    a = [load_report(d) for d in a_dirs]
    b = [load_report(d) for d in b_dirs]
    ids = list(dict.fromkeys(k for r in a + b for k in r.get("areas", {})))
    frames = compare_frames(a_dirs, b_dirs, ids, out)
    ab = compare_reports(a, b, frames)
    (out / "ab.json").write_text(json.dumps(ab, indent=2) + "\n")
    text = ab_text(ab)
    (out / "ab.txt").write_text(text)
    say(text)
    return 1 if ab["flagged"] else 0


def do_ab(args):
    """A and B alternately, --rounds rounds; captures in the first round."""
    out = start_out(args.out)
    rounds = max(1, args.rounds)
    a_dirs, b_dirs = [], []
    status = 0
    for r in range(1, rounds + 1):
        for arm, binary, dirs in (("a", args.a, a_dirs), ("b", args.b, b_dirs)):
            d = out / ("%s%d" % (arm, r))
            print("=== %s round %d: %s" % (arm.upper(), r, binary), flush=True)
            rc = do_run(args, bbhost=binary, out=d, captures=(r == 1), say=lambda m: print(m, flush=True))
            status |= rc
            dirs.append(d)
        if args.dry_run:
            return status
    return do_compare(a_dirs, b_dirs, out) | status


def do_areas(args):
    areas = pick_areas(args.areas)
    for a in areas:
        v = a.view()
        print("%-19s %s, %s (0x%s), lamp %d%s%s" % (a.id, a.name, a.block, a.block_hex(), a.return_point,
                                                    (", view %.2f %.2f %.2f yaw %.3f" % v) if v else "", ", The Old Hunters" if a.dlc else ""))
    if args.app0:
        bad = 0
        for c in check_areas(areas, args.app0):
            bad += bool(c["problems"])
            print("  %-19s %s" % (c["id"], "; ".join(c["problems"]) or ("ok" + (" (%s)" % check_note(c) if check_note(c) else ""))))
        return 1 if bad else 0
    return 0


def add_run_options(p):
    p.add_argument("--config", help="a test config: its [paths] app0 and eboot are used (nothing else of it)")
    p.add_argument("--app0", help="the game folder (the one that contains dvdroot_ps4)")
    p.add_argument("--eboot", help="the 1.09 eboot")
    p.add_argument("--seed", required=True, help="the data folder to copy for each pass (its saves/ hold the character)")
    p.add_argument("--out", required=True, help="a new folder for the run")
    p.add_argument("--areas", default="all", help="comma list of places (ids or prefixes), in order; default all")
    p.add_argument("--settle", type=float, default=DEFAULTS["settle"], help="seconds at a place before the window")
    p.add_argument("--measure", type=float, default=DEFAULTS["measure"], help="seconds of the window")
    p.add_argument("--dumps", type=int, default=DEFAULTS["dumps"], help="frames dumped after the window, 2 s apart")
    p.add_argument("--captures", type=int, default=DEFAULTS["captures"],
                   help="pixel shaders captured at each place for the lift checks; 0: no capture pass")
    p.add_argument("--capture-window", type=float, default=DEFAULTS["capture_window"], help="seconds the captures may take at a place")
    p.add_argument("--quiet", type=float, default=QUIET_RADIUS, help="metres around the view whose enemies are switched off; 0: none")
    p.add_argument("--start", type=float, default=DEFAULTS["start"], help="seconds in the world before the first travel")
    p.add_argument("--fps", type=int, default=DEFAULTS["fps"], choices=(30, 60, 90), help="the game's frame rate (BBHOST_GAME_FPS)")
    p.add_argument("--pipeline-cache", choices=("seed", "off"), default="seed",
                   help="seed: the seed's shader caches and stage manifest, as a player's after an update (default); "
                        "off: BBHOST_PIPELINE_CACHE=0, every shader compiled when first met")
    p.add_argument("--numa-node", default="auto", help="pin the game to this NUMA node (numactl); auto: node 0 when numactl exists; none")
    p.add_argument("--p2p-port", type=int, default=9446, help="the run's own P2P port (never the one a player's game uses)")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--env", action="append", default=[], metavar="BBHOST_NAME=VALUE", help="another setting for the game (repeatable)")
    p.add_argument("--keep-data", action="store_true", help="keep each pass's copy of the seed")
    p.add_argument("--keep-captures", action="store_true", help="keep captures whose lift was shown identical (~50 MB each)")
    p.add_argument("--dry-run", action="store_true", help="check and prepare everything, start nothing")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="one build")
    p.add_argument("--bbhost", required=True, help="the build's bbhost (gcnlift and drawreplay beside it)")
    add_run_options(p)
    p = sub.add_parser("ab", help="two builds, alternately, then compared")
    p.add_argument("--a", required=True, help="build A's bbhost")
    p.add_argument("--b", required=True, help="build B's bbhost")
    p.add_argument("--rounds", type=int, default=2, help="runs of each build (2+: a noise floor)")
    add_run_options(p)
    p = sub.add_parser("compare", help="compare finished runs")
    p.add_argument("--a", nargs="+", required=True, help="build A's run folders")
    p.add_argument("--b", nargs="+", required=True, help="build B's run folders")
    p.add_argument("--out", required=True)
    p = sub.add_parser("report", help="a run's report again from its logs")
    p.add_argument("run")
    p = sub.add_parser("areas", help="list the places (and check them against the game's data)")
    p.add_argument("--app0")
    p.add_argument("--areas", default="all")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "run":
            return do_run(args)
        if args.cmd == "ab":
            return do_ab(args)
        if args.cmd == "compare":
            return do_compare(args.a, args.b, args.out)
        if args.cmd == "report":
            return do_report(args)
        return do_areas(args)
    except (ValueError, OSError, KeyError) as exc:
        print("area_check: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
