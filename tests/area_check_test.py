"""tools/area_check.py's pure parts: the tours, the log's reading (lines made
with the host's own format strings), the window statistics, the frame
comparison and the A/B verdicts. No game, no GPU."""

import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("area_check", Path(__file__).resolve().parents[1] / "tools/area_check.py")
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)


# The host's lines, with the format strings they are printed with.
def frames_line(fps=60, intervals=None, gpu_busy=420, work=8.1):
    iv = intervals or [1000.0 / fps] * fps
    n = len(iv)
    s = sorted(iv)
    return ("[bbhost] frames: %.2f s: %d flips (%.1f/s), displayed %.1f/s, interval avg %.1f p95 %.1f max %.1f ms, %d over 16.7, "
            "%d over 33.3; cpu %.0f%%:%s; main loop waited %.0f ms, slept %.0f ms, file I/O %.0f ms%s%s") % (
        sum(iv) / 1000.0, n, n / (sum(iv) / 1000.0), n / (sum(iv) / 1000.0), sum(iv) / n, s[int(0.95 * n)], s[-1],
        sum(1 for v in iv if v > 17.5), sum(1 for v in iv if v > 34.5), 250, " main:101 90% bb-cp0:102 60%", 100, 500, 0,
        "; main loop work avg %.1f p95 %.1f max %.1f ms, %d over 16.7" % (work, work + 1, work + 3, 0),
        "; gpu busy %.0f ms/s (%.0f%%)" % (gpu_busy, gpu_busy / 10.0))


def detail_lines(intervals, draws=(6000, 2500, 400, 100), failures=None):
    failures = failures or {}
    why = ""
    for k, v in failures.items():
        why += (" (" if not why else ", ") + "%s %d" % (k, v)
    if why:
        why += ")"
    out = ["[bbhost] frame detail: draws %d: pixel shader lifted %d, translated %d, fallback %d, none %d; draw failures %d%s" % (
        sum(draws), draws[0], draws[1], draws[2], draws[3], sum(failures.values()), why)]
    for at in range(0, len(intervals), 100):
        out.append("[bbhost] frame intervals ms:" + "".join(" %.2f" % v for v in intervals[at:at + 100]))
    return out


def second(intervals, **kw):
    return [frames_line(intervals=intervals, **{k: v for k, v in kw.items() if k in ("gpu_busy", "work")})] + \
        detail_lines(intervals, **{k: v for k, v in kw.items() if k in ("draws", "failures")})


def mapval(fmt, *args):
    """A probe line as bbhost logs a plugin's."""
    return "[bbhost] plugin: " + (fmt % args)


def area_lines(area, t, block, window_intervals, view=None, settle_events=(), dump_dir="frames", kind="perf", captures=()):
    """A place's lines as the probe and the host print them."""
    out = [mapval("mapval step %d travel %s %.3f return point %d block %s -> %s", 1, area.id, t, area.return_point, block, "queued"),
           "[bbhost] loading: begins, flip 1000", "[bbhost] loading: %.2f s (flips %d-%d, %d of them not shown)" % (3.25, 1000, 1200, 10),
           mapval("mapval travel-done %s %.3f %s %.3f %.3f %.3f after %.1f s", area.id, t + 6, block, 1.0, 2.0, 3.0, 6.0)]
    if view:
        out.append(mapval("mapval step %d warp %s %.3f target %.3f %.3f %.3f yaw %.3f -> %s", 2, area.id + ":view", t + 6, *view[:3], view[3], "queued"))
        out.append(mapval("mapval at %s %.3f %s %.3f %.3f %.3f yaw %.3f", area.id + ":view", t + 7.5, block, view[0], view[1], view[2], view[3]))
    out.append(mapval("mapval quiet %.3f %d enemies within %.0f m switched off, %d other characters left", t + 8, 3, 40.0, 1))
    out.append(mapval("mapval camera %.3f %s", t + 8, "reset"))
    out.append(mapval("mapval mark %s %.3f %s %.3f %.3f %.3f yaw %.3f", area.id + ":arrived", t + 8, block, 1.0, 2.0, 3.0, 0.5))
    out.extend(second([16.67] * 60))  # a settle second
    out.extend(settle_events)
    if kind == "perf":
        out.append(mapval("mapval mark %s %.3f %s %.3f %.3f %.3f yaw %.3f", area.id + ":measure", t + 23, block, 1.0, 2.0, 3.0, 0.5))
        for iv in window_intervals:
            out.extend(second(iv))
        out.append(mapval("mapval mark %s %.3f %s %.3f %.3f %.3f yaw %.3f", area.id + ":measured", t + 43, block, 1.0, 2.0, 3.0, 0.5))
        for k in "ab":
            out.append(mapval("mapval dump %s %.3f", "%s-%s" % (area.id, k), t + 44))
            out.append("[bbhost] test: frame dump at flip %d to %s/%s-%s.ppm" % (5000, dump_dir, area.id, k))
    else:
        out.append(mapval("mapval capture %d %s %.3f", 6, area.id, t + 18))
        out.append("[bbhost] test: draw captures armed for %d at flip %d into %s" % (6, 4000, "/c/" + area.id))
        out.append(mapval("mapval mark %s %.3f %s %.3f %.3f %.3f yaw %.3f", area.id + ":capture", t + 18, block, 1.0, 2.0, 3.0, 0.5))
        for name in captures:
            out.append("[bbhost] capture: wrote %s/manifest.json (%s, flip %d, draw %d, %d blobs, %d bytes)" % (
                "/c/%s/%s-f4001-d9" % (area.id, name), name, 4001, 9, 40, 50000000))
        out.append("[bbhost] capture: %s draw %d (flip %d) not captured: %s" % ("aaaa0000+bbbb0000", 12, 4002, "indirect draw: its arguments are GPU data"))
    out.append(mapval("mapval mark %s %.3f %s %.3f %.3f %.3f yaw %.3f", area.id + ":end", t + 47, block, 1.0, 2.0, 3.0, 0.5))
    return out


class TourTest(unittest.TestCase):
    def test_places_and_views(self):
        self.assertEqual(len(ac.AREAS), 10)
        self.assertEqual(len({a.id for a in ac.AREAS}), 10)
        cy = ac.AREA_BY_ID["central-yharnam"]
        self.assertEqual(cy.block_hex(), "18010000")
        self.assertIsNone(cy.view())
        rom = ac.AREA_BY_ID["moonside-lake"]
        x, y, z, yaw = rom.view()
        fx, fy, fz, deg = rom.fog
        # Behind the wall by the stand-off, facing as the wall does.
        self.assertAlmostEqual(math.hypot(x - fx, z - fz), ac.FOG_STANDOFF, places=5)
        self.assertAlmostEqual(yaw, math.radians(deg))
        self.assertAlmostEqual(x - fx, math.sin(yaw) * ac.FOG_STANDOFF)
        self.assertTrue(all(a.dlc for a in ac.AREAS if a.block[:3] in ("m34", "m35", "m36")))

    def test_pick_areas(self):
        self.assertEqual([a.id for a in ac.pick_areas("all")], [a.id for a in ac.AREAS])
        self.assertEqual([a.id for a in ac.pick_areas("old,central")], ["old-yharnam", "central-yharnam"])
        with self.assertRaises(ValueError):
            ac.pick_areas("hunters")  # the Dream or the Nightmare
        with self.assertRaises(ValueError):
            ac.pick_areas("nowhere")
        with self.assertRaises(ValueError):
            ac.pick_areas("central,central-yharnam")

    def test_perf_tour(self):
        areas = ac.pick_areas("central-yharnam,moonside-lake")
        text = ac.tour(areas, dict(ac.DEFAULTS), "perf")
        steps = [l.split()[0] for l in text.splitlines() if l and not l.startswith("#")]
        # Each place begins with its travel: what a failed travel skips to.
        self.assertEqual(steps.count("travel"), 2)
        first = text.splitlines().index(next(l for l in text.splitlines() if l.startswith("travel 2412951")))
        self.assertIn("travel 2412951 18010000 central-yharnam", text)
        self.assertIn("travel 3202950 20000000 moonside-lake 13200040 13200120\n", text)
        self.assertIn("warp moonside-lake:view", text)
        self.assertNotIn("warp central-yharnam:view", text)
        self.assertIn("mark central-yharnam:measure", text)
        self.assertIn("dump central-yharnam-a", text)
        self.assertIn("dump central-yharnam-b", text)
        self.assertNotIn("capture", text)
        self.assertGreater(first, 0)
        # The window is the measure hold.
        lines = text.splitlines()
        k = lines.index("mark central-yharnam:measure")
        self.assertEqual(lines[k + 1], "hold %.1f" % ac.DEFAULTS["measure"])
        self.assertEqual(lines[k + 2], "mark central-yharnam:measured")

    def test_capture_tour_and_settings(self):
        areas = ac.pick_areas("hunters-nightmare")
        opts = dict(ac.DEFAULTS, pipeline_cache=True, env=[])
        text = ac.tour(areas, opts, "capture")
        self.assertIn("capture 6 hunters-nightmare", text)
        self.assertIn("capture 0", text)
        self.assertNotIn("dump", text)
        s = ac.pass_settings(Path("/r/capture"), areas, opts, "capture")
        self.assertEqual(s["BBHOST_CAPTURE_DRAW"], "*ps")
        self.assertEqual(s["BBHOST_CAPTURE_ARMED"], "1")
        self.assertEqual((s["BBHOST_FRAME_STATS"], s["BBHOST_FRAME_DETAIL"]), ("1", "1"))
        self.assertEqual(s["BBHOST_TEST_REQUESTS"], "/r/capture/requests")
        perf = ac.pass_settings(Path("/r/perf"), ac.pick_areas("central-yharnam"), opts, "perf")
        self.assertNotIn("BBHOST_CAPTURE_DRAW", perf)
        with self.assertRaises(ValueError):
            ac.pass_settings(Path("/r"), areas, dict(opts, env=["BBHOST_FRAME_STATS=1"]), "perf")
        with self.assertRaises(ValueError):
            ac.pass_settings(Path("/r"), areas, dict(opts, env=["PATH=/x"]), "perf")
        self.assertEqual(ac.pass_settings(Path("/r"), areas, dict(opts, env=["BBHOST_X=1"]), "perf")["BBHOST_X"], "1")
        # No presses at fixed times: the title is passed with menutap requests (launch).
        self.assertNotIn("BBHOST_AUTOPRESS", perf)

    def test_title_taps_until_the_world(self):
        """launch() asks for the menus' confirm button until the world's first
        frame, and no more after it: a stand-in game answers two requests, then
        logs the world and the end of the tour."""
        game = textwrap.dedent("""
            import os, sys, time
            req = sys.argv[1]
            seen = 0
            t0 = time.monotonic()
            while time.monotonic() - t0 < 20:
                for name in sorted(os.listdir(req)):
                    path = os.path.join(req, name)
                    print("[bbhost] test: " + open(path).read().strip() + " from " + name, flush=True)
                    os.remove(path)
                    seen += 1
                if seen >= 2:
                    print("[bbhost] world: the first in-game frame, flip 1200, 31.0 s after start", flush=True)
                    time.sleep(0.5)
                    leftover = sorted(os.listdir(req))
                    print("[bbhost] leftover %d" % len(leftover), flush=True)
                    print("[bbhost] plugin: mapval tour-done 1.000", flush=True)
                    time.sleep(0.2)
                    sys.exit(0)
                time.sleep(0.02)
            sys.exit(3)
        """)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "requests").mkdir()
            (d / "game.py").write_text(game)
            plan = {"argv": [sys.executable, str(d / "game.py"), str(d / "requests")], "budget_s": 30,
                    "settings": {"BBHOST_TEST_REQUESTS": str(d / "requests")}}
            with mock.patch.object(ac, "TITLE_TAP_FIRST_S", 0.1), mock.patch.object(ac, "TITLE_TAP_EVERY_S", 0.5):
                out = ac.launch(d, plan, lambda *_: None)
            log = (d / "run.log").read_text()
        self.assertEqual(out["stop_reason"], "tour done")
        self.assertIn("test: menutap confirm from title-001.req", log)
        self.assertIn("test: menutap confirm from title-002.req", log)
        # Nothing asked for after the world's first frame.
        self.assertIn("leftover 0", log)
        self.assertNotIn("title-003", log)

    def test_run_toml(self):
        import tomllib
        t = tomllib.loads(ac.run_toml("/g/CUSA00900", "/e/eboot.bin", Path("/r/perf/data"), 9446, 1920, 1080))
        self.assertEqual(t["paths"], {"app0": "/g/CUSA00900", "data": "/r/perf/data", "eboot": "/e/eboot.bin"})
        self.assertEqual(t["online"], {"host": "127.0.0.1", "p2p_port": 9446})
        self.assertFalse(t["plugins"]["hello"])
        self.assertNotIn("patches", t)
        dlc = tomllib.loads(ac.run_toml("/g", "/e", Path("/d"), 9446, 1920, 1080, old_hunters=True))
        self.assertIs(dlc["patches"]["old-hunters"], True)
        # As bbhost reads it: what is between the quotes, as it is.
        self.assertEqual(ac.toml_str("/jeux/Bloodborne é"), '"/jeux/Bloodborne é"')
        for bad in ('/a"b', "C:\\game", "/a\nb"):
            with self.assertRaises(ValueError):
                ac.toml_str(bad)


class LogTest(unittest.TestCase):
    def run_lines(self):
        cy, rom = ac.AREA_BY_ID["central-yharnam"], ac.AREA_BY_ID["moonside-lake"]
        rng = random.Random(7)
        window = [[16.67 + rng.uniform(-0.3, 0.3) for _ in range(60)] for _ in range(20)]
        window[5][10] = 40.0  # one hitch
        lines = ["[bbhost] bbhost v0.2.16-14-g2291ffe+ (2291ffe+, Linux x86-64)",
                 "[bbhost] world: the first in-game frame, flip %d, %.1f s after start" % (1500, 31.0),
                 "[bbhost] plugin: mapval tour-start %.3f" % 100.0]
        lines += area_lines(cy, 100.0, "18010000", window, settle_events=[
            "[bbhost] render: PS a22c7f71 lifted (%d words; translated %d)" % (900, 2000),
            "[bbhost] render: PS 6c5edb96 not lifted, translated shader kept: %s" % "0x01c: ds_swizzle_b32 is not lifted",
            "[bbhost] render: PS 7f03e945 not lifted, translated shader kept: %s" % "0x2a0: ds_swizzle_b32 is not lifted",
            "[bbhost] render: slow pipeline %s%s on %s: vkCreateGraphicsPipelines %d ms; SPIR-V words VS %d, PS %d%s; VS params %d, "
            "PS images %d, buffers VS %d PS %d" % ("8747a367+a22c7f71", " (no-fallback)", "a worker", 120, 3000, 2000, "", 4, 6, 1, 2),
            "[bbhost] stall: flip #%d came %d ms after the previous one (cp-backlog %d); in between cp-thread ms: draw=1; graphics "
            "pipelines 1, draws 1, dispatches 1, texture uploads 1, texture hashes 1 in 1 ms, surfaces replaced 0; texture ms: images 0, "
            "staging 0, untile 0, record 0, views 0; pipeline ms: translate %d, modules %d, create %d; stage translations 1, stages reused 1"
            % (1234, 85, 2, 10, 5, 50),
            "[bbhost] main loop: a %.1f ms frame before flip %d, %.1f ms of it waiting" % (52.0, 1300, 40.0),
            "[bbhost] render: pipeline %s failed to build; its draws are skipped (first at flip %d)" % ("1111+2222", 1300)])
        view = rom.view()
        lines += area_lines(rom, 160.0, "20000000", [[16.7] * 60 for _ in range(20)], view=view)
        lines.append("[bbhost] plugin: mapval tour-done %.3f" % 220.0)
        return lines, window

    def test_probe_lines_with_and_without_the_plugin_prefix(self):
        line = "mapval mark central-yharnam:arrived 108.000 18010000 1.000 2.000 3.000 yaw 0.500"
        for prefix in ("[bbhost] plugin: ", "[bbhost] "):
            m = ac.MARK.search(prefix + line)
            self.assertIsNotNone(m, prefix)
            self.assertEqual((m.group(1), m.group(2)), ("central-yharnam", "arrived"))
        self.assertTrue(ac.TOUR_DONE.search("[bbhost] plugin: mapval tour-done 1.000"))

    def test_places_from_the_log(self):
        lines, window = self.run_lines()
        rep = ac.pass_report(lines, ac.pick_areas("central-yharnam,moonside-lake,hunters-dream"), "perf")
        self.assertTrue(rep["tour_done"])
        self.assertEqual(rep["version"], "v0.2.16-14-g2291ffe+")
        self.assertEqual(rep["world"], {"flip": 1500, "seconds": 31.0})
        cy = rep["areas"]["central-yharnam"]
        self.assertEqual(cy["status"], "ok")
        self.assertEqual(cy["travel"]["seconds"], 6.0)
        self.assertEqual(cy["quiet"], 3)
        w = cy["window"]
        # The first second after the mark straddles it and is left out.
        kept = [v for iv in window[1:] for v in iv]
        self.assertEqual(w["flips"], len(kept))
        self.assertTrue(w["exact_percentiles"])
        s = sorted(round(v, 2) for v in kept)
        self.assertEqual(w["frame_ms"]["p50"], ac.percentile(s, 50))
        self.assertEqual(w["frame_ms"]["max"], 40.0)
        self.assertEqual(w["frame_ms"]["p99"], ac.percentile(s, 99))
        self.assertEqual(w["over_33_3"], 1)
        self.assertAlmostEqual(w["fps"], len(kept) / (sum(s) / 1000.0), places=1)
        self.assertEqual(w["gpu_busy_ms_per_s"]["mean"], 420)
        self.assertEqual(w["draws"]["lifted"], 6000 * 19)
        self.assertAlmostEqual(w["draws"]["lifted_share"], 6000 / 8900, places=4)
        self.assertEqual(w["draws"]["failures"], 0)
        ar = cy["arrival"]
        self.assertEqual(ar["loads"], [3.25])
        self.assertEqual(ar["slow_pipelines"], {"count": 1, "ms": 120, "max_ms": 120})
        self.assertEqual(ar["stalls"]["count"], 1)
        self.assertEqual(ar["stalls"]["pipeline_ms"], 65)
        self.assertEqual(ar["long_frames"]["over_50"], 1)
        self.assertEqual(ar["pipelines_failed"], ["1111+2222"])
        nps = ar["new_pixel_shaders"]
        self.assertEqual((nps["lifted"], nps["translated"]), (1, 2))
        self.assertEqual(nps["reasons"], {"ds_swizzle_b32 is not lifted": 2})
        self.assertEqual(sorted(cy["frames"]), ["a", "b"])
        self.assertTrue(cy["frames"]["a"]["path"].endswith("central-yharnam-a.ppm"))
        rom = rep["areas"]["moonside-lake"]
        self.assertEqual(rom["status"], "ok")
        self.assertTrue(rom["view"]["reached"])
        # Nothing of Central Yharnam's spills into the next place.
        self.assertEqual(rom["arrival"]["slow_pipelines"]["count"], 0)
        self.assertEqual(rep["areas"]["hunters-dream"]["status"], "not reached")

    def test_failed_travel_and_missing_view(self):
        cy, rom = ac.AREA_BY_ID["central-yharnam"], ac.AREA_BY_ID["moonside-lake"]
        lines = [mapval("mapval step %d travel %s %.3f return point %d block %s -> %s", 1, rom.id, 10.0, rom.return_point, "20000000", "queued"),
                 mapval("mapval travel-failed %s %.3f in %s after %.1f s", rom.id, 160.0, "15000000", 150.0)]
        lines += area_lines(cy, 170.0, "18010000", [[16.7] * 60 for _ in range(5)])
        rep = ac.pass_report(lines, [rom, cy], "perf")
        self.assertEqual(rep["areas"]["moonside-lake"]["status"], "travel failed")
        self.assertEqual(rep["areas"]["central-yharnam"]["status"], "ok")
        self.assertFalse(rep["tour_done"])
        # A view that was never reached.
        lines = area_lines(rom, 10.0, "20000000", [[16.7] * 60 for _ in range(5)])
        lines = [l for l in lines if "mapval at " not in l]
        self.assertEqual(ac.pass_report(lines, [rom], "perf")["areas"]["moonside-lake"]["status"], "view not reached")

    def test_capture_pass(self):
        cy = ac.AREA_BY_ID["central-yharnam"]
        lines = area_lines(cy, 10.0, "18010000", [], kind="capture", captures=["8747a367+a22c7f71", "1234abcd+6c5edb96"])
        rep = ac.pass_report(lines, [cy], "capture")
        c = rep["areas"]["central-yharnam"]
        self.assertEqual(c["status"], "ok")
        self.assertEqual(len(c["captures"]["written"]), 2)
        self.assertEqual(c["captures"]["rejected"][0]["why"], "indirect draw: its arguments are GPU data")
        self.assertNotIn("window", c)
        self.assertEqual(ac.capture_ps("1234abcd+6c5edb96-f4001-d9"), "6c5edb96")

    def test_approximate_without_intervals(self):
        recs = [ac.Log([frames_line()]).seconds[0][1] for _ in range(3)]
        w = ac.window_stats(recs)
        self.assertFalse(w["exact_percentiles"])
        self.assertEqual(w["flips"], 180)
        self.assertNotIn("draws", w)

    def test_failure_reasons(self):
        log = ac.Log(second([16.7] * 60, failures={"pixel shader": 2, "descriptor set": 1}))
        d = log.seconds[0][1]["detail"]
        self.assertEqual(d["failures"], 3)
        self.assertEqual(d["reasons"], {"pixel shader": 2, "descriptor set": 1})
        self.assertEqual(ac.reason_key("0x01c: ds_swizzle_b32 is not lifted"), "ds_swizzle_b32 is not lifted")
        self.assertEqual(ac.reason_key("12: backward s_branch"), "backward s_branch")

    def test_percentile(self):
        s = list(range(1, 101))
        self.assertEqual(ac.percentile(s, 50), 50)
        self.assertEqual(ac.percentile(s, 95), 95)
        self.assertEqual(ac.percentile(s, 100), 100)
        self.assertEqual(ac.percentile([5.0], 99), 5.0)
        self.assertIsNone(ac.percentile([], 50))


def image(w, h, fill, seed=None, spot=None):
    rng = random.Random(seed)
    px = bytearray()
    for y in range(h):
        for x in range(w):
            v = fill
            if seed is not None:
                v = max(0, min(255, fill + rng.randint(-3, 3)))
            if spot and spot[0] <= x < spot[2] and spot[1] <= y < spot[3]:
                v = 255
            px += bytes((v, v, v))
    return ac.Image(w, h, bytes(px))


class FrameTest(unittest.TestCase):
    def test_ppm_round_trip(self):
        img = image(5, 3, 40)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.ppm"
            Path(p).write_bytes(b"P6\n# a comment\n5 3\n255\n" + img.rgb)
            back = ac.read_ppm(p)
            self.assertEqual((back.width, back.height, back.rgb), (5, 3, img.rgb))
            ac.write_ppm(p, img)
            self.assertEqual(ac.read_ppm(p).rgb, img.rgb)
        self.assertEqual(ac.image_stats(img)["mean_luma"], 40.0)
        with tempfile.TemporaryDirectory() as d:
            for bad in (b"", b"P6\n5 3", b"P6\n5 3\n255\n" + img.rgb[:10]):
                (Path(d) / "bad.ppm").write_bytes(bad)
                with self.assertRaises(ValueError):
                    ac.read_ppm(Path(d) / "bad.ppm")

    def test_identical_and_changed(self):
        a = image(32, 32, 100)
        same = ac.compare_images(a, a)
        self.assertEqual((same["changed_pct"], same["max_block"], same["blurred_pct"]), (0.0, 0.0, 0.0))
        b = image(32, 32, 100, spot=(0, 0, 8, 8))  # one corner block white
        c = ac.compare_images(a, b)
        self.assertAlmostEqual(c["changed_pct"], 100.0 * 64 / 1024, places=3)
        self.assertAlmostEqual(c["max_block"], 155.0, places=1)
        self.assertGreater(c["blurred_pct"], 5.0)
        self.assertAlmostEqual(c["mean_signed"], 155.0 * 64 / 1024, places=2)
        self.assertEqual(ac.compare_images(a, image(16, 16, 100))["error"][:11], "sizes diffe")

    def test_pure_and_numpy_agree(self):
        if ac.np is None:
            self.skipTest("no numpy")
        a, b = image(40, 24, 90, seed=1), image(40, 24, 95, seed=2, spot=(30, 10, 40, 24))
        fast = ac.compare_images(a, b)
        saved = ac.np
        try:
            ac.np = None
            slow = ac.compare_images(a, b)
        finally:
            ac.np = saved
        for k in ("changed_pct", "mean_abs", "mean_signed", "max_block", "blurred_pct"):
            self.assertAlmostEqual(fast[k], slow[k], places=2, msg=k)
        self.assertEqual(len(fast["blocks"]), ac.GRID)
        tiny = ac.compare_images(image(3, 2, 10), image(3, 2, 20))  # smaller than the grid: a block a pixel
        self.assertEqual((len(tiny["blocks"]), len(tiny["blocks"][0])), (2, 3))
        self.assertAlmostEqual(tiny["max_block"], 10.0, places=3)

    def test_verdict_against_the_noise_floor(self):
        noise = [{"changed_pct": 12.0, "max_block": 3.0, "blurred_pct": 1.0, "mean_signed": 0.4}]
        within = [{"changed_pct": 14.0, "max_block": 3.5, "blurred_pct": 1.2, "mean_signed": 0.5}]
        beyond = [{"changed_pct": 14.0, "max_block": 25.0, "blurred_pct": 9.0, "mean_signed": 0.5}]
        self.assertEqual(ac.frame_verdict(within, noise)["verdict"], "within noise")
        v = ac.frame_verdict(beyond, noise)
        self.assertEqual(v["verdict"], "differs")
        self.assertEqual(v["beyond"], ["blurred_pct", "max_block"])
        self.assertEqual(ac.frame_verdict(beyond, [])["verdict"], "no noise floor")
        self.assertEqual(ac.frame_verdict([], noise)["verdict"], "not compared")


class CompareTest(unittest.TestCase):
    def report(self, fps, p95, busy, lifted_share, status="ok", lift=None, failures=0):
        w = {"fps": fps, "frame_ms": {"p50": 16.7, "p95": p95, "p99": p95 + 1, "max": 20.0}, "over_33_3": 0,
             "gpu_busy_ms_per_s": {"mean": busy}, "gpu_ms_per_flip": busy / fps, "main_loop_work_ms": {"mean": 8.0}, "cpu_pct": 250,
             "draws": {"lifted_share": lifted_share, "failures": failures}}
        a = {"status": status, "window": w, "arrival": {"slow_pipelines": {"count": 3, "ms": 300}, "stalls": {"count": 1},
                                                        "long_frames": {"count": 0}, "draw_failures": 0}}
        if lift:
            a["lift"] = lift
        return {"dir": "x", "version": "v1", "areas": {"central-yharnam": a}}

    def test_noise_floor(self):
        d = ac.metric_delta([59.8, 60.0], [59.9, 59.95], True, 0.3)
        self.assertEqual(d["verdict"], "within noise")
        d = ac.metric_delta([59.8, 60.0], [55.0, 55.2], True, 0.3)
        self.assertEqual(d["verdict"], "worse")
        self.assertAlmostEqual(d["delta"], -4.8)
        d = ac.metric_delta([17.0, 17.1], [16.0, 16.1], False, 0.3)
        self.assertEqual(d["verdict"], "better")
        self.assertEqual(ac.metric_delta([60.0], [55.0], True, 0.3)["verdict"], "no noise floor")
        self.assertEqual(ac.metric_delta([60.0], [59.9], True, 0.3)["verdict"], "same")
        self.assertIsNone(ac.metric_delta([None], [60.0], True, 0.3))

    def test_reports(self):
        lift_a = {"totals": {"identical": 3, "rejected": 2}, "rows": [{"name": "1+aaaa0001-f1-d1", "status": "rejected"},
                                                                    {"name": "1+aaaa0002-f1-d2", "status": "identical"}]}
        lift_b = {"totals": {"identical": 4, "different": 1}, "rows": [{"name": "1+aaaa0001-f9-d1", "status": "identical"},
                                                                      {"name": "1+aaaa0002-f9-d2", "status": "different"}]}
        a = [self.report(60.0, 17.0, 400, 0.70, lift=lift_a), self.report(59.9, 17.1, 410, 0.70)]
        b = [self.report(55.0, 19.0, 300, 0.80, lift=lift_b, failures=2), self.report(55.1, 19.2, 305, 0.80, failures=2)]
        ab = ac.compare_reports(a, b)
        e = ab["areas"]["central-yharnam"]
        self.assertEqual(e["perf"]["fps"]["verdict"], "worse")
        self.assertEqual(e["perf"]["GPU busy ms/s"]["verdict"], "better")
        self.assertEqual(e["lifted_share"]["b_mean"], 0.8)
        self.assertEqual(e["lift"]["changed"], {"aaaa0001": {"a": "rejected", "b": "identical"},
                                                "aaaa0002": {"a": "identical", "b": "different"}})
        self.assertEqual(e["lift"]["b_different"], ["aaaa0002"])
        text = "\n".join(ab["flagged"])
        self.assertIn("fps worse", text)
        self.assertIn("render differently in B: aaaa0002", text)
        self.assertIn("more draw failures", text)
        self.assertIn("GPU busy ms/s better", "\n".join(ab["improved"]))
        self.assertIn("flagged:", ac.ab_text(ab))

    def test_summary_text(self):
        lines, _ = LogTest().run_lines()
        perf = ac.pass_report(lines, ac.pick_areas("central-yharnam,moonside-lake"), "perf")
        rep = {"dir": "/r", "run": {"version": perf["version"], "numa_node": 0, "fps": 60}, "passes": {"perf": {
            "stop_reason": "tour done", "elapsed_s": 300, "world": perf["world"], "tour_done": True, "fatal": []}},
               "areas": ac.merge(perf, None, None, ac.pick_areas("central-yharnam,moonside-lake"))}
        rep["problems"] = ac.problems_of(rep)
        self.assertEqual(rep["problems"], [])
        text = ac.summary_text(rep)
        self.assertIn("central-yharnam", text)
        self.assertIn("pixel shaders first met here: 1 lifted, 2 translated", text)
        self.assertIn("fps", text)


# A stand-in for the game: it follows the tour as the probe would and prints
# the probe's and the host's lines, writes the frames the dump requests ask
# for and the captures the capture requests allow - in no time - then waits
# to be stopped, as the game does.
GAME = r"""
import json, os, sys, time
from pathlib import Path
tour = Path(os.environ["BBHOST_MAPVAL_TOUR"]).read_text().splitlines()
dumps, captures = Path(os.environ["BBHOST_MAPVAL_DUMPS"]), Path(os.environ["BBHOST_MAPVAL_CAPTURES"])
assert os.environ["BBHOST_FRAME_STATS"] == "1" and os.environ["BBHOST_FRAME_DETAIL"] == "1" and os.environ["BBHOST_TEST_WARP"] == "1"
out = sys.stdout
def log(s):
    out.write("[bbhost] " + s + "\n")
log("bbhost v9.9.9-test (abc, Linux x86-64)")
log("world: the first in-game frame, flip 900, 30.0 s after start")
t, flip, block, step = 1000.0, 1000, "15000000", 0
def second(fps=60):
    iv = [1000.0 / fps] * fps
    log("frames: 1.00 s: %d flips (%.1f/s), displayed %.1f/s, interval avg 16.7 p95 16.7 max 16.7 ms, 0 over 16.7, 0 over 33.3; "
        "cpu 250%%: main:1 90%%; main loop waited 1 ms, slept 1 ms, file I/O 0 ms; main loop work avg 8.0 p95 9.0 max 10.0 ms, "
        "0 over 16.7; gpu busy 400 ms/s (40%%)" % (fps, fps, fps))
    log("frame detail: draws 9000: pixel shader lifted 6000, translated 2900, fallback 50, none 50; draw failures 0")
    log("frame intervals ms:" + "".join(" %.2f" % v for v in iv))
for line in tour:
    w = line.split()
    if not w or w[0].startswith("#"):
        continue
    step += 1
    if w[0] == "travel":
        block = w[2]
        log("mapval step %d travel %s %.3f return point %s block %s -> queued" % (step, w[3], t, w[1], block))
        log("loading: 3.00 s (flips %d-%d, 0 of them not shown)" % (flip, flip + 90))
        log("render: PS %08x lifted (100 words; translated 200)" % (step,))
        t += 6
        log("mapval travel-done %s %.3f %s 1.000 2.000 3.000 after 6.0 s" % (w[3], t, block))
    elif w[0] == "warp":
        log("mapval at %s %.3f %s %s %s %s yaw %s" % (w[1], t, block, w[2], w[3], w[4], w[5]))
    elif w[0] == "quiet":
        log("mapval quiet %.3f 2 enemies within %s m switched off, 0 other characters left" % (t, w[1]))
    elif w[0] == "camera":
        log("mapval camera %.3f reset" % t)
    elif w[0] == "mark":
        log("mapval mark %s %.3f %s 1.000 2.000 3.000 yaw 0.500" % (w[1], t, block))
    elif w[0] == "hold":
        for _ in range(int(float(w[1]) + 0.5)):
            second()
            t += 1
            flip += 60
    elif w[0] == "dump":
        p = dumps / (w[1] + ".ppm")
        p.write_bytes(b"P6\n4 4\n255\n" + bytes([100]) * 48)
        log("test: frame dump at flip %d to %s" % (flip, p))
    elif w[0] == "capture" and int(w[1]) > 0:
        for k in range(min(2, int(w[1]))):
            d = captures / w[2] / ("%04x+%08x-f%d-d%d" % (k, 0xa0000000 + step * 16 + k, flip, k))
            d.mkdir(parents=True)
            (d / "manifest.json").write_text(json.dumps({"bound": "lifted" if k else "translated"}))
            log("capture: wrote %s/manifest.json (%s, flip %d, draw %d, 3 blobs, 100 bytes)" % (d, d.name.split("-")[0], flip, k))
log("mapval tour-done %.3f" % t)
out.flush()
time.sleep(60)
"""


class EndToEndTest(unittest.TestCase):
    """`run` twice and `compare`, with the stand-in game and the lift_verify
    stand-ins: the launching, the log's reading, the passes merged, the lift
    checks, the reports and the A/B."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.d = d
        build = d / "build"
        build.mkdir()
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import lift_verify_test as lv
        lv.script(build / "bbhost", GAME)
        lv.script(build / "gcnlift", lv.GCNLIFT)
        lv.script(build / "drawreplay", lv.DRAWREPLAY)
        (d / "app0" / "dvdroot_ps4").mkdir(parents=True)
        (d / "eboot.bin").write_bytes(b"not the game")
        (d / "seed" / "saves" / "SPRJ0005").mkdir(parents=True)
        (d / "seed" / "saves" / "SPRJ0005" / "userdata0000").write_bytes(b"save")
        self.saved = (ac.EBOOT_109_SHA256, ac.check_areas)
        ac.EBOOT_109_SHA256 = ac.sha256_file(d / "eboot.bin")
        ac.check_areas = lambda areas, app0: [{"id": a.id, "problems": []} for a in areas]

    def tearDown(self):
        ac.EBOOT_109_SHA256, ac.check_areas = self.saved
        self.tmp.cleanup()

    def run_once(self, name):
        d = self.d
        args = ["run", "--bbhost", str(d / "build" / "bbhost"), "--app0", str(d / "app0"), "--eboot", str(d / "eboot.bin"),
                "--seed", str(d / "seed"), "--out", str(d / name), "--areas", "central-yharnam,moonside-lake", "--numa-node", "none",
                "--captures", "2", "--capture-window", "3", "--settle", "3", "--measure", "6"]
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return ac.main(args)

    def test_run_and_compare(self):
        self.assertEqual(self.run_once("a1"), 0, (self.d / "a1" / "summary.txt").read_text())
        rep = json.loads((self.d / "a1" / "report.json").read_text())
        self.assertEqual(rep["problems"], [])
        self.assertTrue(rep["run"]["seed_unchanged"])
        self.assertEqual(rep["run"]["version"], "v9.9.9-test")
        cy = rep["areas"]["central-yharnam"]
        self.assertEqual(cy["status"], "ok")
        self.assertEqual(cy["window"]["flips"], 5 * 60)  # 6 s, the first left out
        self.assertTrue(cy["window"]["exact_percentiles"])
        self.assertEqual(cy["window"]["draws"]["lifted"], 5 * 6000)
        self.assertEqual(cy["frames"]["a"]["stats"]["mean_luma"], 100.0)
        self.assertEqual(cy["arrival"]["new_pixel_shaders"]["lifted"], 1)
        self.assertEqual(cy["capture_pass"]["status"], "ok")
        self.assertEqual(cy["lift"]["totals"], {"identical": 2})
        self.assertTrue(rep["areas"]["moonside-lake"]["view"]["reached"])
        self.assertFalse((self.d / "a1" / "perf" / "data").exists())  # the seed's copy goes
        self.assertIn("central-yharnam", (self.d / "a1" / "summary.txt").read_text())
        # The report again from the logs.
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ac.main(["report", str(self.d / "a1")]), 0)
        again = json.loads((self.d / "a1" / "report.json").read_text())
        self.assertEqual(again["areas"]["central-yharnam"]["window"], cy["window"])
        # Two runs compared: the same numbers, the same frames.
        self.assertEqual(self.run_once("b1"), 0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ac.main(["compare", "--a", str(self.d / "a1"), "--b", str(self.d / "b1"), "--out", str(self.d / "ab")]), 0)
        ab = json.loads((self.d / "ab" / "ab.json").read_text())
        self.assertEqual(ab["flagged"], [])
        f = ab["areas"]["central-yharnam"]["frames"]["a"]
        self.assertEqual(f["verdict"], "no noise floor")
        self.assertEqual(f["ab"]["changed_pct"], 0.0)
        self.assertEqual(ab["areas"]["central-yharnam"]["perf"]["fps"]["verdict"], "same")
        self.assertTrue((self.d / "ab" / "ab.txt").is_file())


if __name__ == "__main__":
    unittest.main()
