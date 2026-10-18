"""tools/full_tour.py without the game: the stops' spread, a chunk's tour,
a chunk's log read back into segments, and the union of the chunks'
manifests that keeps every recipe in use."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import full_tour as ft  # noqa: E402
import pipeline_list as pl  # noqa: E402
from pipeline_list_test import ORIGIN, recipe  # noqa: E402


def plan(items):
    return {"options": {"stop_hold": 1.0, "headings": 6, "heading_s": 0.5, "remo_skip": 40, "remo_max": 120}, "items": items}


# The stops' spread (the plan step) runs on numpy, which a CI runner need not have.
@unittest.skipUnless(importlib.util.find_spec("numpy"), "the spread needs numpy")
class SpreadTest(unittest.TestCase):
    def test_covers_and_stops(self):
        import numpy as np
        import random
        grid = np.array([[x * 10.0, 0.0, z * 10.0] for x in range(20) for z in range(20)])
        picks = ft.spread(grid, 60.0, 100, random.Random(1))
        chosen = grid[[i for i, _ in picks]]
        far = max(min(float(((p - c) ** 2).sum()) ** 0.5 for c in chosen) for p in grid)
        self.assertLess(far, 60.0)  # every point within the radius of a stop
        self.assertEqual(picks, ft.spread(grid, 60.0, 100, random.Random(1)))  # the seed decides it
        self.assertEqual(len(ft.spread(grid, 60.0, 3, random.Random(1))), 3)  # at most max_n
        self.assertGreaterEqual(len(ft.spread(grid, 1000.0, 50, random.Random(1), min_n=5)), 5)  # at least min_n

    def test_height_counts_twice(self):
        import numpy as np
        import random
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 40.0, 0.0]])  # 40 m apart in height: 80 m by the weighting
        self.assertEqual(len(ft.spread(pts, 60.0, 10, random.Random(1), first=0)), 2)


class TourTest(unittest.TestCase):
    def test_chunk_tour(self):
        items = [
            {"id": "m24_01_00_00", "kind": "block", "block": "m24_01_00_00", "return_point": 2412950, "stops": [
                {"name": "s00", "pos": [1.0, 2.0, 3.0]}, {"name": "s01", "pos": [4.0, 5.0, 6.0]}], "remos": [24010010]},
            {"id": "m29_10_90_00", "kind": "chalice", "block": "m29_00_00_00", "map_id": "1d0a5a00", "slot": 6, "return_point": 2902950,
             "stops": [{"name": "s00", "pos": [7.0, 8.0, 9.0]}], "remos": []},
            {"id": "m24_01_00_00@t0", "kind": "state", "block": "m24_01_00_00", "return_point": 2412950,
             "flags": [[9800, 0], [9801, 0], [9802, 0]], "stops": [], "remos": []},
            {"id": "twelve:moonside-lake", "kind": "twelve", "block": "m32_00_00_00", "return_point": 3202950, "door_flags": [13200040, 13200120],
             "view": [1.0, 2.0, 3.0, 0.5], "stops": [], "remos": []},
        ]
        p = plan(items)
        text = ft.chunk_tour(p, {"n": 3, "items": [i["id"] for i in items]}, None).splitlines()
        self.assertEqual(text[1:3], ["god hide", "flaglog 9800 3"])
        self.assertIn("travel 2412950 18010000 m24_01_00_00", text)
        self.assertIn("warp m24_01_00_00:s01 4.000 5.000 6.000 0 1.0", text)
        self.assertIn("turn 6 0.50", text)
        self.assertIn("dump m24_01_00_00-s00", text)
        self.assertIn("remo 24010010 0 40 120", text)
        self.assertIn("chalice 1d0a5a00 6 m29_10_90_00", text)
        at = text.index("flags 9800=0")
        self.assertEqual(text[at:at + 4], ["flags 9800=0", "flags 9801=0", "flags 9802=0", "flaglog 9800 3"])
        self.assertTrue(text[at + 4].startswith("travel 2412950 18010000 m24_01_00_00@t0"))
        self.assertIn("travel 3202950 20000000 twelve:moonside-lake 13200040 13200120", text)
        self.assertEqual(text[-2:], ["exit", "wait 30"])

    def test_estimate(self):
        it = {"stops": [{}] * 10, "remos": [1, 2]}
        self.assertEqual(ft.item_estimate(it), round(ft.EST["travel"] + 10 * ft.EST["stop"] + 2 * ft.EST["remo"], 1))


LOG = """[bbhost] render: pipeline manifest: new recipe 00000000000000aa 1111+2222 (1 in this run) at flip 3
[bbhost] plugin: mapval step 4 travel m24_01_00_00 1792000000.000 return point 2412950 block 18010000 -> queued
[bbhost] FS trace: /app0/dvdroot_ps4/map/mapstudio/m24_01_00_01.msb.dcx (no overlay file)
[bbhost] plugin: mapval travel-done m24_01_00_00 1792000020.000 18010000 1.0 2.0 3.0 after 20.0 s, 9.5 s of it loading
[bbhost] render: pipeline manifest: new recipe 00000000000000ab 3333+4444 (2 in this run) at flip 900
[bbhost] plugin: mapval step 6 warp m24_01_00_00:s00 1792000021.000 target 1 2 3 yaw 0 -> queued
[bbhost] render: pipeline manifest: new recipe 00000000000000ac 3333+5555 tessellated (3 in this run) at flip 950
[bbhost] render: pipeline build 3333+5555 (no-fallback) on the command processor at flip 950: 12.50 ms (built, linked): shader libraries 1 in 12 ms
[bbhost] render: CP translates PS 00000000000000ff (100 words, no-fallback) at flip 950
[bbhost] stall: flip #951 came 75 ms after the one before
[bbhost] frame detail: draws 900: pixel shader lifted 800, translated 100, fallback 0, none 0; draw failures 2 (no pipeline 2)
[bbhost] FS: /app0/dvdroot_ps4/map/m24/m24_9999.tpf.dcx is not in the game folder (no mapping)
[bbhost] FS: /app0/dvdroot_ps4/chr/c9999.chrbnd.dcx is not in the game folder (no mapping)
[bbhost] plugin: mapval step 7 warp m24_01_00_00:s01 1792000030.000 target 4 5 6 yaw 0 -> queued
[bbhost] plugin: mapval remo-request 24010010 1792000040.000 mode 0 -> queued (1)
[bbhost] plugin: mapval remo-start 24010010 1792000042.000 after 2.0 s
[bbhost] render: pipeline manifest: new recipe 00000000000000ad 6666+7777 (4 in this run) at flip 1200
[bbhost] plugin: mapval remo-skip 24010010 1792000082.000 press 1
[bbhost] plugin: mapval remo-end 24010010 1792000083.000 played 41.0 s, skip pressed 1 times
[bbhost] test: exit asked for at flip 2000
[bbhost] gpu: draw failures by reason: no pipeline 2
"""


class ReadTest(unittest.TestCase):
    def test_segments(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "run.log"
            p.write_text(LOG)
            segs, items, exitlines, _flags, _turns = ft.read_chunk(p, 5)
        by = {(s["item"], s["stop"]): s for s in segs}
        self.assertEqual(by[("(start)", "(start)")]["new_recipes"], 1)
        travel = by[("m24_01_00_00", "(travel)")]
        self.assertEqual((travel["new_recipes"], travel["msb"]), (1, ["m24_01_00_01.msb"]))
        s0 = by[("m24_01_00_00", "s00")]
        self.assertEqual((s0["new_recipes"], s0["new_tess"], s0["builds_over_5ms"], s0["translations"], s0["stalls"], s0["stall_ms"]),
                         (1, 1, 1, 1, 1, 75))
        self.assertEqual((s0["draw_failures"], s0["failure_reasons"]), (2, {"no pipeline": 2}))
        self.assertEqual(len(s0["missing"]), 2)
        self.assertEqual(by[("m24_01_00_00", "remo24010010")]["new_recipes"], 1)
        self.assertEqual(items["m24_01_00_00"]["travel_s"], 20.0)
        self.assertEqual(items["m24_01_00_00"]["remos"][24010010]["ended"], "end")
        self.assertTrue(any("draw failures by reason" in x for x in exitlines))
        self.assertTrue(ft.KNOWN_PROBES.search(s0["missing"][0]))
        self.assertFalse(ft.KNOWN_PROBES.search(s0["missing"][1]))


class UnionTest(unittest.TestCase):
    def test_union_keeps_everything_in_use(self):
        vs, ps = b"\x01\x00\x00\x00" * 3, b"\x02\x00\x00\x00" * 2
        with tempfile.TemporaryDirectory() as d:
            a, b, union, mine = (Path(d) / n for n in ("a.bin", "b.bin", "union.bin", "mine.bin"))
            pl.write_file(a, 3, ORIGIN, [recipe(1, 3, vs, ps), recipe(2, 1, vs, ps)])
            pl.write_file(b, 20, ORIGIN, [recipe(2, 20, vs, ps), recipe(3, 20, vs, ps)])
            self.assertEqual(ft.merge_union(union, a), 2)
            self.assertEqual(ft.merge_union(union, b), 3)
            self.assertEqual(ft.refresh_union(union, mine), 3)
            run, _o, recipes = pl.read_file(mine)
            self.assertEqual(run, 20)
            self.assertEqual(sorted(rc.key for rc in recipes), [1, 2, 3])
            self.assertTrue(all(rc.last_run == 20 for rc in recipes))  # none idle: bbhost drops nothing at its save


if __name__ == "__main__":
    unittest.main()
