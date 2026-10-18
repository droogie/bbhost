"""tools/pipeline_list.py on synthetic records: bbhost's hash, a manifest read
and written back byte for byte, merge keeping the newest of a key, and a list
naming its programs instead of holding them."""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pipeline_list as pl  # noqa: E402

ORIGIN = {"translator": "f" * 32, "vendor": 0x10DE, "device": 0x2786, "driver": 1, "uuid": "00" * 16, "libraries": True}


def recipe(key, last_run, vs, ps, vs_name="0000aaaa", ps_name="0000bbbb", vertex_input=True):
    """One record laid out as render.cpp's recipe_file::write lays it out."""
    flags = (pl.HAS_VERTEX_INPUT if vertex_input else 0) | pl.RELINK
    state = bytes(19 * 4 + 8 + 57 * 4 + pl.GX_BLEND + pl.GX_DEPTH + pl.GX_RASTER)
    tail = struct.pack("<I", 0) * 4 + struct.pack("<I", 1) + struct.pack("<4I", 0, 1, 4, 0x8C) + struct.pack("<I", 0) * 2 + \
        struct.pack("<Q", 0x1234) + bytes(128 + 128 + 4)
    tail += struct.pack("<I", 0) + bytes(9 * 4 + 16 + 8 + 48 + 8)  # no tessellation
    body = b"".join([struct.pack("<IQIII", flags, key, 1, last_run, key), state, struct.pack("<I", len(vs) // 4), vs,
                     struct.pack("<I", 0), struct.pack("<I", len(ps) // 4), ps, pl.wstr(vs_name), pl.wstr(ps_name), tail])
    return pl.Recipe().parse(body)


class PipelineListTest(unittest.TestCase):
    def test_hash_is_bbhosts(self):
        # Values from host/gpu.cpp's fnv1a over the same words.
        w = struct.pack("<5I", 0xBEEFCAFE, 1, 2, 3, 0xFFFFFFFF)
        self.assertEqual(pl.fnv1a(w[:0]), 1469598103934665603)
        self.assertEqual(pl.fnv1a(w[:8]), 1910414784848688967)
        self.assertEqual(pl.fnv1a(w), 5040822000519442178)

    def test_round_trip_and_merge(self):
        vs, ps = struct.pack("<3I", 1, 2, 3), struct.pack("<2I", 7, 8)
        with tempfile.TemporaryDirectory() as d:
            a, b, out = Path(d) / "a.bin", Path(d) / "b.bin", Path(d) / "m.bin"
            pl.write_file(a, 3, ORIGIN, [recipe(1, 3, vs, ps), recipe(2, 1, vs, ps)])
            run, origin, recs = pl.read_file(a)
            self.assertEqual((run, origin["device"], [r.key for r in recs]), (3, 0x2786, [1, 2]))
            pl.write_file(b, 0, origin, recs)
            self.assertEqual(a.read_bytes()[12:], b.read_bytes()[12:])  # all but the run, as it was
            pl.write_file(b, 5, ORIGIN, [recipe(2, 5, vs, struct.pack("<I", 9)), recipe(3, 5, vs, ps)])
            sys.argv = ["pipeline_list.py", "merge", "--out", str(out), str(a), str(b)]
            self.assertEqual(pl.main(), 0)
            run, _, recs = pl.read_file(out)
            by = {r.key: r for r in recs}
            self.assertEqual((run, sorted(by)), (5, [1, 2, 3]))
            self.assertEqual(by[2].ps, struct.pack("<I", 9))  # the newer start's record of key 2

    def test_list_names_programs(self):
        vs, ps = struct.pack("<3I", 1, 2, 3), struct.pack("<2I", 7, 8)
        rc = recipe(1, 1, vs, ps)
        rc.flags |= pl.VS_REF | pl.PS_REF
        rc.vs_ref, rc.ps_ref = (3, pl.fnv1a(vs)), (2, pl.fnv1a(ps))
        rc.vs = rc.ps = None
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "list.bin"
            pl.write_file(path, 0, ORIGIN, [rc])
            _, _, recs = pl.read_file(path)
            self.assertEqual((recs[0].vs, recs[0].ps), (None, None))
            self.assertEqual(recs[0].vs_ref, (3, pl.fnv1a(vs)))
            self.assertEqual((recs[0].vs_name, recs[0].ps_name), ("0000aaaa", "0000bbbb"))


    def test_tessellated_record(self):
        vs, ps, ls, hs = (struct.pack("<2I", 1, 2), struct.pack("<I", 3), struct.pack("<3I", 4, 5, 6), struct.pack("<I", 7))
        rc = recipe(9, 1, vs, ps)
        # A tessellated one: the LS and the hull after the tessellation state.
        tess = struct.pack("<I", 1 | pl.TESS_HAS_LS | pl.TESS_HAS_HS) + bytes(9 * 4 + 16 + 8 + 48 + 8)
        body = rc.body()[:-len(rc.tess)] + tess + struct.pack("<I", 3) + ls + pl.wstr("0000cccc") + struct.pack("<I", 2) + \
            struct.pack("<I", 1) + hs + pl.wstr("0000dddd") + struct.pack("<I", 0xFFFFFFFF)
        rc = pl.Recipe().parse(body)
        self.assertEqual((rc.ls, rc.hs, rc.ls_name, rc.hs_name, rc.ls_sgpr), (ls, hs, "0000cccc", "0000dddd", 2))
        self.assertEqual(rc.body(), body)
        rc.flags |= pl.LS_REF
        rc.ls_ref, rc.ls = (3, pl.fnv1a(ls)), None
        again = pl.Recipe().parse(rc.body())
        self.assertEqual((again.ls, again.ls_ref, again.hs), (None, (3, pl.fnv1a(ls)), hs))

    def test_ls_pass_record(self):
        """Version 3: a tessellated record with the host tessellator's LS passes
        (each its registers, its LS, its fetch shader) reads and writes back as
        it was, and a list names each LS and keeps the fetch shaders' words."""
        vs, ps, lsp, fetch = (struct.pack("<2I", 1, 2), struct.pack("<I", 3), struct.pack("<3I", 4, 5, 6), struct.pack("<2I", 8, 9))
        rc = recipe(10, 1, vs, ps)
        tess = struct.pack("<I", 1 | pl.TESS_LS_PASSES) + bytes(9 * 4 + 16 + 8 + 48 + 8)
        entry = lambda name, words: struct.pack("<I", 0) + struct.pack("<5I", 0x11, 0x22, 1, 64, 3) + struct.pack("<I", len(words) // 4) + words + \
            pl.wstr(name) + struct.pack("<I", 2) + struct.pack("<I", 1) + struct.pack("<I", 2) + fetch
        body = rc.body()[:-len(rc.tess)] + tess + struct.pack("<I", 2) + entry("0000eeee", lsp) + entry("0000ffff", lsp + lsp)
        rc = pl.Recipe().parse(body)
        self.assertEqual([(e["prog"], e["name"], e["sgpr"], e["fetch"]) for e in rc.lsps],
                         [(lsp, "0000eeee", 2, fetch), (lsp + lsp, "0000ffff", 2, fetch)])
        self.assertEqual(rc.body(), body)
        progs = {"0000aaaa": vs, "0000bbbb": ps, "0000eeee": lsp, "0000ffff": lsp + lsp}
        out, left, _plain, _patched = pl.make_list([rc], progs)
        self.assertEqual((len(out), sum(left.values())), (1, 0))
        listed = pl.Recipe().parse(out[0].body())
        self.assertEqual([(e["prog"], e["ref"], e["fetch"]) for e in listed.lsps],
                         [(None, (3, pl.fnv1a(lsp)), fetch), (None, (6, pl.fnv1a(lsp + lsp)), fetch)])
        out, left, _plain, _patched = pl.make_list([pl.Recipe().parse(body)], progs, fetch_words=False)
        self.assertEqual((len(out), sum(left.values())), (0, 1))  # --no-fetch-words leaves it out

    def test_versions_agree(self):
        """The record format pipeline_list.py writes is the one bbhost writes
        (render.cpp recipe_file::kVersion), and the one the pinned list is."""
        src = (ROOT / "src" / "host" / "render.cpp").read_text()
        at = src.index("namespace recipe_file {")
        version = int(src[at:at + 400].split("kVersion = ")[1].split(";")[0])
        self.assertEqual(version, pl.VERSION)
        fetch = (ROOT / "tools" / "fetch_pipeline_list.sh").read_text()
        self.assertIn("LIST_FORMAT=%d\n" % pl.VERSION, fetch)


if __name__ == "__main__":
    unittest.main()
