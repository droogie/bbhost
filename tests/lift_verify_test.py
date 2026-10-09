"""tools/lift_verify.py with stand-ins for gcnlift and drawreplay: a captured
translation is replayed against the lift, a captured lift (BBHOST_DECOMP)
against the translation - never against itself."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]

# gcnlift <capture> <out>: the capture's manifest says which module it bound.
GCNLIFT = textwrap.dedent("""\
    import json, sys
    from pathlib import Path
    cap, out = Path(sys.argv[1]), Path(sys.argv[2])
    kind = json.loads((cap / "manifest.json").read_text())["bound"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "lifted.spv").write_bytes(b"L")
    (out / "reference.spv").write_bytes(b"T")
    (out / "report.json").write_text(json.dumps({"pipeline": cap.name, "reference_matches_capture": True,
        "captured_module": kind, "lifted": True, "spirv_valid": True, "reference_words": 10, "lifted_words": 5}))
""")
# drawreplay <capture> --ps <file> --no-live --out <dir>: identical when the
# other module differs from the bound one, as a real comparison needs.
DRAWREPLAY = textwrap.dedent("""\
    import json, sys
    from pathlib import Path
    cap = Path(sys.argv[1])
    ps = Path(sys.argv[sys.argv.index("--ps") + 1])
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    kind = json.loads((cap / "manifest.json").read_text())["bound"]
    other = ps.read_bytes()
    (out / "used.txt").write_text(ps.name)
    real = (kind == "lifted" and other == b"T") or (kind == "translated" and other == b"L")
    (out / "report.json").write_text(json.dumps({"pass": True, "comparisons": {"A_vs_variant": {
        "bytes_identical": real, "pass": real}}}))
""")


def script(path, body):
    path.write_text("#!%s\n%s" % (sys.executable, body))
    os.chmod(path, 0o755)
    return path


class LiftVerifyTest(unittest.TestCase):
    def test_the_other_compiler_is_replayed(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            gcnlift = script(d / "gcnlift", GCNLIFT)
            drawreplay = script(d / "drawreplay", DRAWREPLAY)
            caps = d / "captures"
            for name, bound in (("1111+aaaa0001-f1-d1", "translated"), ("2222+bbbb0002-f1-d2", "lifted")):
                (caps / name).mkdir(parents=True)
                (caps / name / "manifest.json").write_text(json.dumps({"bound": bound}))
            r = subprocess.run([sys.executable, str(ROOT / "tools/lift_verify.py"), str(caps), "--out", str(d / "out"),
                                "--gcnlift", str(gcnlift), "--drawreplay", str(drawreplay)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            summary = json.loads((d / "out" / "summary.json").read_text())
            self.assertEqual(summary["totals"], {"identical": 2})
            rows = {row["name"]: row for row in summary["rows"]}
            self.assertEqual(rows["1111+aaaa0001-f1-d1"]["compared"], "captured translation, lift replayed")
            self.assertEqual(rows["2222+bbbb0002-f1-d2"]["compared"], "captured lift, translation replayed")
            self.assertEqual((d / "out" / "2222+bbbb0002-f1-d2" / "replay" / "used.txt").read_text(), "reference.spv")
            self.assertEqual((d / "out" / "1111+aaaa0001-f1-d1" / "replay" / "used.txt").read_text(), "lifted.spv")


if __name__ == "__main__":
    unittest.main()
