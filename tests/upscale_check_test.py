#!/usr/bin/env python3
"""Tests for tools/upscale_check.py on log lines as host/upscale_pass.cpp's
check_counts_read_locked writes them."""
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import upscale_check  # noqa: E402

CAMERA = "[bbhost] upscaler: the camera's view block at 0x143e70040 (tangents 0.934 0.394, eye -9.6 -4.5 -18.9)\n"


def check_line(flip, ratio, red, green, blue, bands):
    return ("[bbhost] upscaler: the reprojection check over 30 frames to flip %d (2560x1080): red/(red+green) %s; red %.3f%%, "
            "green %.3f%%, blue %.3f%% of the pixels; red/(red+green) by sixteenth of the width from the left:%s\n" % (
                flip, ratio, red, green, blue, "".join(" " + b for b in bands)))


def run(text, *args):
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "run.log"
        log.write_text(text)
        out = io.StringIO()
        old = sys.argv
        sys.argv = ["upscale_check.py", str(log), *args]
        try:
            with redirect_stdout(out):
                rc = upscale_check.main()
        finally:
            sys.argv = old
        return rc, out.getvalue()


class UpscaleCheckTest(unittest.TestCase):
    def test_moving_windows_summed(self):
        still = check_line(100, "0.9000", 0.9, 0.1, 0.0, ["0.900"] * 16)
        pan = check_line(130, "0.0500", 3.0, 57.0, 2.5, ["0.100", "0.080"] + ["0.040"] * 12 + ["0.060", "0.120"])
        walk = check_line(160, "0.0400", 2.0, 48.0, 0.0, ["0.050"] * 16)
        rc, out = run(CAMERA + still + pan + walk)
        self.assertEqual(rc, 0)
        self.assertIn("tangents 0.934 0.394 (aspect 2.371)", out)
        self.assertIn("2 of 3 windows moving: r/(r+g) 0.045", out)  # (3 + 2) / (3 + 57 + 2 + 48)
        self.assertNotIn("flip     100", out)  # the still window left out of the list
        self.assertIn("edges 0.070, middle 0.045", out)  # (0.075 + 0.065 + 0.055 + 0.085) / 4, (0.04 + 0.05) / 2

    def test_no_camera_said(self):
        _, out = run(check_line(100, "1.0000", 70.0, 0.0, 0.0, ["1.000"] * 16))
        self.assertIn("NONE", out)

    def test_bands_without_change(self):
        rc, out = run(CAMERA + check_line(100, "0.0500", 3.0, 57.0, 0.0, ["-"] * 4 + ["0.050"] * 12), "--all")
        self.assertEqual(rc, 0)
        self.assertIn("bands  --   --   --   --  0.05 0.05", out)

    def test_nothing_moving_or_no_lines(self):
        rc, _ = run(CAMERA + check_line(100, "0.5000", 1.0, 1.0, 0.0, ["0.500"] * 16))
        self.assertEqual(rc, 1)
        rc, out = run(CAMERA)
        self.assertEqual(rc, 2)
        self.assertIn("no reprojection check line", out)


if __name__ == "__main__":
    unittest.main()
