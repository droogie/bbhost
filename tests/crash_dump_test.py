"""Native Windows integration tests; only benign exceptions and owned memory."""
import argparse
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import time


def validate(path, result, code, full=True, concurrent=False):
    data = path.read_bytes()
    assert len(data) >= 32
    signature, _, count, directory = struct.unpack_from("<IIII", data)
    assert signature == 0x504D444D and 0 < count < 100
    assert directory + count * 12 <= len(data)
    streams = {}
    for i in range(count):
        kind, size, rva = struct.unpack_from("<III", data, directory + i * 12)
        if kind == 0:  # DbgHelp can reserve an unused directory slot.
            assert size == 0 and rva == 0
            continue
        assert size and rva + size <= len(data), (kind, size, rva)
        streams[kind] = (size, rva)
    assert 3 in streams and 6 in streams
    size, rva = streams[6]
    assert size >= 168
    tid, actual = struct.unpack_from("<I", data, rva)[0], struct.unpack_from("<I", data, rva + 8)[0]
    assert actual == code, (hex(actual), hex(code))
    if not concurrent:
        assert tid == int(re.search(r"tid=(\d+)", result.stdout)[1])
    ctxsize, ctx = struct.unpack_from("<II", data, rva + 160)
    assert ctxsize >= 256 and ctx + ctxsize <= len(data)
    assert struct.unpack_from("<Q", data, ctx + 248)[0]  # original RIP
    if not full:
        return
    assert struct.unpack_from("<Q", data, 24)[0] & 2
    assert all(kind in streams for kind in (9, 16, 17))
    size, rva = streams[9]
    n, offset = struct.unpack_from("<QQ", data, rva)
    assert size >= 16 + n * 16
    address = int(re.search(r"sentinel=([0-9a-f]+)", result.stdout)[1], 16)
    marker = b"BBHOST-CAPTURE-CHECK\0"
    found = False
    for i in range(n):
        start, length = struct.unpack_from("<QQ", data, rva + 16 + i * 16)
        assert offset + length <= len(data)
        if start <= address and address + len(marker) <= start + length:
            at = offset + address - start
            assert data[at:at + len(marker)] == marker
            found = True
        offset += length
    assert found
    assert not Path(str(path) + ".partial").exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--helper", required=True, type=Path)
    parser.add_argument("--slow-helper", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="bbhost crash ü ") as temp:
        root = Path(temp)
        # Unicode/spaces in both the helper location and output path.
        normal = root / "normal ü"
        normal.mkdir()
        shutil.copy2(args.fixture, normal / "fixture.exe")
        shutil.copy2(args.helper, normal / "crash_dump_helper.exe")
        env = os.environ.copy()
        env.pop("BBHOST_FULL_DUMP", None)
        cases = 0

        def run(name, mode="exception", folder=normal, expected=139, path=None, enabled=True, extra_env=None):
            nonlocal cases
            path = path or root / (name + ".dmp")
            child_env = env.copy()
            child_env.update(extra_env or {})
            if enabled:
                child_env["BBHOST_FULL_DUMP"] = str(path)
            result = subprocess.run([str(folder / "fixture.exe"), mode], env=child_env,
                                    capture_output=True, text=True, timeout=25)
            assert result.returncode == expected, (name, result.returncode, result.stderr)
            cases += 1
            print(f"PASS {name}: exit={result.returncode}", flush=True)
            return path, result

        for name, mode, code, exitcode in (
            ("early-guest", "early", 0xC0000094, 99),
            ("exception", "exception", 0xE0424242, 139),
            ("abort", "abort", 0xE0000001, 134),
            ("concurrent", "concurrent", 0xE0000001, 0),
        ):
            path, result = run(name, mode, expected=exitcode)
            validate(path, result, code, concurrent=mode == "concurrent")
            validate(Path(str(path) + ".triage.dmp"), result, code, full=False,
                     concurrent=mode == "concurrent")
            assert "triage dump: complete error=0" in result.stderr
            assert "full dump: complete error=0" in result.stderr

        for mode in ("clean", "forced-exit"):
            path, _ = run(mode, mode, expected=0)
            # Forced parent exit lets the independently waiting helper clean up.
            deadline = time.monotonic() + 5
            while Path(str(path) + ".partial").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not path.exists() and not Path(str(path) + ".partial").exists()

        path, _ = run("disabled", enabled=False)
        assert not path.exists()
        run("missing-directory", path=root / "absent" / "dump.dmp", expected=78)
        for suffix in ("", ".partial", ".triage.dmp", ".triage.dmp.partial", ".capturing"):
            path = root / ("collision-" + str(cases) + ".dmp")
            existing = Path(str(path) + suffix)
            existing.write_bytes(b"KEEP")
            run("collision" + suffix, path=path, expected=78)
            assert existing.read_bytes() == b"KEEP"

        missing = root / "missing-helper"
        missing.mkdir()
        shutil.copy2(args.fixture, missing / "fixture.exe")
        run("missing-helper", folder=missing, expected=78)

        dead = root / "dead-helper"
        dead.mkdir()
        shutil.copy2(args.fixture, dead / "fixture.exe")
        shutil.copy2(args.fixture, dead / "crash_dump_helper.exe")
        path, result = run("helper-exit", folder=dead)
        assert not path.exists() and "FAILED error=1067" in result.stderr
        path, result = run("helper-not-ready", folder=dead, expected=78,
                           extra_env={"BBHOST_TEST_HELPER_NOT_READY": "1"})
        assert not path.exists() and "startup error=1460" in result.stderr

        slow = root / "slow-helper"
        slow.mkdir()
        shutil.copy2(args.fixture, slow / "fixture.exe")
        shutil.copy2(args.slow_helper, slow / "crash_dump_helper.exe")
        path, result = run("full-timeout", folder=slow)
        assert not path.exists() and "full dump: FAILED error=1460" in result.stderr
        validate(Path(str(path) + ".triage.dmp"), result, 0xE0424242, full=False)
        assert "triage dump: complete error=0" in result.stderr

        path, result = run("triage-failure", "triage-failure")
        validate(path, result, 0xE0424242)
        assert "triage dump: FAILED error=" in result.stderr
        assert not Path(str(path) + ".triage.dmp").exists()
        path, result = run("full-final-collision", "full-final-collision")
        assert path.read_bytes() == b"KEEP"
        assert "full dump: FAILED error=" in result.stderr
        validate(Path(str(path) + ".triage.dmp"), result, 0xE0424242, full=False)
        print(f"PASS {cases} collector cases; exception/context, directory bounds and memory sentinel verified")


if __name__ == "__main__":
    main()
