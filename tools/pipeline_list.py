#!/usr/bin/env python3
"""Pipeline manifests and lists: what bbhost recorded, and a list made from them.

bbhost records every graphics pipeline a draw creates in
<data>/bbhost/pipeline-manifest.bin (render.cpp, PipelineRecipe) and builds
them all again at the next start, so no pipeline is compiled on first sight in
an area an earlier run has seen. A list is the same records made from one or
more players' or harness runs' manifests - for instance a tour of every area -
which bbhost reads beside its executable (pipeline-list.bin) or from
BBHOST_PIPELINE_LIST, so a first start has them too.

A manifest holds the programs' words (the game's shader code, from the
player's own files). A list holds none: each program is named as the game's
files name it (its footer's hash), with its word count and hash, and bbhost
takes the words from the player's own files at the start - the bundles, or
the eboot for the eboot's own programs (YEBIS's, Scaleform's), with --eboot -
patched as the draws read them, dropping a record whose hash does not match.
A fetch shader's words are made by the engine at run time and are in no file:
a list carries them as the manifest does (the owner accepted that on
2026-10-17, so tessellated pipelines are listed too); --no-fetch-words leaves
those records out instead. A tessellated record may also carry the host
tessellator's LS passes (version 3): each LS is named like the other programs.
Version 4 adds compute records: a dispatch's compute pipeline (its program,
named the same way in a list, its resource registers, thread counts and the
images and samplers its key was made from), which a start builds before the
dispatch - the title's own compute shaders among them.

  pipeline_list.py stats <file>...
  pipeline_list.py merge --out <manifest> <manifest>...
  pipeline_list.py list --app0 <game folder> [--eboot <eboot>] [--no-fetch-words] --out <list> <manifest>...
  pipeline_list.py check --app0 <game folder> [--eboot <eboot>] <list>

Whether a list is shipped with releases is the owner's decision; this tool
only makes one.
"""
import argparse
import os
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shaderbnd  # noqa: E402

MAGIC = 0x4D504242  # "BBPM"
VERSION = 4  # 2: tessellated pipelines (their LS and hull programs, the LS's fetch shader); 3: the host tessellator's LS passes;
#             4: compute records (ComputeRecipe)
READS = (2, 3, 4)
COMPUTE_RECORD = 1 << 30  # a record's flags: a compute record
VS_REF, PS_REF, LS_REF, HS_REF = 256, 512, 1024, 2048
TESS_HAS_LS, TESS_HAS_HS, TESS_HAS_FETCH, TESS_LS_PASSES = 64, 128, 256, 1024
HAS_VERTEX_INPUT, RELINK = 4, 8
GX_BLEND, GX_DEPTH, GX_RASTER = 66, 16, 20  # sizeof GxBlendDesc, GxDepthStencilDesc, GxRasterDesc
BUNDLES = ("gxshader.shaderbnd.dcx", "gxrenderershader.shaderbnd.dcx", "gxposteffect.shaderbnd.dcx", "gxgui.shaderbnd.dcx",
           "gxffxshader.shaderbnd.dcx", "gxdecal.shaderbnd.dcx", "gxflvershader.shaderbnd.dcx", "gxflvershader.shaderbnddebug.dcx")


def fnv1a(data, h=1469598103934665603):
    """bbhost's fnv1a (host/gpu.cpp): eight bytes at a time, with a fold."""
    m = (1 << 64) - 1
    n8 = len(data) // 8 * 8
    for (v,) in struct.iter_unpack("<Q", data[:n8]):
        h = ((h ^ v) * 1099511628211) & m
        h ^= h >> 32
    for b in data[n8:]:
        h = ((h ^ b) * 1099511628211) & m
    return h


class Reader:
    def __init__(self, data, at=0, end=None):
        self.d, self.p, self.end = data, at, len(data) if end is None else end

    def take(self, n):
        if self.p + n > self.end:
            raise ValueError("short record")
        b = self.d[self.p:self.p + n]
        self.p += n
        return b

    def u8(self):
        return self.take(1)[0]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def words(self):
        n = self.u32()
        return self.take(4 * n)

    def str(self):
        return self.take(self.u32()).decode("latin-1")


def w32(v):
    return struct.pack("<I", v)


def wstr(s):
    b = s.encode("latin-1")
    return w32(len(b)) + b


def program(r, by_ref):
    """A program's words (bytes), or None and its (count, hash) when named."""
    if by_ref:
        return None, (r.u32(), r.u64())
    return r.words(), None


def program_bytes(words, ref):
    if words is None:
        return w32(ref[0]) + struct.pack("<Q", ref[1])
    return w32(len(words) // 4) + words


class ComputeRecipe:
    """A compute record (version 4, render.cpp ComputeRecipe): flags, key and
    use as a pipeline's record has them, the program (named in a list: flag
    256), its name, resource registers and thread counts, and the images and
    samplers its key was made from - kept as bytes."""
    compute = True

    def parse(self, body):
        r = Reader(body)
        self.flags = r.u32()
        self.key = r.u64()
        self.uses = r.u32()
        self.last_run = r.u32()
        self.order = r.u32()
        self.prog, self.ref = program(r, self.flags & VS_REF)
        self.name = r.str()
        self.regs = r.take(5 * 4)  # rsrc1, rsrc2, threads x y z
        self.dims = r.take(5 * r.u32())  # (dim, arrayed) each
        self.modes = r.take(r.u32())
        if r.p != r.end:
            raise ValueError("a compute record longer than its fields")
        return self

    def body(self):
        return b"".join([w32(self.flags), struct.pack("<Q", self.key), w32(self.uses), w32(self.last_run), w32(self.order),
                         program_bytes(self.prog, self.ref), wstr(self.name), self.regs, w32(len(self.dims) // 5), self.dims,
                         w32(len(self.modes)), self.modes])


def parse_record(body):
    """A record of either kind, by its flags."""
    flags = struct.unpack_from("<I", body, 0)[0] if len(body) >= 4 else 0
    return (ComputeRecipe() if flags & COMPUTE_RECORD else Recipe()).parse(body)


class Recipe:
    """One record, kept as the bytes around its programs so it is written back as it was."""
    compute = False

    def parse(self, body):
        r = Reader(body)
        self.flags = r.u32()
        self.key = r.u64()
        self.uses = r.u32()
        self.last_run = r.u32()
        self.order = r.u32()
        state_at = r.p
        r.take(19 * 4 + 8 + (8 + 8 + 32 + 8 + 1) * 4 + GX_BLEND + GX_DEPTH + GX_RASTER)
        self.state = body[state_at:r.p]
        self.vs, self.vs_ref = program(r, self.flags & VS_REF)
        self.fetch = r.words()
        self.ps, self.ps_ref = program(r, self.flags & PS_REF)
        self.vs_name = r.str()
        self.ps_name = r.str()
        mid_at = r.p
        # dims and modes of both stages, the vertex input, the relink's specialization
        for size in (5, 5, 1, 1, 16, 12, 16):
            r.take(size * r.u32())
        r.take(8 + 128 + 128 + 4)
        self.mid = body[mid_at:r.p]
        tess_at = r.p
        self.tess_flags = r.u32()
        r.take(9 * 4 + 16 + 8 + 48 + 8)  # its registers, factors, constants and constant buffers
        self.tess = body[tess_at:r.p]
        self.ls = self.ls_ref = self.hs = self.hs_ref = None
        self.ls_name = self.hs_name = ""
        self.ls_sgpr = self.hs_sgpr = 0
        if self.tess_flags & TESS_HAS_LS:
            self.ls, self.ls_ref = program(r, self.flags & LS_REF)
            self.ls_name, self.ls_sgpr = r.str(), r.u32()
        if self.tess_flags & TESS_HAS_HS:
            self.hs, self.hs_ref = program(r, self.flags & HS_REF)
            self.hs_name, self.hs_sgpr = r.str(), r.u32()
        self.ls_fetch = r.words() if self.tess_flags & TESS_HAS_FETCH else None
        # The host tessellator's LS passes (version 3, flag 1024): a count,
        # then each one's registers, its LS (named in a list), its fetch
        # shader's words.
        self.lsps = []
        if self.tess_flags & TESS_LS_PASSES:
            for _ in range(r.u32()):
                ref = r.u32() & 1
                regs = r.take(5 * 4)
                prog, pref = program(r, ref)
                name, sgpr = r.str(), r.u32()
                fetch = r.words() if r.u32() else None
                self.lsps.append({"regs": regs, "prog": prog, "ref": pref, "name": name, "sgpr": sgpr, "fetch": fetch})
        if r.p != r.end:
            raise ValueError("a record longer than its fields")
        return self

    def body(self):
        out = [w32(self.flags), struct.pack("<Q", self.key), w32(self.uses), w32(self.last_run), w32(self.order), self.state,
               program_bytes(self.vs, self.vs_ref), w32(len(self.fetch) // 4) + self.fetch, program_bytes(self.ps, self.ps_ref),
               wstr(self.vs_name), wstr(self.ps_name), self.mid, self.tess]
        if self.tess_flags & TESS_HAS_LS:
            out += [program_bytes(self.ls, self.ls_ref), wstr(self.ls_name), w32(self.ls_sgpr)]
        if self.tess_flags & TESS_HAS_HS:
            out += [program_bytes(self.hs, self.hs_ref), wstr(self.hs_name), w32(self.hs_sgpr)]
        if self.tess_flags & TESS_HAS_FETCH:
            out.append(w32(len(self.ls_fetch) // 4) + self.ls_fetch)
        if self.tess_flags & TESS_LS_PASSES:
            out.append(w32(len(self.lsps)))
            for e in self.lsps:
                out += [w32(1 if e["prog"] is None else 0), e["regs"], program_bytes(e["prog"], e["ref"]), wstr(e["name"]), w32(e["sgpr"])]
                out.append(w32(1) + w32(len(e["fetch"]) // 4) + e["fetch"] if e["fetch"] is not None else w32(0))
        return b"".join(out)


def read_file(path):
    d = Path(path).read_bytes()
    r = Reader(d)
    if r.u32() != MAGIC or r.u32() not in READS:
        raise ValueError("%s: not a pipeline manifest this tool reads (versions %s)" % (path, READS))
    run = r.u32()
    origin = {"translator": r.str(), "vendor": r.u32(), "device": r.u32(), "driver": r.u32(), "uuid": r.take(16).hex(),
              "libraries": bool(r.u8())}
    recipes = []
    for _ in range(r.u32()):
        n = r.u32()
        recipes.append(parse_record(r.take(n)))
    return run, origin, recipes


def write_file(path, run, origin, recipes):
    head = [w32(MAGIC), w32(VERSION), w32(run), wstr(origin["translator"]), w32(origin["vendor"]), w32(origin["device"]),
            w32(origin["driver"]), bytes.fromhex(origin["uuid"]), bytes([1 if origin["libraries"] else 0]), w32(len(recipes))]
    body = b"".join(w32(len(b)) + b for b in (rc.body() for rc in recipes))
    tmp = str(path) + ".tmp"
    Path(tmp).write_bytes(b"".join(head) + body)
    os.replace(tmp, path)


def bundle_programs(app0):
    """name -> code words (bytes) of every shader in the game's bundles, as bbhost's index names them."""
    out = {}
    for b in BUNDLES:
        path = Path(app0) / "dvdroot_ps4" / "shader" / b
        if not path.is_file():
            continue
        plain = shaderbnd.dcx_decompress(path.read_bytes())
        for name, size, off, ident, flags in shaderbnd.bnd4_entries(plain):
            c = plain[off:off + size]
            foot = c.rfind(b"OrbShdr")
            if foot < 0 or foot + 20 > len(c):
                continue
            h0 = struct.unpack_from("<I", c, foot + 16)[0]
            key = "%08x" % h0
            if key in out:
                continue
            # The code: the container's ShaderFileHeader gives the code's offset
            # (gcn::shader_code): 'Shdr' at s, header size in dwords at s + 9,
            # the code right after it; its length in the footer's word at +8.
            s = c.find(b"Shdr")
            first = c.find(b"OrbShdr", s) if s >= 0 else -1
            if s < 0 or s + 16 > len(c) or first < 0 or first + 12 > len(c):
                continue
            length = struct.unpack_from("<I", c, first + 8)[0] >> 8
            code_at = s + 16 + c[s + 9] * 4
            if code_at + length > first or length == 0 or length % 4:
                continue
            out[key] = c[code_at:code_at + length]
    return out


def eboot_programs(eboot):
    """name -> code words of the shader containers the eboot carries itself
    (host/shader_bundles.cpp eboot_shaders: a 'Shdr' header of a vertex, pixel
    or compute shader, the first 'OrbShdr' footer after it)."""
    out = {}
    if not eboot or not Path(eboot).is_file():
        return out
    d = Path(eboot).read_bytes()
    at = 0
    while True:
        s = d.find(b"Shdr", at)
        if s < 0:
            break
        at = s + 1
        if s >= 3 and d[s - 3:s] == b"Orb":
            continue
        if s + 0x60 > len(d) or (d[s + 8] & 0xF) not in (1, 2, 4):
            continue
        foot = d.find(b"OrbShdr", s + 16, s + (1 << 20))
        if foot < 0 or foot + 28 > len(d):
            continue
        name = "%08x" % struct.unpack_from("<I", d, foot + 16)[0]
        if name in out:
            continue
        length = struct.unpack_from("<I", d, foot + 8)[0] >> 8
        code_at = s + 16 + d[s + 9] * 4
        if code_at + length > foot or length == 0 or length % 4:
            continue
        out[name] = d[code_at:code_at + length]
    return out


def describe(path, run, origin, records):
    recipes = [rc for rc in records if not rc.compute]
    computes = [c for c in records if c.compute]
    names = Counter()
    for rc in recipes:
        names[rc.vs_name] += 1
        names[rc.ps_name] += 1
    for c in computes:
        names[c.name] += 1
    words = sum(len(rc.vs or b"") + len(rc.ps or b"") + len(rc.fetch) + len(rc.ls or b"") + len(rc.hs or b"") + len(rc.ls_fetch or b"")
                for rc in recipes) + sum(len(c.prog or b"") for c in computes)
    print("%s: run %d, %d pipelines (%d with an optimized relink's specialization, %d on vertex input, %d tessellated, %d with LS passes) "
          "and %d compute pipelines, %d programs by name; "
          "%.1f KiB of program words; made by translator %s on device %04x:%04x driver %08x%s" % (
              path, run, len(recipes), sum(1 for rc in recipes if rc.flags & RELINK),
              sum(1 for rc in recipes if rc.flags & HAS_VERTEX_INPUT), sum(1 for rc in recipes if rc.tess_flags & 1 or rc.tess_flags & 32),
              sum(1 for rc in recipes if rc.lsps), len(computes),
              len(names), words / 1024.0, origin["translator"][:12],
              origin["vendor"], origin["device"], origin["driver"], ", with pipeline libraries" if origin["libraries"] else ", whole pipelines"))
    used = Counter(rc.uses for rc in records)
    print("  starts that used each: %s" % ", ".join("%d: %d" % (k, used[k]) for k in sorted(used)))


def make_list(recipes, progs, eboot_given=True, fetch_words=True):
    """The list records: each program named by its footer hash, count and hash
    instead of its words. (out, left Counter, plain, patched)."""
    out, left = [], Counter()
    plain = patched = 0
    for rc in recipes:
        if rc.compute:
            if rc.prog is not None and rc.name not in progs:
                left["a compute program neither the bundles nor the eboot hold" if eboot_given else
                     "a compute program the bundles do not hold (the eboot's own: give --eboot)"] += 1
                continue
            if rc.prog is not None:
                if progs[rc.name] == rc.prog:
                    plain += 1
                else:
                    patched += 1
                rc.flags |= VS_REF
                rc.ref, rc.prog = (len(rc.prog) // 4, fnv1a(rc.prog)), None
            rc.uses = rc.last_run = 0
            out.append(rc)
            continue
        if rc.vs_name not in progs or (rc.ps is not None and rc.ps and rc.ps_name not in progs):
            left["a program neither the bundles nor the eboot hold" if eboot_given else
                 "a program the bundles do not hold (the eboot's own: give --eboot)"] += 1
            continue
        if not fetch_words and ((rc.fetch and not rc.flags & HAS_VERTEX_INPUT) or rc.ls_fetch is not None or
                                any(e["fetch"] is not None for e in rc.lsps)):
            left["a fetch shader's words (made at run time; --no-fetch-words)"] += 1
            continue
        if (rc.ls is not None and rc.ls_name not in progs) or (rc.hs is not None and rc.hs_name not in progs) or \
                any(e["prog"] is not None and e["name"] not in progs for e in rc.lsps):
            left["a tessellation program neither the bundles nor the eboot hold"] += 1
            continue
        for prog, name in ((rc.vs, rc.vs_name), (rc.ps, rc.ps_name)):
            if prog:
                if progs[name] == prog:
                    plain += 1
                else:
                    patched += 1  # bbhost patched it (host/shader_patch.cpp): the hash is the patched words'
        rc.flags |= VS_REF
        rc.vs_ref = (len(rc.vs) // 4, fnv1a(rc.vs))
        rc.vs = None
        if rc.ps:
            rc.flags |= PS_REF
            rc.ps_ref = (len(rc.ps) // 4, fnv1a(rc.ps))
            rc.ps = None
        if rc.flags & HAS_VERTEX_INPUT:
            rc.fetch = b""  # unused on vertex input (vs_translate_options)
        if rc.ls is not None:
            rc.flags |= LS_REF
            rc.ls_ref, rc.ls = (len(rc.ls) // 4, fnv1a(rc.ls)), None
        if rc.hs is not None:
            rc.flags |= HS_REF
            rc.hs_ref, rc.hs = (len(rc.hs) // 4, fnv1a(rc.hs)), None
        for e in rc.lsps:
            if e["prog"] is not None:
                e["ref"], e["prog"] = (len(e["prog"]) // 4, fnv1a(e["prog"])), None
        rc.uses = rc.last_run = 0
        out.append(rc)
    return out, left, plain, patched


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats")
    p.add_argument("files", nargs="+")
    p = sub.add_parser("merge")
    p.add_argument("--out", required=True)
    p.add_argument("files", nargs="+")
    p = sub.add_parser("list")
    p.add_argument("--app0", required=True)
    p.add_argument("--eboot", help="the 1.09 eboot: its own programs are named too (else left out)")
    p.add_argument("--no-fetch-words", action="store_true", help="leave out the records that need a fetch shader's words")
    p.add_argument("--out", required=True)
    p.add_argument("files", nargs="+")
    p = sub.add_parser("check")
    p.add_argument("--app0", required=True)
    p.add_argument("--eboot")
    p.add_argument("file")
    args = ap.parse_args()

    if args.cmd == "stats":
        for f in args.files:
            describe(f, *read_file(f))
        return 0

    if args.cmd in ("merge", "list"):
        merged, origin, run = {}, None, 0
        for f in args.files:
            r, o, recipes = read_file(f)
            origin = origin or o
            run = max(run, r)
            for rc in recipes:
                have = merged.get((rc.compute, rc.key))
                if have is None or rc.last_run > have.last_run:
                    merged[(rc.compute, rc.key)] = rc
        recipes = list(merged.values())
        if args.cmd == "merge":
            write_file(args.out, run, origin, recipes)
            describe(args.out, *read_file(args.out))
            return 0
        progs = bundle_programs(args.app0)
        if not progs:
            print("pipeline_list: no shader bundles under %s" % args.app0, file=sys.stderr)
            return 2
        for name, code in eboot_programs(args.eboot).items():
            progs.setdefault(name, code)
        out, left, plain, patched = make_list(recipes, progs, bool(args.eboot), not args.no_fetch_words)
        write_file(args.out, 0, origin, out)
        size = Path(args.out).stat().st_size
        print("%s: %d of %d pipelines (%.1f KiB, no program words: %d programs as the game's files have them, %d as bbhost patches them)" % (
            args.out, len(out), len(recipes), size / 1024.0, plain, patched))
        for why, n in left.most_common():
            print("  left out: %d with %s" % (n, why))
        return 0

    run, origin, recipes = read_file(args.file)
    progs = bundle_programs(args.app0)
    for name, code in eboot_programs(args.eboot).items():
        progs.setdefault(name, code)
    ok = bad = patched = 0
    for rc in recipes:
        if rc.compute:
            refs = [(rc.prog is None, rc.ref, rc.name)]
        else:
            refs = [(rc.flags & flag, ref, name) for flag, ref, name in
                    ((VS_REF, rc.vs_ref, rc.vs_name), (PS_REF, rc.ps_ref, rc.ps_name), (LS_REF, rc.ls_ref, rc.ls_name),
                     (HS_REF, rc.hs_ref, rc.hs_name))]
            refs += [(e["prog"] is None, e["ref"], e["name"]) for e in rc.lsps]
        for named, ref, name in refs:
            if not named:
                continue
            code = progs.get(name)
            if code is None:
                bad += 1
            elif len(code) // 4 == ref[0] and fnv1a(code) == ref[1]:
                ok += 1
            else:
                patched += 1
    print("%s: %d pipelines and %d compute pipelines; program references: %d match the game's files, %d differ (bbhost's shader patches, "
          "checked at its start), %d not in these files" % (args.file, sum(1 for rc in recipes if not rc.compute),
                                                            sum(1 for rc in recipes if rc.compute), ok, patched, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
