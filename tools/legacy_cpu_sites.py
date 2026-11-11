#!/usr/bin/env python3
"""The game's BMI1, LZCNT, TZCNT and MOVBE instructions, for core/legacy_cpu.cpp: a processor that lacks
them (Intel's 2nd and 3rd generation Core, AMD FX before Excavator) runs each through a routine of
ours, which bbhost writes at load in the instruction's place.

    tools/legacy_cpu_sites.py EBOOT [--bn-functions symbols/1.09-functions.txt.gz] [--objdump objdump]
                              [--write src/core/legacy_cpu_sites.inc | --check src/core/legacy_cpu_sites.inc]
                              [--sites FILE]
    tools/legacy_cpu_sites.py check     the build's test: the checkout's table against the eboot (BBHOST_EBOOT,
                                        or eboot-109-decrypted.bin in the checkout), with objdump when there is
                                        one; exit 77 without the eboot or capstone

The code is the functions the eboot's .eh_frame lists (its call-frame table: every compiled function,
with its start and length), plus those Binary Ninja found that it does not (--bn-functions: one start a
line, as Binary Ninja addresses; each runs to the next start). A function is disassembled from its start
with capstone up to its first switch table: the compiler puts a function's tables of case offsets after
its code, inside the function's length (the `lea; movsxd; add; jmp` switches, read here entry by entry),
and nothing branches past them. A second disassembler, GNU objdump (--objdump), sweeps the same functions
on its own and must find the same instructions of the four families; the gaps between the functions are
swept too and what they hold is listed.

Every instruction of the code is also checked against the processors bbhost supports: SSE up to 4.2,
POPCNT and AVX are the floor, and an instruction needing anything newer (AVX2, FMA, F16C, BMI2, AES, ...)
other than the four families here is an error, because no routine covers it.

A routine is reached by a 5-byte `jmp`, written over the instruction. A site shorter than that (some
LZCNT, TZCNT and MOVBE: 4 bytes) takes whole instructions next to it into its routine as well - the ones
after it, else the ones before it - so that the jump fits. That is sound only when nothing enters the
taken bytes anywhere but at their first instruction, which the tool checks against every direct branch
and call in the code, the switch tables' cases, the code addresses the relocations put in data, the
functions' starts and the exception tables' landing pads. A taken instruction must also run from the
routine as it ran in place: no call (its return address would be ours) and no loop/jrcxz (no rel32 form).
A run of short sites shares one routine. Binary Ninja's cross-references to the taken instructions are a
third check, made by hand from the list --sites writes.

The table written (--write) is the address (Binary Ninja's), length and family of every site, and the
bytes of whole instructions a short site's routine takes before and after it; nothing of the game's
bytes. bbhost checks at load that each site still decodes to its family, and that the taken bytes still
decode to whole instructions, before it writes anything there. --check compares a table with what the
eboot gives.

Exit status: 0 when the table was written or matches, 1 when something above failed, 2 for bad arguments.
"""
import argparse
import bisect
import collections
import gzip
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys

try:
    import capstone
except ImportError:
    capstone = None

EBOOT_109 = "941f887a562aae054fac35af8cc8f27cf075f3d4cc2e029fb5ae2a663aaa5ae7"
BN_BASE = 0x400000  # Binary Ninja places the ELF image here; bbhost's tables use its addresses
PT_LOAD, PT_DYNAMIC, PT_GNU_EH_FRAME = 1, 2, 0x6474E550
PT_SCE_DYNLIBDATA = 0x61000000
DT_SCE_RELA, DT_SCE_RELASZ, DT_SCE_RELAENT = 0x61000033, 0x61000031, 0x6100002F
R_X86_64_RELATIVE = 8
JUMP = 5  # a jmp rel32

# The families, in core/legacy_cpu_code.h's order (LegacyFamily).
FAMILIES = ["andn", "bextr", "blsi", "blsmsk", "blsr", "lzcnt", "tzcnt", "movbe"]
FAMILY_NAMES = {"andn": "kAndn", "bextr": "kBextr", "blsi": "kBlsi", "blsmsk": "kBlsmsk", "blsr": "kBlsr",
                "lzcnt": "kLzcnt", "tzcnt": "kTzcnt", "movbe": "kMovbe"}

# Instructions past the supported floor (SSE4.2, POPCNT, AVX: Intel 2nd gen Core, AMD FX). VEX, EVEX and
# XOP ones by capstone's groups, the legacy encodings by mnemonic.
BEYOND_GROUPS = {
    "X86_GRP_AVX2": "AVX2", "X86_GRP_FMA": "FMA", "X86_GRP_FMA4": "FMA4", "X86_GRP_F16C": "F16C",
    "X86_GRP_BMI2": "BMI2", "X86_GRP_ADX": "ADX", "X86_GRP_SHA": "SHA", "X86_GRP_AVX512": "AVX-512",
    "X86_GRP_XOP": "XOP", "X86_GRP_TBM": "TBM", "X86_GRP_SSE4A": "SSE4a", "X86_GRP_3DNOW": "3DNow!",
    "X86_GRP_RTM": "RTM", "X86_GRP_HLE": "HLE", "X86_GRP_FSGSBASE": "FSGSBASE", "X86_GRP_SGX": "SGX",
    "X86_GRP_AES": "AES", "X86_GRP_PCLMUL": "PCLMUL",
}
BEYOND_MNEMONICS = {
    "rdrand": "RDRAND", "rdseed": "RDSEED", "adcx": "ADX", "adox": "ADX", "clflushopt": "CLFLUSHOPT",
    "clwb": "CLWB", "prefetchw": "PRFCHW", "prefetchwt1": "PREFETCHWT1", "xsavec": "XSAVEC",
    "xsaves": "XSAVES", "xrstors": "XSAVES", "rdpid": "RDPID", "movdiri": "MOVDIRI", "movdir64b": "MOVDIR64B",
    "invpcid": "INVPCID",
}
LEGACY_BEYOND_FEATURES = [("aes", "AES"), ("pclmul", "PCLMUL"), ("vpclmul", "PCLMUL"), ("extrq", "SSE4a"),
                          ("insertq", "SSE4a"), ("movntss", "SSE4a"), ("movntsd", "SSE4a"), ("sha1", "SHA"),
                          ("sha256", "SHA")]
LEGACY_BEYOND = tuple(p for p, _ in LEGACY_BEYOND_FEATURES)
SWITCH_LOAD = re.compile(r"\w+, dword ptr \[(\w+) \+ \w+\*4\]$")
RIP_LEA = re.compile(r"(\w+), \[rip \+ (0x[0-9a-f]+)\]$")
REG_COPY = re.compile(r"(r\w+), (r\w+)$")


def s32(b, o):
    return struct.unpack_from("<i", b, o)[0]


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def uleb(b, o):
    v = shift = 0
    while True:
        x = b[o]
        o += 1
        v |= (x & 0x7F) << shift
        shift += 7
        if not x & 0x80:
            return v, o


def sleb(b, o):
    v = shift = 0
    while True:
        x = b[o]
        o += 1
        v |= (x & 0x7F) << shift
        shift += 7
        if not x & 0x80:
            if x & 0x40:
                v -= 1 << shift
            return v, o


class Image:
    """The ELF's loadable segments, read by ELF address."""

    def __init__(self, blob):
        self.blob = blob
        e_phoff = struct.unpack_from("<Q", blob, 0x20)[0]
        e_phentsize, e_phnum = struct.unpack_from("<HH", blob, 0x36)
        self.phdrs = []
        for i in range(e_phnum):
            t, fl, off, va, _pa, fsz, msz = struct.unpack_from("<IIQQQQQ", blob, e_phoff + i * e_phentsize)
            self.phdrs.append((t, fl, off, va, fsz, msz))
        self.loads = [p for p in self.phdrs if p[0] == PT_LOAD]
        code = [p for p in self.loads if p[1] & 1]
        if len(code) != 1:
            sys.exit("expected one executable segment")
        _t, _fl, self.code_off, self.code_va, self.code_size, _msz = code[0]

    def off(self, va):
        for _t, _fl, off, pva, fsz, _msz in self.loads:
            if pva <= va < pva + fsz:
                return off + va - pva
        return None

    def read(self, va, n):
        o = self.off(va)
        return self.blob[o:o + n] if o is not None else b""

    def in_code(self, va):
        return self.code_va <= va < self.code_va + self.code_size


def pointer(enc, img, va):
    """A DW_EH_PE-encoded pointer at ELF address va: (value, size)."""
    fmt = enc & 0x0F
    b = img.read(va, 8)
    if fmt == 0x0B:
        v, n = s32(b, 0), 4
    elif fmt == 0x03:
        v, n = u32(b, 0), 4
    elif fmt in (0x04, 0x0C):
        v, n = struct.unpack_from("<q" if fmt == 0x0C else "<Q", b, 0)[0], 8
    else:
        raise ValueError("pointer encoding %#x" % enc)
    if enc & 0x70 == 0x10:
        v += va
    return v, n


def functions_from_eh_frame(img):
    """Each FDE's [start, end) and its LSDA (0 when none), from .eh_frame_hdr's table."""
    hdr = [p for p in img.phdrs if p[0] == PT_GNU_EH_FRAME][0][3]
    b = img.read(hdr, 12)
    if b[0] != 1 or b[2] != 0x03 or b[3] != 0x3B:
        sys.exit(".eh_frame_hdr: unexpected encodings")
    count = u32(b, 8)
    table = img.read(hdr + 12, count * 8)
    cies = {}
    out = []
    for i in range(count):
        fde = hdr + s32(table, i * 8 + 4)
        fb = img.read(fde, 16)
        if u32(fb, 0) == 0xFFFFFFFF:
            sys.exit("64-bit FDE at %#x" % fde)
        cie = fde + 4 - u32(fb, 4)
        if cie not in cies:
            cb = img.read(cie, 128)
            o = 9
            aug = b""
            while cb[o]:
                aug += bytes([cb[o]])
                o += 1
            o += 1
            _, o = uleb(cb, o)  # code alignment
            _, o = sleb(cb, o)  # data alignment
            _, o = uleb(cb, o)  # return address register
            fde_enc, lsda_enc = 0, None
            if aug.startswith(b"z"):
                _, o = uleb(cb, o)
                for c in aug[1:]:
                    if c == ord("R"):
                        fde_enc = cb[o]
                        o += 1
                    elif c == ord("L"):
                        lsda_enc = cb[o]
                        o += 1
                    elif c == ord("P"):
                        penc = cb[o]
                        _, n = pointer(penc, img, cie + o + 1)
                        o += 1 + n
            cies[cie] = (fde_enc, lsda_enc, aug)
        fde_enc, lsda_enc, aug = cies[cie]
        start, n = pointer(fde_enc, img, fde + 8)
        size, n2 = pointer(fde_enc & 0x0F, img, fde + 8 + n)
        lsda = 0
        if aug.startswith(b"z") and lsda_enc is not None and lsda_enc != 0xFF:
            o = fde + 8 + n + n2
            alen, o2 = uleb(img.read(o, 8), 0)
            if alen:
                lsda, _ = pointer(lsda_enc, img, o + o2)
        out.append((start, start + size, lsda))
    return out


def landing_pads(img, start, lsda):
    """The landing pads of one function's LSDA (the exception table's call sites)."""
    b = img.read(lsda, 8192)
    o = 0
    lpstart = start
    if b[o] != 0xFF:
        lpstart, n = pointer(b[o], img, lsda + 1)
        o += n
    o += 1
    if b[o] != 0xFF:
        o += 1
        _, o = uleb(b, o)
    else:
        o += 1
    cs_enc = b[o]
    o += 1
    cs_len, o = uleb(b, o)
    end = o + cs_len
    pads = []
    while o < end:
        vals = []
        for _ in range(3):
            if cs_enc == 0x01:
                v, o = uleb(b, o)
            elif cs_enc in (0x03, 0x0B):
                v = u32(b, o) if cs_enc == 0x03 else s32(b, o)
                o += 4
            else:
                raise ValueError("call-site encoding %#x" % cs_enc)
            vals.append(v)
        _, o = uleb(b, o)  # action
        if vals[2]:
            pads.append(lpstart + vals[2])
    return pads


def relocation_targets(img):
    """The code addresses R_X86_64_RELATIVE relocations write into the image (vtables, tables of functions)."""
    dynlib = [p for p in img.phdrs if p[0] == PT_SCE_DYNLIBDATA][0][2]
    dyn = [p for p in img.phdrs if p[0] == PT_DYNAMIC][0]
    rela = relasz = relaent = 0
    for o in range(dyn[2], dyn[2] + dyn[4], 16):
        tag, val = struct.unpack_from("<QQ", img.blob, o)
        if tag == DT_SCE_RELA:
            rela = val
        elif tag == DT_SCE_RELASZ:
            relasz = val
        elif tag == DT_SCE_RELAENT:
            relaent = val
    if rela == 24 and relaent > 24:  # this eboot swaps the two (src/core/elf.cpp)
        rela, relaent = relaent, rela
    out = set()
    base = dynlib + rela
    for o in range(base, base + relasz, 24):
        _off, info, addend = struct.unpack_from("<QQq", img.blob, o)
        if info & 0xFFFFFFFF == R_X86_64_RELATIVE and img.in_code(addend):
            out.add(addend)
    return out


class Code:
    """The decoded instructions of every function, and what reaches into them."""

    def __init__(self, img, functions, isa_audit):
        self.img = img
        self.insns = {}  # va -> (size, mnemonic, op_str)
        self.function_of = {}  # va -> function start (sites and their neighbours)
        self.targets = collections.defaultdict(set)  # va -> where it is reached from
        self.sites = []
        self.beyond = collections.Counter()
        self.beyond_where = {}
        self.tables = 0
        self.cases = 0
        self.problems = []
        self.code_end = {}  # function start -> where its code ends (its first switch table, else its end)
        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        detail = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        detail.detail = True
        groups = {getattr(capstone.x86_const, g): name for g, name in BEYOND_GROUPS.items()}
        code = img.blob
        for start, end in functions:
            o = img.off(start)
            seq = list(md.disasm_lite(code[o:o + (end - start)], start))
            # The switch tables: movsxd X, dword ptr [REG + I*4] where REG was last given an address by a
            # rip-relative lea (or a copy of one). The tables follow the code, so a pattern past the first
            # table found is not looked at (it is the tables' bytes read as instructions).
            tables = []
            lea_of = {}
            cut = end
            for k, (va, size, mn, ops) in enumerate(seq):
                if va >= cut:
                    break
                if mn == "lea":
                    lm = RIP_LEA.match(ops)
                    if lm:
                        lea_of[lm.group(1)] = (va, va + size + int(lm.group(2), 16))
                    continue
                if mn == "mov":
                    mm = REG_COPY.match(ops)
                    if mm and mm.group(2) in lea_of:
                        lea_of[mm.group(1)] = lea_of[mm.group(2)]
                    continue
                if mn != "movsxd":
                    continue
                m = SWITCH_LOAD.match(ops)
                if not m or m.group(1) not in lea_of:
                    continue
                lea, table = lea_of[m.group(1)]
                tables.append((table, lea))
                if start <= table < cut and table > va:
                    cut = table
            first_table = min([t for t, _ in tables if start <= t < end], default=end)
            self.code_end[start] = first_table
            last = start
            for va, size, mn, ops in seq:
                if va >= first_table or va + size > first_table:
                    break
                last = va + size
                self.insns[va] = (size, mn, ops)
                if mn in FAMILIES:
                    self.sites.append(va)
                    self.function_of[va] = (start, first_table)
                if (mn[0] == "j" or mn == "call") and ops.startswith("0x"):
                    self.targets[int(ops, 16)].add("a %s at %#x" % (mn, va + BN_BASE))
                if isa_audit:
                    b0 = code[img.off(va)]
                    vex = b0 in (0xC4, 0xC5, 0x62) or (b0 == 0x8F and code[img.off(va) + 1] & 0x1F >= 8)
                    if vex or mn in BEYOND_MNEMONICS or mn.startswith(LEGACY_BEYOND):
                        feat = BEYOND_MNEMONICS.get(mn) or next(
                            (f for p, f in LEGACY_BEYOND_FEATURES if mn.startswith(p)), None)
                        if not feat and vex:
                            ins = next(detail.disasm(code[img.off(va):img.off(va) + size], va), None)
                            for g in ins.groups if ins is not None else ():
                                if g in groups:
                                    feat = groups[g]
                                    break
                        if feat and mn not in FAMILIES:
                            self.beyond[feat] += 1
                            self.beyond_where.setdefault(feat, (va, mn, ops))
            if last < first_table and first_table - last > 15 and first_table == end:
                # Stopped short of the end with no table to explain it: say so (padding is shorter).
                tail = code[img.off(last):img.off(end)]
                if tail.strip(b"\x00\x90\xcc"):
                    self.problems.append("%#x: decoded to %#x of %#x" % (start + BN_BASE, last + BN_BASE, end + BN_BASE))
            # The cases: each table's entries, read while they land in the function's code. A bound from a
            # `cmp` ahead of the switch is not trusted (the nearest cmp of the index is often another
            # test's); reading on past a table's end only adds addresses to check.
            for table, lea in tables:
                if not img.in_code(table):
                    continue
                self.tables += 1
                for i in range(4096):
                    b = img.read(table + 4 * i, 4)
                    if len(b) < 4:
                        break
                    t = table + s32(b, 0)
                    if not start <= t < first_table:
                        break
                    self.targets[t].add("a case of the switch at %#x" % (lea + BN_BASE))
                    self.cases += 1
        self.sites.sort()
        self.starts = sorted(self.insns)


def regions(code, sites, entries):
    """Each site's routine region: (site, before, after) - the bytes of whole instructions taken before and
    after it. A short site takes the instructions after it until the jump fits, else the ones before it;
    sites inside another's region come back as (site, None, None)."""
    out = []
    problems = []
    covered_to = -1
    site_set = set(sites)

    def movable(va):
        size, mn, ops = code.insns[va]
        if va in site_set:
            return True
        if mn == "call" or mn.startswith("loop") or mn in ("jrcxz", "jecxz"):
            return False
        return True

    def why_not_inside(va):
        return sorted(entries.get(va, ()))

    for s in sites:
        if s < covered_to:
            out.append((s, None, None))
            continue
        size = code.insns[s][0]
        if size >= JUMP:
            out.append((s, 0, 0))
            covered_to = s + size
            continue
        fstart, fend = code.function_of[s]
        # After: whole instructions up to the jump's length, none of them entered from elsewhere.
        after = 0
        va = s + size
        ok = True
        while size + after < JUMP:
            if va not in code.insns or va >= fend or why_not_inside(va) or not movable(va):
                ok = False
                break
            n = code.insns[va][0]
            mn = code.insns[va][1]
            after += n
            va += n
            if size + after < JUMP and (mn in ("jmp", "ret") or mn.startswith("ret")):
                ok = False  # the region would run on past an unconditional exit into what follows
                break
        if ok:
            out.append((s, 0, after))
            covered_to = s + size + after
            continue
        # Before: the instructions ahead of it, the site itself not entered from elsewhere.
        i = bisect.bisect_left(code.starts, s)
        before = 0
        ok = not why_not_inside(s)
        j = i - 1
        while ok and size + before < JUMP:
            if j < 0:
                ok = False
                break
            va = code.starts[j]
            n = code.insns[va][0]
            if va + n != s - before or va < fstart or va < covered_to or not movable(va):
                ok = False
                break
            before += n
            if size + before < JUMP and why_not_inside(va):
                ok = False
                break
            j -= 1
        if ok:
            out.append((s, before, 0))
            covered_to = s + size
            continue
        problems.append("%#x: no region - the instructions after it (%s) and before it are entered from elsewhere"
                        % (s + BN_BASE, ", ".join(why_not_inside(s + size)) or "or not movable"))
        out.append((s, 0, 0))
    return out, problems


def objdump_sites(objdump, img, functions, tmpdir):
    """The family instructions GNU objdump finds in the functions, each disassembled on its own: the
    functions laid end to end in one file, 16 bytes of int3 between them so that its sweep starts each
    function in step."""
    pad = b"\xcc" * 16
    parts, where = [], []
    pos = 0
    for start, end in functions:
        parts.append(pad)
        pos += len(pad)
        where.append((pos, start, end))
        parts.append(img.read(start, end - start))
        pos += end - start
    path = os.path.join(tmpdir, "legacy_cpu_functions_%d.bin" % os.getpid())
    with open(path, "wb") as f:
        f.write(b"".join(parts))
    found = {}
    try:
        p = subprocess.Popen([objdump, "-D", "-b", "binary", "-m", "i386:x86-64", path], stdout=subprocess.PIPE,
                             text=True, errors="replace")
        line_re = re.compile(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2} )+\s*(\w+)")
        offs = [w[0] for w in where]
        for line in p.stdout:
            m = line_re.match(line)
            if not m or m.group(2) not in FAMILIES:
                continue
            o = int(m.group(1), 16)
            i = bisect.bisect_right(offs, o) - 1
            if i >= 0:
                pos0, start, end = where[i]
                if o - pos0 < end - start:
                    found[start + o - pos0] = m.group(2)
        p.wait()
    finally:
        os.unlink(path)
    return found


def gap_sites(img, functions):
    """Family instructions a linear sweep finds between the known functions (data, or code no table names)."""
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.skipdata = True
    out = []
    pos = img.code_va
    for start, end in functions + [(img.code_va + img.code_size, 0)]:
        if start > pos:
            o = img.off(pos)
            for va, size, mn, ops in md.disasm_lite(img.blob[o:o + (start - pos)], pos):
                if mn in FAMILIES:
                    out.append((va, mn, ops))
        pos = max(pos, end)
    return out


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "check":
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        eboot = os.environ.get("BBHOST_EBOOT", os.path.join(root, "eboot-109-decrypted.bin"))
        if not os.path.exists(eboot) or capstone is None:
            print("skipped: %s" % ("no capstone" if capstone is None else "no eboot at %s (BBHOST_EBOOT)" % eboot))
            return 77
        argv = [eboot, "--check", os.path.join(root, "src", "core", "legacy_cpu_sites.inc")]
        objdump = shutil.which("objdump")
        if objdump:
            argv += ["--objdump", objdump]
        sys.argv[1:] = argv
    if capstone is None:
        print("capstone is not installed (pip install capstone)")
        return 2
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("eboot")
    ap.add_argument("--bn-functions", help="Binary Ninja's function starts (one hex address a line, .gz or text)")
    ap.add_argument("--objdump", help="GNU objdump, for the second disassembly")
    ap.add_argument("--write", help="write the table here")
    ap.add_argument("--check", help="compare this table with the eboot's")
    ap.add_argument("--sites", help="write the sites, the regions and the gaps' finds here")
    ap.add_argument("--no-isa", action="store_true", help="skip the check of what needs more than the floor")
    a = ap.parse_args()

    blob = open(a.eboot, "rb").read()
    sha = hashlib.sha256(blob).hexdigest()
    if sha != EBOOT_109:
        print("%s is not the 1.09 eboot (sha256 %s)" % (a.eboot, sha))
        return 2
    img = Image(blob)
    problems = []

    fdes = functions_from_eh_frame(img)
    ranges = sorted({(s, e) for s, e, _ in fdes if e > s and img.in_code(s)})
    fde_starts = {s for s, _ in ranges}
    pads = set()
    for s, _e, lsda in fdes:
        if lsda:
            pads.update(landing_pads(img, s, lsda))
    bn_starts = set()
    if a.bn_functions:
        opener = gzip.open if a.bn_functions.endswith(".gz") else open
        with opener(a.bn_functions, "rt") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    bn_starts.add(int(line, 16) - BN_BASE)
    # Binary Ninja's functions outside every FDE's run to the next known start.
    starts_sorted = sorted(fde_starts | bn_starts)
    fde_lo = [s for s, _ in ranges]
    extra = []
    for s in sorted(bn_starts - fde_starts):
        i = bisect.bisect_right(fde_lo, s) - 1
        if i >= 0 and s < ranges[i][1]:
            continue
        j = bisect.bisect_right(starts_sorted, s)
        e = starts_sorted[j] if j < len(starts_sorted) else img.code_va + img.code_size
        extra.append((s, min(e, s + 0x10000)))
    functions = sorted(ranges + extra)

    code = Code(img, functions, not a.no_isa)
    problems += code.problems
    sites = code.sites
    reloc = relocation_targets(img)
    entries = collections.defaultdict(set)
    for va, why in code.targets.items():
        entries[va] |= why
    for va in reloc:
        entries[va].add("an address in the relocations")
    for va in fde_starts | bn_starts:
        entries[va].add("a function's start")
    for va in pads:
        entries[va].add("a landing pad")

    print("functions: %d from .eh_frame, %d more from Binary Ninja (of its %d); switch tables %d, cases %d; "
          "relocated code addresses %d; landing pads %d"
          % (len(ranges), len(extra), len(bn_starts), code.tables, code.cases, len(reloc), len(pads)))
    fam = collections.Counter(code.insns[s][1] for s in sites)
    print("sites: %d" % len(sites))
    for f in FAMILIES:
        if fam[f]:
            forms = collections.Counter(("rip" if "rip" in code.insns[s][2] else "mem" if "[" in code.insns[s][2]
                                         else "reg") for s in sites if code.insns[s][1] == f)
            sizes = collections.Counter(code.insns[s][0] for s in sites if code.insns[s][1] == f)
            print("  %-6s %5d  forms %s  lengths %s" % (f, fam[f], dict(sorted(forms.items())),
                                                       dict(sorted(sizes.items()))))
    if code.beyond:
        for feat, n in sorted(code.beyond.items()):
            va, mn, ops = code.beyond_where[feat]
            problems.append("needs %s: %d instructions (first %#x: %s %s)" % (feat, n, va + BN_BASE, mn, ops))
    elif not a.no_isa:
        print("nothing else in the code needs more than SSE4.2, POPCNT and AVX")

    regs, rproblems = regions(code, sites, entries)
    problems += rproblems
    short = [r for r in regs if r[1] is not None and code.insns[r[0]][0] < JUMP]
    inside = [r for r in regs if r[1] is None]
    print("short sites: %d with a region of their own (%d take what follows, %d what precedes), %d inside "
          "another's" % (len(short), sum(1 for r in short if r[2]), sum(1 for r in short if r[1]), len(inside)))

    if a.objdump:
        tmpdir = os.environ.get("TMPDIR", "/tmp")
        other = objdump_sites(a.objdump, img, functions, tmpdir)
        mine = {s: code.insns[s][1] for s in sites}
        # objdump runs on past a function's switch tables; what it finds there is not code.
        lo = [f[0] for f in functions]
        other = {v: m for v, m in other.items() if v < code.code_end[functions[bisect.bisect_right(lo, v) - 1][0]]}
        only_mine = sorted(set(mine) - set(other))
        only_other = sorted(set(other) - set(mine))
        differ = sorted(v for v in set(mine) & set(other) if mine[v] != other[v])
        for v in only_mine[:20]:
            problems.append("objdump does not find %#x %s" % (v + BN_BASE, mine[v]))
        for v in only_other[:20]:
            problems.append("objdump finds %#x %s, capstone does not" % (v + BN_BASE, other[v]))
        for v in differ[:20]:
            problems.append("%#x: capstone %s, objdump %s" % (v + BN_BASE, mine[v], other[v]))
        if only_mine or only_other or differ:
            problems.append("the disassemblers differ: %d only capstone's, %d only objdump's, %d of another family"
                            % (len(only_mine), len(only_other), len(differ)))
        else:
            print("objdump finds the same %d sites in the same functions" % len(other))

    gaps = gap_sites(img, functions)
    print("between the functions a linear sweep finds %d instructions of the families (data or unreachable)"
          % len(gaps))

    if a.sites:
        with open(a.sites, "w") as f:
            for s, before, after in regs:
                size, mn, ops = code.insns[s]
                form = "rip" if "rip" in ops else "mem" if "[" in ops else "reg"
                f.write("site %#x %d %s %s %s" % (s + BN_BASE, size, mn, form, ops))
                if before is None:
                    f.write(" | inside the previous region\n")
                    continue
                if before or after:
                    taken = []
                    va = s - before
                    while va < s + size + after:
                        n, m2, o2 = code.insns[va]
                        if va != s:
                            taken.append("%#x %s %s" % (va + BN_BASE, m2, o2))
                        va += n
                    f.write(" | region %#x+%d takes: %s" % (s - before + BN_BASE, before + size + after,
                                                           "; ".join(taken)))
                f.write("\n")
            for va, mn, ops in gaps:
                f.write("gap %#x %s %s\n" % (va + BN_BASE, mn, ops))

    table = render(code, regs)
    if a.write:
        with open(a.write, "w") as f:
            f.write(table)
        print("wrote %s" % a.write)
    if a.check:
        have = open(a.check).read()
        if have != table:
            problems.append("%s is not what the eboot gives (run with --write)" % a.check)
        else:
            print("%s matches the eboot" % a.check)
    for p in problems:
        print("error: " + p)
    return 1 if problems else 0


def render(code, regs):
    out = ["// Generated by tools/legacy_cpu_sites.py from the 1.09 eboot (sha256 %s)." % EBOOT_109,
           "// Every BMI1, LZCNT, TZCNT and MOVBE instruction in its code: the address (Binary Ninja's), the",
           "// length and the family, and for one shorter than a jump the bytes of whole instructions its",
           "// routine takes before and after it (a site inside another's taken bytes has none of its own).",
           "// core/legacy_cpu.cpp decodes each before it writes there. %d sites." % len(regs)]
    row = []
    for s, before, after in regs:
        size, mn, _ = code.insns[s]
        row.append("{0x%07x, %d, %s, %d, %d}," % (s + BN_BASE, size, FAMILY_NAMES[mn], before or 0, after or 0))
        if len(row) == 4:
            out.append("    " + " ".join(row))
            row = []
    if row:
        out.append("    " + " ".join(row))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
