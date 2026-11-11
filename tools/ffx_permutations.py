#!/usr/bin/env python3
"""The shader permutation headers of AMD's FidelityFX SDK, for its Vulkan backend on Linux.

    tools/ffx_permutations.py --sdk KITS/FidelityFX --dxc DXC --out DIR [--spirv-val SPIRV_VAL] [--jobs N]
                              [--compress] [--effect fsr3upscaler]

The SDK's effects take their compute shaders from headers generated per pass
(`<pass>_permutations.h`, `<pass>_wave64_permutations.h`,
`<pass>_16bit_permutations.h`, `<pass>_wave64_16bit_permutations.h`), which
its `*_shaderblobs.cpp` include. AMD writes them with its FidelityFX_SC tool,
which is Windows-only and not in SDK 2.x; this writes the same headers from
the same HLSL with DXC's SPIR-V back end:

  - the passes and the four variants are the `#include <..._permutations.h>`
    lines of the effect's `_shaderblobs.cpp`; the options, in key order, the
    `key.<OPTION> = ...` lines of its POPULATE_PERMUTATION_KEY macro, each
    0 or 1 (one bit). Every pass's key has the same options in that order, so
    a pass that reads its table with another pass's key type (FSR 3's luma
    instability and debug view take the shading change pyramid's) gets the
    same index.
  - an option that the pass's HLSL and its includes never name changes
    nothing: its keys share the blob of the key with it at 0 (FidelityFX_SC's
    rule), so only the options a pass names are compiled.
  - DXC: SPIR-V for Vulkan 1.1, cs_6_2, entry CS, FFX_GPU=1 and FFX_HLSL=1,
    the effect's fixed defines, -DOPTION=0|1. The 16-bit variants add
    -enable-16bit-types and FFX_HALF=1 (32-bit: FFX_HALF=0). Registers become
    bindings in set 0: t# at #, u# at 128+#, b# at 256+#, s# at 320+# - the
    backend binds its two static samplers at 320 and 321
    (backend/vk/ffx_vk.cpp SAMPLER_BINDING_SHIFT), the rest from the
    reflection below, so only the samplers' place is fixed.
  - the wave64 variants are the 32-bit ones: Vulkan has no wave size in
    SPIR-V (DXC: "Wave size is not supported by Vulkan SPIR-V"), and the
    Vulkan backend reports shader model 5.1, so the effects never pick them
    there. Their tables point at the same blobs.
  - each distinct blob (by SHA-256) is written once, `<pass>_<hash>.h`, with
    its reflection: per kind (constant buffers, SRV and UAV textures and
    buffers, samplers, acceleration structures) the names, bindings, counts
    and descriptor sets, sorted by set and binding - what the backend's
    CreatePipelineVK makes its descriptor set layout of, and what the effect
    matches by name to its resources.
  - checked before anything is written: every blob passes spirv-val (with
    --spirv-val), every resource has a name, a binding and set 0, no two
    share a binding, samplers sit at 320 and 321, and every name is one the
    effect's own binding tables know (ffx_fsr3upscaler.cpp's
    srvTextureBindingTable, uavTextureBindingTable and
    constantBufferBindingTable), each of its kind - so the backend can bind
    everything a pipeline declares.

--compress stores each blob deflated behind the SDK's FFXZ header
(api/internal/ffx_blob_compression.h), which the backend inflates before
vkCreateShaderModule. The output is the same for the same SDK, DXC and
options. A summary is written to DIR/ffx_permutations.json.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import zlib

# The effects this knows. `defines` are FidelityFX_SC's fixed arguments for
# the effect (SDK 1.1.4's sdk/include/FidelityFX/gpu/<effect>/
# CMakeCompile*Shaders.txt); `tables` is the file holding the effect's
# name-to-resource tables.
EFFECTS = {
    "fsr3upscaler": {
        "blobs": "upscalers/fsr3/internal/ffx_fsr3upscaler_shaderblobs.cpp",
        "shaders": "upscalers/fsr3/internal/shaders",
        "includes": ["upscalers/fsr3/include/gpu"],
        "tables": "upscalers/fsr3/internal/ffx_fsr3upscaler.cpp",
        "defines": [
            "FFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0",
            "FFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0",
            "FFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1",
            "FFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0",
            "FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2",
        ],
    },
}

VARIANTS = ["", "_wave64", "_16bit", "_wave64_16bit"]
SHIFT = {"t": 0, "u": 128, "b": 256, "s": 320}
SAMPLER_BASE, SAMPLER_SLOTS = 320, 2
COMMON_ARGS = ["-spirv", "-fspv-target-env=vulkan1.1", "-T", "cs_6_2", "-E", "CS", "-DFFX_GPU=1", "-DFFX_HLSL=1",
               "-DFFX_HLSL_SM=62", "-Wno-for-redefinition", "-Wno-ambig-lit-shift"]
for _reg, _shift in SHIFT.items():
    COMMON_ARGS += ["-fvk-%s-shift" % _reg, str(_shift), "0"]
HALF_ARGS = {False: ["-DFFX_HALF=0"], True: ["-DFFX_HALF=1", "-enable-16bit-types"]}
FFXZ_MAGIC = 0x465A5846  # 'FFXZ' (ffx_blob_compression.h)
KINDS = [  # (header name, struct prefix, struct count member)
    ("CBV", "constantBuffer", "numConstantBuffers"),
    ("TextureSRV", "srvTexture", "numSRVTextures"),
    ("TextureUAV", "uavTexture", "numUAVTextures"),
    ("BufferSRV", "srvBuffer", "numSRVBuffers"),
    ("BufferUAV", "uavBuffer", "numUAVBuffers"),
    ("Sampler", "sampler", "numSamplers"),
    ("RTAccelerationStructure", "rtAccelerationStructure", "numRTAccelerationStructures"),
]


def fail(msg):
    sys.exit("ffx_permutations: " + msg)


# --- the effect's own sources -------------------------------------------------------------------

def read_shaderblobs(path):
    """The passes (in include order) with their variants, and the options in key order."""
    text = open(path, encoding="utf-8").read()
    passes = {}
    for name in re.findall(r"#include\s*<(\w+)_permutations\.h>", text):
        for v in sorted(VARIANTS, key=len, reverse=True):
            if v and name.endswith(v):
                base = name[: -len(v)]
                break
        else:
            base, v = name, ""
        passes.setdefault(base, set()).add(v)
    for base, vs in passes.items():
        if vs != set(VARIANTS):
            fail("%s: %s has the variants %s, not the four" % (path, base, sorted(vs)))
    # The macro's lines: its #define and each continued one (a trailing backslash).
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(r"\s*#define\s+POPULATE_PERMUTATION_KEY\b", l)), None)
    if start is None:
        fail("%s: no POPULATE_PERMUTATION_KEY" % path)
    body = []
    for l in lines[start:]:
        body.append(l)
        if not l.rstrip().endswith("\\"):
            break
    options = [o for o in re.findall(r"key\.(\w+)\s*=", "\n".join(body)) if o != "index"]
    if not passes or not options:
        fail("%s: no passes or no options" % path)
    return list(passes), options


def read_tables(path):
    """The effect's name tables: kind -> set of names."""
    text = open(path, encoding="utf-8").read()
    out = {}
    for kind, table in (("TextureSRV", "srvTextureBindingTable"), ("TextureUAV", "uavTextureBindingTable"),
                        ("CBV", "constantBufferBindingTable"), ("BufferSRV", "srvBufferBindingTable"),
                        ("BufferUAV", "uavBufferBindingTable")):
        m = re.search(r"\b%s\[\]\s*=\s*\{(.*?)\};" % table, text, re.S)
        out[kind] = set(re.findall(r'L"(\w+)"', m.group(1))) if m else set()
    return out


def names_in_closure(hlsl, include_dirs, options):
    """The options named anywhere in a shader and the files it includes (FidelityFX_SC's scan)."""
    found, seen, todo = set(), set(), [os.path.abspath(hlsl)]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for line in open(path, encoding="utf-8", errors="replace"):
            s = line.strip()
            m = re.match(r'#\s*include\s*["<]([^">]+)[">]', s)
            if m:
                for d in [os.path.dirname(path)] + include_dirs:
                    p = os.path.abspath(os.path.join(d, m.group(1)))
                    if os.path.isfile(p):
                        todo.append(p)
                        break
            for o in options:
                if o in s:
                    found.add(o)
    return found, sorted(seen)


# --- SPIR-V reflection --------------------------------------------------------------------------

def spirv_string(words):
    b = b"".join(struct.pack("<I", w) for w in words)
    return b.split(b"\0", 1)[0].decode("utf-8")


def reflect(spv):
    """kind -> [(name, binding, count, set)], and the entry point's name."""
    if len(spv) % 4 or len(spv) < 20:
        raise ValueError("not SPIR-V")
    w = struct.unpack("<%dI" % (len(spv) // 4), spv)
    if w[0] != 0x07230203:
        raise ValueError("not SPIR-V")
    names, deco, member_deco, types, ptrs, consts, variables = {}, {}, {}, {}, {}, {}, []
    entry = None
    i = 5
    while i < len(w):
        op, n = w[i] & 0xFFFF, w[i] >> 16
        if n == 0:
            raise ValueError("a zero-length instruction")
        a = w[i + 1:i + n]
        if op == 15:  # OpEntryPoint
            entry = (a[0], spirv_string(a[2:]))
        elif op == 5:  # OpName
            names[a[0]] = spirv_string(a[1:])
        elif op == 71:  # OpDecorate
            deco.setdefault(a[0], {})[a[1]] = a[2] if len(a) > 2 else True
        elif op == 72:  # OpMemberDecorate
            member_deco.setdefault(a[0], []).append(a[2])
        elif op == 25:  # OpTypeImage: result, sampled type, dim, depth, arrayed, ms, sampled, format
            types[a[0]] = ("image", a[2], a[6])
        elif op == 26:
            types[a[0]] = ("sampler",)
        elif op == 27:
            types[a[0]] = ("sampled_image",)
        elif op == 28:  # OpTypeArray
            types[a[0]] = ("array", a[1], a[2])
        elif op == 29:
            types[a[0]] = ("runtime_array", a[1])
        elif op == 30:
            types[a[0]] = ("struct", a[0])
        elif op == 5341:
            types[a[0]] = ("accel",)
        elif op == 32:  # OpTypePointer: result, storage class, type
            ptrs[a[0]] = (a[1], a[2])
        elif op == 43:  # OpConstant: result type, result, value
            consts[a[1]] = a[2]
        elif op == 59:  # OpVariable: result type, result, storage class
            variables.append((a[1], a[0], a[2]))
        i += n
    out = {k[0]: [] for k in KINDS}
    for var, ptr_type, storage in variables:
        if storage not in (0, 2, 12) or 33 not in deco.get(var, {}):  # UniformConstant, Uniform, StorageBuffer
            continue
        binding = deco[var][33]
        dset = deco[var].get(34, 0)
        name = names.get(var, "")
        if not name:
            raise ValueError("a resource at binding %d has no name" % binding)
        t = ptrs[ptr_type][1]
        count = 1
        while types.get(t, ("",))[0] in ("array", "runtime_array"):
            if types[t][0] == "runtime_array":
                raise ValueError("%s: a runtime array, which the backend cannot bind" % name)
            count *= consts[types[t][2]]
            t = types[t][1]
        kind = types.get(t, ("?",))
        if kind[0] == "sampler":
            k = "Sampler"
        elif kind[0] == "accel":
            k = "RTAccelerationStructure"
        elif kind[0] == "image":
            if kind[1] == 5:  # Dim Buffer: a texel buffer
                raise ValueError("%s: a texel buffer, which the backend cannot bind" % name)
            if kind[2] == 1:
                k = "TextureSRV"
            elif kind[2] == 2:
                k = "TextureUAV"
            else:
                raise ValueError("%s: an image neither sampled nor storage" % name)
        elif kind[0] == "struct":
            block = 2 in deco.get(t, {})
            buffer_block = 3 in deco.get(t, {})
            if storage == 2 and block:
                k = "CBV"
            elif storage == 12 or buffer_block:
                writable = not (deco.get(var, {}).get(24) or (member_deco.get(t) and all(d == 24 for d in member_deco[t])))
                k = "BufferUAV" if writable else "BufferSRV"
            else:
                raise ValueError("%s: a struct that is no buffer" % name)
        else:
            raise ValueError("%s: a resource type the backend cannot bind (%s)" % (name, kind[0]))
        out[k].append((name, binding, count, dset))
    for k in out:
        out[k].sort(key=lambda r: (r[3], r[1]))
    if not entry:
        raise ValueError("no entry point")
    return out, entry[1]


def check_reflection(where, refl, tables):
    seen = {}
    for kind, rows in refl.items():
        for name, binding, count, dset in rows:
            if dset != 0:
                raise ValueError("%s: %s in descriptor set %d" % (where, name, dset))
            for b in range(binding, binding + count):
                if b in seen:
                    raise ValueError("%s: %s and %s share binding %d" % (where, name, seen[b], b))
                seen[b] = name
            if kind == "Sampler":
                if not SAMPLER_BASE <= binding < SAMPLER_BASE + SAMPLER_SLOTS:
                    raise ValueError("%s: sampler %s at %d, not where the backend binds its samplers" % (where, name, binding))
            elif kind in tables and name not in tables[kind]:
                raise ValueError("%s: %s (%s) is not in the effect's binding tables" % (where, name, kind))
            elif kind not in tables:
                raise ValueError("%s: %s is a %s, which the effect binds none of" % (where, name, kind))


# --- compiling ----------------------------------------------------------------------------------

def dxc_version(dxc):
    r = subprocess.run([dxc, "--version"], capture_output=True, text=True)
    return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else "unknown"


def compile_one(job):
    dxc, hlsl, includes, defines, half, tmpdir = job
    fd, out = tempfile.mkstemp(suffix=".spv", dir=tmpdir)
    os.close(fd)
    args = [dxc] + COMMON_ARGS + HALF_ARGS[half] + ["-D" + d for d in defines]
    for inc in includes:
        args += ["-I", inc]
    args += ["-Fo", out, hlsl]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        os.unlink(out)
        raise RuntimeError("DXC failed on %s with %s:\n%s" % (hlsl, " ".join(defines), r.stderr.strip()))
    data = open(out, "rb").read()
    os.unlink(out)
    return data, r.stderr.strip()


def validate(spirv_val, spv, tmpdir, where):
    fd, path = tempfile.mkstemp(suffix=".spv", dir=tmpdir)
    with os.fdopen(fd, "wb") as f:
        f.write(spv)
    r = subprocess.run([spirv_val, "--target-env", "vulkan1.1", path], capture_output=True, text=True)
    os.unlink(path)
    if r.returncode != 0:
        raise ValueError("%s: spirv-val: %s" % (where, (r.stdout + r.stderr).strip()))


# --- the headers --------------------------------------------------------------------------------

def c_string_list(items):
    return "{ " + ", ".join('"%s"' % s for s in items) + " }"


def c_uint_list(items):
    return "{ " + ", ".join(str(int(x)) for x in items) + " }"


def write_if_changed(path, text):
    try:
        if open(path, encoding="utf-8").read() == text:
            return
    except OSError:
        pass
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def blob_header(sym, origin, spv, refl, compress):
    lines = ["// %s.h - %s" % (sym, origin), "#pragma once", ""]
    for kind, _, _ in KINDS:
        rows = refl[kind]
        if not rows:
            continue
        lines.append("static const char* g_%s_%sResourceNames[] = %s;" % (sym, kind, c_string_list(r[0] for r in rows)))
        lines.append("static const uint32_t g_%s_%sResourceBindings[] = %s;" % (sym, kind, c_uint_list(r[1] for r in rows)))
        lines.append("static const uint32_t g_%s_%sResourceCounts[] = %s;" % (sym, kind, c_uint_list(r[2] for r in rows)))
        lines.append("static const uint32_t g_%s_%sResourceSets[] = %s;" % (sym, kind, c_uint_list(r[3] for r in rows)))
        lines.append("")
    data = spv
    if compress:
        data = struct.pack("<II", FFXZ_MAGIC, len(spv)) + zlib.compress(spv, 9)
    lines.append("static const uint32_t g_%s_size = %d;" % (sym, len(data)))
    lines.append("")
    # Aligned for vkCreateShaderModule's uint32_t words (the backend hands an
    # uncompressed blob over as it is).
    lines.append("alignas(4) static const unsigned char g_%s_data[] = {" % sym)
    for off in range(0, len(data), 24):
        lines.append(",".join("0x%02x" % b for b in data[off:off + 24]) + ("," if off + 24 < len(data) else ""))
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def permutations_header(shader, origin, options, blob_syms, table, refls):
    lines = ["// %s_permutations.h - %s" % (shader, origin), ""]
    for sym in blob_syms:
        lines.append('#include "%s.h"' % sym)
    lines += ["", "typedef union %s_PermutationKey {" % shader, "    struct {"]
    for o in options:
        lines.append("        uint32_t %s : 1;" % o)
    lines += ["    };", "    uint32_t index;", "} %s_PermutationKey;" % shader, ""]
    lines += ["typedef struct %s_PermutationInfo {" % shader, "    const uint32_t       blobSize;",
              "    const unsigned char* blobData;", "    const char*          entryName;", ""]
    for _, prefix, count in KINDS:
        lines += ["    const uint32_t  %s;" % count, "    const char**    %sNames;" % prefix,
                  "    const uint32_t* %sBindings;" % prefix, "    const uint32_t* %sCounts;" % prefix,
                  "    const uint32_t* %sSpaces;" % prefix, ""]
    lines += ["} %s_PermutationInfo;" % shader, ""]
    lines.append("static const uint32_t g_%s_IndirectionTable[] = {" % shader)
    lines += ["    %d," % t for t in table]
    lines += ["};", ""]
    lines.append("static const %s_PermutationInfo g_%s_PermutationInfo[] = {" % (shader, shader))
    for sym in blob_syms:
        refl, entry = refls[sym]
        parts = ["g_%s_size" % sym, "g_%s_data" % sym, '"%s"' % entry]
        for kind, _, _ in KINDS:
            n = len(refl[kind])
            if n:
                parts += [str(n)] + ["g_%s_%sResource%s" % (sym, kind, f) for f in ("Names", "Bindings", "Counts", "Sets")]
            else:
                parts += ["0", "0", "0", "0", "0"]
        lines.append("    { " + ", ".join(parts) + " },")
    lines += ["};", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sdk", required=True, help="the SDK's Kits/FidelityFX folder")
    ap.add_argument("--dxc", required=True, help="DXC's command-line compiler")
    ap.add_argument("--out", required=True)
    ap.add_argument("--effect", default="fsr3upscaler", choices=sorted(EFFECTS))
    ap.add_argument("--spirv-val", default="", help="spirv-val, to check every blob (recommended)")
    ap.add_argument("--jobs", type=int, default=len(os.sched_getaffinity(0)), help="DXC runs at once (default: the CPUs this may use)")
    ap.add_argument("--compress", action="store_true", help="store each blob deflated behind the SDK's FFXZ header")
    ap.add_argument("--stamp", default="", help="a file written last, for the build's dependency")
    a = ap.parse_args()

    eff = EFFECTS[a.effect]
    sdk = os.path.abspath(a.sdk)
    passes, options = read_shaderblobs(os.path.join(sdk, eff["blobs"]))
    tables = read_tables(os.path.join(sdk, eff["tables"]))
    includes = [os.path.join(sdk, d) for d in eff["includes"]]
    dxc = os.path.abspath(a.dxc)
    version = dxc_version(dxc)
    os.makedirs(a.out, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix=".ffx-", dir=a.out)
    nbits = len(options)

    # Which keys each pass compiles: the options it names; the others share key 0's value.
    plan = {}
    for p in passes:
        hlsl = os.path.join(sdk, eff["shaders"], p + ".hlsl")
        if not os.path.isfile(hlsl):
            fail("no %s for the pass %s" % (hlsl, p))
        found, _ = names_in_closure(hlsl, includes, options)
        mask = sum(1 << i for i, o in enumerate(options) if o in found)
        plan[p] = (hlsl, mask)

    jobs = {}
    for p, (hlsl, mask) in plan.items():
        for half in (False, True):
            for key in range(1 << nbits):
                if key & ~mask:
                    continue
                defines = eff["defines"] + ["%s=%d" % (o, (key >> i) & 1) for i, o in enumerate(options)]
                jobs[(p, half, key)] = (dxc, hlsl, includes, defines, half, tmpdir)

    results = {}
    warnings = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        futures = {pool.submit(compile_one, j): k for k, j in jobs.items()}
        for f in concurrent.futures.as_completed(futures):
            k = futures[f]
            try:
                spv, warn = f.result()
            except RuntimeError as e:
                fail(str(e))
            results[k] = spv
            if warn:
                warnings.add(warn.splitlines()[0])

    # Distinct blobs, their reflection and checks.
    blobs, refls = {}, {}
    for (p, half, key), spv in sorted(results.items()):
        sym = "%s_%s" % (p, hashlib.sha256(spv).hexdigest()[:16])
        if sym in blobs:
            continue
        where = "%s %s key %d" % (p, "16-bit" if half else "32-bit", key)
        try:
            if a.spirv_val:
                validate(a.spirv_val, spv, tmpdir, where)
            refl, entry = reflect(spv)
            check_reflection(where, refl, tables)
        except ValueError as e:
            fail(str(e))
        if entry != "CS":
            fail("%s: entry point %s, not CS" % (where, entry))
        blobs[sym] = spv
        refls[sym] = (refl, entry)
    os.rmdir(tmpdir)

    origin = "generated by bbhost's tools/ffx_permutations.py from AMD FidelityFX SDK's %s (MIT) with DXC %s" % (
        os.path.relpath(eff["shaders"]), version)
    written = set()
    summary = {"effect": a.effect, "dxc": version, "options": options, "compress": a.compress, "passes": {}}
    for p, (hlsl, mask) in plan.items():
        per = {"options_named": [o for i, o in enumerate(options) if mask >> i & 1], "variants": {}}
        for v in VARIANTS:
            half = v.endswith("_16bit")
            shader = p + v
            syms, table, index = [], [], {}
            for key in range(1 << nbits):
                spv = results[(p, half, key & mask)]
                sym = "%s_%s" % (p, hashlib.sha256(spv).hexdigest()[:16])
                if sym not in index:
                    index[sym] = len(syms)
                    syms.append(sym)
                table.append(index[sym])
            for sym in syms:
                if sym not in written:
                    text = blob_header(sym, origin, blobs[sym], refls[sym][0], a.compress)
                    write_if_changed(os.path.join(a.out, sym + ".h"), text)
                    written.add(sym)
            write_if_changed(os.path.join(a.out, shader + "_permutations.h"),
                             permutations_header(shader, origin, options, syms, table, refls))
            per["variants"][v or "_wave32"] = {"blobs": len(syms), "keys": len(table)}
        summary["passes"][p] = per
    total = sum(len(b) for b in blobs.values())
    summary["distinct_blobs"] = len(blobs)
    summary["compiles"] = len(jobs)
    summary["spirv_bytes"] = total
    summary["spirv_val"] = bool(a.spirv_val)
    summary["dxc_warnings"] = sorted(warnings)
    write_if_changed(os.path.join(a.out, "ffx_permutations.json"), json.dumps(summary, indent=1, sort_keys=True) + "\n")
    # Headers of blobs no longer made (an older run's) go, so the folder is this run's alone.
    for f in os.listdir(a.out):
        if f.endswith(".h") and not f.endswith("_permutations.h") and f[:-2] not in written:
            os.unlink(os.path.join(a.out, f))
    if a.stamp:
        write_if_changed(a.stamp, "%s %d blobs\n" % (a.effect, len(blobs)))
    print("ffx_permutations: %s: %d passes x %d variants, %d compiles, %d distinct blobs (%.1f MiB of SPIR-V%s)%s, DXC %s"
          % (a.effect, len(passes), len(VARIANTS), len(jobs), len(blobs), total / 1048576.0,
             ", stored deflated" if a.compress else "", ", all through spirv-val" if a.spirv_val else "", version))


if __name__ == "__main__":
    main()
