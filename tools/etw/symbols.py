"""Names for the addresses in a profile.py pickle (report.py, callees.py).

- bbhost.exe: its own symbols (llvm-nm on the build with symbols, build-win/bbhost.exe;
  the stripped package keeps none), moved by where Windows put the image.
- The game's code, which bbhost runs in place: an address in the eboot's image
  (its slide is read from the run's log, "guest slide is 0x...") becomes
  eboot!sub_<va>, the function's start as Binary Ninja numbers it (the 0x400000
  base), found by the prologue every function the game's compiler emitted starts
  with (push rbp; mov rbp, rsp, 16-byte aligned).
- Other modules: the nearest export below the address (system DLLs, the driver's
  exported entry points), from the file's own export table.
- Kernel addresses: [kernel].
"""
import bisect, os, re, shutil, struct, subprocess

KERNEL = 0xffff800000000000
ELF_BASE = 0x400000  # Binary Ninja's base for the eboot, the address manifests and docs use


def pe_image_base(path):
    with open(path, 'rb') as f:
        data = f.read(0x400)
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    opt = pe + 24
    return struct.unpack_from('<Q', data, opt + 24)[0] if struct.unpack_from('<H', data, opt)[0] == 0x20b else \
        struct.unpack_from('<I', data, opt + 28)[0]


def pe_exports(path):
    """[(rva, name)] sorted, or [] (no file, no exports)."""
    try:
        data = open(path, 'rb').read()
        pe = struct.unpack_from('<I', data, 0x3c)[0]
        opt = pe + 24
        dd = opt + (112 if struct.unpack_from('<H', data, opt)[0] == 0x20b else 96)
        exp_rva = struct.unpack_from('<I', data, dd)[0]
        if not exp_rva:
            return []
        nsec = struct.unpack_from('<H', data, pe + 6)[0]
        sec_off = opt + struct.unpack_from('<H', data, pe + 20)[0]
        secs = [struct.unpack_from('<8sIIII', data, sec_off + 40 * i) for i in range(nsec)]

        def off(rva):
            for _, vsize, va, rsize, raw in secs:
                if va <= rva < va + max(vsize, rsize):
                    return rva - va + raw
            raise ValueError(rva)

        e = off(exp_rva)
        nfuncs, nnames, afuncs, anames, aords = struct.unpack_from('<IIIII', data, e + 20)
        funcs = [struct.unpack_from('<I', data, off(afuncs) + 4 * i)[0] for i in range(nfuncs)]
        out = []
        for i in range(nnames):
            nrva = struct.unpack_from('<I', data, off(anames) + 4 * i)[0]
            ordi = struct.unpack_from('<H', data, off(aords) + 2 * i)[0]
            o = off(nrva)
            name = data[o:data.index(b'\0', o)].decode('ascii', 'replace')
            out.append((funcs[ordi], name))
        return sorted(out)
    except Exception:
        return []


def eboot_functions(eboot, cache_dir):
    """Function starts (Binary Ninja addresses) in the eboot's executable segment."""
    st = os.stat(eboot)
    cache = os.path.join(cache_dir, f'eboot-{st.st_size}-{int(st.st_mtime)}.funcs')
    if os.path.exists(cache):
        return [int(l, 16) for l in open(cache)]
    data = open(eboot, 'rb').read()
    phoff = struct.unpack_from('<Q', data, 0x20)[0]
    phentsize, phnum = struct.unpack_from('<HH', data, 0x36)
    starts = []
    for i in range(phnum):
        p_type, p_flags, p_off, p_va, _, p_filesz, _, _ = struct.unpack_from('<IIQQQQQQ', data, phoff + i * phentsize)
        if p_type != 1 or not (p_flags & 1):  # PT_LOAD, executable
            continue
        seg = data[p_off:p_off + p_filesz]
        at = seg.find(b'\x55\x48\x89\xe5')
        while at >= 0:
            va = p_va + at
            if va % 16 == 0:
                starts.append(va + ELF_BASE)
            at = seg.find(b'\x55\x48\x89\xe5', at + 1)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache, 'w') as f:
        f.writelines(f'{s:x}\n' for s in starts)
    return starts


def exe_symbols(exe, cache_dir):
    """[(va, name)] of the exe's functions at its preferred base (llvm-nm, demangled)."""
    st = os.stat(exe)
    cache = os.path.join(cache_dir, f'{os.path.basename(exe)}-{st.st_size}-{int(st.st_mtime)}.syms')
    if not os.path.exists(cache):
        nm = shutil.which('llvm-nm') or r'C:\msys64\ucrt64\bin\llvm-nm.exe'
        out = subprocess.run([nm, '-n', '-C', '--defined-only', exe], capture_output=True, text=True,
                             errors='replace').stdout
        rows = []
        for line in out.splitlines():
            m = re.match(r'([0-9a-fA-F]+) ([tTwW]) (.+)$', line)
            if m:
                rows.append((int(m.group(1), 16), m.group(3)))
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache, 'w', encoding='utf-8') as f:
            f.writelines(f'{a:x} {n}\n' for a, n in rows)
    syms = []
    for line in open(cache, encoding='utf-8'):
        a, _, n = line.rstrip('\n').partition(' ')
        syms.append((int(a, 16), n))
    return syms


def log_slide(log):
    for line in open(log, encoding='utf-8', errors='replace'):
        m = re.search(r'guest slide is (0x[0-9a-fA-F]+)', line) or re.search(r'\bslide (0x[0-9a-fA-F]+)', line)
        if m:
            return int(m.group(1), 16)
    return None


class Resolver:
    def __init__(self, profile, exe=None, eboot=None, log=None, slide=None, image_size=0x56d30f4,
                 cache_dir=os.path.join('build', 'perf', 'etw-cache')):
        self.mods = sorted(profile['modules'])
        self.mod_bases = [b for b, _, _ in self.mods]
        self.proc = profile.get('process', 'bbhost.exe').lower()
        self.syms, self.sym_addrs, self.exe_delta = [], [], 0
        if exe:
            self.syms = exe_symbols(exe, cache_dir)
            self.sym_addrs = [a for a, _ in self.syms]
            run_base = next((b for b, _, p in self.mods if os.path.basename(p).lower() == self.proc), None)
            if run_base is not None:
                self.exe_delta = pe_image_base(exe) - run_base
        self.slide = slide if slide is not None else (log_slide(log) if log else None)
        self.image_size = image_size
        self.funcs = eboot_functions(eboot, cache_dir) if eboot and self.slide is not None else []
        self.exports = {}
        self.cache = {}

    def module(self, a):
        i = bisect.bisect_right(self.mod_bases, a) - 1
        if i >= 0 and a < self.mods[i][1]:
            return self.mods[i]
        return None

    def resolve(self, a):
        """-> (module, function or None)."""
        r = self.cache.get(a)
        if r is not None:
            return r
        if a >= KERNEL:
            r = ('[kernel]', None)
        elif self.slide is not None and self.slide <= a < self.slide + self.image_size:
            va = a - self.slide + ELF_BASE
            j = bisect.bisect_right(self.funcs, va) - 1
            r = ('eboot', f'eboot!sub_{self.funcs[j]:x}' if j >= 0 else f'eboot!0x{va:x}')
        else:
            m = self.module(a)
            if m is None:
                r = ('[no module]', None)
            else:
                base, _, path = m
                name = os.path.basename(path)
                if name.lower() == self.proc and self.syms:
                    j = bisect.bisect_right(self.sym_addrs, a + self.exe_delta) - 1
                    r = (name, self.syms[j][1] if j >= 0 else None)
                else:
                    if path not in self.exports:
                        local = path
                        if local.lower().startswith('\\systemroot\\'):
                            local = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), local[12:])
                        local = re.sub(r'^\\\\\?\\', '', local)
                        ex = pe_exports(local)
                        self.exports[path] = (ex, [x for x, _ in ex])
                    ex, addrs = self.exports[path]
                    j = bisect.bisect_right(addrs, a - base) - 1
                    r = (name, f'{name}!{ex[j][1]}' if j >= 0 else None)
        self.cache[a] = r
        return r

    def label(self, a):
        mod, fn = self.resolve(a)
        return fn if fn is not None else mod
