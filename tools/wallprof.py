#!/usr/bin/env python3
"""Sum the stacks BBHOST_SAMPLE wrote (src/host/sampler.h).

    tools/wallprof.py build/samples [--flips 1300:1600] [--thread bb-cp0] [--top 40]

Samples are wall-clock: a thread blocked in a futex or a driver wait is
sampled like one that is running, so the tables show where its time goes,
not only its CPU. Printed per thread:

  inclusive  functions on the stack, by share of samples
  self       the innermost function
  blocked    for samples whose innermost frame is a wait (futex, ioctl,
             poll, nanosleep...), the first bbhost frame that led there

Addresses are symbolized with addr2line against the modules in <out>.maps;
a module without symbols prints as module+offset. A Windows run's maps (the
sampler under wine) name PE modules; bbhost.exe carries DWARF, addr2line reads
it at the link addresses the maps' offsets give.
"""
import argparse
import bisect
import collections
import os
import re
import shutil
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument('prefix', help='BBHOST_SAMPLE_OUT (without .txt/.maps)')
ap.add_argument('--flips', help='first:last flip to include')
ap.add_argument('--thread', help='only threads whose name starts with this')
ap.add_argument('--tid', type=int, help='only this thread id')
ap.add_argument('--top', type=int, default=40)
ap.add_argument('--children', help='also print what this function (a substring of its name) spends its samples in')
ap.add_argument('--callers', help='print the call chains (our frames, up to four) that lead to this function (a substring of its name)')
a = ap.parse_args()

first, last = 0, 1 << 62
if a.flips:
    f0, f1 = a.flips.split(':')
    first, last = int(f0 or 0), int(f1 or 1 << 62)

names = {}
samples = collections.defaultdict(list)  # thread name -> [stack]
for line in open(a.prefix + '.txt'):
    if line.startswith('#thread'):
        _, tid, name = line.split(None, 2)
        names[int(tid)] = name.strip()
        continue
    parts = line.split()
    if len(parts) < 3:
        continue
    flip, tid = int(parts[0]), int(parts[1])
    if not first <= flip <= last:
        continue
    name = names.get(tid, str(tid))
    if a.thread and not name.startswith(a.thread):
        continue
    if a.tid and tid != a.tid:
        continue
    # Guest addresses (a leading 'g') are Binary Ninja's; kept negative here
    # so they never collide with a host address.
    # A frame cut off mid-write (a bare "g" at the end of a line) is dropped.
    samples['%s:%d' % (name, tid)].append([-int(x[1:], 16) if x[0] == 'g' else int(x, 16)
                                           for x in parts[2:] if x and x != 'g'])

maps = []
for line in open(a.prefix + '.maps'):
    m = re.match(r'([0-9a-f]+)-([0-9a-f]+) (\S+) ([0-9a-f]+) \S+ \S+\s*(.*)', line)
    if not m:
        continue
    start, end, perm, off, path = int(m[1], 16), int(m[2], 16), m[3], int(m[4], 16), m[5].strip()
    # A Windows run (the sampler under wine) lists PE modules by their wine
    # paths, with the module's link base as the offset: Z: is this machine's
    # root, so bbhost.exe is found and addr2line takes its link addresses.
    if re.match(r'^[Zz]:[\\/]', path):
        path = '/' + path[3:].replace('\\', '/')
    maps.append((start, end, perm, off, path))
maps.sort()
base = {}
link = {}
for start, end, perm, off, path in maps:
    if path.startswith('/') and path not in base:
        base[path] = start - off  # the first mapping of a file sits at its ELF address 0
        link[path] = off  # macOS: the image's __TEXT link address
HAVE_ADDR2LINE = shutil.which('addr2line') is not None
HAVE_ATOS = not HAVE_ADDR2LINE and shutil.which('atos') is not None
execs = [(s, e, p) for s, e, perm, off, p in maps if 'x' in perm]
starts = [m[0] for m in execs]


def module_of(addr):
    if addr < 0:
        return None
    i = bisect.bisect_right(starts, addr) - 1
    if i >= 0 and execs[i][0] <= addr < execs[i][1]:
        return execs[i][2]
    return None


# Symbolize every distinct address (callers at addr-1, so a call's line is its own).
wanted = collections.defaultdict(set)
for stacks in samples.values():
    for st in stacks:
        for k, addr in enumerate(st):
            path = module_of(addr)
            if path and path.startswith('/'):
                wanted[path].add(addr if k == 0 else addr - 1)
sym = {}
for path, addrs in wanted.items():
    addrs = sorted(addrs)
    rel = [x - base.get(path, 0) for x in addrs]
    out = []
    if os.path.exists(path) and HAVE_ADDR2LINE:
        r = subprocess.run(['addr2line', '-f', '-C', '-e', path] + ['%x' % x for x in rel],
                           capture_output=True, text=True)
        out = r.stdout.splitlines()
    elif os.path.exists(path) and HAVE_ATOS:
        # macOS: a map's offset is the image's __TEXT link address, so load
        # atos there. Its one line per address becomes addr2line's pairs.
        r = subprocess.run(['atos', '-o', path, '-arch', 'x86_64', '-l', '%x' % link.get(path, 0)] +
                           ['%x' % x for x in rel], capture_output=True, text=True)
        for ln in r.stdout.splitlines():
            fn = re.sub(r' \(in [^)]*\)', '', ln).strip()
            fn = re.sub(r' \([^()]*:\d+\)$', '', fn)
            out += ['??' if re.match(r'^0x[0-9a-f]+$', fn) else fn, '']
    short = os.path.basename(path)
    for i, x in enumerate(addrs):
        fn = out[2 * i] if 2 * i < len(out) else '??'
        if fn == '??':
            fn = '%s+%x' % (short, rel[i])
        elif not short.startswith('bbhost'):
            fn = '%s!%s' % (short.split('.so')[0], fn)
        sym[x] = fn


def name_of(addr, k):
    if addr < 0:
        return 'guest:%x' % -addr
    key = addr if k == 0 else addr - 1
    if key in sym:
        return sym[key]
    path = module_of(addr)
    return '%s+%x' % (os.path.basename(path), addr - base.get(path, 0)) if path else 'unknown@%x' % addr


WAITS = re.compile(r'futex|__lll_lock_wait|pthread_cond_(timed)?wait|ioctl|__poll|ppoll|epoll_wait|'
                   r'nanosleep|clock_nanosleep|sched_yield|__GI___select|read$|__libc_read|syscall|'
                   # macOS (libsystem_kernel's traps)
                   r'__psynch_cvwait|__psynch_mutexwait|__ulock_wait|semaphore_.*wait_trap|mach_msg2?_trap|'
                   r'__semwait_signal|__workq_kernreturn|kevent|__select|swtch_pri|thread_switch')

for thread, stacks in sorted(samples.items(), key=lambda kv: -len(kv[1])):
    n = len(stacks)
    incl, self_, blocked = collections.Counter(), collections.Counter(), collections.Counter()
    waiting = 0
    for st in stacks:
        fns = [name_of(x, k) for k, x in enumerate(st)]
        if not fns:
            continue
        self_[fns[0]] += 1
        for fn in set(fns):
            incl[fn] += 1
        if WAITS.search(fns[0]) or any(WAITS.search(f) for f in fns[:3]):
            waiting += 1
            ours = next((f for f in fns if not ('!' in f or '.so' in f)), fns[-1])
            chain = ' < '.join([f for f in fns if not ('!' in f or '.so' in f)][:3])
            blocked[chain or ours] += 1
    if a.callers:
        chains = collections.Counter()
        within = 0
        for st in stacks:
            fns = [name_of(x, k) for k, x in enumerate(st)]
            for i, fn in enumerate(fns):
                if a.callers in fn:
                    within += 1
                    ours = [f for f in fns[i + 1:] if not ('!' in f or '.so' in f)][:4]
                    chains[' < '.join(f.split('(')[0] for f in ours) or '(none)'] += 1
                    break
        print('== %s: %s is on %.1f%% of the samples; reached from:' % (thread, a.callers, 100.0 * within / max(n, 1)))
        for fn, k in chains.most_common(a.top):
            print('  %5.1f%%  %s' % (100.0 * k / n, fn[:220]))
        continue
    if a.children:
        kids = collections.Counter()
        within = 0
        for st in stacks:
            fns = [name_of(x, k) for k, x in enumerate(st)]
            for i, fn in enumerate(fns):
                if a.children in fn:
                    within += 1
                    kids[fns[i - 1] if i > 0 else '(self)'] += 1
                    break
        print('== %s: %s is on %.1f%% of the samples; its callees:' % (thread, a.children, 100.0 * within / max(n, 1)))
        for fn, k in kids.most_common(a.top):
            print('  %5.1f%%  %s' % (100.0 * k / n, fn[:160]))
        continue
    print('== %s: %d samples, %.1f%% waiting' % (thread, n, 100.0 * waiting / max(n, 1)))
    for title, c in (('inclusive', incl), ('self', self_), ('blocked (our frames that led to the wait)', blocked)):
        print('-- %s' % title)
        for fn, k in c.most_common(a.top):
            print('  %5.1f%%  %s' % (100.0 * k / n, fn[:160]))
    print()
