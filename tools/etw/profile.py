"""One process's CPU profile out of a WPR trace (tools/etw/record.ps1), for report.py.

    python tools/etw/profile.py build/perf/trace.etl build/perf/trace.pickle [--process bbhost.exe]

Runs xperf's dumper over the trace and keeps, for the process: every sampled
stack by thread (a sample a millisecond while a thread is on a core), each
thread's on-CPU time from the context switches and how often it was switched
out, with the wait reason it gave, the thread names and start addresses, and
the modules it had loaded (with their paths: report.py names system DLL
frames by their exports). And its waits: each time one of its threads is
switched out waiting, how long until it ran again, by the wait reason it gave
- off-CPU time, which samples cannot show (a waiting thread is on no core).
WPR's context-switch stacks are the incoming thread's and kernel frames only,
so they cannot name the wait's caller; the samples' kernel frames under each
wait call do (callees.py).
After the KyoPS4x ETW tools (Kyo).

Frames in code without unwind information - the game's own, which bbhost runs
in place in its image - end the stack walk: a game thread's samples carry the
leaf and the kernel frames only, which is still where its time goes.
"""
import argparse, collections, os, pickle, subprocess, sys, time

XPERF = r'C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\xperf.exe'

ap = argparse.ArgumentParser()
ap.add_argument('etl')
ap.add_argument('out')
ap.add_argument('--process', default='bbhost.exe')
a = ap.parse_args()
proc = a.process.lower()

stacks = collections.Counter()        # (tid, frames) -> samples; frames leaf first
cpu_us = collections.Counter()        # tid -> on-CPU microseconds (context switches)
switches = collections.Counter()      # tid -> times switched out
wait_reasons = collections.Counter()  # (tid, reason) -> times switched out waiting for it
names, starts, pids = {}, {}, set()
modules = []                          # (base, end, path)
running = {}                          # cpu -> (tid of ours or None, since)
first_ts = last_ts = None
cur = None                            # the sample whose stack lines follow: (tid, frames, pc)
open_waits = {}                       # tid -> (switched out at, reason)
off_cpu = collections.Counter()       # (tid, (), reason) -> microseconds off the CPU waiting
lines = 0
t0 = time.time()


def ours(field):
    """'bbhost.exe (1234)' -> True when it is the process (and note its pid)."""
    f = field.strip().lower()
    if not f.startswith(proc):
        return False
    try:
        pids.add(int(f[f.rindex('(') + 1:f.rindex(')')]))
    except ValueError:
        pass
    return True


cmd = [XPERF, '-i', a.etl, '-tle', '-tti', '-quiet', '-a', 'dumper', '-stacktimeshifting']
xp = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, errors='replace', bufsize=1 << 20)
for line in xp.stdout:
    lines += 1
    comma = line.find(',')
    if comma < 0:
        continue
    kind = line[:comma].strip()
    if kind == 'Stack':
        if cur is not None:
            f = line.split(',', 5)
            try:
                if int(f[2]) == cur[0]:
                    cur[1].append(int(f[4], 16))
            except ValueError:
                pass
        continue
    if cur is not None:
        stacks[(cur[0], tuple(cur[1]) if cur[1] else (cur[2],))] += 1
        cur = None
    if kind == 'SampledProfile':
        f = line.split(',', 7)
        if ours(f[2]):
            try:
                ts = int(f[1])
                cur = (int(f[3]), [], int(f[4], 16))
            except ValueError:
                continue
            first_ts = ts if first_ts is None else first_ts
            last_ts = ts
    elif kind == 'CSwitch':
        f = line.split(',')
        try:
            ts, cpu = int(f[1]), int(f[16])
        except (ValueError, IndexError):
            continue
        prev = running.get(cpu)
        if prev is not None and prev[0] is not None:
            cpu_us[prev[0]] += ts - prev[1]
        if ours(f[8]):
            old = int(f[9])
            switches[old] += 1
            if f[12].strip() == 'Waiting':
                reason = f[13].strip()
                wait_reasons[(old, reason)] += 1
                open_waits[old] = (ts, reason)
        if ours(f[2]):
            new = int(f[3])
            w = open_waits.pop(new, None)
            if w is not None:
                off_cpu[(new, (), w[1])] += ts - w[0]
        running[cpu] = (int(f[3]) if ours(f[2]) else None, ts)
    elif kind == 'ThreadName':
        f = line.split(',', 4)
        if ours(f[2]):
            names[int(f[3])] = f[4].strip().strip('"')
    elif kind in ('T-DCStart', 'T-Start'):
        f = line.split(',')
        if ours(f[2]):
            starts[int(f[3])] = f[-1].strip()
    elif kind in ('I-DCStart', 'I-Start'):
        f = line.split(',')
        if ours(f[2]):
            try:
                modules.append((int(f[3], 16), int(f[4], 16), f[8].strip().strip('"')))
            except (ValueError, IndexError):
                pass
    if lines % 20_000_000 == 0:
        print(f'{lines / 1e6:.0f}M lines, {time.time() - t0:.0f} s, {sum(stacks.values())} samples',
              file=sys.stderr, flush=True)
if cur is not None:
    stacks[(cur[0], tuple(cur[1]) if cur[1] else (cur[2],))] += 1
xp.wait()
if not stacks:
    sys.exit(f'no samples of {a.process} in {a.etl} (was it running? xperf exit {xp.returncode})')
with open(a.out, 'wb') as f:
    pickle.dump(dict(process=a.process, pids=sorted(pids), stacks=stacks, cpu_us=cpu_us, switches=switches,
                     wait_reasons=wait_reasons, off_cpu=off_cpu, names=names, starts=starts, modules=sorted(set(modules)),
                     first_ts=first_ts, last_ts=last_ts), f)
print(f'{lines} lines, {sum(stacks.values())} samples, {len(stacks)} distinct stacks, '
      f'{len(names)} named threads, {time.time() - t0:.0f} s -> {a.out}', file=sys.stderr)
