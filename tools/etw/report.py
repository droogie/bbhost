"""Where one process's CPU went, from profile.py's pickle: by thread, by module, and each busy
thread's functions (self and inclusive), with what its threads waited for.

    python tools/etw/report.py build/perf/trace.pickle --exe build-win/bbhost.exe \
        --eboot <the eboot> --log build/perf/NAME.log [--top 25] [--min 300] [--thread NAME]

A sample is a millisecond on a core: a thread at 1000 samples a second is a whole
core, and the frame rate cannot rise past what the busiest thread on the frame's
path can do in a second. After the KyoPS4x ETW tools (Kyo).
"""
import argparse, collections, os, pickle, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symbols import Resolver  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('pickle')
ap.add_argument('--exe', help='the bbhost.exe with symbols that ran (build-win/bbhost.exe)')
ap.add_argument('--eboot', help='the eboot that ran (its functions name the game\'s samples)')
ap.add_argument('--log', help="the run's log (its guest slide)")
ap.add_argument('--slide', type=lambda s: int(s, 0))
ap.add_argument('--top', type=int, default=25)
ap.add_argument('--min', type=int, default=300, help='threads with fewer samples get no function list')
ap.add_argument('--thread', help='only threads whose name contains this')
ap.add_argument('--depth', type=int, default=6, help='frames named for a wait (above the kernel)')
ap.add_argument('--wait-threads', type=int, default=12, help='how many of the busiest threads get their waits listed')
a = ap.parse_args()

d = pickle.load(open(a.pickle, 'rb'))
r = Resolver(d, exe=a.exe, eboot=a.eboot, log=a.log, slide=a.slide)
dur_s = (d['last_ts'] - d['first_ts']) / 1e6
per_thread = collections.Counter()
for (tid, _), n in d['stacks'].items():
    per_thread[tid] += n
total = sum(per_thread.values())


def tname(tid):
    return d['names'].get(tid) or d['starts'].get(tid, '')


print(f"{d['process']} (pid {', '.join(map(str, d['pids']))}): {dur_s:.1f} s traced, {total} samples "
      f"= {total / dur_s / 1000:.2f} cores busy on average; guest slide "
      f"{hex(r.slide) if r.slide is not None else 'unknown (no --log/--slide: game code shows as [no module])'}\n")
print(f'{"tid":>6} {"samples":>8} {"core%":>6} {"on-cpu ms":>9} {"switch/s":>8}  name')
for tid, n in per_thread.most_common():
    if n < a.min / 10:
        break
    print(f'{tid:>6} {n:>8} {100 * n / (dur_s * 1000):>5.1f}% {d["cpu_us"].get(tid, 0) / 1000:>9.0f} '
          f'{d["switches"].get(tid, 0) / dur_s:>8.0f}  {tname(tid)}')

mod_self = collections.Counter()
for (tid, frames), n in d['stacks'].items():
    mod_self[r.resolve(frames[0])[0]] += n
print('\nself time by module, the whole process:')
for m, n in mod_self.most_common(14):
    print(f'  {100 * n / total:5.1f}%  {m}')

for tid, n in per_thread.most_common():
    if n < a.min:
        break
    if a.thread and a.thread.lower() not in tname(tid).lower():
        continue
    self_c, incl_c, mods = collections.Counter(), collections.Counter(), collections.Counter()
    for (t, frames), c in d['stacks'].items():
        if t != tid:
            continue
        self_c[r.label(frames[0])] += c
        mods[r.resolve(frames[0])[0]] += c
        seen = set()
        for x in frames:
            lab = r.label(x)
            if lab not in seen:
                seen.add(lab)
                incl_c[lab] += c
    print(f'\n=== {tname(tid) or "(unnamed)"} (tid {tid}): {n} samples, {100 * n / (dur_s * 1000):.1f}% of a core; '
          + ', '.join(f'{m} {100 * c / n:.0f}%' for m, c in mods.most_common(5)))
    print('  -- self --')
    for lab, c in self_c.most_common(a.top):
        print(f'  {100 * c / n:5.1f}%  {lab[:150]}')
    print('  -- inclusive (bbhost and system frames; a game frame ends the walk) --')
    for lab, c in incl_c.most_common(a.top):
        print(f'  {100 * c / n:5.1f}%  {lab[:150]}')

wr = collections.defaultdict(collections.Counter)
for (tid, reason), c in d['wait_reasons'].items():
    wr[tid][reason] += c
print('\nwhat the busiest threads waited for (times switched out, by wait reason):')
for tid, n in per_thread.most_common(16):
    print(f'  {tid:>6} {tname(tid)[:28]:28} {dict(wr[tid].most_common(4))}')

# Off the CPU: how long each busy thread spent waiting, by wait reason (which
# call it waited in: callees.py on the kernel samples under the wait).
oc = d.get('off_cpu') or {}
if oc:
    by_thread = collections.defaultdict(collections.Counter)
    for (tid, _, reason), us in oc.items():
        by_thread[tid][reason] += us
    print('\noff the CPU, waiting (ms a second, by wait reason):')
    shown = 0
    for tid, n in per_thread.most_common():
        if shown >= a.wait_threads:
            break
        if a.thread and a.thread.lower() not in tname(tid).lower():
            continue
        waits = by_thread.get(tid)
        if not waits:
            continue
        shown += 1
        tot = sum(waits.values())
        parts = ', '.join(f'{r} {us / 1000 / dur_s:.0f}' for r, us in waits.most_common(4))
        print(f'  {tname(tid)[:28]:28} {tot / 1000 / dur_s:6.0f} ms/s waiting, {100 * n / (dur_s * 1000):3.0f}% running: {parts}')
