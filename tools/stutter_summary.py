#!/usr/bin/env python3
"""The stutters of a tools/stutter_tour.sh run, over the tour's window.

    tools/stutter_summary.py build/NAME/run.log [more logs...]

Per log:
  frames       frames over 33 ms (the per-second lines), the worst, mean fps
  main loop    long frames (> 33 ms: `main loop:` lines), their time and how
               much of it the main loop waited, and - with BBHOST_HLE_COUNT=1 -
               that wait by the call site it waited at (the innermost caller,
               Binary Ninja addresses): 0x2599365 the render thread at frame
               end, 0x2157d38 the parallel copy of a streamed resource (gone
               since C11), 0x1d17ecd / 0x1d18285 the character jobs
  CP stalls    `stall:` lines by cause: shader/pipeline (translations and
               pipeline creation on the command processor), texture work,
               lock waits, driver calls (the command processor in a timed
               driver call, or waiting for the queue: host/slow_calls.h), the
               command processor busy with draws, or idle (the main loop or
               the game was the slow one)
  slow calls   `slow call:` lines by call, and the longest
  settle       with BBHOST_GX_RATE=1: seconds from each warp until GX
               resources stop being made (20 or more a second) - how long
               what a warp brings in takes to arrive
"""
import re
import sys
from collections import Counter


def window(lines):
    s = next((i for i, l in enumerate(lines) if 'mapval tour-start' in l), 0)
    e = next((i for i, l in enumerate(lines) if 'mapval tour-done' in l), len(lines))
    return lines[s:e]


def summary(path):
    lines = open(path, errors='replace').read().splitlines()
    seg = window(lines)
    secs, longs, stalls, slow = [], [], [], []
    for l in seg:
        if l.startswith('[bbhost] frames:'):
            m = re.search(r'\(([0-9.]+)/s\).*max ([0-9.]+) ms, \d+ over 16.7, (\d+) over 33.3', l)
            if m:
                secs.append((float(m.group(1)), float(m.group(2)), int(m.group(3))))
        elif 'main loop: a ' in l:
            m = re.search(r'a ([0-9.]+) ms frame before flip (\d+), ([0-9.]+) ms of it waiting(.*)', l)
            if m:
                longs.append((float(m.group(1)), float(m.group(3)), m.group(4)))
        elif l.startswith('[bbhost] slow call:'):
            m = re.search(r'slow call: (\S+) ([0-9.]+) ms on (\S+)', l)
            if m:
                slow.append((m.group(1), float(m.group(2)), m.group(3).rsplit(':', 1)[0]))
        elif l.startswith('[bbhost] stall:'):
            d = {k: int(v) for k, v in re.findall(r'(\w[\w-]*)=(\d+)', l)}
            d['gap'] = int(re.search(r'came (\d+) ms', l).group(1))
            for key, pat in (('hash', r'texture hashes \d+ in (\d+) ms'), ('pl', r'pipeline ms: translate (\d+), modules (\d+), create (\d+)'),
                             ('tex', r'texture ms: images (\d+), staging (\d+), untile (\d+), record (\d+)')):
                m = re.search(pat, l)
                d[key] = sum(int(x) for x in m.groups()) if m else 0
            stalls.append(d)
    print(f'{path}:')
    if not secs:
        print('  no frame statistics in the tour window')
        return
    print(f'  frames: {len(secs)} s, over 33 ms {sum(s[2] for s in secs)}, worst {max(s[1] for s in secs):.0f} ms, '
          f'mean fps {sum(s[0] for s in secs) / len(secs):.1f}')
    t = sum(x[0] for x in longs)
    w = sum(x[1] for x in longs)
    sites = Counter()
    for _, _, tail in longs:
        for site, ms in re.findall(r'(0x[0-9a-f<x]+)(?: \(\w+\))? ([0-9.]+) ms;', tail):
            sites[site.split('<')[-1]] += float(ms)
    print(f'  main loop: {len(longs)} long frames, {t:.0f} ms, waiting {w:.0f} ms' +
          (('; by caller ' + ', '.join(f'{k} {v:.0f}' for k, v in sites.most_common(5))) if sites else ''))
    causes = Counter()
    for d in stalls:
        c = []
        if d['pl'] >= 10:
            c.append('shader/pipeline')
        if d['hash'] + d['tex'] >= 10:
            c.append('texture')
        if d.get('lock-wait', 0) >= 10:
            c.append('lock-wait')
        if d.get('driver', 0) >= 10:
            c.append('driver')
        if not c:
            c.append('cp-busy' if d.get('exec', 0) >= d['gap'] * 0.6 else 'cp-idle')
        causes.update(c)
    print(f'  CP stalls: {len(stalls)}' + (' (' + ', '.join(f'{k} {v}' for k, v in causes.most_common()) + ')' if stalls else ''))
    if slow:
        calls = Counter(f'{call} on {thread}' for call, _, thread in slow)
        worst = max(slow, key=lambda x: x[1])
        print(f'  slow calls: {len(slow)} lines (' + ', '.join(f'{k} {v}' for k, v in calls.most_common(6)) +
              f'), the longest {worst[0]} {worst[1]:.0f} ms on {worst[2]}')
    warps = [float(m.group(1)) for l in lines for m in [re.search(r'mapval step \d+ warp \S+ ([0-9.]+)', l)] if m]
    done = [float(m.group(1)) for l in lines for m in [re.search(r'mapval tour-done ([0-9.]+)', l)] if m]
    rates = [(int(m.group(2)), int(m.group(1))) for l in lines for m in [re.search(r'gx-resources: rate: (\d+) made \(\d+ KiB\) in second (\d+)', l)] if m]
    if warps and rates:
        ends = warps[1:] + (done[:1] or [1e18])
        settle = []
        for t0, t1 in zip(warps, ends):
            busy = [s for s, n in rates if int(t0) <= s < int(t1) and n >= 20]
            settle.append(max(busy) + 1 - t0 if busy else 0.0)
        ss = sorted(settle)
        print(f'  settle: {len(settle)} warps, mean {sum(settle) / len(settle):.2f} s, median {ss[len(ss) // 2]:.1f}, max {ss[-1]:.1f}')


for p in sys.argv[1:]:
    summary(p)
