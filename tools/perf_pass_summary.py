#!/usr/bin/env python3
"""Summarises a perf pass (tools/perf_pass_win.sh): the run's log and the
perfmon samples (tools/win/perfmon.ps1), over the steady seconds - from
--after seconds after the world's first frame to the end.

  perf_pass_summary.py RUN.log RUN.perfmon [--after 20]
"""
import re
import statistics
import sys


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    after = 20.0
    if '--after' in sys.argv:
        after = float(sys.argv[sys.argv.index('--after') + 1])
        args = [a for a in args if a != sys.argv[sys.argv.index('--after') + 1]]
    log_path, mon_path = args[0], args[1]
    lines = open(log_path, encoding='utf-8', errors='replace').read().split('\n')
    version = next((l for l in lines[:4] if 'bbhost v' in l), '?').replace('[bbhost] ', '')
    world_s = None
    for l in lines:
        m = re.search(r'world: the first in-game frame, flip \d+, ([0-9.]+) s after start', l)
        if m:
            world_s = float(m.group(1))
            break
    print('%s: %s' % (log_path, version))
    if world_s is None:
        print('  the run did not reach the world')
    start = (world_s or 0.0) + after
    rows = []
    for l in open(mon_path, encoding='utf-8', errors='replace'):
        kv = dict(x.split('=', 1) for x in l.split() if '=' in x)
        try:
            t = float(kv['t'])
        except (KeyError, ValueError):
            continue
        if t >= start:
            # Counters can come with the locale's thousands separator ("1,066.5").
            rows.append({k: float(v.replace(',', '')) for k, v in kv.items()})
    # Frame statistics lines after the world: one a second.
    fps = []
    busy = []  # ms a second our command buffers kept the GPU busy (host/gpu_busy.cpp)
    seen_world = False
    for l in lines:
        if 'world: the first in-game frame' in l:
            seen_world = True
        if seen_world and l.startswith('[bbhost] frames:'):
            m = re.search(r'\(([0-9.]+)/s\)', l)
            if m:
                fps.append(float(m.group(1)))
                g = re.search(r'gpu busy ([0-9.]+) ms/s', l)
                busy.append(float(g.group(1)) if g else None)
    fps = fps[int(after):]
    busy = [b for b in busy[int(after):] if b is not None]
    if rows:
        def avg(k):
            return statistics.mean(r[k] for r in rows)

        def p95(k):
            v = sorted(r[k] for r in rows)
            return v[min(len(v) - 1, int(0.95 * len(v)))]
        print('  steady %d s (from %.0f s): GPU 3D %.1f%% (p95 %.1f), compute %.1f%%, copy %.1f%%; '
              'dedicated %.0f MiB (max %.0f), shared %.0f MiB (max %.0f); working set %.0f MiB; CPU %.0f%% of a core'
              % (len(rows), start, avg('3d'), p95('3d'), avg('compute'), avg('copy'), avg('dedicated'),
                 max(r['dedicated'] for r in rows), avg('shared'), max(r['shared'] for r in rows), avg('ws'), avg('cpu')))
    else:
        print('  no perfmon samples in the steady window')
    if fps:
        print('  fps mean %.2f, min %.1f, seconds under 58: %d of %d' % (statistics.mean(fps), min(fps),
                                                                        sum(1 for f in fps if f < 58), len(fps)))
    if busy:
        # bbhost's own GPU busy (the union of its command buffers on the GPU's
        # clock): the per-process 3D column above stays at 0 on AMD's driver.
        b = sorted(busy)
        print('  gpu busy (bbhost\'s timestamps) mean %.0f ms/s (%.1f%%), p95 %.0f, max %.0f; %.2f ms a flip'
              % (statistics.mean(b), statistics.mean(b) / 10.0, b[min(len(b) - 1, int(0.95 * len(b)))], b[-1],
                 statistics.mean(b) / statistics.mean(fps) if fps and statistics.mean(fps) > 0 else 0.0))
    for key in ('gpu: memory by heap', 'gpu: memory by site', 'image heap:', 'gpu profile', 'gpu: frame', 'gpu busy:'):
        last = [l for l in lines if key in l]
        if last:
            print('  ' + last[-1].strip()[:600])
    tail = [l for l in lines if l.startswith('exit ') or 'device lost' in l]
    for l in tail[-3:]:
        print('  ' + l.strip())


if __name__ == '__main__':
    main()
