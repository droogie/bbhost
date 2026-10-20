#!/usr/bin/env python3
"""What fx_enumerate.c's log says: per animation and effect id, the requests
made, how long they lived, and the ones that outlived the rest or never went.

    tools/fx_enumerate/report.py RUN.log [--base BASE.log] [--name NAME]

A request "outlived" when it lived past max(FLOOR_S, 3x the typical life of
its id - the lower quartile of its lives in the run, or the base's median
with --base); one still alive when the run's tail ran out "never went".
With --base (the same boss at another rate, or another build), each id's
median life beside the base's, and the ids whose requests now outlive or
never go when they did not there.
"""
import argparse
import re
import statistics
import sys
from collections import defaultdict

FLOOR_S = 10.0
NEW = re.compile(r'fxe: f(\d+) new a(-?\d+) (\d+) req ([0-9a-f]+)')
GONE = re.compile(r'fxe: f(\d+) gone a(-?\d+) (\d+) req ([0-9a-f]+) after ([\d.]+) s \(state (\d+)(, retired)?\)')
STILL = re.compile(r'fxe: f(\d+) still a(-?\d+) (\d+) req ([0-9a-f]+) alive ([\d.]+) s')
PLAY = re.compile(r'fxe: f(\d+) play (\d+)/(\d+) a(\d+)')


def parse(text):
    """{(anim, id): {'lives': [s...], 'still': [s...], 'made': n}}, animations played."""
    st = defaultdict(lambda: {'lives': [], 'still': [], 'made': 0})
    played = []
    for line in text.splitlines():
        m = NEW.search(line)
        if m:
            st[(int(m.group(2)), int(m.group(3)))]['made'] += 1
            continue
        m = GONE.search(line)
        if m:
            st[(int(m.group(2)), int(m.group(3)))]['lives'].append(float(m.group(5)))
            continue
        m = STILL.search(line)
        if m:
            st[(int(m.group(2)), int(m.group(3)))]['still'].append(float(m.group(5)))
            continue
        m = PLAY.search(line)
        if m:
            played.append(int(m.group(4)))
    return st, played


def lives_by_id(st):
    by_id = defaultdict(list)
    for (_a, i), v in st.items():
        by_id[i] += v['lives']
    return {i: sorted(l) for i, l in by_id.items() if l}


def medians(st):
    return {i: statistics.median(l) for i, l in lives_by_id(st).items()}


def typical(st):
    """Each id's lower-quartile life: what its requests live when half of them do not end."""
    return {i: l[len(l) // 4] for i, l in lives_by_id(st).items()}


def report(text, name='', base_text=None):
    st, played = parse(text)
    med = medians(st)
    typ = medians(parse(base_text)[0]) if base_text is not None else typical(st)
    lines = ['fx_enumerate %s: %d animations played, %d effect ids, %d requests' %
             (name, len(played), len({i for (_a, i) in st}), sum(v['made'] for v in st.values()))]
    flagged = []
    for (anim, i), v in sorted(st.items()):
        limit = max(FLOOR_S, 3 * typ.get(i, med.get(i, 0.0)))
        out = [s for s in v['lives'] if s > limit]
        if out or v['still']:
            flagged.append((anim, i, len(out), len(v['still']), max(out + v['still']), v['made']))
    lines.append('outlived (past max(%.0f s, 3x the id\'s typical life)) or never went:' % FLOOR_S)
    for anim, i, nout, nstill, worst, made in flagged:
        lines.append('  a%d %d: %d of %d outlived, %d never went; longest %.1f s (median %.2f s)' % (anim, i, nout, made, nstill, worst, med.get(i, 0.0)))
    if not flagged:
        lines.append('  none')
    if base_text is not None:
        bst, _ = parse(base_text)
        bmed = medians(bst)
        lines.append('median life by id, this run against the base:')
        for i in sorted(set(med) | set(bmed)):
            lines.append('  %d: %s s against %s s' % (i, '%.2f' % med[i] if i in med else '-', '%.2f' % bmed[i] if i in bmed else '-'))
        bflag = {(a, i) for (a, i, v) in ((a, i, v) for (a, i), v in bst.items()) if v['still']}
        new = [(a, i) for (a, i, _o, ns, _w, _m) in flagged if ns and (a, i) not in bflag]
        lines.append('never went here but did in the base: %s' % (', '.join('a%d %d' % k for k in new) or 'none'))
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--base')
    ap.add_argument('--name', default='')
    a = ap.parse_args()
    base = open(a.base, errors='replace').read() if a.base else None
    sys.stdout.write(report(open(a.log, errors='replace').read(), a.name, base))


if __name__ == '__main__':
    main()
