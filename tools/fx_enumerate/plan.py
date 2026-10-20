#!/usr/bin/env python3
"""The effect animations of the bosses plugins/boss_rush reaches, from their TAE.

    tools/fx_enumerate/plan.py [--app0 DIR] [--boss N|NAME ...] [--json]

For each boss of the boss rush's story order (its kBosses table, read from
plugins/boss_rush/boss_rush.cpp; round N is the Nth): its model (the health
bar's name id / 100: Ebrietas' 251000 is c2510), its TAE (chr/cMMMM.anibnd.dcx,
else the .chrbnd.dcx) and every animation with effect events: types 96, 100,
109 and 122 (an effect id first: 625102, 650091, ...), 112 (a behaviour id
first - Ebrietas' 1251700 makes the effects 1251701 and 1251721 - 1000000 +
model * 100 + n * 100 for the bosses' own) and 118 with an effect id (100000
and up), their first parameter listed. A TAE3: the animation table (id u64,
offset u64) the header points at;
an animation +0 event headers, +0x20 their count; an event header: start time
offset, end time offset, data offset (u64 each); the data: type i32, +8 the
parameters' offset.

The plan file fx_enumerate.c reads (--out FILE): one line a boss and one an
animation:
    boss <round> <model> <name>
    anim <id> <seconds the TAE's events span> <effect id>,<effect id>...
"""
import argparse
import os
import re
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bbparam import bnd4, dcx  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')


def bosses():
    """(round, name, model, fog xyz, The Old Hunters') for the boss rush's story order."""
    src = open(os.path.join(ROOT, 'plugins', 'boss_rush', 'boss_rush.cpp'), encoding='utf-8').read()
    table = src[src.index('const Boss kBosses[] = {'):]
    table = table[:table.index('\n};')]
    out = []
    for m in re.finditer(r'\{"([^"]+)", (\d+), (\d+), (true|false), (\d+), (\d+), \{([-\d.f, ]+)\}', table):
        name, name_id = m.group(1), int(m.group(5))
        fog = [float(v.strip().rstrip('f')) for v in m.group(7).split(',')]
        out.append((len(out) + 1, name, name_id // 100, fog, m.group(4) == 'true'))
    return out


def tae_files(app0, model):
    for name in ('c%04d.anibnd.dcx' % model, 'c%04d.chrbnd.dcx' % model):
        p = os.path.join(app0, 'dvdroot_ps4', 'chr', name)
        if not os.path.exists(p):
            continue
        found = {n: b for n, b in bnd4(dcx(p)).items() if n.endswith('.tae')}
        if found:
            return p, found
    return None, {}


def events(b):
    """(animation id, event type, start s, end s, first six int parameters) of a TAE3."""
    n = len(b)
    u64 = lambda o: struct.unpack_from('<Q', b, o)[0]
    i32 = lambda o: struct.unpack_from('<i', b, o)[0]
    f32 = lambda o: struct.unpack_from('<f', b, o)[0]
    table = None
    for o in range(0, 0x100, 8):
        p = u64(o)
        if not 0x40 <= p < n:
            continue
        k, prev = 0, 0
        while p + 16 * k + 16 <= n:
            aid, off = u64(p + 16 * k), u64(p + 16 * k + 8)
            if not (aid < 10**7 and prev < off < n):
                break
            prev, k = off, k + 1
        if k > 5 and (table is None or k > table[1]):
            table = (p, k)
    out = []
    if not table:
        return out
    p, count = table
    for a in range(count):
        aid, off = u64(p + 16 * a), u64(p + 16 * a + 8)
        ev_off, ev_count = u64(off), i32(off + 32)
        if not (0 < ev_off < n and 0 <= ev_count < 5000):
            continue
        for e in range(ev_count):
            h = ev_off + 24 * e
            if h + 24 > n:
                break
            st, en, data = u64(h), u64(h + 8), u64(h + 16)
            if not (0 < data < n and 0 < st < n and 0 < en < n):
                continue
            etype, params = i32(data), u64(data + 8)
            if not 0 < params < n:
                continue
            ints = [i32(params + 4 * k) for k in range(6) if params + 4 * k + 4 <= n]
            out.append((aid, etype, f32(st), f32(en), ints))
    return out


# The event types that make effects, and the least first parameter that is an
# effect (or behaviour) id rather than a dummy poly or a count.
EFFECT_TYPES = {96: 1000, 100: 1000, 109: 1000, 112: 1000, 122: 1000, 118: 100000}


def plan(app0, model):
    """{animation id: (seconds the TAE spans, [effect ids])} of a model, and the event types seen."""
    path, taes = tae_files(app0, model)
    ids, span, types = {}, {}, set()
    for _name, b in sorted(taes.items()):
        for (aid, t, st, en, ints) in events(b):
            span[aid] = max(span.get(aid, 0.0), en)
            if t in EFFECT_TYPES and ints and ints[0] >= EFFECT_TYPES[t]:
                ids.setdefault(aid, set()).add(ints[0])
                types.add(t)
    return path, sorted(types), {aid: (span.get(aid, 0.0), sorted(l)) for aid, l in sorted(ids.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--app0', default=os.environ.get('BBHOST_APP0', ''))
    ap.add_argument('--boss', action='append', default=[], help='a round number or a name (substring); default: every boss')
    ap.add_argument('--out', help='write the plan file fx_enumerate.c reads')
    a = ap.parse_args()
    if not a.app0:
        sys.exit('plan.py: --app0 (or BBHOST_APP0): the dump, the folder holding dvdroot_ps4')
    chosen = []
    for rnd, name, model, fog, _dlc in bosses():
        if a.boss and not any(b == str(rnd) or b.lower() in name.lower() for b in a.boss):
            continue
        chosen.append((rnd, name, model, fog))
    lines = []
    for rnd, name, model, fog in chosen:
        path, types, anims = plan(a.app0, model)
        ids = sorted({i for (_, l) in anims.values() for i in l})
        print('%2d %-40s c%04d: %3d animations with effect events (types %s), %3d effect ids%s' %
              (rnd, name, model, len(anims), ','.join(map(str, types)) or '-', len(ids), '' if path else ' - no TAE found'))
        lines.append('boss %d %d %s' % (rnd, model, name))
        for aid, (span, l) in anims.items():
            lines.append('anim %d %.2f %s' % (aid, span, ','.join(map(str, l))))
    if a.out:
        with open(a.out, 'w') as f:
            f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
