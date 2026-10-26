#!/usr/bin/env python3
"""Compare the draw digest's frames (BBHOST_DRAW_DIGEST, src/host/draw_digest.h): what the command processor drew, in
order, one hash a draw.

    tools/draw_digest.py DIGEST --every N        a run whose switch flips every N flips (BBHOST_RQ_SPLIT_EVERY=N):
                                                 its frames on against its frames off
    tools/draw_digest.py DIGEST_A DIGEST_B       two runs' frames against each other
    --flips A-B                                  only the frames that end between those flips (a place's measure window,
                                                 where the view holds still; tools/area_check.py's report.json has them)

A frame's sequence is its draws' hashes in the order they were drawn. With --every, a frame is on when the flips done at
its end, divided by N, are odd; frames within --margin flips (4) of a change are left out, since the switch is read when
the frame's draws were recorded, a flip or two before the command processor draws them. Each frame of one side is
matched by its sequence against the other side's frames; a frame no frame of the other side matches is compared with
the nearest one of the other side by sequence (difflib): what was added, removed or moved. The same is reported for the
frames of one side against each other, which is what the scene itself varies by (an effect's draws come and go, a
target alternates between two).

The order test: the draws whose hash comes exactly once in every frame of both sides (the scene's stable draws: its
meshes, its passes) are taken out of each frame in the order drawn. The game moves some of them itself from frame to
frame (segments whose order keys tie play in the order they were queued, which the job timing decides): the draws any
frame of the reference side (off, or A) has out of that side's commonest order are its movers, and are set aside. Equal
when every frame of both sides has the rest in one order: a change that moved any draw the game keeps in place shows
there. The movers are reported, and the other side's frames are checked to move no others.
"""
import argparse
import collections
import difflib
import struct
import sys


def read(path):
    frames = []  # (flips at the end, [hashes])
    cur = []
    with open(path, 'rb') as f:
        if f.read(16)[:8] != b'DRAWDIG1':
            sys.exit('not a draw digest: ' + path)
        while True:
            b = f.read(16 * 65536)
            if not b:
                break
            for k in range(len(b) // 16):
                kind, frame, digest = struct.unpack_from('<IIQ', b, 16 * k)
                if kind == 0:
                    cur.append(digest)
                else:
                    frames.append((digest, tuple(cur)))
                    cur = []
    return frames


def describe(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    added = removed = changed = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'insert':
            added += j2 - j1
        elif op == 'delete':
            removed += i2 - i1
        elif op == 'replace':
            changed += max(i2 - i1, j2 - j1)
    return added, removed, changed


def compare(side_a, side_b, name_a, name_b):
    seqs_b = collections.Counter(s for _, s in side_b)
    hit = sum(1 for _, s in side_a if s in seqs_b)
    worst = []
    for flips, s in side_a:
        if s in seqs_b:
            continue
        # the nearest frame of the other side, by flips
        near = min(side_b, key=lambda fb: abs(fb[0] - flips))
        worst.append((flips, len(s), len(near[1]), describe(near[1], s)))
    print('  %s frames whose sequence a %s frame has: %d of %d' % (name_a, name_b, hit, len(side_a)))
    for flips, n, m, (add, rem, chg) in worst[:6]:
        print('    flip %d: %d draws against %d: %d added, %d removed, %d changed' % (flips, n, m, add, rem, chg))
    return hit, len(side_a), worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('digests', nargs='+')
    ap.add_argument('--every', type=int, default=0)
    ap.add_argument('--margin', type=int, default=4)
    ap.add_argument('--flips', action='append', default=[], help='A-B (repeatable)')
    a = ap.parse_args()
    spans = [tuple(int(x) for x in f.split('-')) for f in a.flips]
    keep = (lambda fl: any(lo <= fl <= hi for lo, hi in spans)) if spans else (lambda fl: True)
    if a.every:
        frames = [f for f in read(a.digests[0]) if keep(f[0])]
        on, off = [], []
        for flips, s in frames:
            phase = flips % a.every
            if phase < a.margin or phase >= a.every - a.margin:
                continue
            (on if (flips // a.every) & 1 else off).append((flips, s))
        sides = (('on', on), ('off', off))
    else:
        sides = (('A', [f for f in read(a.digests[0]) if keep(f[0])]), ('B', [f for f in read(a.digests[1]) if keep(f[0])]))
    (na, fa), (nb, fb) = sides
    print('%s: %d frames, %d distinct sequences, %.0f draws a frame; %s: %d frames, %d distinct, %.0f draws a frame' % (
        na, len(fa), len(set(s for _, s in fa)), sum(len(s) for _, s in fa) / max(1, len(fa)),
        nb, len(fb), len(set(s for _, s in fb)), sum(len(s) for _, s in fb) / max(1, len(fb))))
    print('across:')
    ha, ta, wa = compare(fa, fb, na, nb)
    hb, tb, wb = compare(fb, fa, nb, na)
    # the scene's own variation: each side's frames against the side's other frames (half against half)
    print('within (the scene itself):')
    wi = []
    for name, side in sides:
        half = len(side) // 2
        h, t, w = compare(side[:half], side[half:], name + ' first half', name + ' second half')
        wi.append((h, t))
    across = (ha + hb) / max(1, ta + tb)
    within = sum(h for h, _ in wi) / max(1, sum(t for _, t in wi))
    print('whole frames matched across %.1f%%, within %.1f%%' % (100 * across, 100 * within))
    # the order test over the stable draws
    every = fa + fb
    once = None
    for _, seq in every:
        c = collections.Counter(seq)
        mine = {d for d, n in c.items() if n == 1}
        once = mine if once is None else once & mine
    once = once or set()
    orders = collections.Counter()
    by_side = {na: collections.Counter(), nb: collections.Counter()}
    for name, side in sides:
        for _, seq in side:
            r = tuple(d for d in seq if d in once)
            orders[r] += 1
            by_side[name][r] += 1
    top = orders.most_common(1)[0][0] if orders else ()
    off_top = {name: sum(n for r, n in c.items() if r != top) for name, c in by_side.items()}
    print('the stable draws: %d a frame (of %.0f); orders seen %d; frames not in the commonest order: %s' % (
        len(once), sum(len(s) for _, s in every) / max(1, len(every)), len(orders),
        ', '.join('%s %d of %d' % (k, v, sum(by_side[k].values())) for k, v in off_top.items())))
    for r, n in orders.most_common(4)[1:]:
        add, rem, chg = describe(top, r)
        sides_of = [k for k, c in by_side.items() if r in c]
        print('  another order, %d frames (%s): against the commonest %d added, %d removed, %d moved or changed' % (n, '/'.join(sides_of), add, rem, chg))
    # the reference side's movers: draws out of its commonest order in any of its frames
    ref = nb if a.every else na  # off with --every, else A
    ref_frames = [r for r, n in by_side[ref].items() for _ in range(n)]
    ref_top = by_side[ref].most_common(1)[0][0] if by_side[ref] else ()
    movers = set()
    for r in by_side[ref]:
        sm = difflib.SequenceMatcher(None, ref_top, r, autojunk=False)
        kept = set()
        for blk in sm.get_matching_blocks():
            kept.update(ref_top[blk.a:blk.a + blk.size])
        movers |= set(ref_top) - kept
    fixed = [d for d in ref_top if d not in movers]
    fixed_set = set(fixed)
    bad = {name: 0 for name in by_side}
    for name, c in by_side.items():
        for r, n in c.items():
            if tuple(d for d in r if d in fixed_set) != tuple(fixed):
                bad[name] += n
    print('the %s side moves %d of the stable draws itself (tied segments); the other %d: frames not in their order %s' % (
        ref, len(movers), len(fixed), ', '.join('%s %d of %d' % (k, v, sum(by_side[k].values())) for k, v in bad.items())))
    equal = all(v == 0 for v in bad.values())
    print('order: %s' % ('EQUAL - every frame of both sides draws the stable draws the game keeps in place in one order' if equal
                         else 'NOT equal'))
    return 0 if equal else 1


if __name__ == '__main__':
    sys.exit(main())
