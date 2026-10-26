#!/usr/bin/env python3
"""Read the render queue's census (BBHOST_RQ_CENSUS=1 with BBHOST_RQ_CENSUS_OUT=path, src/decomp/gx/render_queue.cpp)
and say, per place of an area check's run, what the GX workers record of a frame's draws and what holds the frame:

    tools/rq_census.py RUN_DIR CENSUS_FILE [--place ID] [--json OUT]

RUN_DIR is an area check's run (tools/area_check.py; its report.json gives each place's measure window, which needs a
frame dump after it: --dumps 1). Per place:
  - the parallel-fors a flip, their chunks, and the phase: the partition's call to the last chunk job's end;
  - by the partition table's entry (category/sub-category of the sort key): chunks, items, draw commands, the work
    (ms a flip) and the longest chunk;
  - the critical chunk (the last to end): its entry, items, length, where it started in the phase, its largest item's
    share, the participant it ran on;
  - the checks a cut of a chunk relies on: items that drew with state an earlier item of their chunk set (commands 1,
    2, 6, 7), and other segments of the frame's list whose order keys fall inside a chunk's span (a cut chunk's later
    pieces would sort after them).
"""
import argparse
import bisect
import collections
import json
import os
import struct
import sys

PART = struct.Struct('<QQIIIIIIIIIII' + 'bBHII' * 96 + 'I')
CHUNK = struct.Struct('<QQQIIIIIIIIII')
ITEM = struct.Struct('<IIHHHBB')
APPEND = struct.Struct('<QQIIIIII')
HOW = {0: 'none', 1: 'one', 2: 'split', 3: 'stealing', 4: 'overflow', 5: 'last'}

# The partition's table (0x553ae90): the category and sub-category an entry starts at (-1: the whole category).
STARTS = [(0x00, -1), (0x01, -1), (0x02, -1), (0x03, -1), (0x04, -1), (0x05, -1), (0x0b, -1), (0x11, -1), (0x17, -1),
          (0x1d, -1), (0x23, -1), (0x29, -1), (0x2f, -1), (0x35, -1), (0x36, -1), (0x37, -1), (0x3d, -1), (0x3e, 0),
          (0x3e, 4), (0x3e, 5), (0x3e, 13), (0x3e, 14), (0x3e, 15), (0x3e, 19), (0x3e, 21), (0x3e, 26), (0x3e, 46),
          (0x3e, 51), (0x3e, 52), (0x3e, 53), (0x3f, -1)]


def entry_name(e):
    if e is None or e < 0:
        return 'none'
    c, s = STARTS[e]
    return '%02x' % c if s < 0 else '%02x/%02x' % (c, s)


def read(path):
    parts, chunks, appends = [], [], []
    with open(path, 'rb') as f:
        if f.read(8) != b'RQCENS1\n':
            sys.exit('not a census file: ' + path)
        while True:
            tag = f.read(1)
            if tag == b'P':
                b = f.read(PART.size)
                if len(b) < PART.size:
                    break
                v = PART.unpack(b)
                runs = [dict(entry=v[13 + 5 * k], how=HOW.get(v[14 + 5 * k]), chunks=v[15 + 5 * k], start=v[16 + 5 * k],
                             n=v[17 + 5 * k]) for k in range(min(v[11], 96))]
                parts.append(dict(t=v[0], ret=v[2], gen=v[9], flip=v[10], runs=runs))
            elif tag == b'C':
                b = f.read(CHUNK.size)
                if len(b) < CHUNK.size:
                    break
                v = CHUNK.unpack(b)
                ib = f.read(ITEM.size * v[6])
                chunks.append(dict(t0=v[0], t1=v[1], first=v[3], last=v[4], gen=v[7], worker=v[8], key=v[9], work=v[10],
                                   items=[ITEM.unpack_from(ib, 16 * k) for k in range(len(ib) // 16)]))
            elif tag == b'A':
                b = f.read(APPEND.size)
                if len(b) < APPEND.size:
                    break
                v = APPEND.unpack(b)
                appends.append(dict(t=v[0], list=v[1], key=v[2] if v[2] != 0xffffffff else v[3]))
            elif not tag:
                break
            else:
                sys.exit('bad record at %d' % f.tell())
    return parts, chunks, appends


def windows(run_dir):
    rep = json.load(open(os.path.join(run_dir, 'report.json')))
    out = []
    for pid, ar in rep.get('areas', {}).items():
        try:
            fa, w = ar['frames']['a']['flip'], ar['window']
            out.append((pid, fa - w['flips'], fa - 1, w['flips'], w['seconds'], w.get('fps') or 0.0))
        except (KeyError, TypeError):
            pass
    return out


def place(parts, chunks, appends, win):
    pid, f0, f1, flips, secs, fps = win
    gens = {p['gen']: dict(part=p, chunks=[]) for p in parts if p['gen'] and f0 <= p['flip'] <= f1}
    for c in chunks:
        if c['gen'] in gens:
            gens[c['gen']]['chunks'].append(c)
    gl = [g for g in gens.values() if g['chunks']]
    if not gl:
        return None
    t_lo = min(g['part']['t'] for g in gl)
    t_hi = max(max(c['t1'] for c in g['chunks']) for g in gl)
    tpm = (t_hi - t_lo) / (secs * 1000.0)  # ticks a ms, from the window's own length
    by = collections.defaultdict(lambda: dict(chunks=0, items=0, draws=0, work=0, longest=0))
    crit, phase, inherits = [], 0, 0
    for g in gl:
        p, cs = g['part'], g['chunks']
        runs = sorted((r['start'], r) for r in p['runs'])

        def run_of(first):
            hit = None
            for s, r in runs:
                if s <= first:
                    hit = r
            return hit
        end = max(c['t1'] for c in cs)
        phase += end - p['t']
        for c in cs:
            r = run_of(c['first'])
            d = by[r['entry'] if r else -1]
            d['chunks'] += 1
            d['items'] += len(c['items'])
            d['draws'] += sum(it[3] for it in c['items'])
            d['work'] += c['work']
            d['longest'] = max(d['longest'], c['t1'] - c['t0'])
            inherits += sum(1 for it in c['items'] if it[5] & 2)
        cc = max(cs, key=lambda c: c['t1'])
        r = run_of(cc['first'])
        drawn = [it[1] for it in cc['items'] if not it[5] & 1]
        crit.append(dict(entry=r['entry'] if r else -1, items=cc['last'] - cc['first'], ms=cc['t1'] - cc['t0'],
                         start=cc['t0'] - p['t'], largest=(max(drawn) / cc['work']) if drawn and cc['work'] else 0,
                         worker=cc['worker'], run_items=r['n'] if r else 0, run_chunks=r['chunks'] if r else 0))
    # the frame's segments: consecutive appends to one list; a chunk's span runs from its key to its last item's
    frames, cur, cur_list = [], [], None
    for a in appends:
        if a['list'] != cur_list and cur:
            frames.append(cur)
            cur = []
        cur_list = a['list']
        cur.append(a)
    if cur:
        frames.append(cur)
    inside = 0
    gl_by_t = sorted(gl, key=lambda g: g['part']['t'])
    starts = [g['part']['t'] for g in gl_by_t]
    n_frames = 0
    for fr in frames:
        lo_t, hi_t = fr[0]['t'], fr[-1]['t']
        k0 = bisect.bisect_left(starts, lo_t)
        k1 = bisect.bisect_right(starts, hi_t)
        if k0 == k1:
            continue
        n_frames += 1
        keys = sorted(a['key'] for a in fr)
        for g in gl_by_t[k0:k1]:
            for c in g['chunks']:
                if not c['items']:
                    continue
                hi = (c['items'][-1][0] & 0xffff0000) | ((c['first'] + len(c['items']) - 1) & 0xffff)
                inside += bisect.bisect_right(keys, hi) - bisect.bisect_right(keys, c['key'])
    n = len(crit)
    top = collections.Counter()
    for c in crit:
        top[c['entry']] += c['ms']
    return dict(place=pid, flips=flips, fps=fps, parallel_fors=len(gl) / flips, phase_ms=phase / tpm / flips,
                by_entry={entry_name(e): dict(chunks=d['chunks'] / flips, items=d['items'] / flips, draws=d['draws'] / flips,
                                              work_ms=d['work'] / tpm / flips, longest_chunk_ms=d['longest'] / tpm)
                          for e, d in sorted(by.items(), key=lambda kv: -kv[1]['work'])},
                critical=dict(ms=sum(c['ms'] for c in crit) / n / tpm, items=sum(c['items'] for c in crit) / n,
                              start_ms=sum(c['start'] for c in crit) / n / tpm, largest_item=sum(c['largest'] for c in crit) / n,
                              run_items=sum(c['run_items'] for c in crit) / n, run_chunks=sum(c['run_chunks'] for c in crit) / n,
                              by_entry={entry_name(e): t / tpm / flips for e, t in top.most_common(4)},
                              on_main_thread=sum(1 for c in crit if c['worker'] == 0) / n),
                items_with_inherited_state=inherits, frames=n_frames, others_inside_chunk_spans=inside)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('census')
    ap.add_argument('--place')
    ap.add_argument('--json')
    a = ap.parse_args()
    parts, chunks, appends = read(a.census)
    res = {}
    for w in windows(a.run_dir):
        if a.place and w[0] != a.place:
            continue
        r = place(parts, chunks, appends, w)
        if not r:
            continue
        res[w[0]] = r
        c = r['critical']
        print('%s: %.1f fps; %.2f parallel-fors a flip, phase %.2f ms a flip' % (r['place'], r['fps'], r['parallel_fors'], r['phase_ms']))
        print('  critical chunk %.2f ms, %.0f items, starts %.2f ms in, largest item %.0f%%; its run %.0f items in %.1f chunks; by entry %s; '
              'on the main thread %.0f%%' % (c['ms'], c['items'], c['start_ms'], 100 * c['largest_item'], c['run_items'], c['run_chunks'],
                                             ', '.join('%s %.2f' % kv for kv in c['by_entry'].items()), 100 * c['on_main_thread']))
        print('  entry: work ms a flip, chunks, items, draw commands, longest chunk ms')
        for k, d in list(r['by_entry'].items())[:10]:
            print('    %-6s %6.2f %5.1f %6.1f %6.1f %6.2f' % (k, d['work_ms'], d['chunks'], d['items'], d['draws'], d['longest_chunk_ms']))
        print('  a cut keeps: items drawing with an earlier item\'s state %d; other segments inside a chunk\'s span %d (over %d frames)' % (
            r['items_with_inherited_state'], r['others_inside_chunk_spans'], r['frames']))
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
