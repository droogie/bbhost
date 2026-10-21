#!/usr/bin/env python3
"""What fx_player.c's log and word table say, per family and effect id.

    tools/fx_player/report.py RUN_DIR [--base RUN_DIR ...] [--off RUN_DIR] [--converted 13|19|0x..,0x..]

RUN_DIR is a tools/fx_player/run.py output (ac/perf/run.log, words.tsv,
families.json). Per shot: whether it made any effect. Per family and effect id:
its instances' lives (median, game seconds), state edges and emissions (per
life and per game second), the frames its containers ran (with a whole step;
between steps with the frame's own step - "smooth" - or held - "kept"), the
classes behind a kept frame that are not on the converted list (the build's:
its exit report names how many; --converted 13, 19 or a 0x.. list to say),
and the frame words: in each class's objects, the words that change on 90% or
more of the frames with a whole step ("frame words"), and how many of those
change on 90% or more of the frames between steps (moving) or on 10% or less
(held).

--base: the same plan at the game's own 30 (given more than once, the bases'
spread is the reference: a second run of the same arm is the noise floor of
what a birth's random draws move); each id's count of lives, median life,
edges and emissions per life against the bases'. --off: the same plan at the
same rate with the smooth steps off. The pass rule for the smooth steps: no
frame of the family was kept on the held path, and what its effects count -
the count of lives, a life within a tick (1/30 s), the edges and emissions
within one - is what they count with the steps off (with --off; without it,
against the base). A rate differs from 30 in what the game around the effects
does at it (the bullets' flight, the camera, the random draws they share), with
the steps off as much as on, so with --off the base is shown, not judged.
Held frame words of converted classes are shown, not judged: some of them count
once a whole step by design (the light's flicker phase).
"""
import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# The converted classes (decomp/sfx/step_gate.cpp's kConverted) as the smooth steps' first two batches list them - the
# 13 of the first, then the FFX light and the FFX-format GX types and wind source - by the count the build's exit report
# names ("(N classes on it)"); and the composite motion, converted when its two parts are (the step gate looks through it).
FIRST_CONVERTED = {0x570e650, 0x570e790, 0x570e710, 0x56eb2e0, 0x5756e50, 0x56e3090, 0x56e3100, 0x56eb1d0, 0x56eb120, 0x56eb220, 0x56ea9b0,
                 0x56eaa90, 0x56eb290}
SECOND_CONVERTED = FIRST_CONVERTED | {0x56ebba0, 0x56e35d0, 0x56e3530, 0x56e3580, 0x56ebe20, 0x56eadf0}
CONVERTED_BY_COUNT = {13: FIRST_CONVERTED, 19: SECOND_CONVERTED}
COMPOSITE = 0x570e750
TICK = 1.0 / 30
FRAME_WORD = 0.9   # a word that changes on this share of whole-step frames or more is a frame word
MOVING, HELD = 0.9, 0.1
MIN_OBS = 8
PLACE_LIFE = 20.0  # an id no family expects that lives longer (game seconds) or past the end is the place's, not the shot's

READY = re.compile(r'fxp: f(\d+) ready in (\w+) .* (\d+) shots, (\d+) frames a game second')
SHOT = re.compile(r'fxp: f(\d+) shot s(\d+) (\d+)/(\d+) (\S+) (\S+) (\w+) (-?\d+) dummy (-?\d+) pitch (\S+) repeat (\d+)/(\d+): (-?\d+)')
FALLBACK = re.compile(r'fxp: f(\d+) shot s(\d+): no (\d+) at dummy poly (-?\d+)')
SHOT_DONE = re.compile(r'fxp: f(\d+) shot s(\d+) done after ([\d.]+) s')
CAPPED = re.compile(r'fxp: f(\d+) shot s(\d+): (\d+) request\(s\) live at')
END = re.compile(r'fxp: f(\d+) (gone|still) s(-?\d+) (\S+) (\d+) (?:req|inst) ([0-9a-f]+) life (\d+) frames (\d+) steps retired (-?\d+) nodes (\d+) lost (\d+) '
                 r'edges (\d+) emissions (\d+) frames whole (\d+) kept (\d+) smooth (\d+) none (\d+) calls whole (\d+) held (\d+) smooth (\d+) '
                 r'step whole (\S+) between (\S+) classes (\S+) path (\S+)')
DONE = re.compile(r'fxp: f(\d+) done: (\d+) shots made, (\d+) frames with a whole step, (\d+) between')
IDS = re.compile(r'ids checked against requests (\d+), differing (\d+)')
REQ_GONE = re.compile(r'fxp: f(\d+) req gone s(-?\d+) (\S+) (\d+) req ([0-9a-f]+) life (\d+) frames')
SMOOTH_ON = re.compile(r'frame rate: effect steps smooth: ')
SIM_TIME = re.compile(r'frame rate: BBHOST_SIM_FRAME_TIME=1')
SIM_FPS = re.compile(r'BBHOST_SFX_SIM_FPS=(\d+): the effects')
KEPT_LINE = re.compile(r'decomp: the smooth effect steps: (\d+) instance updates took them, (\d+) stayed on the held frames.*\((\d+) classes on it\)')
KEPT_CLASS = re.compile(r'decomp:   kept by class 0x([0-9a-f]+): (\d+) instance updates')


def read_words(path):
    """{(id, kind, class): {'obs': [whole, between], 'words': {offset: [whole, between]}}} of a words.tsv; {} when absent."""
    out = defaultdict(lambda: {'obs': [0, 0], 'words': {}})
    p = Path(path)
    if not p.exists():
        return {}
    for line in p.open():
        f = line.rstrip('\n').split('\t')
        if line.startswith('#obs'):
            k = (int(f[1]), f[2], int(f[3], 16))
            out[k]['obs'] = [int(f[4]), int(f[5])]
        elif line.startswith('#') or line.startswith('id\t'):
            continue
        elif len(f) >= 8:
            k = (int(f[0]), f[1], int(f[2], 16))
            out[k]['obs'] = [int(f[4]), int(f[5])]
            out[k]['words'][int(f[3], 16)] = [int(f[6]), int(f[7])]
    return dict(out)


def parse(text):
    r = {'fps': 30, 'shots': {}, 'ends': [], 'fallback': set(), 'capped': set(), 'smooth': bool(SMOOTH_ON.search(text)),
         'sim_time': bool(SIM_TIME.search(text)), 'sim_fps': None, 'done': None, 'kept_total': None, 'kept_classes': {},
         'converted_count': None, 'ids_checked': None, 'reqs': []}
    m = SIM_FPS.search(text)
    if m:
        r['sim_fps'] = int(m.group(1))
    for line in text.splitlines():
        m = READY.search(line)
        if m:
            r['fps'] = int(m.group(4))
            continue
        m = SHOT.search(line)
        if m:
            k = int(m.group(2))
            r['shots'][k] = {'frame': int(m.group(1)), 'family': m.group(5), 'name': m.group(6), 'kind': m.group(7), 'id': int(m.group(8)),
                             'dummy': int(m.group(9)), 'rc': int(m.group(13)), 'repeat': int(m.group(11)), 'time': None}
            continue
        m = FALLBACK.search(line)
        if m:
            r['fallback'].add(int(m.group(2)))
            continue
        m = SHOT_DONE.search(line)
        if m and int(m.group(2)) in r['shots']:
            r['shots'][int(m.group(2))]['time'] = float(m.group(3))
            continue
        m = CAPPED.search(line)
        if m:
            r['capped'].add(int(m.group(2)))
            continue
        m = END.search(line)
        if m:
            g = m.groups()
            e = {'how': g[1], 'shot': int(g[2]), 'family': g[3], 'id': int(g[4]), 'life': int(g[6]), 'steps': int(g[7]), 'retired': int(g[8]),
                 'nodes': int(g[9]), 'edges': int(g[11]), 'emissions': int(g[12]), 'f_whole': int(g[13]), 'f_kept': int(g[14]),
                 'f_smooth': int(g[15]), 'calls_held': int(g[18]), 'calls_smooth': int(g[19]),
                 'classes': [] if g[22] == '-' else [int(c, 16) for c in g[22].split(',')],
                 'path': [] if g[23] == '-' else [tuple(float(v) for v in p.split('/')) for p in g[23].split(';')]}
            r['ends'].append(e)
            continue
        m = REQ_GONE.search(line)
        if m:
            r['reqs'].append({'shot': int(m.group(2)), 'family': m.group(3), 'id': int(m.group(4)), 'life': int(m.group(6))})
            continue
        m = DONE.search(line)
        if m:
            r['done'] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
            m2 = IDS.search(line)
            if m2:
                r['ids_checked'] = (int(m2.group(1)), int(m2.group(2)))
            continue
        m = KEPT_LINE.search(line)
        if m:
            r['kept_total'] = (int(m.group(1)), int(m.group(2)))
            r['converted_count'] = int(m.group(3))
            continue
        m = KEPT_CLASS.search(line)
        if m:
            r['kept_classes'][int(m.group(1), 16)] = int(m.group(2))
    return r


def med(v):
    return statistics.median(v) if v else None


def by_id(run):
    """{id: [ends]} and {id: family} of a parsed run."""
    ids, fam = defaultdict(list), {}
    for e in run['ends']:
        ids[e['id']].append(e)
        fam.setdefault(e['id'], e['family'])
    return ids, fam


def id_stats(ends, fps):
    lives = [e['life'] / fps for e in ends]
    per_life = lambda k: [e[k] for e in ends]
    rate = lambda k: [e[k] / (e['life'] / fps) for e in ends if e['life'] > 0]
    return {'n': len(ends), 'life': med(lives), 'steps': med([e['steps'] * TICK for e in ends]), 'edges': med(per_life('edges')),
            'emissions': med(per_life('emissions')), 'edges_s': med(rate('edges')), 'emissions_s': med(rate('emissions')),
            'f_whole': sum(e['f_whole'] for e in ends), 'f_kept': sum(e['f_kept'] for e in ends), 'f_smooth': sum(e['f_smooth'] for e in ends),
            'classes': sorted({c for e in ends for c in e['classes']}), 'still': sum(1 for e in ends if e['how'] == 'still')}


def word_verdicts(words, eid, converted):
    """Per class of an id: (kind, class, converted?, whole obs, between obs, frame words, moving, held)."""
    out = []
    for (i, kind, cls), v in sorted(words.items()):
        if i != eid:
            continue
        wo, bo = v['obs']
        fw = [off for off, (wc, _bc) in v['words'].items() if wo >= MIN_OBS and wc >= FRAME_WORD * wo]
        moving = [off for off in fw if bo >= MIN_OBS and v['words'][off][1] >= MOVING * bo]
        held = [off for off in fw if bo >= MIN_OBS and v['words'][off][1] <= HELD * bo]
        out.append((kind, cls, cls in converted or cls == COMPOSITE, wo, bo, fw, moving, held))
    return out


def fmt(v, f='%.2f'):
    return '-' if v is None else f % v


def report(text, words, fams, base=None, name='', converted=None, off=None):
    """base: a (log text, words) pair or a list of them; converted: the classes (default: by the build's count); off: a
    (log text, words) pair of the same rate with the smooth steps off - with it, a family passes when none of its frames was
    kept and what its effects count (lives, edges, emissions) is what they count with the steps off, and the base is shown
    beside it (a rate's own difference from 30, which the steps off have too)."""
    bases = [] if base is None else (list(base) if isinstance(base, list) else [base])
    run = parse(text)
    if converted is None:
        converted = CONVERTED_BY_COUNT.get(run['converted_count'], FIRST_CONVERTED)
    fps = run['fps'] or 30
    ids, fam_of = by_id(run)
    lines = ['fx_player %s: %d frames a game second; effect steps smooth %s%s%s' %
             (name, fps, 'ON' if run['smooth'] else 'off', ', the whole game a target a frame' if run['sim_time'] else '',
              ', the effects\' schedule as at %d (SIM)' % run['sim_fps'] if run['sim_fps'] else '')]
    if run['done']:
        lines.append('  %d shots made; frames with a whole effect step %d, between %d' % run['done'])
        if run['ids_checked']:
            lines.append('  effect ids (named at their creation) checked against their requests\' ids: %d checks, %d differ' % run['ids_checked'])
    else:
        lines.append('  THE PLAN DID NOT FINISH (no "done" line): %d shots made' % len(run['shots']))
    lines.append('  the converted list: %d classes%s' % (len(converted), ' (the build names %d)' % run['converted_count'] if run['converted_count'] else ''))
    if run['kept_total']:
        lines.append('  the binary\'s count: %d instance updates took the smooth steps, %d were kept (%s)' %
                     (run['kept_total'] + (', '.join('0x%x %d' % kv for kv in sorted(run['kept_classes'].items(), key=lambda kv: -kv[1])) or '-',)))

    # shots: did each make anything?
    made = defaultdict(set)
    for e in run['ends']:
        made[e['shot']].add(e['id'])
    for q in run['reqs']:
        made[q['shot']].add(q['id'])
    req_lives = defaultdict(list)
    for q in run['reqs']:
        req_lives[q['id']].append(q['life'] / fps)
        fam_of.setdefault(q['id'], q['family'])
    lines.append('')
    lines.append('shots (spawned = the shot made effect instances):')
    for k, s in sorted(run['shots'].items()):
        got = made.get(k, set())
        lines.append('  s%-3d %-20s %-44s %-8s %9d: %s%s%s' % (
            k, s['family'], s['name'][:44], s['kind'], s['id'], ('spawned %s' % ' '.join(map(str, sorted(got)))) if got else 'NOTHING SPAWNED',
            ' (body dummy poly)' if k in run['fallback'] else '', ' (capped)' if k in run['capped'] else ''))

    bstats = None  # {id: [stats of each base]}
    if bases:
        bstats = defaultdict(list)
        for btext, _bwords in bases:
            brun = parse(btext)
            bids_, _ = by_id(brun)
            for i, ends in bids_.items():
                bstats[i].append(id_stats(ends, brun['fps'] or 30))
    ostats = None
    if off is not None:
        orun = parse(off[0])
        oids, _ = by_id(orun)
        ostats = {i: [id_stats(e, orun['fps'] or 30)] for i, e in oids.items()}
    lines.append('')
    lines.append('per family and effect id (lives and rates: medians; frames: summed over the lives; words: per class, '
                 'frame words W / moving on the frames between M / held there H):')
    order = list(fams.keys()) if fams else []
    for f in sorted(set(fam_of.values()) | set(order), key=lambda f: (order.index(f) if f in order else 99, f)):
        fid = sorted(i for i, ff in fam_of.items() if ff == f)
        expected = set(fams.get(f, {}).get('expected', [])) if fams else set()
        missing = sorted(expected - set(fid))
        lines.append('== %s: %d effect ids seen%s' % (f, len(fid), ('; expected, not seen: ' + ' '.join(map(str, missing))) if missing else ''))
        fam_pass, fam_held, fam_kept, fam_measured, fam_place = True, False, 0, 0, 0
        for i in fid:
            if i not in ids:
                rl = sorted(req_lives.get(i, []))
                lines.append('  %8d%s request only: %d lives, life %s s - its instance ran no node updates, nothing to measure between steps' % (
                    i, '' if i in expected or not expected else '+', len(rl), fmt(med(rl))))
                continue
            st = id_stats(ids[i], fps)
            if expected and i not in expected and (st['still'] or (st['life'] or 0) > PLACE_LIFE):
                lines.append('  %8d+ lives %d, life %s s%s: the place\'s own, most likely (born during a shot, alive long past it) - not counted' % (
                    i, st['n'], fmt(st['life']), ', %d still alive at the end' % st['still'] if st['still'] else ''))
                fam_place += 1
                continue
            between = st['f_kept'] + st['f_smooth']
            off_list = [c for c in st['classes'] if c not in converted and c != COMPOSITE]
            wv = word_verdicts(words, i, converted)
            conv_held = sum(len(h) for (_k, _c, conv, _wo, _bo, _fw, _m, h) in wv if conv)
            conv_fw = sum(len(fw) for (_k, _c, conv, _wo, bo, fw, _m, _h) in wv if conv and bo >= MIN_OBS)
            conv_moving = sum(len(m) for (_k, _c, conv, _wo, _bo, _fw, m, _h) in wv if conv)
            row = '  %8d%s lives %d, life %s s (whole steps %s s), edges %s / emissions %s a life (%s / %s a s); frames whole %d, between %d: smooth %d, kept %d' % (
                i, '' if i in expected or not expected else '+', st['n'], fmt(st['life']), fmt(st['steps']), fmt(st['edges'], '%.1f'),
                fmt(st['emissions'], '%.1f'), fmt(st['edges_s'], '%.1f'), fmt(st['emissions_s'], '%.1f'), st['f_whole'], between, st['f_smooth'],
                st['f_kept'])
            if st['still']:
                row += '; %d STILL ALIVE at the end' % st['still']
            lines.append(row)
            lines.append('           classes %s%s' % (' '.join('0x%x' % c for c in st['classes']) or '-',
                                                    ('; off the converted list: ' + ' '.join('0x%x' % c for c in off_list)) if off_list else ''))
            wl = ['%s 0x%x%s W%d M%d H%d' % (k, c, '' if conv else '*', len(fw), len(m), len(h)) for (k, c, conv, wo, bo, fw, m, h) in wv if fw]
            if wl:
                lines.append('           words (* off the list): ' + ', '.join(wl))
            verdict = []
            if st['f_whole'] + between == 0:
                verdict.append('its nodes never updated (no container call) - nothing measured')
            elif between:
                if st['f_kept'] == 0 and st['f_smooth'] > 0:
                    verdict.append('smooth on every frame between steps')
                elif st['f_smooth'] == 0:
                    verdict.append('HELD on every frame between steps' + (' (kept by %s)' % ' '.join('0x%x' % c for c in off_list) if run['smooth'] and off_list else ''))
                else:
                    verdict.append('kept on %d of %d frames between steps' % (st['f_kept'], between))
                if conv_fw:
                    verdict.append('converted classes\' frame words: %d of %d move between steps, %d held' % (conv_moving, conv_fw, conv_held))
            else:
                verdict.append('no frame between steps (every frame a whole step)')
            if st['f_whole'] + between:
                fam_measured += 1
            if run['smooth'] and between and st['f_kept']:
                fam_pass = False
            if between and st['f_smooth'] == 0:
                fam_held = True
            fam_kept += st['f_kept']
            for label, stats, decides in (('the steps off', ostats, True), ('the base' + ('s' if len(bases) > 1 else ''), bstats, ostats is None)):
                if stats is None:
                    continue
                bl = stats.get(i, [])
                if not bl:
                    verdict.append('not in %s' % label)
                    continue
                def within(v, key, tol):
                    vals = [b[key] for b in bl if b[key] is not None]
                    if v is None or not vals:
                        return True, '-'
                    lo, hi = min(vals), max(vals)
                    span = ('%.2f' % lo) if lo == hi else ('%.2f-%.2f' % (lo, hi))
                    return lo - tol - 1e-6 <= v <= hi + tol + 1e-6, span
                ok_n, sn = within(st['n'], 'n', 1)
                # a life still running at the end is the run's length, not the effect's
                ok_l, sl = (True, 'still') if st['still'] or any(b['still'] for b in bl) else within(st['life'], 'life', TICK)
                ok_e, se = within(st['edges'], 'edges', 1)
                ok_m, sm = within(st['emissions'], 'emissions', 1)
                ok = ok_n and ok_l and ok_e and ok_m
                verdict.append('against %s: lives %s, life %s s, edges %s, emissions %s a life: %s' % (
                    label, sn.split('.')[0] if '-' not in sn else sn, sl, se, sm,
                    'within a tick' if ok else 'DIFFERS (%s)' % ', '.join(n for n, o in (('lives', ok_n), ('life', ok_l), ('edges', ok_e),
                                                                                       ('emissions', ok_m)) if not o)))
                if decides and not ok:
                    fam_pass = False
            lines.append('           ' + '; '.join(verdict))
        if len(fid) == fam_place:
            lines.append('   %s: NOTHING SPAWNED%s' % (f, ' (only the place\'s own effects in its shots\' windows)' if fid else ''))
        elif not fam_measured:
            lines.append('   %s: spawned, but nothing measured (no effect of it updated its nodes)' % f)
        elif run['smooth']:
            lines.append('   %s: %s' % (f, 'PASS (smooth, kept 0%s)' % (', counted as with the steps off' if ostats is not None else
                                                                        ', the base agrees' if bstats is not None else '') if fam_pass else
                                       'not yet: %d frames kept%s' % (fam_kept, '' if fam_kept else ', see above')))
        else:
            lines.append('   %s: %s' % (f, 'held between whole steps (effect steps 30, as today)' if fam_held else 'every frame a whole step'))
    if run['ends']:
        flights = [(e['shot'], e['id'], e['path']) for e in run['ends']
                   if len(e['path']) >= 3 and run['shots'].get(e['shot'], {}).get('kind') in ('behavior', 'goods')]
        if flights:
            lines.append('')
            lines.append('the first frames of the bullets\' effects (root node position):')
            seen = set()
            for k, i, p in flights:
                if (k, i) in seen:
                    continue
                seen.add((k, i))
                d = [p[-1][j] - p[0][j] for j in range(3)]
                lines.append('  s%d %d: from (%.2f %.2f %.2f) by (%+.2f %+.2f %+.2f)' % (k, i, p[0][0], p[0][1], p[0][2], d[0], d[1], d[2]))
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--base', action='append', default=[], help='a run of the same plan to compare with; more than one: their spread')
    ap.add_argument('--off', help='the same plan at the same rate with the smooth steps off: what a family counts must match it')
    ap.add_argument('--converted', help='the converted classes, 0x.. comma list (default: the list of as many classes as the build\'s exit report names)')
    a = ap.parse_args()
    d = Path(a.run)
    text = (d / 'ac' / 'perf' / 'run.log').read_text(errors='replace')
    fams = json.loads((d / 'families.json').read_text()) if (d / 'families.json').exists() else {}
    base = [((Path(b) / 'ac' / 'perf' / 'run.log').read_text(errors='replace'), read_words(Path(b) / 'words.tsv')) for b in a.base] or None
    conv = None
    if a.converted:
        conv = CONVERTED_BY_COUNT.get(int(a.converted)) if a.converted.isdigit() else {int(x, 16) for x in a.converted.split(',')}
    off = None
    if a.off:
        off = ((Path(a.off) / 'ac' / 'perf' / 'run.log').read_text(errors='replace'), read_words(Path(a.off) / 'words.tsv'))
    sys.stdout.write(report(text, read_words(d / 'words.tsv'), fams, base=base, name=d.name, converted=conv, off=off))


if __name__ == '__main__':
    main()
