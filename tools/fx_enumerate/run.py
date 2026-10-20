#!/usr/bin/env python3
"""Play every effect animation of bosses and log their effects, one game run a boss.

    tools/fx_enumerate/run.py --bbhost BUILD --config TEST.toml --seed DATA --out DIR
        [--boss N|NAME ...] [--fps 60] [--anims attacks|all] [--first N] [--env BBHOST_X=Y ...]

Each boss: plan.py's plan of it, a run of tools/area_check.py's machinery
whose tour is only `god` and a hold long enough for the plan, plugins/boss_rush
taking the player to its fog wall (one round, its own) and fx_enumerate.c
(built here) playing the plan; then report.py on the run's log, into
DIR/<round>-c<model>/report.txt. A run that faults stops the rest. The
config is a test config (its [paths]); the seed's character must reach the
bosses (the boss rush's own rules).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tools'))
import area_check  # noqa: E402
import plan as fxplan  # noqa: E402
import report as fxreport  # noqa: E402

FAULT = re.compile(r'device is lost|device is taken for lost|device lost|SIGSEGV|SIGABRT|dumped core')


def build_plugin(out_so):
    out_so.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([os.environ.get('CC', 'cc'), '-shared', '-fPIC', '-O2', '-Wall', '-I' + str(ROOT / 'include'), '-o', str(out_so),
                        str(HERE / 'fx_enumerate.c'), '-lm'], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit('fx_enumerate.c did not build:\n' + r.stderr)


def one(a, rnd, name, model, fog, dlc, plugin, app0):
    out = Path(a.out) / ('%02d-c%04d' % (rnd, model))
    if out.exists():
        raise SystemExit('exists: %s' % out)
    out.parent.mkdir(parents=True, exist_ok=True)
    path, _types, anims = fxplan.plan(app0, model)
    if a.anims != 'all':
        anims = {k: v for k, v in anims.items() if 3000 <= k < 4000}
    if a.first:
        anims = dict(list(anims.items())[:a.first])
    if not anims:
        print('%2d %s: no effect animations to play - skipped' % (rnd, name))
        return True
    plan_file = Path(a.out) / ('%02d-c%04d.plan' % (rnd, model))
    plan_file.write_text('boss %d %d %s\n' % (rnd, model, name) +
                         ''.join('anim %d %.2f %s\n' % (k, s, ','.join(map(str, l))) for k, (s, l) in anims.items()))
    # the hold: the boss rush's travel, then each animation's span (capped) and its effects' lives
    hold = 120 + sum(min(s, 8.0) + 4.0 for (s, _l) in anims.values()) + 30
    tour = '# fx_enumerate: %s (c%04d), %d animations\ngod\nmark fxe:start\nhold %.0f\nmark fxe:end\nwait 2\n' % (name, model, len(anims), hold)
    area_check.tour = lambda areas, opts, kind: tour
    area_check.pass_budget = lambda areas, opts, kind: hold + 240
    orig_toml, orig_prepare = area_check.run_toml, area_check.prepare_pass

    def run_toml(*args, **kw):
        t = orig_toml(*args, **kw)
        t = t.replace('[plugins]\n', '[plugins]\nfx_enumerate = true\nboss_rush = true\n', 1)
        t += '\n'.join(['', '[boss_rush]', 'auto_start = true', 'start_delay = 5', 'order = "story"', 'first = %d' % rnd, 'rounds = 1',
                        'awake = true', 'dlc = true', 'chalice = false', ''])
        if dlc and 'old-hunters' not in t:
            t += '\n[patches]\nold-hunters = true\n'
        return t

    def prepare_pass(pass_dir, kind, areas, opts, paths):
        p = orig_prepare(pass_dir, kind, areas, opts, paths)
        plugs = pass_dir / 'data' / 'plugins'
        shutil.copy2(plugin, plugs / 'fx_enumerate.so')
        shutil.copy2(Path(a.bbhost).resolve().parent / 'plugins' / 'boss_rush.so', plugs / 'boss_rush.so')
        return p

    area_check.run_toml, area_check.prepare_pass = run_toml, prepare_pass
    os.environ.update({'FXE_PLAN': str(plan_file.resolve()), 'FXE_FOG': '%g,%g,%g' % tuple(fog), 'FXE_ANIMS': a.anims})
    argv = ['run', '--bbhost', a.bbhost, '--config', a.config, '--seed', a.seed, '--out', str(out), '--areas', 'cathedral-ward',
            '--captures', '0', '--p2p-port', str(a.p2p_port), '--numa-node', a.numa_node, '--fps', str(a.fps), '--dumps', '0',
            '--width', str(a.width), '--height', str(a.height)]
    for e in a.env:
        argv += ['--env', e]
    print('%2d %s (c%04d): %d animations, hold %.0f s' % (rnd, name, model, len(anims), hold), flush=True)
    rc = area_check.main(argv)
    area_check.run_toml, area_check.prepare_pass = orig_toml, orig_prepare
    log = out / 'perf' / 'run.log'
    text = log.read_text(errors='replace') if log.exists() else ''
    (out / 'report.txt').write_text(fxreport.report(text, '%s (c%04d)' % (name, model)))
    m = FAULT.search(text)
    print('%2d %s: exit %d%s' % (rnd, name, rc, (', FAULT: ' + m.group(0)) if m else ''), flush=True)
    return m is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bbhost', required=True)
    ap.add_argument('--config', required=True, help='a test config (its [paths] app0 and eboot)')
    ap.add_argument('--seed', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--boss', action='append', default=[], help='a round number or a name (substring); default every boss')
    ap.add_argument('--fps', type=int, default=60)
    ap.add_argument('--anims', choices=('attacks', 'all'), default='attacks', help='attacks: 3000-3999 (default)')
    ap.add_argument('--first', type=int, default=0, help='only the first N animations of each boss (a short check of the tool)')
    ap.add_argument('--env', action='append', default=[])
    ap.add_argument('--numa-node', default='0')
    ap.add_argument('--p2p-port', type=int, default=9811)
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    a = ap.parse_args()
    app0 = os.environ.get('BBHOST_APP0') or area_check.read_paths(a.config)[0] or ''
    if not app0:
        raise SystemExit('run.py: no app0 ([paths] app0 in --config, or BBHOST_APP0)')
    plugin = Path(a.out) / 'fx_enumerate.so'
    Path(a.out).mkdir(parents=True, exist_ok=True)
    build_plugin(plugin)
    for rnd, name, model, fog, dlc in fxplan.bosses():
        if a.boss and not any(b == str(rnd) or b.lower() in name.lower() for b in a.boss):
            continue
        if not one(a, rnd, name, model, fog, dlc, plugin, app0):
            print('a fault - the rest not run')
            return 9
    return 0


if __name__ == '__main__':
    sys.exit(main())
