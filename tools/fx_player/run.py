#!/usr/bin/env python3
"""Make the player's combat effects at one place and measure them, one game run.

    tools/fx_player/run.py --bbhost BUILD --config TEST.toml --seed DATA --out DIR
        [--place central-yharnam] [--stand X,Y,Z,DEG] [--fps 30|60|120] [--smooth 0|1]
        [--sim-frame-time] [--family NAME ...] [--repeat N] [--dumps 0.3,1.5]
        [--env BBHOST_X=Y ...] [--base RUN_DIR]

plan.py's plan (every family, or --family), then a run of tools/area_check.py's
machinery whose tour travels to --place (an area_check place: its lamp, its
view; Central Yharnam's lamp square by default - the Hunter's Dream lawn draws
at under a frame a second on a software renderer), stands the player there
(--stand: a spot of its own, in the map's frame), switches the enemies near it
off, puts the camera behind the player and holds; fx_player.c (built here) waits until the player has been in the
place's block FXP_SETTLE frames with no loading screen, makes the plan's shots
one after another, and ends the run when they are done. Then report.py on the
run's log, into DIR/report.txt (against --base's run when given).

--fps is the game's Frame rate (BBHOST_GAME_FPS). On a slow renderer every
clock of the game falls behind a converted rate's: --sim-frame-time
(BBHOST_SIM_FRAME_TIME=1, a build that has it) steps the whole game a target's
length a frame, effects, bullets and SpEffects alike, so the effects meet the
game's objects as at that rate; the game's own 30 needs none. --smooth sets
BBHOST_SFX_SMOOTH (the effect steps' frame domain; omitted: the build's
default). The config is a test config (its [paths]); the seed's character must
be able to travel to the place.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tools'))
import area_check  # noqa: E402
import report as fxreport  # noqa: E402

# plan.py has fx_enumerate's plan.py's name: load ours by path
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('fx_player_plan', str(HERE / 'plan.py'))
fxplan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fxplan)

FAULT = ('device is lost', 'device is taken for lost', 'device lost', 'SIGSEGV', 'SIGABRT', 'dumped core')


def build_plugin(out_so):
    out_so.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([os.environ.get('CC', 'cc'), '-shared', '-fPIC', '-O2', '-Wall', '-I' + str(ROOT / 'include'), '-o', str(out_so),
                        str(HERE / 'fx_player.c'), '-lm', '-lpthread'], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit('fx_player.c did not build:\n' + r.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bbhost', required=True)
    ap.add_argument('--config', required=True, help='a test config (its [paths] app0 and eboot)')
    ap.add_argument('--seed', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--place', default='central-yharnam', help='an area_check place (its lamp and view)')
    ap.add_argument('--stand', help='X,Y,Z,DEG: a spot of its own in the place, the map\'s frame')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--smooth', choices=('0', '1'), help='BBHOST_SFX_SMOOTH')
    ap.add_argument('--sim-frame-time', action='store_true', help='BBHOST_SIM_FRAME_TIME=1: the whole game a target\'s length a frame')
    ap.add_argument('--family', action='append', default=[])
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--per-family', type=int, default=0, help='only the first N shots of each family (a short check of the tool)')
    ap.add_argument('--speffect-s', type=float, default=3.0)
    ap.add_argument('--dumps', default='', help='game seconds after each shot to dump the displayed frame at ("0.3,1.5")')
    ap.add_argument('--hold', type=float, default=0.0, help='the tour\'s hold, real seconds (the plugin ends the run when its plan is '
                    'done; default: the plan\'s game seconds at --slow-fps rendered frames a second)')
    ap.add_argument('--slow-fps', type=float, default=4.0, help='the slowest frames a second the default hold allows for (lavapipe: 4)')
    ap.add_argument('--env', action='append', default=[])
    ap.add_argument('--numa-node', default='0')
    ap.add_argument('--p2p-port', type=int, default=9811)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=360)
    ap.add_argument('--base', help='a run of the same plan to compare with (report.py --base)')
    ap.add_argument('--converted', help='report.py --converted: 13, 19 or a 0x.. list (default: as the build names)')
    ap.add_argument('--dry-run', action='store_true', help='area_check --dry-run: everything but the game')
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit('exists: %s' % out)
    out.mkdir(parents=True)
    app0 = os.environ.get('BBHOST_APP0') or area_check.read_paths(a.config)[0] or ''
    if not app0:
        raise SystemExit('run.py: no app0 ([paths] app0 in --config, or BBHOST_APP0)')
    place = area_check.AREA_BY_ID.get(a.place)
    if not place:
        raise SystemExit('run.py: unknown place %s (have %s)' % (a.place, ', '.join(sorted(area_check.AREA_BY_ID))))

    shots, fams = fxplan.build(app0, a.repeat, a.speffect_s)
    if a.family:
        shots = [s for s in shots if s[0] in a.family]
    order = {k: i for i, k in enumerate(fxplan.ORDER)}
    shots.sort(key=lambda s: order.get(s[0], 99))
    if a.per_family:
        kept, count = [], {}
        for s in shots:
            count[s[0]] = count.get(s[0], 0) + 1
            if count[s[0]] <= a.per_family:
                kept.append(s)
        shots = kept
    fams = {k: v for k, v in fams.items() if k in {s[0] for s in shots}}
    plan_file = out / 'plan.txt'
    plan_file.write_text('# fx_player plan: %d shots\n' % len(shots) +
                         ''.join('shot %s %s %d %g %.1f %d %s %.3f\n' % s for s in shots))
    import json
    (out / 'families.json').write_text(json.dumps(fams, indent=1, sort_keys=True) + '\n')
    plugin = out / 'fx_player.so'
    build_plugin(plugin)

    lines = ['# fx_player: %s, %d shots' % (place.id, len(shots)),
             ' '.join(['travel %d %s %s' % (place.return_point, place.block_hex(), place.id)] + ['%d' % f for f in place.flags]),
             'god']
    if a.stand:
        x, y, z, deg = (float(v) for v in a.stand.split(','))
        lines.append('warp %s:stand %.3f %.3f %.3f %.4f 1.0' % (place.id, x, y, z, __import__('math').radians(deg)))
    elif place.view():
        lines.append('warp %s:view %.3f %.3f %.3f %.4f 1.0' % ((place.id,) + tuple(place.view())))
    if not a.hold:
        # about 4 game seconds a shot and a minute to settle, each game second a.fps frames
        a.hold = max(600.0, (4.0 * len(shots) + 60.0) * a.fps / a.slow_fps)
    lines += ['quiet 40', 'camera', 'mark fxp:start', 'hold %.0f' % a.hold, 'mark fxp:end', 'wait 2']
    tour = '\n'.join(lines) + '\n'
    area_check.tour = lambda areas, opts, kind: tour
    area_check.pass_budget = lambda areas, opts, kind: a.hold + 900
    orig_toml, orig_prepare = area_check.run_toml, area_check.prepare_pass

    def run_toml(*args, **kw):
        return orig_toml(*args, **kw).replace('[plugins]\n', '[plugins]\nfx_player = true\n', 1)

    def prepare_pass(pass_dir, kind, areas, opts, paths):
        p = orig_prepare(pass_dir, kind, areas, opts, paths)
        shutil.copy2(plugin, pass_dir / 'data' / 'plugins' / 'fx_player.so')
        if a.dumps:
            (pass_dir / 'fxp-frames').mkdir(exist_ok=True)
        return p

    area_check.run_toml, area_check.prepare_pass = run_toml, prepare_pass
    perf = out / 'ac' / 'perf'  # area_check's pass folder
    env = {'FXP_PLAN': str(plan_file.resolve()), 'FXP_FPS': str(a.fps), 'FXP_BLOCK': place.block_hex(), 'FXP_EXIT': '1',
           'FXP_WORDS': str((out / 'words.tsv').resolve()), 'FXP_SETTLE': os.environ.get('FXP_SETTLE', '240')}
    if a.dumps:
        env.update({'FXP_DUMPS': str((perf / 'fxp-frames').resolve()), 'FXP_DUMP_S': a.dumps})
    os.environ.update(env)
    argv = ['run', '--bbhost', a.bbhost, '--config', a.config, '--seed', a.seed, '--out', str(out / 'ac'), '--areas', place.id,
            '--captures', '0', '--p2p-port', str(a.p2p_port), '--numa-node', a.numa_node, '--fps', str(a.fps), '--dumps', '0',
            '--width', str(a.width), '--height', str(a.height), '--start', '8']
    extra = list(a.env)
    if a.smooth is not None:
        extra.append('BBHOST_SFX_SMOOTH=' + a.smooth)
    if a.sim_frame_time:
        extra.append('BBHOST_SIM_FRAME_TIME=1')
    for e in extra:
        argv += ['--env', e]
    if a.dry_run:
        argv.append('--dry-run')
    print('fx_player: %s, %d shots at %d fps%s%s' % (place.id, len(shots), a.fps, ', smooth %s' % a.smooth if a.smooth else '',
                                                     ', game time a target a frame' if a.sim_frame_time else ''), flush=True)
    rc = area_check.main(argv)
    area_check.run_toml, area_check.prepare_pass = orig_toml, orig_prepare
    log = out / 'ac' / 'perf' / 'run.log'
    text = log.read_text(errors='replace') if log.exists() else ''
    base = None
    if a.base:
        bl = Path(a.base) / 'ac' / 'perf' / 'run.log'
        base = (bl.read_text(errors='replace'), fxreport.read_words(Path(a.base) / 'words.tsv'))
    conv = None
    if a.converted:
        conv = fxreport.CONVERTED_BY_COUNT.get(int(a.converted)) if a.converted.isdigit() else {int(x, 16) for x in a.converted.split(',')}
    rep = fxreport.report(text, fxreport.read_words(out / 'words.tsv'), fams, base=base, name='%s %d fps' % (place.id, a.fps), converted=conv)
    (out / 'report.txt').write_text(rep)
    fault = next((f for f in FAULT if f in text), None)
    done = 'fxp: f' in text and ' done: ' in text
    print('fx_player: area_check exit %d; %s%s; report %s' % (rc, 'plan done' if done else 'PLAN NOT DONE', ', FAULT: ' + fault if fault else '',
                                                              out / 'report.txt'), flush=True)
    return 0 if done and not fault else 1


if __name__ == '__main__':
    sys.exit(main())
