#!/usr/bin/env python3
"""The player's combat effects, family by family, from the game's own data.

    tools/fx_player/plan.py --app0 DIR [--out PLAN] [--json FAMILIES.json] [--family NAME ...] [--repeat N]

The families and how fx_player.c spawns each through the game's paths:

  firearms     every gun's BehaviorParam_PC rows 100000000 + its
               behaviorVariationId x 1000 + 800 / 810 (the TAE's bullet
               events' judges: a shot and a charged one) with refType 1: the
               bullet and its HitBulletID chain, whose sfxId_Bullet are the
               bullets' effects and sfxId_Hit / sfxId_Flick the impacts.
               Shot down at the ground ahead, so the impacts play. One shot
               per distinct chain (Evelyn shares the Hunter Pistol's).
  muzzle       the player's TAE (chr/c0000.anibnd.dcx): type-96 effect events
               (effect id, dummy poly) whose id is a gun effect 6100-6899
               ending in 02-04 - the muzzle flashes, each at its event's dummy
               poly (a weapon's, 20215, falls back to the body's when the
               player holds no such weapon).
  goods        Molotovs and the hunter tools: EquipParamGoods refCategory 1
               (refId a bullet, its chain as above) or 2 (refId a SpEffect),
               and the cast effect the item's use animation makes - a00's
               animation 100000 + (goodsUseAnim + 12) x 10, its type-96
               events.
  buffs        the papers and the shell: EquipParamGoods refCategory 2, the
               SpEffect applied to the player and taken off again after
               --speffect-s game seconds; its stateInfo names the
               SpEffectVfxParam row whose midstSfxId is the effect.

The plan file fx_player.c reads (FXP_PLAN), one line a shot:
    shot <family> <kind> <id> <dummy poly> <pitch degrees> <repeat> <name> <hold>
kind: sfx (an effect id at a dummy poly), behavior (a BehaviorParam_PC row),
goods (an EquipParamGoods row), speffect (a SpEffectParam row; the dummy poly
field holds the game seconds it stays on). pitch: a bullet's launch below the
level (positive: down). hold: an sfx shot's TAE event length in seconds - the
effect is ended then, as the game ends a character's effect when its event
ends (0 for the other kinds). --json writes each family's expected effect ids (the
params' and the TAE's; the effects' own children come from the run).
"""
import argparse
import importlib.util
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import bbparam as P  # noqa: E402

# fx_enumerate's TAE reader (its plan.py, by path: this file has the same name)
_spec = importlib.util.spec_from_file_location('fx_enumerate_plan', os.path.join(HERE, '..', 'fx_enumerate', 'plan.py'))
tae_plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tae_plan)

# The goods of the families (EquipParamGoods ids; names from the game's item names).
GOODS = {
    'molotov': [1200, 1201, 1250, 1251],                    # Molotov Cocktail, Delayed, Rope, Delayed Rope
    'executioners-gloves': [2080],
    'a-call-beyond': [2010],
    'augur-of-ebrietas': [2000],
    'old-hunter-bone': [2060],
    'beast-roar': [2020],
    'tiny-tonitrus': [2070],
    'fire-paper': [1300],
    'bolt-paper': [1320],
    'empty-phantasm-shell': [1310],
}
BUFFS = ('fire-paper', 'bolt-paper', 'empty-phantasm-shell')
ORDER = ['muzzle', 'firearms', 'molotov', 'executioners-gloves', 'a-call-beyond', 'augur-of-ebrietas', 'old-hunter-bone', 'beast-roar',
         'tiny-tonitrus', 'fire-paper', 'bolt-paper', 'empty-phantasm-shell']
JUDGES = (800, 810)      # the TAE's gun bullet events' behaviour judges
PITCH_GUN = 20.0         # down at the ground ~4 m ahead from the launch height
PITCH_THROW = -20.0      # a throw: up
BODY_DUMMY = 1           # a body dummy poly every cast uses


def fmg(b):
    """{id: text} of an FMG (format 2: groups at 0x28, 16 bytes each; string offsets from the u64 at 0x18)."""
    groups = struct.unpack_from('<I', b, 0x0c)[0]
    soff = struct.unpack_from('<Q', b, 0x18)[0]
    out = {}
    for g in range(groups):
        idx, first, last, _ = struct.unpack_from('<IiiI', b, 0x28 + 16 * g)
        for k in range(last - first + 1):
            o = struct.unpack_from('<Q', b, soff + 8 * (idx + k))[0]
            if not o:
                continue
            e = o
            while b[e:e + 2] != b'\0\0':
                e += 2
            out[first + k] = b[o:e].decode('utf-16le', 'replace')
    return out


def names(app0, which):
    """The English names of a msg FMG ('アイテム名' items, '武器名' weapons)."""
    bnd = P.bnd4(P.dcx(os.path.join(app0, 'dvdroot_ps4', 'msg', 'engus', 'item.msgbnd.dcx')))
    for n, b in bnd.items():
        if n.startswith(which):
            return fmg(b)
    return {}


def chain(bullets, bid, depth=0, seen=None):
    """A bullet and its HitBulletID chain: [(bullet id, sfxId_Bullet, sfxId_Hit, sfxId_Flick)]."""
    seen = [] if seen is None else seen
    if bid <= 0 or bid not in bullets or depth > 8 or bid in [s[0] for s in seen]:
        return seen
    r = bullets[bid]
    seen.append((bid, r['sfxId_Bullet'], r['sfxId_Hit'], r['sfxId_Flick']))
    return chain(bullets, r['HitBulletID'], depth + 1, seen)


def effects_of(ch):
    """The bullets' effects and the impacts of a chain."""
    flight = sorted({s for (_b, s, _h, _f) in ch if s > 0})
    hits = sorted({x for (_b, _s, h, f) in ch for x in (h, f) if x > 0} - set(flight))
    return flight, hits


def tae_effects(app0):
    """[(anim id, event type, effect id, dummy poly, TAE file, start s, end s)] of the player's TAE effect events (type 96)."""
    _path, taes = tae_plan.tae_files(app0, 0)
    out = []
    for name, b in sorted(taes.items()):
        for (aid, t, st, en, ints) in tae_plan.events(b):
            if t == 96 and len(ints) >= 2 and ints[0] >= 1000:
                out.append((aid, t, ints[0], ints[1], os.path.basename(name), st, en))
    return out


def build(app0, repeat=1, speffect_s=3.0):
    goods = dict(P.load_table('EquipParamGoods', app0))
    beh = dict(P.load_table('BehaviorParam_PC', app0))
    bul = dict(P.load_table('Bullet', app0))
    wep = dict(P.load_table('EquipParamWeapon', app0))
    spe = dict(P.load_table('SpEffectParam', app0))
    vfx = dict(P.load_table('SpEffectVfxParam', app0))
    items, weapons = names(app0, 'アイテム名'), names(app0, '武器名')
    tae = tae_effects(app0)
    shots, fams = [], {}

    def fam(key):
        return fams.setdefault(key, {'name': key, 'expected': [], 'shots': 0})

    def expect(key, ids):
        f = fam(key)
        f['expected'] = sorted(set(f['expected']) | {i for i in ids if i > 0})

    # Muzzle flashes: the TAE's gun effects ending in 02-04, each once, at the first dummy poly it is made at.
    muzzle = {}
    for (aid, _t, eid, dmy, tname, st, en) in tae:
        if 6100 <= eid < 6900 and eid % 100 in (2, 3, 4) and eid not in muzzle:
            muzzle[eid] = (dmy, tname, aid, max(en - st, 0.0))
    for eid, (dmy, tname, aid, hold) in sorted(muzzle.items()):
        shots.append(('muzzle', 'sfx', eid, dmy, 0.0, repeat, '%s-a%d' % (tname[:-4], aid), hold))
        fam('muzzle')['shots'] += 1
    expect('muzzle', muzzle)

    # Firearms: one shot per distinct bullet chain of every gun's 800 / 810 behaviours.
    done = set()
    for wid, w in sorted(wep.items()):
        v = w.get('behaviorVariationId', 0)
        if wid % 10000 or v <= 0:
            continue
        for j in JUDGES:
            bid = 100000000 + v * 1000 + j
            r = beh.get(bid)
            if not r or r['refType'] != 1:
                continue
            ch = chain(bul, r['refId'])
            key = tuple(c[0] for c in ch)
            if not ch or key in done:
                continue
            done.add(key)
            flight, hits = effects_of(ch)
            if not flight and not hits:
                continue
            nm = weapons.get(wid, 'weapon %d' % wid).replace(' ', '_')
            shots.append(('firearms', 'behavior', bid, -1, PITCH_GUN, repeat, '%s-%d' % (nm, j), 0.0))
            fam('firearms')['shots'] += 1
            expect('firearms', flight + hits)

    # Goods: the bullet or the SpEffect, and the use animation's cast effects (type-96 events of a00's 100000 + (anim + 12) x 10).
    for key in ORDER:
        for gid in GOODS.get(key, ()):
            g = goods.get(gid)
            if not g:
                continue
            nm = items.get(gid, 'goods %d' % gid).replace(' ', '_')
            anim = 100000 + (g['goodsUseAnim'] + 12) * 10
            casts = [(eid, dmy, max(en - st, 0.0)) for (aid, _t, eid, dmy, tname, st, en) in tae if tname == 'a00.tae' and aid == anim]
            if key not in BUFFS and key != 'molotov':
                for eid, dmy, hold in casts:
                    shots.append((key, 'sfx', eid, dmy if 0 <= dmy < 10000 else BODY_DUMMY, 0.0, repeat, '%s-cast-a%d' % (nm, anim), hold))
                    fam(key)['shots'] += 1
                    expect(key, [eid])
            if g['refCategory'] == 1:
                ch = chain(bul, g['refId'])
                flight, hits = effects_of(ch)
                pitch = PITCH_THROW if key == 'molotov' else 0.0
                shots.append((key, 'goods', gid, -1, pitch, repeat, nm, 0.0))
                fam(key)['shots'] += 1
                expect(key, flight + hits)
            elif g['refCategory'] == 2:
                sid = g['refId']
                state = spe.get(sid, {}).get('stateInfo', 0)
                row = vfx.get(state, {})
                ids = [row.get(k, -1) for k in ('midstSfxId', 'initSfxId', 'finishSfxId')]
                shots.append((key, 'speffect', sid, speffect_s, 0.0, repeat, '%s-speffect%d-vfx%d' % (nm, sid, state), 0.0))
                fam(key)['shots'] += 1
                expect(key, ids)
    return shots, fams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--app0', default=os.environ.get('BBHOST_APP0', ''))
    ap.add_argument('--out', help='the plan file fx_player.c reads')
    ap.add_argument('--json', help="each family's expected effect ids and shots")
    ap.add_argument('--family', action='append', default=[], help='only these families (default all: %s)' % ', '.join(ORDER))
    ap.add_argument('--repeat', type=int, default=1, help='each shot this many times')
    ap.add_argument('--speffect-s', type=float, default=3.0, help='game seconds a buff stays on before it is taken off')
    a = ap.parse_args()
    if not a.app0:
        sys.exit('plan.py: --app0 (or BBHOST_APP0): the dump, the folder holding dvdroot_ps4')
    shots, fams = build(a.app0, a.repeat, a.speffect_s)
    if a.family:
        unknown = [f for f in a.family if f not in ORDER]
        if unknown:
            sys.exit('plan.py: unknown families %s (have %s)' % (', '.join(unknown), ', '.join(ORDER)))
        shots = [s for s in shots if s[0] in a.family]
        fams = {k: v for k, v in fams.items() if k in a.family}
    order = {k: i for i, k in enumerate(ORDER)}
    shots.sort(key=lambda s: order.get(s[0], 99))
    lines = ['# fx_player plan: %d shots' % len(shots)]
    for (f, kind, ident, dmy, pitch, rep, name, hold) in shots:
        lines.append('shot %s %s %d %g %.1f %d %s %.3f' % (f, kind, ident, dmy, pitch, rep, name, hold))
    for key in sorted(fams, key=lambda k: order.get(k, 99)):
        v = fams[key]
        print('%-22s %3d shots, %3d expected effect ids: %s' % (key, v['shots'], len(v['expected']), ' '.join(map(str, v['expected']))))
    if a.out:
        with open(a.out, 'w') as fo:
            fo.write('\n'.join(lines) + '\n')
    if a.json:
        with open(a.json, 'w') as fo:
            json.dump(fams, fo, indent=1, sort_keys=True)


if __name__ == '__main__':
    main()
