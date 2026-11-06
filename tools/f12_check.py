#!/usr/bin/env python3
"""Check an F12 dump for the rendering bug patterns found so far.

    tools/f12_check.py <capture folder | f12-<flip>-draws.txt> [run.log]

F12 writes each capture into a folder of its own (logs/f12-<date>-<time>/ in
a package, build/f12-<date>-<time>/ in a checkout; the log names it): the
frame and the render targets as PNG, the draw list as f12-<flip>-draws.txt.

Each check is a pattern that has broken frames before:

  size       a T# whose render target is another size. Smaller T#: the
             target's top-left region is what the hardware reads (the host
             copies it, BBHOST_RT_REGIONS). Larger T#: the target is too small
             and what the game drew past its edge was clipped - the
             new-character magenta (588x221 drawn into 588x11), the 2560x1440
             glare drawn to the right of its source (640x360 levels made
             523x360 wide). Not counted when the memory is a live GX texture
             of the T#'s size now, which the host samples instead: the
             capture's "registered:" line says so (a YEBIS texture's memory
             never is); a capture without those lines is judged by a surface
             of the T#'s size there.
  lost-clear a target cleared, then re-created by the next draw: the clear
             went to the old image (the metallic huntsman's blood layer).
             Since 2026-09-29 the new image starts with that clear
             (BBHOST_RT_REFILL); those are counted apart, not listed.
  stale      a texture uploaded from memory a drawn target or a GPU-written
             texture holds newer bits for (the host's stale-read check).
  layers     RGBA8 textures filled by copy tokens (the game's runtime blood
             layers): which mips a copy reached, which were refused.
  floats     with the run's log: the float targets' largest value and their
             infinite and NaN texel counts, as F12 wrote them.
  cp-stale   a texture the command processor or the GPU copied into after its
             last upload (copy tokens "as memory"): the new bytes never reached
             the image (the opening movie's green chroma plane).
"""
import glob
import os
import re
import sys


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    if os.path.isdir(path):
        lists = sorted(glob.glob(os.path.join(path, 'f12-*-draws.txt')))
        if not lists:
            sys.exit('%s: no f12-<flip>-draws.txt in it' % path)
        for n, draws in enumerate(lists):
            if len(lists) > 1:
                print('%s== %s' % ('\n' if n else '', draws))
            check(draws)
        return
    check(path)


def check(path):
    L = open(path, errors='replace').read().split('\n')
    start = next((i for i, l in enumerate(L) if re.match(r'^\d+ textures', l)), len(L))
    events = [l for l in L if re.match(r'^flip \d+: ', l)]

    # Texture records: the T#, the render target or surface lines, the history.
    texs, cur = [], None
    for l in L[start + 1:]:
        m = re.match(r'^tex (0x[0-9a-f]+): T# dfmt (\d+) nfmt (\d+) (\d+)x(\d+).* levels (\d+)\.\.(\d+)', l)
        if m:
            cur = dict(addr=m.group(1), dfmt=int(m.group(2)), nfmt=int(m.group(3)), w=int(m.group(4)), h=int(m.group(5)),
                       last_level=int(m.group(7)), rt=None, hist=[])
            texs.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r'\s+render target vkformat (\d+) (\d+)x(\d+)', l)
        if m:
            cur['rt'] = (int(m.group(2)), int(m.group(3)), int(m.group(1)))
            continue
        m = re.match(r'\s+surface vkformat \d+ (\d+)x(\d+)', l)
        if m:
            cur['surface'] = (int(m.group(1)), int(m.group(2)))
        m = re.match(r'\s+registered: (YEBIS|GX) texture (\d+)x(\d+)', l)
        if m:
            cur['registered'] = (m.group(1), int(m.group(2)), int(m.group(3)))
        if l.startswith('      '):
            cur['hist'].append(l.strip())
        else:
            m = re.search(r'(\d+) upload\(s\), last checked (\d+) ms ago', l)
            if m:
                cur['uploads'], cur['checked_ms'] = int(m.group(1)), int(m.group(2))

    print('%s: %d textures, %d events (flips %s..%s)' % (path, len(texs), len(events),
          events[0].split(':')[0][5:] if events else '-', events[-1].split(':')[0][5:] if events else '-'))

    # size
    smaller, larger, reused = [], [], 0
    for t in texs:
        if not t['rt']:
            continue
        rw, rh, _ = t['rt']
        if (rw, rh) == (t['w'], t['h']):
            continue
        reg = t.get('registered')
        live_texture = reg == ('GX', t['w'], t['h']) if 'registered' in t else t.get('surface') == (t['w'], t['h'])
        if (t['w'] > rw or t['h'] > rh) and live_texture:
            # Memory an old target was drawn into, now a texture of the T#'s
            # size: the host samples the texture (the exit report's "read as
            # the live texture" binds), so nothing is clipped.
            reused += 1
            continue
        (larger if t['w'] > rw or t['h'] > rh else smaller).append('%s T# %dx%d rt %dx%d' % (t['addr'], t['w'], t['h'], rw, rh))
    print('\nsize: %d T#s smaller than their target (region read), %d LARGER (target too small: drawing clipped), '
          '%d over an old target now a texture of their size' % (len(smaller), len(larger), reused))
    for s in larger[:12]:
        print('  LARGER  ' + s)
    for s in smaller[:12]:
        print('  smaller ' + s)

    # lost-clear
    fills, lost, refilled = {}, {}, 0
    for l in events:
        m = re.match(r'^flip (\d+): fill token (0x[0-9a-f]+) (0x[0-9a-f]+) bytes with (.*?)( \(render target\))?$', l)
        if m:
            fills[m.group(2)] = (int(m.group(1)), m.group(4), bool(m.group(5)))
            continue
        m = re.match(r'^flip (\d+): render target (0x[0-9a-f]+) (\d+x\d+) x\d+ format (\d+) created(.*)$', l)
        if m and m.group(2) in fills:
            f = fills[m.group(2)]
            if int(m.group(1)) - f[0] <= 2 and f[2] and 'carried' not in m.group(5):
                lost[m.group(2)] = '%s %s format %s at flip %s, cleared at %d with %s' % (m.group(2), m.group(3), m.group(4), m.group(1), f[0], f[1])
            continue
        # Since 2026-09-29 the new image starts with the fill (rt_image): not lost.
        m = re.match(r'^flip \d+: render target (0x[0-9a-f]+) .* starts with the fill its old image took', l)
        if m and lost.pop(m.group(1), None):
            refilled += 1
    print('\nlost-clear: %d targets re-created right after a clear found their old image (%d more started with that clear)'
          % (len(lost), refilled))
    for s in list(lost.values())[:12]:
        print('  ' + s)

    # stale
    stale = [l for l in events if 'STALE?' in l]
    print('\nstale: %d uploads from memory the GPU holds newer bits for' % len(stale))
    for s in stale[:12]:
        print('  ' + s[:190])

    # layers
    layers = [t for t in texs if t['dfmt'] == 10 and any('copy token into' in h for h in t['hist'])]
    print('\nlayers: %d RGBA8 textures filled by copy tokens' % len(layers))
    for t in layers[:16]:
        mips = sorted(set(int(m.group(1)) for h in t['hist'] for m in [re.search(r'mip (\d+) \(', h)] if m))
        refused = sum(1 for h in t['hist'] if 'REFUSED' in h)
        want = list(range(t['last_level'] + 1))
        note = '' if not mips or mips == want else '  (levels 0..%d, copied %s)' % (t['last_level'], mips)
        print('  %s %dx%d nfmt %d: mips copied %s, refused %d%s' % (t['addr'], t['w'], t['h'], t['nfmt'], mips or '-', refused, note))

    # cp-stale
    cp = []
    for t in texs:
        last_up = max((int(m.group(1)) for h in t['hist'] for m in [re.match(r'flip (\d+): upload ', h)] if m), default=-1)
        writes = [int(m.group(1)) for h in t['hist'] for m in [re.match(r'flip (\d+): copy (token|dispatch) .* as memory', h)] if m]
        after = [f for f in writes if f > last_up]
        if after and not t['rt']:
            cp.append('%s %dx%d: %d copies after its last upload (flip %d), %s uploads, last checked %s ms ago' % (
                t['addr'], t['w'], t['h'], len(after), last_up, t.get('uploads', '?'), t.get('checked_ms', '?')))
    print('\ncp-stale: %d textures copied into by the GPU after their last upload' % len(cp))
    for s_ in cp[:12]:
        print('  ' + s_)

    # floats
    if len(sys.argv) > 2:
        flip = re.search(r'f12-(\d+)-draws', os.path.basename(path))
        rows = []
        for l in open(sys.argv[2], errors='replace'):
            m = re.search(r'f12-%s-rt-([0-9a-f]+)\.(?:ppm|png) float channels: max (\S+) (\S+) (\S+), infinite (\d+) (\d+) (\d+), NaN (\d+) (\d+) (\d+)'
                          % (flip.group(1) if flip else r'\d+'), l)
            if m:
                inf = sum(int(m.group(i)) for i in (5, 6, 7))
                nan = sum(int(m.group(i)) for i in (8, 9, 10))
                rows.append((nan, inf, '0x' + m.group(1), m.group(2), m.group(3), m.group(4)))
        bad = [r for r in rows if r[0] or r[1]]
        print('\nfloats: %d float targets, %d with infinite or NaN texels' % (len(rows), len(bad)))
        for nan, inf, a, r, g, b in sorted(bad, reverse=True)[:12]:
            print('  %s NaN %d infinite %d (max %s %s %s)' % (a, nan, inf, r, g, b))


if __name__ == '__main__':
    main()
