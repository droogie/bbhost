#!/usr/bin/env python3
"""Drive the game's menus with the real mouse, headless, the way a player would.

It reads what the game says rather than assuming: with BBHOST_POINTER_LOG=1
every list that gains focus logs its shape and each item's on-screen
rectangle (engine/menu_pointer.cpp), and every step here acts on those - so a
click lands on an item because the game said the item is there.

    tools/menu_drive.sh title click:3 wait hover shot:system

Steps (run in order):
    title          popup (if any) -> Play Offline -> the main menu, each screen
                   recognised by its list's shape. The unclean-exit popup only
                   appears after a run that did not exit through the menu, and
                   Play Online is never clicked - it logs into the network.
                   It ends *on* the main menu, so act on that with clicknow:N
                   (click:N would wait for the list after it).
    wait           wait for the next list to gain focus
    click:N        wait for the next list, click its item N
    clicknow:N     click item N of the newest list
    hover          hover every item of the lists that last appeared, one
                   screenshot each (hover-<n>-<list>-<item>.png)
    item:K,I       move to item I of the K-th list that last appeared (-1 the last)
    at:X,Y         click at a stage point      move:X,Y   move there
    dclick:X,Y     double click there (two presses 120 ms apart)
    hold:X,Y,S     press there for S seconds   rclick     right click
    wheel:up|down  one wheel notch             key:NAME   an X key (focused)
    keyhold:NAME,S hold an X key for S seconds
    type:TEXT      type TEXT into the focused window (a text field)
    keys:A+B+C,S   hold several X keys together for S seconds (run: w+space)
    lclick:N       N left clicks 0.7 s apart (in the world: light attacks)
    pan:DX,S       move the mouse DX pixels every 20 ms for S seconds (turns
                   the camera in the world)
    padplug / padunplug   a virtual controller arrives / leaves (uinput; run
                   the game with BBHOST_GAMEPAD_MATCH=045e:02ea)
    pad:BUTTON[,S] tap a pad button (a b x y lb rb back start ls rs)
    down:A+B       press X keys and keep them held (until up:)
    up:A+B         release X keys held by down:
    padhold:BUTTON / padrelease:BUTTON   hold a pad button / let it go
    padaxis:AXIS,VALUE   set a pad axis and leave it there (0 centres it)
    padsweep:X0:Y0:X1:Y1,S   turn the left stick from (X0,Y0) to (X1,Y1)
                   along the arc between their angles over S seconds, as a
                   thumb does (+-32767; a step every 8 ms), and leave it there
    padstick:AXIS,VALUE,S  hold an axis (lx ly rx ry: +-32767, lt rt: 0-255,
                   dx dy: +-1 for the d-pad) for S seconds
    req:VERB[,ARG[,ARG2]]  a test request (BBHOST_TEST_REQUESTS): req:keys,FILE
                   types FILE's key script into the game (host/window.h)
    shot:NAME      screenshot NAME.png         sleep:S
    stamp:LABEL    print 'stamp LABEL <CLOCK_MONOTONIC s>' - the host's clock
                   (ajm t=, host_log ms): marks a stretch of play in its log
    burst:NAME,N   N screenshots back to back, NAME-<ms since the burst began>.png
                   (each takes about two seconds: for timing use record)
    record:NAME,S  S seconds of the screen at 20 fps, in the background, as
                   NAME-<frame>.jpg (frame n is n/20 s after it starts)
    title:online   the same, clicking Play Online - only against a server on
                   this machine: MENU_DRIVE_ONLINE_CONFIG must name the run's
                   bbhost.toml and its [online] host (and auth_server, if
                   set) must be 127.0.0.1 / localhost, or the step refuses
                   and the script stops (tools/map_validation.sh)
    waitlog:REGEX  wait until the run log matches
    waitn:N,REGEX  wait until it matches at least N times (a second world load)
    waitnew:REGEX  wait for a match that appears after this step began
    waitnewt:SECS,REGEX  the same with its own cap
    wheelto:N      scroll the focused list to row N, verifying each notch from
                   the pointer log (frame-rate independent)
    expect:REGEX   the same, but stop the whole script if it does not within
                   30 s - gate every key sequence on the screen it is meant
                   for (a focused list's shape, a section's "opened" line), or
                   keys meant for one screen land on another

Positions are stage pixels (1920x1080), mapped to the screen through the game
window's geometry and the picture's letterboxed rectangle in it, so any window
size or display works (DISPLAY_DRIVE picks the display).
"""
import os, re, subprocess, sys, time
out = sys.argv[1]
log = os.path.join(out, 'run.log')
env = dict(os.environ, DISPLAY=os.environ.get('DISPLAY', ':77'))

def blocks():
    cur = None; res = []
    for l in open(log, errors='replace').read().splitlines():
        if 'pointer: list' in l and 'count' in l and 'no focus' not in l:
            cur = {'head': l.split('pointer: ')[1], 'items': []}; res.append(cur)
        m = re.search(r'pointer:   item (\d+) .* at (-?\d+),(-?\d+)-(-?\d+),(-?\d+)', l)
        if m and cur is not None:
            i, x1, y1, x2, y2 = map(int, m.groups())
            cur['items'].append((i, (x1 + x2) // 2, (y1 + y2) // 2))
    return res

def x(*a): subprocess.run(['xdotool', *a], env=env, stderr=subprocess.DEVNULL)
def shot(name): subprocess.run(['import', '-display', env['DISPLAY'], '-window', 'root', os.path.join(out, name + '.png')], env=env, stderr=subprocess.DEVNULL)
def picture():
    """The picture's size: the last render size the log names (1920x1080 before any)."""
    w, h = 1920, 1080
    for l in open(log, errors='replace'):
        m = re.search(r'resolution: (?:display buffers \d+x\d+, rendering|\d+x\d+ ->) (\d+)x(\d+)', l)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
    return w, h
def to_screen(px, py):
    """A stage point (1920x1080) on the screen, through the game window's
    real geometry and the picture's letterboxed rectangle inside it (the
    presenter's fit_picture) - so a window of any size, tiled by a window
    manager or fullscreen on another resolution, is driven at the same items.
    A picture wider than 16:9 has the stage in its middle (Scaleform's
    show-all), so the stage is fitted into the picture's rectangle too."""
    wid = os.environ.get('WID')
    if not wid:
        return px, py
    r = subprocess.run(['xdotool', 'getwindowgeometry', '--shell', wid], env=env, capture_output=True, text=True)
    g = dict(l.split('=', 1) for l in r.stdout.split() if '=' in l)
    try:
        wx, wy, ww, wh = (int(g[k]) for k in ('X', 'Y', 'WIDTH', 'HEIGHT'))
    except (KeyError, ValueError):
        return px, py
    pw, ph = picture()
    a = pw / ph
    fw, fh = (ww, ww / a) if ww / wh < a else (wh * a, wh)  # the picture, centred
    s = min(fw / 1920, fh / 1080)  # the stage in it, centred
    return round(wx + (ww - 1920 * s) / 2 + px * s), round(wy + (wh - 1080 * s) / 2 + py * s)
def move(px, py):
    ax, ay = to_screen(px - 12, py - 5); bx, by = to_screen(px, py)
    x('mousemove', str(ax), str(ay)); time.sleep(0.25); x('mousemove', str(bx), str(by))

# A virtual controller (uinput, an Xbox One S pad's ids 045e:02ea), for the
# pad steps: run the game with BBHOST_GAMEPAD_MATCH=045e:02ea so it opens this
# one and not a controller on the desk. Plugged in on the first pad step.
vpad = None
PAD_BUTTONS = {'a': 'BTN_A', 'b': 'BTN_B', 'x': 'BTN_X', 'y': 'BTN_Y', 'lb': 'BTN_TL', 'rb': 'BTN_TR',
               'back': 'BTN_SELECT', 'start': 'BTN_START', 'ls': 'BTN_THUMBL', 'rs': 'BTN_THUMBR'}
PAD_AXES = {'lx': 'ABS_X', 'ly': 'ABS_Y', 'rx': 'ABS_RX', 'ry': 'ABS_RY', 'lt': 'ABS_Z', 'rt': 'ABS_RZ',
            'dx': 'ABS_HAT0X', 'dy': 'ABS_HAT0Y'}
def pad_plug():
    global vpad
    if vpad: return
    import evdev
    from evdev import UInput, AbsInfo, ecodes as e
    stick = AbsInfo(0, -32768, 32767, 16, 128, 0)
    cap = {e.EV_KEY: [getattr(e, b) for b in PAD_BUTTONS.values()] + [e.BTN_MODE],
           e.EV_ABS: [(e.ABS_X, stick), (e.ABS_Y, stick), (e.ABS_RX, stick), (e.ABS_RY, stick),
                      (e.ABS_Z, AbsInfo(0, 0, 255, 0, 0, 0)), (e.ABS_RZ, AbsInfo(0, 0, 255, 0, 0, 0)),
                      (e.ABS_HAT0X, AbsInfo(0, -1, 1, 0, 0, 0)), (e.ABS_HAT0Y, AbsInfo(0, -1, 1, 0, 0, 0))]}
    vpad = UInput(cap, name='Microsoft X-Box One S pad', vendor=0x045e, product=0x02ea, version=0x110, bustype=e.BUS_USB)
    time.sleep(1.5)  # for the game to see it arrive
def pad_unplug():
    global vpad
    if vpad: vpad.close(); vpad = None
    time.sleep(1.0)
def pad_button(name, secs):
    from evdev import ecodes as e
    pad_plug()
    code = getattr(e, PAD_BUTTONS[name])
    vpad.write(e.EV_KEY, code, 1); vpad.syn(); time.sleep(secs)
    vpad.write(e.EV_KEY, code, 0); vpad.syn()
def pad_axis(axis, value):
    from evdev import ecodes as e
    pad_plug()
    vpad.write(e.EV_ABS, getattr(e, PAD_AXES[axis]), value); vpad.syn()
def pad_hold(name, down):
    from evdev import ecodes as e
    pad_plug()
    vpad.write(e.EV_KEY, getattr(e, PAD_BUTTONS[name]), 1 if down else 0); vpad.syn()
def pad_sweep(x0, y0, x1, y1, secs):
    import math
    from evdev import ecodes as e
    pad_plug()
    a0, a1 = math.atan2(y0, x0), math.atan2(y1, x1)
    r0, r1 = math.hypot(x0, y0), math.hypot(x1, y1)
    d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi  # the short way round
    n = max(1, int(secs / 0.008))
    for i in range(1, n + 1):
        t = i / n
        a, r = a0 + d * t, r0 + (r1 - r0) * t
        x = max(-32767, min(32767, int(round(r * math.cos(a)))))
        y = max(-32767, min(32767, int(round(r * math.sin(a)))))
        vpad.write(e.EV_ABS, e.ABS_X, x); vpad.write(e.EV_ABS, e.ABS_Y, y); vpad.syn()
        time.sleep(secs / n)
def pad_stick(axis, value, secs):
    from evdev import ecodes as e
    pad_plug()
    code = getattr(e, PAD_AXES[axis])
    vpad.write(e.EV_ABS, code, value); vpad.syn(); time.sleep(secs)
    vpad.write(e.EV_ABS, code, 0); vpad.syn()

seen = 0
batch = []
def wait_new(timeout=90):
    global seen, batch
    t = time.time()
    while time.time() - t < timeout:
        b = blocks()
        if len(b) > seen:
            time.sleep(2.5)
            b = blocks(); batch = b[seen:]; seen = len(b)
            for bl in batch: print('list:', bl['head'], bl['items'], flush=True)
            return b[-1]
        time.sleep(0.5)
    print('no new list', flush=True); return None

def screen_of(b):
    """Which title screen a list is, from its shape - never from its order.
    The unclean-exit popup only appears after a run that did not exit through
    the menu, so the first list may be it or may already be Play Online /
    Play Offline."""
    h = b['head']
    ys = [py for _, _, py in b['items']]
    if 'count 1 cells 2 lines 1' in h:
        return 'popup'
    # The title movie has six lines since bbhost added Quit Game (five before),
    # and the main list five or six rows (Log In and Quit Game come and go).
    if re.search(r'count 2 cells 1 lines [56]\b', h) and len(ys) == 2:
        return 'online'
    if re.search(r'count [56] cells 1 lines [56]\b', h) and len(ys) >= 5 and ys[0] == ys[1] - 40:
        return 'main'
    return None


def online_allowed():
    """Play Online logs into whatever server the run's config names. The
    drivers never do that against a real one: only when the config the run
    uses (MENU_DRIVE_ONLINE_CONFIG) points every online endpoint at this
    machine."""
    path = os.environ.get('MENU_DRIVE_ONLINE_CONFIG', '')
    if not path or not os.path.isfile(path):
        print('title:online refused: MENU_DRIVE_ONLINE_CONFIG does not name the run config', flush=True)
        return False
    section, online = None, {}
    for raw in open(path, errors='replace'):
        line = raw.split('#', 1)[0].strip()
        m = re.match(r'^\[(.+)\]$', line)
        if m:
            section = m.group(1).strip(); continue
        m = re.match(r'^([A-Za-z_]+)\s*=\s*"?([^"]*)"?$', line)
        if m and section == 'online':
            online[m.group(1)] = m.group(2).strip()
    local = ('127.0.0.1', 'localhost', '::1')
    def is_local(v):
        v = re.sub(r'^[a-z]+://', '', v).split('/')[0]
        v = v.rsplit(':', 1)[0] if v.count(':') == 1 else v
        return v in local or v.startswith('127.')   # all of 127/8 is this machine
    bad = [k for k in ('host', 'auth_server', 'np_server') if k in online and not is_local(online[k])]
    if 'host' not in online or bad:
        print('title:online refused: %s has [online] %s not on this machine' % (path, ', '.join(bad) or 'host unset'), flush=True)
        return False
    return True


def to_main_menu(online=False):
    """Popup (if any) -> Play Offline -> the main menu, checking each screen.
    Play Online is clicked only for title:online (it logs into the network,
    so only against a server meant for the run)."""
    global seen
    for _ in range(4):
        b = wait_new()
        if not b:
            return False
        kind = screen_of(b)
        items = dict((i, (px, py)) for i, px, py in b['items'])
        print('screen:', kind, flush=True)
        if kind == 'popup':
            # At 60 fps the first click on the popup is sometimes not taken:
            # click again until the next screen's list appears.
            for _ in range(4):
                move(*items[0]); time.sleep(0.8); x('click', '1')
                t = time.time()
                while time.time() - t < 12 and len(blocks()) <= seen:
                    time.sleep(0.5)
                if len(blocks()) > seen:
                    break
                print('popup still up; clicking again', flush=True)
        elif kind == 'online':
            i = 0 if online else 1   # Play Online is index 0, Play Offline 1
            move(*items[i]); time.sleep(0.8); x('click', '1')
            print('clicked', 'Play Online' if online else 'Play Offline', flush=True)
            if online:
                # Going online shows the server's notices as message boxes,
                # which are not lists: confirm them until the main menu's
                # list appears.
                for k in range(6):
                    if wait_new(timeout=20):
                        seen -= 1   # let the loop's own wait_new take this list
                        break
                    shot('online-notice-%d' % k)
                    # The box's OK is a glyph, not a list: click where it
                    # sits, then the confirm key as well.
                    # One way at a time, each followed by a look for the
                    # next list: a confirm key that lands after the box has
                    # gone would act on the main menu (Return is Continue).
                    def gone():
                        time.sleep(1.5)
                        return len(blocks()) > seen
                    move(845, 735); time.sleep(0.6); x('click', '1')
                    if not gone():
                        # The Interact key is Cross everywhere; Return is
                        # Cross while a game menu is open. Held for a few
                        # frames: an instant `xdotool key` can fall between
                        # two polls of a slow frame.
                        if os.environ.get('WID'): x('windowactivate', '--sync', os.environ['WID'])
                        for k in ('e', 'Return'):
                            x('keydown', k); time.sleep(0.2); x('keyup', k)
                            if gone():
                                break
                    print('confirmed a notice', flush=True)
        elif kind == 'main':
            return True
        else:
            print('unknown screen, stopping:', b['head'], flush=True)
            return False
    return False


recorders = []
expect_from = {}
# The log size before each step: waitnew searches from the mark taken before
# the *previous* step, so a list that opens within the milliseconds between a
# key and the wait is not missed.
mark_prev = mark_now = 0
for step in sys.argv[2:]:
    kind, _, arg = step.partition(':')
    mark_prev = mark_now
    try: mark_now = os.path.getsize(log)
    except OSError: mark_now = 0
    if kind == 'title':
        if arg == 'online' and not online_allowed():
            print('did not reach the main menu; aborting', flush=True)
            break
        if not to_main_menu(online=(arg == 'online')):
            print('did not reach the main menu; aborting', flush=True)
            break
    elif kind == 'wait':
        wait_new()
    elif kind in ('click', 'clicknow'):
        b = wait_new() if kind == 'click' else blocks()[-1]
        if not b: continue
        items = dict((i, (px, py)) for i, px, py in b['items'])
        i = int(arg)
        if i not in items: print('no item', i, flush=True); continue
        move(*items[i]); time.sleep(0.8); shot('before-click-%d' % seen)
        x('click', '1'); print('clicked', i, items[i], flush=True)
    elif kind == 'item':
        # item:K,I - move to item I of the K-th list that appeared in the last wait (K -1: the last one)
        k, i = (int(v) for v in arg.split(','))
        lists = batch or blocks()[-1:]
        items = dict((n, (px, py)) for n, px, py in lists[k]['items']) if lists else {}
        if i in items:
            move(*items[i]); print('moved to item', i, 'of list', k, items[i], flush=True)
        else:
            print('no item', i, 'in list', k, flush=True)
    elif kind == 'hover':
        for k, b in enumerate(batch or blocks()[-1:]):
            for i, px, py in b['items']:
                move(px, py); time.sleep(1.0); shot('hover-%d-%d-%d' % (seen, k, i))
    elif kind == 'at':
        px, py = map(int, arg.split(','))
        move(px, py); time.sleep(0.8); x('click', '1'); print('clicked at', px, py, flush=True)
    elif kind == 'dclick':
        px, py = map(int, arg.split(','))
        move(px, py); time.sleep(0.8)
        x('click', '--repeat', '2', '--delay', '120', '1'); print('double clicked at', px, py, flush=True)
    elif kind == 'hold':
        px, py, secs = arg.split(',')
        move(int(px), int(py)); time.sleep(0.8)
        x('mousedown', '1'); time.sleep(float(secs)); x('mouseup', '1'); print('held at', px, py, secs, flush=True)
    elif kind == 'move':
        px, py = map(int, arg.split(','))
        move(px, py); print('moved to', px, py, flush=True)
    elif kind == 'wheel':
        x('click', '4' if arg == 'up' else '5'); print('wheel', arg, flush=True)
    elif kind == 'rclick':
        x('click', '3'); print('right click', flush=True)
    elif kind == 'key':
        if os.environ.get('WID'): x('windowactivate', '--sync', os.environ['WID'])
        x('keydown', arg); time.sleep(0.15); x('keyup', arg); print('key', arg, flush=True)
    elif kind == 'type':
        x('type', '--delay', '40', arg)
    elif kind == 'pan':
        # pan:dx,secs or pan:dx:dy,secs - a relative move every 20 ms
        d, secs = arg.split(',')
        dx, _, dy = d.partition(':')
        t0 = time.time()
        n = 0
        while time.time() - t0 < float(secs):
            x('mousemove_relative', '--', dx, dy or '0'); time.sleep(0.02)
            n += 1
        print('panned', dx, dy or '0', 'x', n, secs, flush=True)
    elif kind == 'keyhold':
        name, secs = arg.split(',')
        if os.environ.get('WID'): x('windowactivate', '--sync', os.environ['WID'])
        x('keydown', name); time.sleep(float(secs)); x('keyup', name); print('held key', name, secs, flush=True)
    elif kind == 'keys':
        names, secs = arg.rsplit(',', 1)
        if os.environ.get('WID'): x('windowactivate', '--sync', os.environ['WID'])
        for name in names.split('+'): x('keydown', name)
        time.sleep(float(secs))
        for name in reversed(names.split('+')): x('keyup', name)
        print('held keys', names, secs, flush=True)
    elif kind == 'lclick':
        for _ in range(int(arg)):
            x('click', '1'); time.sleep(0.7)
        print('left clicks', arg, flush=True)
    elif kind == 'padplug':
        pad_plug(); print('pad plugged in', flush=True)
    elif kind == 'padunplug':
        pad_unplug(); print('pad unplugged', flush=True)
    elif kind == 'pad':
        name, _, secs = arg.partition(',')
        pad_button(name, float(secs or 0.15)); print('pad', name, flush=True)
    elif kind == 'down':
        if os.environ.get('WID'): x('windowactivate', '--sync', os.environ['WID'])
        for name in arg.split('+'): x('keydown', name)
        print('down', arg, flush=True)
    elif kind == 'up':
        for name in reversed(arg.split('+')): x('keyup', name)
        print('up', arg, flush=True)
    elif kind == 'padhold':
        pad_hold(arg, True); print('pad hold', arg, flush=True)
    elif kind == 'padrelease':
        pad_hold(arg, False); print('pad release', arg, flush=True)
    elif kind == 'padaxis':
        axis, value = arg.split(',')
        pad_axis(axis, int(value)); print('pad axis', axis, value, flush=True)
    elif kind == 'padsweep':
        pts, secs = arg.rsplit(',', 1)
        x0, y0, x1, y1 = (int(v) for v in pts.split(':'))
        pad_sweep(x0, y0, x1, y1, float(secs)); print('pad sweep', pts, secs, flush=True)
    elif kind == 'padstick':
        axis, value, secs = arg.split(',')
        pad_stick(axis, int(value), float(secs)); print('pad stick', axis, value, secs, flush=True)
    elif kind == 'shot':
        shot(arg)
    elif kind == 'record':
        name, secs = arg.split(',')
        recorders.append(subprocess.Popen(
            ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'x11grab', '-framerate', '20', '-video_size', '1920x1080',
             '-i', env['DISPLAY'], '-t', secs, '-q:v', '3', os.path.join(out, name + '-%03d.jpg')], env=env))
        time.sleep(0.5)  # started before the next step acts
        print('recording', name, secs, 's', flush=True)
    elif kind == 'burst':
        name, n = arg.split(',')
        t0 = time.time()
        for _ in range(int(n)):
            shot('%s-%04d' % (name, int((time.time() - t0) * 1000)))
    elif kind == 'expect':
        t = time.time()
        seen_from = expect_from.get(arg, 0)
        ok = False
        while time.time() - t < 30:
            text = open(log, errors='replace').read()
            m = re.compile(arg).search(text, seen_from)
            if m:
                expect_from[arg] = m.end()  # the same expect later needs a new line
                ok = True
                break
            time.sleep(0.3)
        print('expect', arg, 'ok' if ok else 'NOT SEEN - stopping', flush=True)
        if not ok:
            break
    elif kind == 'waitlog':
        t = time.time()
        while time.time() - t < 400:
            if re.search(arg, open(log, errors='replace').read()): break
            time.sleep(0.5)
        print('waitlog', arg, 'ok' if time.time() - t < 400 else 'timeout', flush=True)
    elif kind in ('waitnew', 'waitnewt'):
        # waitnew:REGEX - wait (up to 400 s) for a match that appears after
        # this step began: the list that is about to open, the next world
        # load, the next room - without counting earlier ones.
        # waitnewt:SECS,REGEX - the same with its own cap.
        cap = 400
        if kind == 'waitnewt':
            secs, _, arg = arg.partition(',')
            cap = float(secs)
        start = mark_prev
        t = time.time()
        hit = False
        while time.time() - t < cap:
            with open(log, errors='replace') as f:
                f.seek(start)
                if re.search(arg, f.read()): hit = True; break
            time.sleep(0.3)
        print('waitnew', arg, 'ok' if hit else 'timeout', flush=True)
    elif kind == 'wheelto':
        # wheelto:N - scroll the focused list to row N one notch at a time,
        # reading the cursor back from the pointer log ("pad cursor a -> b")
        # after each notch, so a slow frame that swallows or doubles a notch
        # is corrected instead of counted. Needs BBHOST_POINTER_LOG=1.
        target = int(arg)
        rx = re.compile(r'pad cursor (\d+) -> (\d+)')
        def cursor():
            m = rx.findall(open(log, errors='replace').read())
            return int(m[-1][1]) if m else 0
        cur = cursor()
        moves = 0
        stuck = 0
        t0 = time.time()
        while cur != target and time.time() - t0 < 600:
            x('click', '5' if cur < target else '4')
            moves += 1
            tw = time.time()
            nxt = cur
            while time.time() - tw < 0.8:
                nxt = cursor()
                if nxt != cur: break
                time.sleep(0.03)
            if nxt == cur:
                stuck += 1
                if stuck > 60: break
                continue
            stuck = 0
            cur = nxt
        print('wheelto', target, 'at', cur, 'after', moves, 'notches', 'ok' if cur == target else 'FAILED', flush=True)
    elif kind == 'waitn':
        # waitn:N,REGEX - wait (up to 400 s) until the run log matches REGEX at
        # least N times: the second world load, the second room, in one run.
        n, rx = arg.split(',', 1)
        t = time.time()
        while time.time() - t < 400:
            if len(re.findall(rx, open(log, errors='replace').read())) >= int(n): break
            time.sleep(0.5)
        print('waitn', n, rx, 'ok' if time.time() - t < 400 else 'timeout', flush=True)
    elif kind == 'req':
        # req:VERB[,ARG[,ARG2]] - a test request into BBHOST_TEST_REQUESTS
        # (hle/video.cpp), carried out at the game's next flip:
        # req:keys,/abs/script plays a key script (host/window.h).
        reqs = os.environ.get('BBHOST_TEST_REQUESTS', '')
        if not reqs:
            print('req', arg, 'refused: BBHOST_TEST_REQUESTS is not set', flush=True); continue
        req_n = globals().get('req_n', 0) + 1
        globals()['req_n'] = req_n
        tmp = os.path.join(reqs, 'drive-%04d.tmp' % req_n)
        with open(tmp, 'w') as f: f.write(' '.join(arg.split(',')) + '\n')
        os.rename(tmp, os.path.join(reqs, 'drive-%04d.req' % req_n))  # whole when the host lists it
        print('req', arg, flush=True)
    elif kind == 'sleep':
        time.sleep(float(arg))
    elif kind == 'stamp':
        print('stamp', arg, '%.3f' % time.monotonic(), flush=True)
for r in recorders:
    r.wait()
