"""Each thread's CPU over a few seconds of a running bbhost, with the thread names it set: no
trace, no administrator - the quick look before a WPR trace (tools/etw/record.ps1).

    python tools/etw/thread_cpu.py [pid | process name, default bbhost.exe] [seconds, 10] [rows, 25]

After the KyoPS4x ETW tools (Kyo).
"""
import ctypes, ctypes.wintypes as wt, subprocess, sys, time

target = sys.argv[1] if len(sys.argv) > 1 else 'bbhost.exe'
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 10
rows_n = int(sys.argv[3]) if len(sys.argv) > 3 else 25
if target.isdigit():
    pid = int(target)
else:
    out = subprocess.run(['tasklist', '/fi', f'imagename eq {target}', '/fo', 'csv', '/nh'], capture_output=True,
                         text=True).stdout.strip().splitlines()
    if not out or not out[0].startswith('"'):
        sys.exit(f'{target} is not running')
    pid = int(out[0].split('","')[1])

k32 = ctypes.WinDLL('kernel32', use_last_error=True)


class THREADENTRY32(ctypes.Structure):
    _fields_ = [('dwSize', wt.DWORD), ('cntUsage', wt.DWORD), ('th32ThreadID', wt.DWORD),
                ('th32OwnerProcessID', wt.DWORD), ('tpBasePri', ctypes.c_long), ('tpDeltaPri', ctypes.c_long),
                ('dwFlags', wt.DWORD)]


k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
k32.OpenThread.restype = wt.HANDLE
snap = k32.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD
te = THREADENTRY32()
te.dwSize = ctypes.sizeof(te)
tids = []
ok = k32.Thread32First(snap, ctypes.byref(te))
while ok:
    if te.th32OwnerProcessID == pid:
        tids.append(te.th32ThreadID)
    ok = k32.Thread32Next(snap, ctypes.byref(te))
k32.CloseHandle(snap)

handles, names = {}, {}
for tid in tids:
    h = k32.OpenThread(0x0800 | 0x0040, False, tid)  # QUERY_LIMITED_INFORMATION | QUERY_INFORMATION
    if not h:
        continue
    handles[tid] = h
    p = ctypes.c_wchar_p()
    if k32.GetThreadDescription(h, ctypes.byref(p)) >= 0 and p.value:
        names[tid] = p.value


def times(h):
    c, e, kt, ut = wt.FILETIME(), wt.FILETIME(), wt.FILETIME(), wt.FILETIME()
    k32.GetThreadTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(kt), ctypes.byref(ut))
    f = lambda t: (t.dwHighDateTime << 32 | t.dwLowDateTime) / 1e7
    return f(kt), f(ut)


before = {t: times(h) for t, h in handles.items()}
time.sleep(secs)
rows = []
for t, h in handles.items():
    k1, u1 = times(h)
    k0, u0 = before[t]
    rows.append(((k1 - k0 + u1 - u0) / secs * 100, (k1 - k0) / secs * 100, t))
rows.sort(reverse=True)
print(f'pid {pid}, {len(handles)} threads over {secs:g} s; % of one core')
print(f'{"cpu%":>6} {"kern%":>6} {"tid":>6}  name')
for tot, kern, t in rows[:rows_n]:
    print(f'{tot:6.1f} {kern:6.1f} {t:6}  {names.get(t, "")}')
print(f'{sum(r[0] for r in rows):6.1f}        total')
