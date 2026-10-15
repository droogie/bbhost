"""Under one function of one thread: what it calls (paths, a few frames deep), and, for the
thread's kernel samples, which of its own frames each came from.

    python tools/etw/callees.py build/perf/trace.pickle <thread name or tid> <function substring>
        --exe build-win/bbhost.exe --eboot <the eboot> --log build/perf/NAME.log [--depth 2]

After the KyoPS4x ETW tools (Kyo).
"""
import argparse, collections, os, pickle, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symbols import Resolver  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('pickle')
ap.add_argument('thread')
ap.add_argument('needle')
ap.add_argument('--exe')
ap.add_argument('--eboot')
ap.add_argument('--log')
ap.add_argument('--slide', type=lambda s: int(s, 0))
ap.add_argument('--depth', type=int, default=2)
a = ap.parse_args()

d = pickle.load(open(a.pickle, 'rb'))
r = Resolver(d, exe=a.exe, eboot=a.eboot, log=a.log, slide=a.slide)
tids = {t for (t, _) in d['stacks']}
want = {int(a.thread)} if a.thread.isdigit() else \
    {t for t in tids if a.thread.lower() in (d['names'].get(t) or d['starts'].get(t, '')).lower()}
if not want:
    sys.exit(f'no thread matches {a.thread}')

paths, kern = collections.Counter(), collections.Counter()
under = total = 0
for (t, frames), c in d['stacks'].items():
    if t not in want:
        continue
    total += c
    labs = [r.label(x) for x in frames]
    if labs[0] == '[kernel]':
        kern[next((l for l in labs if l != '[kernel]'), '(none)')[:110]] += c
    hit = [i for i, l in enumerate(labs) if a.needle in l]
    if not hit:
        continue
    under += c
    i = hit[-1]  # the outermost frame that matches
    chain, j = [], i - 1
    while j >= 0 and len(chain) < a.depth:
        if labs[j] != labs[j + 1]:
            chain.append(labs[j][:90])
        j -= 1
    paths[' > '.join(chain) or '(self)'] += c
print(f'{a.needle}: {under} of {total} samples ({100 * under / max(total, 1):.1f}%) in {len(want)} thread(s)')
for p, c in paths.most_common(30):
    print(f'  {100 * c / max(under, 1):5.1f}%  {p}')
print(f'\nkernel-leaf samples by the first frame above the kernel ({sum(kern.values())}):')
for p, c in kern.most_common(15):
    print(f'  {100 * c / max(total, 1):5.1f}% of the thread  {p}')
