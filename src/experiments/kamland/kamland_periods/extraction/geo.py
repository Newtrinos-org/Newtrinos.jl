"""KamLAND 2013 Fig. 6 (middle panel): U and Th geo-ν̄e prompt spectra (red dashed / dotted), shapes only.
x: plot frame 0.9 – 2.7 MeV; y: zero line of the middle panel."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parse import parse
paths, texts, H = parse('../fig6_trace.xml')
blk = [s for p in paths if p['color'] == '0' for s in p['segs']]
xs = sorted({round(q[0], 2) for s in blk for q in s})
# frame: the widest horizontal black segments
hor = [s for s in blk if len(s) >= 2 and abs(s[0][1] - s[-1][1]) < 0.01]
L = min(min(q[0] for q in s) for s in hor); R = max(max(q[0] for q in s) for s in hor)
print('frame x', L, R)
red = [p for p in paths if p['color'] == '1 0 0']
for p in red:
    pts = [q for s in p['segs'] for q in s]
    print(p['dash'], p['lw'], len(pts), round(min(q[0] for q in pts), 1), round(max(q[0] for q in pts), 1), round(min(q[1] for q in pts), 1), round(max(q[1] for q in pts), 1))
# shapes (arbitrary units per 0.2 MeV, efficiency included): zero line of the middle panel at the curves' endpoints
Y0 = max(q[1] for p in red[:2] for s in p['segs'] for q in s)
E = lambda x: 0.9 + (x - L) / (R - L) * 1.8
def curve(p):
    pts = sorted({(round(E(q[0]), 4), round(Y0 - q[1], 4)) for s in p['segs'] for q in s})
    return pts
U, Th = curve(red[0]), curve(red[1])
import bisect
def interp(c, e):
    xs = [q[0] for q in c]; i = bisect.bisect_left(xs, e)
    if i == 0: return c[0][1]
    if i >= len(c): return c[-1][1]
    (x0, y0), (x1, y1) = c[i - 1], c[i]
    return y0 + (y1 - y0) * (e - x0) / (x1 - x0)
out = '../../../../Newtrinos.jl/src/experiments/kamland/kamland_periods/data/geo_shapes.csv'
out = '/home/iwsatlas1/peller/claude/Newtrinos.jl/src/experiments/kamland/kamland_periods/data/geo_shapes.csv'
with open(out, 'w') as f:
    f.write('# KamLAND 2013 (PRD 88, 033001), Fig. 6 middle panel: U (dashed) and Th (dotted) geo-ν̄e contributions vs prompt energy,\n'
            '# all periods, efficiency included, arbitrary units (shapes only); from the vector graphics (extraction/geo.py)\n')
    f.write('E_MeV,U,Th\n')
    for k in range(0, 181):
        e = 0.9 + 0.01 * k
        f.write('%.2f,%.5f,%.5f\n' % (e, max(interp(U, e), 0), max(interp(Th, e), 0)))
print('Y0', Y0, 'U peak at', max(U, key = lambda q: q[1]), 'Th peak at', max(Th, key = lambda q: q[1]))
