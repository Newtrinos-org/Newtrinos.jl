"""KamLAND 2013 (PRD 88, 033001, arXiv:1303.4667), Fig. 3: per-period prompt spectra (events / 0.425 MeV / day) from the
vector graphics (ps2pdf -dEPSCrop fig3.eps; mutool trace). x: 0 MeV at 49.75 pt, 43.4 pt/MeV (0.2 MeV minor ticks);
y: 1105.7 pt per event/0.425MeV/day, zero lines at 286.75, 485.75, 684.75 pt (Periods 1, 2, 3)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parse import parse
paths, texts, H = parse(sys.argv[1] if len(sys.argv) > 1 else '../fig3_trace.xml')
X0, SX = 49.75, (414.0 - 49.75) / 8.4          # minor ticks at 0.2 MeV: 49.75 … 414.0 = 0 … 8.4 MeV
SY = 193.5 / 0.175
ZERO = {1: 286.75, 2: 485.75, 3: 684.75}
E = lambda x: (x - X0) / SX
edges = [0.9 + 0.425 * k for k in range(19)]
COLORS = {'osc': '.350098 .330078 .849609', 'acc': '1 .599609 .599609', 'c13': '.599609 1 .599609',
          'total': '.399902 .599609 1', 'geo': '0 0 1'}
def hsegs(color, p):
    z = ZERO[p]
    out = []
    for q in paths:
        if q['color'] != color: continue
        for s in q['segs']:
            for a, b in zip(s[:-1], s[1:]):
                if abs(a[1] - b[1]) < 0.05 and z - 199 < a[1] <= z + 0.3 and abs(abs(a[0] - b[0]) - 0.425 * SX) < 2.0 + 0.0 or \
                   (abs(a[1] - b[1]) < 0.05 and z - 199 < a[1] <= z + 0.3 and abs(a[0] - b[0]) > 0.425 * SX * 0.9):
                    v = (z - a[1]) / SY
                    if (0.5 * (a[0] + b[0]) > 285 or p == 3) and v > 0.05: continue      # inset panels, legend
                    out.append((min(a[0], b[0]), max(a[0], b[0]), v))
    return out
def hist(color, p, pick = max):
    segs = hsegs(color, p); vals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        c = [v for (x0, x1, v) in segs if E(x0) <= lo + 0.05 and E(x1) >= hi - 0.05 - (0.08 if hi > 8.4 else 0)]
        vals.append(pick(c) if c else 0.0)
    return vals
def datapts(p):
    """data points: horizontal error bars drawn as two halves around the marker, vertical bars above/below it"""
    z = ZERO[p]; pts = {}
    blk = [s for q in paths if q['color'] == '0' for s in q['segs'] if len(s) == 2]
    for lo, hi in zip(edges[:-1], edges[1:]):
        xl, xr = X0 + lo * SX, X0 + hi * SX; xc = 0.5 * (xl + xr)
        left = [a[1] for (a, b) in blk if abs(a[1] - b[1]) < 0.05 and abs(min(a[0], b[0]) - xl) < 0.6 and abs(max(a[0], b[0]) - xc) < 4 and z - 199 < a[1] < z + 1]
        if not left: continue
        y = left[0]
        v = [(a, b) for (a, b) in blk if abs(a[0] - b[0]) < 0.05 and abs(a[0] - xc) < 0.6 and z - 199 < min(a[1], b[1]) and max(a[1], b[1]) < z + 1]
        ys = [q[1] for sg in v for q in sg]
        pts[lo] = dict(y = (z - y) / SY, hi = (z - min(ys)) / SY if ys else (z - y) / SY, lo = (z - max(ys)) / SY if ys else (z - y) / SY)
    return pts
for p in (1, 2, 3):
    h = {k: hist(c, p) for k, c in COLORS.items()}
    d = datapts(p)
    print('Period', p, 'data points', len(d))
    with open('period%d.csv' % p, 'w') as f:
        f.write('E_lo,E_hi,data,data_lo,data_hi,osc,acc_cum,c13_cum,geo_cum,total\n')
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            dd = d.get(lo, dict(y = float('nan'), lo = float('nan'), hi = float('nan')))
            f.write('%.3f,%.3f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' % (lo, hi, dd['y'], dd['lo'], dd['hi'], h['osc'][i], h['acc'][i], h['c13'][i], h['geo'][i], h['total'][i]))
    print(open('period%d.csv' % p).read())
