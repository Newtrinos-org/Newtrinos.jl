"""Borexino Phase-I ⁷Be spectral fit (PRL 107, 141302 (2011), arXiv:1104.1816, Fig. 1 top, MC_final.pdf): data points
with errors, Borexino's total fit and fitted components, from the vector graphics (mutool trace MC_final.pdf > mc_trace.xml).
Axis calibration from the tick marks; units evt / (1000 keV × ton × day), 10 keV bins."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parse import parse
paths, texts, H = parse(sys.argv[1] if len(sys.argv) > 1 else 'mc_trace.xml')
E = lambda X: 200 + (X - 56.2042) / (69.2405 / 200)
V = lambda Y: 10 ** (-2 + (344.9014 - Y) / 83.2875)
# data: central markers (horizontal 1-pt segments, lw 0.75) and error bars (vertical lw 1.5 segments from the centre)
marks = sorted({(round(s[0][0] + 0.5, 3), round(s[0][1], 3)) for p in paths if p['color'] == '0 0 0' and p['lw'] == 0.75
                for s in p['segs'] if len(s) == 2 and abs(s[0][1] - s[1][1]) < 1e-3 and abs(abs(s[1][0] - s[0][0]) - 1.0) < 0.05})
bars = [s for p in paths if p['color'] == '0 0 0' and p['lw'] == 1.5 for s in p['segs'] if abs(s[0][0] - s[1][0]) < 1e-3]
data = []
for x, y in marks:
    # error bars start at the marker height; their x is the bin centre (the marker segment is offset by ~0.5 pt)
    mine = [b for b in bars if abs(b[0][0] - x) < 0.8 and (abs(b[0][1] - y) < 0.05 or abs(b[1][1] - y) < 0.05)]
    xc = mine[0][0][0] if mine else x
    ends = [b[1][1] if abs(b[0][1] - y) < 0.05 else b[0][1] for b in mine]
    up = min([e for e in ends if e < y], default=y); dn = max([e for e in ends if e > y], default=y)
    data.append((E(xc), V(y), V(y) - V(dn), V(up) - V(y)))
def curve(color, lw=2.25, minpts=10):
    pts = sorted((E(x), V(y)) for p in paths if p['color'] == color and p['lw'] == lw for s in p['segs'] if len(s) >= minpts for x, y in s)
    return pts
def interp(pts, x):
    if not pts or x < pts[0][0] or x > pts[-1][0]: return 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0: return y0
            t = (x - x0) / (x1 - x0); return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))
    return 0.0
def chains(color, lw=2.25, minpts=10):
    return [sorted((E(x), V(y)) for x, y in s) for p in paths if p['color'] == color and p['lw'] == lw for s in p['segs'] if len(s) >= minpts]
comps = {'total': chains('0 0 0'), 'Be7': chains('1 0 0'), 'Kr85': chains('0 0 1'), 'Bi210': chains('0 1 0'),
         'C11': chains('.8 0 1'), 'Po210': chains('.6 .4 0'), 'ext': chains('.8 .8 .2', minpts=3), 'pp_pep_cno': chains('0 1 1')}
for k, v in comps.items(): print(k, len(v), 'chains', [len(c) for c in v], 'E range', [ (round(c[0][0]), round(c[-1][0])) for c in v])
names = list(comps)
with open('../phase1_be7_fig1.csv', 'w') as f:
    f.write('# Borexino Phase I, 7Be analysis (PRL 107, 141302 (2011), arXiv:1104.1816), Fig. 1 top: MC-based fit, 270-1600 keV,\n'
            '# extracted from the vector graphics. Units: events / (1000 keV x ton x day); 10 keV bins. err_lo/err_hi: error bars.\n'
            '# total = Borexino fit; components are summed over the drawn curves of each colour (pp_pep_cno: the three fixed curves);\n'
            '# 0 = below the plotted range (1e-2).\n')
    f.write('energy_keV,data,err_lo,err_hi,' + ','.join(names) + '\n')
    for e, d, lo, hi in data:
        f.write('%.2f,%.5g,%.4g,%.4g,' % (e, d, lo, hi) + ','.join('%.5g' % sum(interp(c, e) for c in comps[k]) for k in names) + '\n')
print('data points', len(data), 'first', data[0], 'last', data[-1])
