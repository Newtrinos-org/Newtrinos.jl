"""KamLAND 2011 (PRD 83, 052002, arXiv:1009.4771), Fig. 1 (fig1.eps → ps2pdf -dEPSCrop; mutool trace): the expected
no-oscillation prompt spectrum (grey histogram, events / 0.425 MeV) and the energy-dependent selection efficiency
(top panel, red). Axes from the tick marks: x 0 MeV at 56.75 pt, 54.05 pt/MeV (0.2 MeV minor ticks to 8.4 MeV at 510.75);
main panel 0 events at 471.5 pt, 0.8136 pt/event; efficiency 100 % at 4.5 pt, 2.3625 pt per %."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parse import parse
paths, texts, H = parse(sys.argv[1] if len(sys.argv) > 1 else 'fig1_trace.xml')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
E = lambda x: (x - 56.75) / ((510.75 - 56.75) / 8.4)
grey = [s for p in paths if p['color'] == '.300049 .300049 .300049' for s in p['segs']]
steps = sorted({(round(min(a[0], b[0]), 2), round(max(a[0], b[0]), 2), round(a[1], 2)) for s in grey for a, b in zip(s[:-1], s[1:])
                if abs(a[1] - b[1]) < 0.01 and abs(a[0] - b[0]) > 5})
with open(os.path.join(OUT, 'noosc_2011.csv'), 'w') as f:
    f.write('# KamLAND 2011 (PRD 83, 052002), Fig. 1: expected no-oscillation prompt spectrum (2002-2009, 2135 live days),\n'
            '# events per 0.425 MeV bin, from the vector graphics (extraction/extract_2011.py)\nE_lo,E_hi,events\n')
    for x0, x1, y in steps:
        if min(abs(E(x0) - (0.9 + 0.425 * k)) for k in range(19)) > 0.02: continue          # legend sample
        f.write('%.3f,%.3f,%.2f\n' % (E(x0), E(x1), (471.5 - y) / 0.8136))
# efficiency points; the legend sample (a two-point horizontal line at x 340–370 pt) is excluded
red = sorted({(round(q[0], 2), round(q[1], 2)) for p in paths if p['color'] == '1 0 0' for s in p['segs'] for q in s
              if not (len(s) == 2 and abs(s[0][1] - s[1][1]) < 0.01)})
xs = sorted({x for x, _ in red})
with open(os.path.join(OUT, 'efficiency_2011.csv'), 'w') as f:
    f.write('# KamLAND 2011 (PRD 83, 052002), Fig. 1 top panel: selection efficiency (weighted average over five periods),\n'
            '# from the vector graphics (extraction/extract_2011.py)\nE_MeV,efficiency\n')
    for x in xs:
        ys = [y for xx, y in red if xx == x]
        f.write('%.4f,%.4f\n' % (E(x), (100 - (sum(ys) / len(ys) - 4.5) / 2.3625) / 100))
print(open(os.path.join(OUT, 'noosc_2011.csv')).read()); print(open(os.path.join(OUT, 'efficiency_2011.csv')).read()[:600])

# observed spectrum and best-fit components (events / 0.425 MeV; written per live day, 2135 days, like the 2013 periods)
EDG = [0.9 + 0.425 * k for k in range(19)]
def hist(color):
    vals = [0.0] * 18
    for p in paths:
        if p['color'] != color: continue
        for s in p['segs']:
            for a, b in zip(s[:-1], s[1:]):
                if abs(a[1] - b[1]) > 0.01 or abs(a[0] - b[0]) < 15: continue
                lo, hi = E(min(a[0], b[0])), E(max(a[0], b[0]))
                if min(abs(lo - e) for e in EDG) > 0.02: continue            # legend samples
                for k in range(18):
                    if lo - 0.02 <= EDG[k] and EDG[k + 1] <= hi + 0.1: vals[k] = (471.5 - a[1]) / 0.8136
    return vals
blk = [s for p in paths if p['color'] == '0' and not p['dash'] for s in p['segs'] if len(s) == 2]
data, dlo, dhi = [], [], []
for k in range(17):
    xl, xr = 56.75 + EDG[k] * (510.75 - 56.75) / 8.4, 56.75 + EDG[k + 1] * (510.75 - 56.75) / 8.4; xc = 0.5 * (xl + xr)
    left = [a[1] for (a, b) in blk if abs(a[1] - b[1]) < 0.05 and abs(min(a[0], b[0]) - xl) < 0.8 and abs(max(a[0], b[0]) - xc) < 6 and 146 < a[1] < 472]
    v = [q[1] for (a, b) in blk if abs(a[0] - b[0]) < 0.05 and abs(a[0] - xc) < 0.8 and 146 < min(a[1], b[1]) for q in (a, b)]
    y = left[0] if left else float('nan')
    data.append((471.5 - y) / 0.8136); dhi.append((471.5 - min(v)) / 0.8136 if v else data[-1]); dlo.append((471.5 - max(v)) / 0.8136 if v else data[-1])
osc, tot = hist('.399902 .399902 1'), hist('.399902 .599609 1')
acc, c13, geo = hist('1 .599609 .599609'), hist('.599609 1 .599609'), hist('.599609 .800781 1')
DAYS = 2135.0
with open(os.path.join(OUT, 'kamland2011.csv'), 'w') as f:
    f.write('E_lo,E_hi,data,data_lo,data_hi,osc,acc_cum,c13_cum,geo_cum,total\n')
    for k in range(18):
        d = data[k] if k < 17 else float('nan'); l = dlo[k] if k < 17 else float('nan'); h = dhi[k] if k < 17 else float('nan')
        f.write('%.3f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n' % (EDG[k], EDG[k + 1], d / DAYS, l / DAYS, h / DAYS,
                osc[k] / DAYS, acc[k] / DAYS, c13[k] / DAYS, geo[k] / DAYS, tot[k] / DAYS))
print('data counts', [round(x, 2) for x in data], 'sum', round(sum(data), 1))
print('sums: osc %.1f acc %.1f c13 %.1f geo %.1f rest %.1f' % (sum(osc), sum(acc), sum(c13) - sum(acc), sum(geo) - sum(c13), sum(tot) - sum(osc) - sum(geo)))
