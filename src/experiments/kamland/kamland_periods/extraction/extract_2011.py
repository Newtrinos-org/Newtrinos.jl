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
red = sorted({(round(q[0], 2), round(q[1], 2)) for p in paths if p['color'] == '1 0 0' for s in p['segs'] for q in s})
xs = sorted({x for x, _ in red})
with open(os.path.join(OUT, 'efficiency_2011.csv'), 'w') as f:
    f.write('# KamLAND 2011 (PRD 83, 052002), Fig. 1 top panel: selection efficiency (weighted average over five periods),\n'
            '# from the vector graphics (extraction/extract_2011.py)\nE_MeV,efficiency\n')
    for x in xs:
        ys = [y for xx, y in red if xx == x]
        f.write('%.4f,%.4f\n' % (E(x), (100 - (sum(ys) / len(ys) - 4.5) / 2.3625) / 100))
print(open(os.path.join(OUT, 'noosc_2011.csv')).read()); print(open(os.path.join(OUT, 'efficiency_2011.csv')).read()[:600])
