"""Borexino Phase-I ⁸B spectrum (PRD 82, 033006 (2010), arXiv:0808.2868, Fig. 7, 7_compare.eps): background-subtracted
counts per 2 MeV in 345.3 live days (100 t), from the vector graphics (ps2pdf -dEPSCrop 7_compare.eps b8.pdf;
mutool trace b8.pdf > b8_trace.xml). Axes from the tick marks: x 3 MeV at 56.2 pt, 37.82 pt/MeV; y 0 at 345.0 pt,
6.8267 pt per count."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parse import parse
paths, texts, H = parse(sys.argv[1] if len(sys.argv) > 1 else 'b8_trace.xml')
E = lambda X: 3 + (X - 56.2) / ((510.0 - 56.2) / 12)
N = lambda Y: (345.0 - Y) / ((345.0 - 37.8) / 45)
red = [s for p in paths if p['color'] == '1 0 0' for s in p['segs'] if len(s) == 2]
horiz = [s for s in red if abs(s[0][1] - s[1][1]) < 1e-3 and abs(s[0][0] - s[1][0]) > 10]          # bin half-widths
vert = [s for s in red if abs(s[0][0] - s[1][0]) < 1e-3 and abs(s[0][1] - s[1][1]) > 5]             # error bars (not the 3.6 pt bin-edge caps)
rows = []
for xc in sorted({round(s[0][0], 2) for s in horiz}):
    y = [s[0][1] for s in horiz if abs(s[0][0] - xc) < 0.05][0]
    lo = min(min(s[0][0], s[1][0]) for s in horiz if abs(s[0][0] - xc) < 0.05)
    hi = max(max(s[0][0], s[1][0]) for s in horiz if abs(s[0][0] - xc) < 0.05)
    ends = [s[1][1] for s in vert if abs(s[0][0] - xc) < 0.05]
    if not ends: continue        # the legend sample
    rows.append((E(lo), E(hi), N(y), N(y) - N(max(ends)), N(min(ends)) - N(y)))
with open('../phase1_b8_fig7.csv', 'w') as f:
    f.write('# Borexino Phase I, 8B (PRD 82, 033006 (2010), arXiv:0808.2868), Fig. 7: background-subtracted spectrum, counts per\n'
            '# 2 MeV bin in 345.3 live days (100 t fiducial mass), extracted from the vector graphics; err_lo/err_hi: error bars.\n')
    f.write('E_lo_MeV,E_hi_MeV,counts,err_lo,err_hi\n')
    for r in rows: f.write('%.3f,%.3f,%.3f,%.3f,%.3f\n' % r)
print(open('../phase1_b8_fig7.csv').read())
