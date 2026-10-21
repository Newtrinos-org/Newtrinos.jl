"""Extract the fit components of Borexino Phase-II Fig. 6a (arXiv:1707.09279) from the vector PDF.
Axis calibration from the tick marks: x linear in N_h, y logarithmic."""
import sys, math, json; sys.path.insert(0, '.')
from trace_parse import parse
paths, texts, H = parse(sys.argv[1])
nh = lambda X: 100 + (X - 61.35) / 0.58200          # top axis: N_h 100 at x=61.35, 100 per 58.20 pt
val = lambda Y: 10 ** (1 - (Y - 64.16) / 56.6475)   # left axis: 10 at y=64.16, one decade per 56.65 pt
STYLES = {('0 0 0', 1.5, None): 'total', ('1 0 0', 1.5, None): 'solar', ('0 1 0', .75, None): 'Bi210',
          ('0 .4 .4', .75, None): 'C11', ('1 0 1', .75, '3 3'): 'pileup', ('0 0 1', .75, None): 'Kr85',
          ('0 0 1', .75, '3 3'): 'ext', ('1 .8 0', .75, None): 'Po210', ('0 .4 0', .75, None): 'C14'}
def chains(segs, tol=0.05):
    out = []
    for s in segs:
        if out and abs(out[-1][-1][0] - s[0][0]) < tol and abs(out[-1][-1][1] - s[0][1]) < tol:
            out[-1].extend(s[1:])
        else:
            out.append(list(s))
    return out
res = {}
for p in paths:
    k = STYLES.get((p['color'], p['lw'], p['dash']))
    if k: res.setdefault(k, []).extend(p['segs'])
curves = {}
for k, segs in res.items():
    cs = chains(segs)
    curves[k] = [[(nh(x), val(y)) for x, y in c] for c in cs]
    print(k, len(segs), 'segs ->', len(cs), 'chains')
    for c in curves[k][:12]:
        xs = [q[0] for q in c]; ys = [q[1] for q in c]
        print('   %5d pts  N_h %6.1f–%6.1f  y %.2e–%.2e  first %s' % (len(c), min(xs), max(xs), min(ys), max(ys), tuple(round(v, 4) for v in c[0])))
json.dump(curves, open('fig6a_curves_raw.json', 'w'))  # input of build_table.py
