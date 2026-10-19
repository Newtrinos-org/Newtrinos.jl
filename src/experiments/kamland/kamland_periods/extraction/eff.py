"""Fig. 3(a): selection efficiency per period (y: 100 % at 14.0 pt, 1.0536 pt per %; x as in extract.py)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parse import parse
paths, texts, H = parse(sys.argv[1] if len(sys.argv) > 1 else 'fig3_trace.xml')
X0, SX = 49.75, (414.0 - 49.75) / 8.4
C = {1: '.199951 .399902 1', 2: '1 .199951 .599609', 3: '.199951 .800781 .399902'}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'efficiency.csv'), 'w') as f:
    f.write('period,E_MeV,efficiency\n')
    for p, c in C.items():
        # exclude the legend sample (horizontal line at x 330–352 pt)
        segs = [s for q in paths if q['color'] == c for s in q['segs']
                if not (all(329 < x < 353 for x, _ in s) and max(y for _, y in s) - min(y for _, y in s) < 0.01)]
        pts = sorted({(round((x - X0) / SX, 4), round((100 - (y - 14.0) / 1.0536) / 100, 4)) for s in segs for (x, y) in s if x > 88 and y > 13})
        pts = [q for q in pts if q[0] > 0.85]
        for e, v in pts: f.write('%d,%.4f,%.4f\n' % (p, e, v))
        print(p, len(pts), pts[:3], pts[len(pts)//2], pts[-2:])
