"""Event-by-event data from the T2K 2023 paper figures (arXiv:2303.03222, Fig. 'Data_over_Asimov...comb'):
the 2D scatter panels draw each FD data event as a filled circle (two Bezier half-circles)."""
import json, glob, numpy as np, pymupdf
from collections import defaultdict
from pdfhist import num

def label_groups(words):
    groups = defaultdict(list)
    for w in words:
        v = num(w[4])
        if v is None: continue
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        groups[("row", round(w[1]))].append((cx, cy, v))
        groups[("col", round(w[0]))].append((cx, cy, v))
        groups[("colR", round(w[2] / 3))].append((cx, cy, v))
        groups[("rowB", round(w[3] / 3))].append((cx, cy, v))
    return groups

def axis_from(groups, must):
    best = None
    for k, g in groups.items():
        vals = {v for *_, v in g}
        if must <= vals:
            g = np.array([x for x in g if x[2] in must])
            c = 0 if k[0].startswith("row") else 1
            A = np.polyfit(g[:, c], g[:, 2], 1)
            r = np.max(np.abs(np.polyval(A, g[:, c]) - g[:, 2]))
            if best is None or r < best[2]: best = (c, A, r)
    if best: return best
    for k, g in groups.items():
        if False:
            g = np.array(g)
            c = 0 if k[0].startswith("row") else 1
            A = np.polyfit(g[:, c], g[:, 2], 1)
            return c, A, np.max(np.abs(np.polyval(A, g[:, c]) - g[:, 2]))
    raise ValueError(f"no axis with labels {must}")

def circles(dr):
    """centres of the filled circles in the largest all-curve fill path"""
    paths = [d for d in dr if d.get("fill") is not None and d["type"] == "f" and all(it[0] == "c" for it in d["items"]) and len(d["items"]) > 2]
    pts = []
    for d in paths:
        its = d["items"]
        assert len(its) % 2 == 0
        for a, b in zip(its[::2], its[1::2]):
            # half circle a: from P0 to P3 (diametrically opposite); b closes the circle
            p0, p3 = a[1], a[4]
            pts.append(((p0.x + p3.x) / 2, (p0.y + p3.y) / 2, abs(p0.x - p3.x) + abs(p0.y - p3.y)))
    return pts, [len(d["items"]) for d in paths]

out = {}
if __name__ != "__main__": raise SystemExit
E = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}; P = {200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0}
spec = {"numu1R": (E, "Erec"), "numubar1R": (E, "Erec"), "nue1R": (P, "p"), "nuebar1R": (P, "p"), "nue1RD": (P, "p")}
for s, (xlabels, xname) in spec.items():
    f = glob.glob(f"arxiv2023/Figures/OA/ptheta/01-distributions/*_{s}_comb.pdf")[0]
    pg = pymupdf.open(f)[0]; dr = pg.get_drawings(); words = pg.get_text("words")
    g = label_groups(words)
    xa = axis_from(g, xlabels); ya = axis_from(g, {0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0})
    pts, sizes = circles(dr)
    ev = [(float(np.polyval(xa[1], (p[0], p[1])[xa[0]])), float(np.polyval(ya[1], (p[0], p[1])[ya[0]]))) for p in pts]
    diam = sorted({round(p[2], 2) for p in pts})
    print(s, "paths", sizes, "events", len(ev), "marker sizes", diam[:5], "x-axis fit resid", round(xa[2], 4), "y resid", round(ya[2], 3))
    out[s] = {"x": xname, "events": ev}
json.dump(out, open("t2k_data_events.json", "w"), indent=1)
