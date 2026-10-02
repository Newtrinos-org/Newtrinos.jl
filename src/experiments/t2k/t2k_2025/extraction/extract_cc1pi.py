import json, numpy as np, pymupdf
import importlib.util, sys
sys.argv=['x']
exec(open('extract_data.py').read().split('if __name__ != ')[0])
pg = pymupdf.open("arxiv/numucc1pi_twk.pdf")[0]; dr = pg.get_drawings(); words = pg.get_text("words")
g = label_groups(words)
xa = axis_from(g, {1.0, 2.0, 3.0, 4.0}); ya = axis_from(g, {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0})
print("resid", xa[2], ya[2])
X = lambda p: float(np.polyval(xa[1], (p.x, p.y)[xa[0]])); Y = lambda p: float(np.polyval(ya[1], (p.x, p.y)[ya[0]]))
pts = []
for d in dr:
    if d.get("fill") == (0.0, 0.0, 0.0) and d["type"] == "f" and len(d["items"]) == 2 and all(it[0] == "c" for it in d["items"]):
        a = d["items"][0]; c = ((a[1].x + a[4].x) / 2, (a[1].y + a[4].y) / 2)
        pts.append((X(pymupdf.Point(*c)), Y(pymupdf.Point(*c))))
# horizontal error bars (black horizontal lines through the markers) give the bin widths
hl = []
for d in dr:
    if d.get("color") == (0.0, 0.0, 0.0) and len(d["items"]) == 1 and d["items"][0][0] == "l":
        p, q = d["items"][0][1], d["items"][0][2]
        x0, x1, y0, y1 = X(p), X(q), Y(p), Y(q)
        if abs(y1 - y0) < 1e-3 and abs(x1 - x0) > 0.02: hl.append((min(x0, x1), max(x0, x1), y0))
pts.sort()
for x, y in pts:
    seg = [h for h in hl if h[0] - 0.06 <= x <= h[1] + 0.06 and abs(h[2] - y) < 0.05]
    print(round(x, 3), round(y, 3), [(round(a, 3), round(b, 3)) for a, b, _ in seg])
fr = max((d["rect"] for d in dr if len(d["items"]) <= 4 and 200 < d["rect"].width < 500), key=lambda r: r.width * r.height)
y0 = Y(pymupdf.Point(fr.x0, fr.y1))
print("frame-bottom value", y0)
pts = [(x, y - y0) for x, y in pts if not (0.9 < x < 0.97 and y > 11.5)]   # drop the legend marker
data = []
for x, y in pts:
    w = 0.1 if x < 3.0 else 0.5
    data.append({"center": round(x, 3), "lo": round(round((x - w / 2) * 20) / 20, 2), "hi": round(round((x + w / 2) * 20) / 20, 2), "count": y * w / 0.1})
for r in data: print(r)
print("total", sum(r["count"] for r in data))
json.dump(data, open("t2k_cc1pi_data.json", "w"), indent=1)
