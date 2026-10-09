"""Digitise the raster T2K 2025 sin²θ23–Δm² frequentist contours (arXiv:2506.05889, Fig. 4) by colour
and line style: connected pixel blobs of the ordering's colour are classified by size into solid (90%),
dashed (68%) and dotted (99.7%) lines; points are ordered by angle around the best fit."""
import numpy as np, json
from PIL import Image
from scipy import ndimage
im = np.asarray(Image.open("th23dm2.png").convert("RGB")).astype(float)
X = lambda px: 0.3 + (px - 175.5) / (1416.5 - 175.5) * 0.4
Y = lambda py: 2.1 + (856.5 - py) / (856.5 - 80.5) * 0.6
out = {}
inside = np.zeros(im.shape[:2], bool); inside[82:855, 177:1415] = True
inside[600:855, 177:700] = False      # legend (lower left)
inside[600:855, 1050:1415] = False     # legend (lower right)
for mo, col, bf in (("NO", (31, 119, 180), (0.556, 2.503)), ("IO", (255, 127, 14), (0.556, 2.473))):
    mask = (np.linalg.norm(im - np.array(col), axis=2) < 45) & inside
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    ext = ndimage.find_objects(lab)
    cls = {"068": [], "090": [], "997": []}
    for i, (sz, sl) in enumerate(zip(sizes, ext)):
        span = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        if span > 25: k = "090"
        elif 6 <= span <= 12: k = "068"
        elif 2 <= span <= 5 and sz >= 3: k = "997"
        else: continue
        ys, xs = np.nonzero(lab[sl] == i + 1)
        if k == "090":
            cls[k] += [(X(x + sl[1].start), Y(y + sl[0].start)) for x, y in zip(xs, ys)]
        else:   # one point per dash / dot
            cls[k].append((X(xs.mean() + sl[1].start), Y(ys.mean() + sl[0].start)))
    PXb, PYb = 175.5 + (bf[0] - 0.3) / 0.4 * 1241, 856.5 - (bf[1] - 2.1) / 0.6 * 776
    cls = {k: [q for q in v if np.hypot((q[0] - bf[0]) / 0.4 * 1241, (q[1] - bf[1]) / 0.6 * 776) > 15] for k, v in cls.items()}   # best-fit marker
    out[mo] = {}
    for k, p in cls.items():
        p = np.array(p)
        # thin out: mean point per angular bin around the best fit (normalised axes)
        a = np.arctan2((p[:, 1] - bf[1]) / 0.6, (p[:, 0] - bf[0]) / 0.4)
        if k != "090":
            o = np.argsort(a); out[mo][k] = [list(map(float, q)) for q in p[o]]
            print(mo, k, len(p), "sin2", p[:, 0].min().round(3), p[:, 0].max().round(3), "dm2", p[:, 1].min().round(3), p[:, 1].max().round(3)); continue
        bins = np.linspace(-np.pi, np.pi, 361)
        idx = np.digitize(a, bins)
        pts = [p[idx == j].mean(axis=0) for j in range(1, 361) if np.any(idx == j)]
        out[mo][k] = [list(map(float, q)) for q in pts]
        q = np.array(pts); print(mo, k, len(q), "sin2", q[:, 0].min().round(3), q[:, 0].max().round(3), "dm2", q[:, 1].min().round(3), q[:, 1].max().round(3))
json.dump(out, open("t2k_official_th23dm2.json", "w"))
