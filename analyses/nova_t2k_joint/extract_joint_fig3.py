"""Bayesian 1/2/3σ credible regions in δCP–sin²θ23 (NO, IO; NOvA+T2K, reactor constraint) from the vector
Fig. 3 of arXiv:2510.19888 (Nature 646, 818)."""
import pymupdf, json, numpy as np
import sys
pg = pymupdf.open(sys.argv[1] if len(sys.argv) > 1 else "src/figures/Fig3.pdf")[0]; dr = pg.get_drawings()
W = {w[4]: w for w in pg.get_text("words")}
ycal = np.polyfit([102.5, 157.0, 209.5], [0.6, 0.5, 0.4], 1)        # centres of the 0.6 / 0.5 / 0.4 labels
panels = {"NO": (190.5, 79.5, (111, 270)), "IO": (354.0, 79.0, (275, 433))}   # x of δCP = 0, px per π
shades = {"NO": {(0.0, 0.59, 0.75): "1sigma", (0.36, 0.78, 0.89): "2sigma", (0.81, 0.93, 0.96): "3sigma"},
          "IO": {(0.97, 0.53, 0.06): "1sigma", (0.95, 0.72, 0.32): "2sigma", (0.98, 0.85, 0.7): "3sigma"}}
out = {"NO": {}, "IO": {}}
seen = set()
for d in dr:
    if d["type"] not in ("f", "fs") or len(d["items"]) < 10: continue
    r = d["rect"]; col = tuple(round(x, 2) for x in d["fill"])
    if not (95 < r.y0 and r.y1 < 202): continue          # the 2D panels only
    for mo, (x0, ppi, (xa, xb)) in panels.items():
        if xa - 6 <= r.x0 and r.x1 <= xb + 6 and col in shades[mo]:
            key = (mo, col, round(r.x0), round(r.y0), len(d["items"]))
            if key in seen: continue
            seen.add(key)
            # split into closed subpaths
            subs, cur, last = [], [], None
            for it in d["items"]:
                p0, p1 = it[1], it[-1]
                if last is not None and (abs(p0.x - last.x) > 0.01 or abs(p0.y - last.y) > 0.01):
                    subs.append(cur); cur = []
                if not cur: cur.append(p0)
                cur.append(p1); last = p1
            subs.append(cur)
            for s in subs:
                pts = [((p.x - x0) / ppi * np.pi, float(np.polyval(ycal, p.y))) for p in s]
                out[mo].setdefault(shades[mo][col], []).append(pts)
for mo in out:
    for lv, polys in out[mo].items():
        a = np.concatenate([np.array(p) for p in polys])
        print(mo, lv, len(polys), "polys; δCP", a[:, 0].min().round(2), a[:, 0].max().round(2), "ss23", a[:, 1].min().round(3), a[:, 1].max().round(3))
json.dump(out, open("joint_fig3_regions.json", "w"))

import h5py
with h5py.File("joint_fig3_regions.h5", "w") as f:
    f.attrs["description"] = __doc__
    for mo, lv in out.items():
        for l, polys in lv.items():
            for i, p in enumerate(polys):
                p = np.array(p); f[f"{mo}/{l}/{i}/dcp"] = p[:, 0]; f[f"{mo}/{l}/{i}/ssth23"] = p[:, 1]
