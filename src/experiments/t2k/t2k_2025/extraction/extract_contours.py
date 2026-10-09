"""Official T2K 2025 frequentist results (arXiv:2506.05889): Δχ² vs δCP (vector) per mass ordering."""
import pymupdf, numpy as np, json
pg = pymupdf.open("arxiv/contour_dCP_wRC.pdf")[0]; dr = pg.get_drawings(); words = pg.get_text("words")
num = lambda s: float(s) if s.replace(".", "").isdigit() else None
# rotated page: δCP runs along page-y (labels at x≈330), Δχ² along page-x (labels at y≈501)
d = [(((w[1] + w[3]) / 2), num(w[4])) for w in words if 325 < w[0] < 335 and num(w[4]) is not None and w[1] < 270]   # 0, 1, 2, 3
c = [(((w[0] + w[2]) / 2), num(w[4])) for w in words if 495 < w[1] < 505 and num(w[4]) is not None]
A = np.polyfit(*zip(*d), 1); B = np.polyfit(*zip(*c), 1)
print("δCP labels", d, "\nΔχ² labels", c)
out = {}
for col, mo in (((0.11, 0.47, 0.71), "NO"), ((1.0, 0.5, 0.06), "IO")):
    dd = [x for x in dr if x["type"] == "s" and x.get("color") and tuple(round(v, 2) for v in x["color"]) == col and len(x["items"]) > 20][0]
    pts = [dd["items"][0][1]] + [it[-1] for it in dd["items"]]
    dcp = np.polyval(A, [p.y for p in pts]); chi2 = np.polyval(B, [p.x for p in pts])
    o = np.argsort(dcp); out[mo] = {"dcp": list(dcp[o]), "dchi2": list(chi2[o])}
    print(mo, len(pts), "δCP range", dcp.min().round(3), dcp.max().round(3), "min Δχ²", chi2.min().round(3), "at", dcp[np.argmin(chi2)].round(3), "max", chi2.max().round(2))
json.dump(out, open("t2k_official_dcp.json", "w"), indent=1)
