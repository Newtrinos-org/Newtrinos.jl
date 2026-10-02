"""Extract histograms (step polylines) from ROOT-made vector PDFs."""
import pymupdf, re, numpy as np

def num(s):
    s = s.replace("−", "-")
    try: return float(s)
    except ValueError: return None

def load(path):
    pg = pymupdf.open(path)[0]
    return pg, pg.get_drawings(), pg.get_text("words")

def axis_maps(words, xticks, yticks):
    """linear maps page-coord -> data for both axes from numeric tick labels.
    xticks/yticks: lists of expected label values (as strings) to identify the axes."""
    def fit(vals):
        cands = [w for w in words if w[4].replace("−", "-") in vals]
        # group by which coordinate is (nearly) constant
        P = np.array([[(w[0]+w[2])/2, (w[1]+w[3])/2, num(w[4])] for w in cands])
        best = None
        for c in (0, 1):   # coordinate along the axis
            other = 1 - c
            # keep the most populated row of labels (same 'other' coordinate)
            rows = {}
            for p in P: rows.setdefault(round(p[other] / 4), []).append(p)
            row = max(rows.values(), key=len)
            row = np.array(row)
            if len(row) < 3: continue
            A = np.polyfit(row[:, c], row[:, 2], 1)
            res = np.max(np.abs(np.polyval(A, row[:, c]) - row[:, 2]))
            if best is None or res < best[2]: best = (c, A, res, len(row))
        return best
    return fit(xticks), fit(yticks)

def segments(d):
    out = []
    for it in d["items"]:
        if it[0] == "l": out.append((it[1].x, it[1].y, it[2].x, it[2].y))
        elif it[0] == "re":
            r = it[1]; out += [(r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1), (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)]
    return out

def to_data(seg, xm, ym):
    (cx, Ax, *_), (cy, Ay, *_) = xm, ym
    x0 = np.polyval(Ax, seg[cx]); x1 = np.polyval(Ax, seg[2 + cx])
    y0 = np.polyval(Ay, seg[cy]); y1 = np.polyval(Ay, seg[2 + cy])
    return x0, y0, x1, y1

def step_hist(d, xm, ym, tol=1e-6):
    """bins (lo, hi, value) from the horizontal pieces of a step polyline"""
    bins = []
    for s in segments(d):
        x0, y0, x1, y1 = to_data(s, xm, ym)
        if abs(y1 - y0) < 1e-4 * max(1, abs(y0)) and abs(x1 - x0) > tol:
            lo, hi = sorted((x0, x1)); bins.append((lo, hi, (y0 + y1) / 2))
    bins.sort()
    return bins

def frame_fix(dr, xm, ym):
    """re-anchor the intercepts on the plot frame (lower edges at round tick values)"""
    frames = [d for d in dr if d.get("color") == (0.0, 0.0, 0.0) and len(d["items"]) <= 4 and d["rect"].width > 100 and d["rect"].height > 100]
    fr = max(frames, key=lambda d: d["rect"].width * d["rect"].height)["rect"]
    out = []
    for (c, A, *rest) in (xm, ym):
        lo, hi = (fr.x0, fr.x1) if c == 0 else (fr.y0, fr.y1)
        # the frame edge whose value is the axis minimum
        e = min((lo, hi), key=lambda z: np.polyval(A, z))
        v = np.polyval(A, e); step = abs(A[0]) * 5
        vr = round(v / step) * step if abs(v) < step else v
        A = np.array([A[0], A[1] + (vr - v)])
        out.append((c, A, *rest))
    return out
