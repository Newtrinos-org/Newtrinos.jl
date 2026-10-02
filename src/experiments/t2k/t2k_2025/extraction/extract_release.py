"""Extract the T2K 2025 data-release (zenodo 15701867) histograms from the ROOT vector PDFs."""
import json, glob, os, numpy as np
from collections import defaultdict
from pdfhist import load, segments, step_hist, frame_fix, num

def auto_axes(words, xword="Reconstructed"):
    nums = [w for w in words if num(w[4]) is not None]
    groups = defaultdict(list)
    for w in nums:
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        groups[(0, round(w[1]))].append((cx, num(w[4])))   # same top-y: a row, varying x
        groups[(1, round(w[0]))].append((cy, num(w[4])))   # same left-x: a column, varying y
    def good(g):
        v = np.array(g)
        if len(v) < 3 or len(set(v[:, 1])) < len(v): return None
        A = np.polyfit(v[:, 0], v[:, 1], 1)
        return A if np.max(np.abs(np.polyval(A, v[:, 0]) - v[:, 1])) < 0.02 * np.ptp(v[:, 1]) else None
    cands = sorted(((len(g), k, good(g)) for k, g in groups.items() if good(g) is not None), key=lambda t: -t[0])
    xw = [w for w in words if w[4] == xword][0]
    # x-axis labels: the group closest to the axis title, parallel to it
    def dist(k):
        c, pos = k
        return abs(pos - (xw[1] if c == 0 else xw[0]))
    xk = min([c for c in cands], key=lambda t: dist(t[1]))
    yk = [c for c in cands if c[1][0] != xk[1][0]][0]
    # the varying coordinate of a row (const y) is x (0); of a column (const x) is y (1)
    return (xk[1][0], xk[2]), (yk[1][0], yk[2])

def colour_hists(path, min_items=20):
    pg, dr, words = load(path)
    xm, ym = frame_fix(dr, *auto_axes(words))
    hists = {}
    for d in dr:
        if d.get("color") is None or len(d["items"]) < min_items: continue
        b = step_hist(d, xm, ym)
        hists.setdefault(d["color"], b)
    return hists, dr, words

def legend_names(dr, words, colours):
    """colour -> legend text, via the short legend line of that colour"""
    names = {}
    lines = [(" ".join(w[4] for w in words if abs((w[1] + w[3]) / 2 - y) < 3 and w[0] > x), c) for c in colours
             for d in dr if d.get("color") == c and len(d["items"]) == 1
             for (x, y) in [(max(d["rect"].x0, d["rect"].x1), (d["rect"].y0 + d["rect"].y1) / 2)]]
    for text, c in lines:
        if text.strip(): names[c] = text
    return names

def common_edges(hists, tol=0.004):
    pts = sorted(x for b in hists for lo, hi, _ in b for x in (lo, hi))
    edges = []
    for x in pts:
        if not edges or x - edges[-1][-1] > tol: edges.append([x])
        else: edges[-1].append(x)
    return [float(np.mean(e)) for e in edges]

def on_grid(b, edges):
    c = [(a + z) / 2 for a, z in zip(edges[:-1], edges[1:])]
    return [max(next((v for lo, hi, v in b if lo <= x <= hi), 0.0), 0.0) for x in c]

def nice(edges, step=0.05):
    return [round(round(e / step) * step, 4) for e in edges]

def legend_order(dr, colours):
    """legend entries top to bottom = stacking order bottom to top (THStack legend)"""
    ys = {}
    for c in colours:
        ls = [d for d in dr if d.get("color") == c and len(d["items"]) == 1]
        if ls: ys[c] = min(d["rect"].y0 for d in ls)
    return sorted(colours, key=lambda c: ys.get(c, 1e9))

out = {}
rel = "t2k-osc-with-new-sk-cc1pi-samples"
for f in sorted(glob.glob(f"{rel}/*_oscunosc.pdf")):
    s = os.path.basename(f).replace("_oscunosc.pdf", "")
    h, dr, words = colour_hists(f)
    names = legend_names(dr, words, list(h))
    out.setdefault(s, {})
    E = common_edges(h.values())
    label = {(0.0, 0.0, 1.0): "Oscillated", (1.0, 0.0, 0.0): "Unoscillated"}
    for c, b in h.items():
        out[s][label[c]] = {"edges": nice(E), "values": on_grid(b, E)}
for f in sorted(glob.glob(f"{rel}/*_mcprediction_AL.pdf")):
    s = os.path.basename(f).replace("_mcprediction_AL.pdf", "")
    h, dr, words = colour_hists(f)
    names = legend_names(dr, words, list(h))
    E = common_edges(h.values())
    cum = [(np.array(on_grid(h[c], E)), names[c].split(" :")[0]) for c in legend_order(dr, list(h))]
    assert all(np.all(np.diff([v for v, _ in cum], axis=0) >= -1e-3) for _ in [0]), "stack not monotonic"
    comps, prev = {}, np.zeros(len(E) - 1)
    for vals, name in cum:
        comps[name] = {"edges": nice(E), "values": list(np.maximum(vals - prev, 0))}
        prev = vals
    out[s]["breakdown"] = comps
json.dump(out, open("t2k_release_extracted.json", "w"), indent=1)
for s, v in out.items():
    print(s)
    for k, h in v.items():
        if k == "breakdown":
            for n, hh in h.items(): print("    ", n, len(hh["values"]), round(sum(hh["values"]), 2), end=";")
            print()
        else:
            print("  ", k, len(h["values"]), round(sum(h["values"]), 2), h["edges"][:3], h["edges"][-2:])
