"""Per-bin (N_h) component table from the extracted Fig. 6a curves; vertex at x ≈ N + 0.248 belongs to bin N + 1."""
import json
from collections import defaultdict
c = json.load(open('fig6a_curves_raw.json'))
def binned(chain):
    out = {}
    for x, y in chain:
        k = x - 0.248
        if abs(k - round(k)) < 0.15: out[round(k) + 1] = y
    return out
comp = defaultdict(lambda: defaultdict(float)); cover = defaultdict(lambda: defaultdict(int))
for name, chains in c.items():
    for ch in chains:
        if len(ch) <= 2: continue                    # legend samples
        if name == 'solar' and len(ch) <= 4 and ch[0][1] > 0.05: continue   # arrows
        for n, y in binned(ch).items():
            comp[name][n] += y; cover[name][n] += 1
# overlapping chains of one style (several curves drawn with the same colour)
for name in comp: print(name, 'bins', len(comp[name]), 'max overlap', max(cover[name].values()))
tot = comp.pop('total')
rel = sorted(abs(sum(comp[k].get(n, 0) for k in comp) / tot[n] - 1) for n in tot)
print('sum of components vs total: median %.4f, 90%% %.4f, max %.4f' % (rel[len(rel)//2], rel[int(.9*len(rel))], rel[-1]))
worst = sorted(((abs(sum(comp[k].get(n, 0) for k in comp) / tot[n] - 1), n) for n in tot))[-8:]
for r, n in worst: print('  N_h', n, 'rel', round(r, 3), 'total %.3g' % tot[n], {k: round(comp[k].get(n, 0), 5) for k in comp})
json.dump({k: dict(v) for k, v in list(comp.items()) + [('total', tot)]}, open('fig6a_binned.json', 'w'))
