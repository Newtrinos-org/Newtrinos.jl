"""Write ../data/phase2_fig6a_components.csv from fig6a_binned.json (build_table.py) and the Nature 2018 Fig. 2a data."""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
b = json.load(open(os.path.join(here, 'fig6a_binned.json')))
b = {k: {int(n): v for n, v in d.items()} for k, d in b.items()}
rows = [l.split() for l in open(os.path.join(here, '../data/Nature2018_Fig2a_DATA.txt')).read().splitlines()[4:] if l.strip()]
data = {int(r[1]): r for r in rows}
names = ['C14', 'pileup', 'Po210', 'Kr85', 'Bi210', 'C11', 'ext', 'solar']
with open(os.path.join(here, '../data/phase2_fig6a_components.csv'), 'w') as f:
    f.write('# Borexino Phase II, TFC-subtracted spectrum: fit components extracted from the vector graphics of Fig. 6a of\n'
            '# arXiv:1707.09279 (MC method fit, N_h estimator). Units: counts / (day x 100 t x N_h), per 1-N_h bin.\n'
            '# total = Borexino total fit (matches Nature 2018 Fig. 2a data - residual*error to 1e-4); components are the\n'
            '# smooth drawn curves (sum matches total to <0.5% median below N_h 350, 2-5% low above: ext bkg clipped at 2e-4).\n'
            '# solar = pp + 7Be + pep + CNO + 8B (all drawn in red); ext = external backgrounds (up to 3 overlapping curves).\n'
            '# 0 = below the plotted range (2e-4) or not drawn.\n')
    f.write('N_h,energy_keV,data,error,total,' + ','.join(names) + '\n')
    for n in range(93, 951):
        r = data[n]
        f.write('%d,%s,%s,%s,%.6g,' % (n, r[2], r[4], r[5], b['total'].get(n, 0)) + ','.join('%.6g' % b[k].get(n, 0) for k in names) + '\n')
