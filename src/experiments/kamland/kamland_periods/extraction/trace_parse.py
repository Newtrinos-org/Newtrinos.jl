"""Parse `mutool trace` XML of a ROOT vector figure: stroke paths (device coords, y up) and text labels."""
import re, sys, xml.etree.ElementTree as ET
from collections import defaultdict

def tf(m, x, y):
    a, b, c, d, e, f = m
    return a * x + c * y + e, b * x + d * y + f

def parse(path):
    root = ET.parse(path).getroot()
    page = root.find('page'); H = float(page.get('mediabox').split()[3])
    paths, texts = [], []
    for el in root.iter():
        if el.tag == 'stroke_path':
            m = list(map(float, el.get('transform').split()))
            segs, cur = [], []
            for c in el:
                if c.tag == 'moveto':
                    if len(cur) > 1: segs.append(cur)
                    cur = [tf(m, float(c.get('x')), float(c.get('y')))]
                elif c.tag == 'lineto':
                    cur.append(tf(m, float(c.get('x')), float(c.get('y'))))
            if len(cur) > 1: segs.append(cur)
            paths.append(dict(color=el.get('color'), lw=float(el.get('linewidth', 1)), dash=el.get('dash'), segs=segs))
        elif el.tag == 'fill_text':
            m = list(map(float, el.get('transform').split()))
            for sp in el.findall('span'):
                size = float(sp.get('trm').split()[0])
                gs = sp.findall('g')
                s = ''.join(g.get('unicode') for g in gs)
                x0 = float(gs[0].get('x')); x1 = float(gs[-1].get('x')) + float(gs[-1].get('adv')) * size
                texts.append(dict(s=s, p0=tf(m, x0, 0), p1=tf(m, x1, 0), size=size))
    return paths, texts, H

if __name__ == '__main__':
    paths, texts, H = parse(sys.argv[1])
    for t in texts: print('TEXT %-12r (%.1f,%.1f)-(%.1f,%.1f) size %.1f' % (t['s'], *t['p0'], *t['p1'], t['size']))
    g = defaultdict(lambda: [0, 0, [1e9, 1e9, -1e9, -1e9]])
    for p in paths:
        k = (p['color'], p['lw'], p['dash'])
        for s in p['segs']:
            g[k][0] += 1; g[k][1] += len(s)
            for x, y in s:
                b = g[k][2]; b[0] = min(b[0], x); b[1] = min(b[1], y); b[2] = max(b[2], x); b[3] = max(b[3], y)
    for k, (n, npt, b) in sorted(g.items(), key=lambda kv: -kv[1][1]):
        print('PATH color=%-12s lw=%.3f dash=%s: %d segs, %d pts, bbox x %.1f–%.1f y %.1f–%.1f' % (k[0], k[1], k[2], n, npt, *b))
