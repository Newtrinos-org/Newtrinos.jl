"""Extract per-process neutrino cross sections from GENIE XML spline files.

Reads the `gxspl-*small.xml` spline lists published on CVMFS
(/cvmfs/larsoft.opensciencegrid.org/products/genie_xsec/<version>/NULL/<tune>/data) and sums the
splines into the process classes used by `Newtrinos.xsec`, per nucleus, flavour and current:

    CC: QE (incl. hyperon/charm QE), MEC, RES, DIS (incl. charm), COH
    NC: QE (elastic), MEC, RES, DIS, COH

ν–e scattering and IMD are skipped (not ν–nucleus). Cross sections are stored per nucleus in
units of 1e-38 cm² (GENIE stores GeV⁻²: 1 GeV⁻² = 0.3893794e-27 cm²), interpolated (linearly between knots) onto a common 0.01–1000 GeV logarithmic grid; above
the third-to-last knot of a tune σ/E is held constant (boundary artefacts).

usage: python -I extract_genie_splines.py OUT.h5 [CVMFS_ROOT]
"""
import sys, os, re
import numpy as np
import h5py

ROOT = sys.argv[2] if len(sys.argv) > 2 else "/cvmfs/larsoft.opensciencegrid.org/products/genie_xsec"

# (label, path below ROOT, file). Tunes whose inclusive cross sections are identical to a listed one are left
# out (the a/b variants differ only in hadronisation: G18_01b, G18_02b, G18_10b_*; G18_10j = G18_10i), as are
# the SBN argon tunes AR25_20i/AR25_22i, which have no ντ splines.
TUNES = [
    ("G2_12_10_Default_MEC",        "v2_12_10/NULL/DefaultPlusMECWithNC",       "gxspl-FNALsmall.xml"),
    ("G2_12_10_Default_ValenciaMEC", "v2_12_10/NULL/DefaultPlusValenciaMEC",    "gxspl-FNALsmall.xml"),
    ("G00_00b_00_000",  "v3_00_04a/NULL/G0000b00000-k250-e1000", "gxspl-FNALsmall.xml"),
    ("G18_01a_00_000",  "v3_00_04a/NULL/G1801a00000-k250-e1000", "gxspl-FNALsmall.xml"),
    ("G18_02a_00_000",  "v3_00_04a/NULL/G1802a00000-k250-e1000", "gxspl-FNALsmall.xml"),
    ("G18_10a_00_000",  "v3_00_04a/NULL/G1810a00000-k250-e1000", "gxspl-FNALsmall.xml"),
    ("G18_10a_02_11a",  "v3_00_04a/NULL/G1810a0211a-k250-e1000", "gxspl-FNALsmall.xml"),
    ("G18_10i_00_000",  "v3_00_04a/NULL/G1810i00000-k250-e1000", "gxspl-FNALsmall.xml"),
    ("G18_10a_02_11b",  "v3_04_00/NULL/G1810a0211b-k250-e1000",  "gxspl-FNALsmall.xml"),
    ("AR23_20i_00_000", "v3_04_00/NULL/AR2320i00000-k250-e1000", "gxspl-FNALsmall.xml"),
]

TARGETS = {1000010010: "H1", 1000060120: "C12", 1000080160: "O16", 1000180400: "Ar40", 1000260560: "Fe56"}
FLAVOURS = {12: "nue", -12: "nuebar", 14: "numu", -14: "numubar", 16: "nutau", -16: "nutaubar"}
PROCS = ("QE", "MEC", "RES", "DIS", "COH")
E_GRID = np.logspace(-2, 3, 301)
GEV2_TO_1E38CM2 = 0.3893794e-27 / 1e-38

name_re = re.compile(r'<spline name="([^"]+)" nknots="(\d+)"')
nu_re = re.compile(r"nu:(-?\d+);tgt:(\d+);")
proc_re = re.compile(r"proc:Weak\[(CC|NC)\],(\w+);")
knot_re = re.compile(r"<E>\s*([-+0-9.eE]+)\s*</E>\s*<xsec>\s*([-+0-9.eE]+)\s*</xsec>")

PROC_MAP = {"QES": "QE", "MEC": "MEC", "RES": "RES", "DIS": "DIS", "COH": "COH"}


def classify(name):
    m = nu_re.search(name)
    if m is None:
        return None
    nu, tgt = int(m.group(1)), int(m.group(2))
    if tgt not in TARGETS or nu not in FLAVOURS:
        return None
    p = proc_re.search(name)
    if p is None:  # ν-e scattering (CC+NC+interference), IMD, ...
        return None
    cur, proc = p.group(1), PROC_MAP.get(p.group(2))
    if proc is None:
        return None
    return FLAVOURS[nu], TARGETS[tgt], cur, proc


def to_grid(E, s):
    E, s = np.asarray(E), np.asarray(s)
    out = np.zeros_like(E_GRID)
    if not (s > 0).any():
        return out
    # linear between the (dense, log-spaced) knots, which also keeps thresholds; above the third-to-last knot σ/E
    # is held constant (the last knots of GENIE spline lists carry boundary artefacts)
    i_end = len(E) - 3
    inside = E_GRID <= E[i_end]
    out[inside] = np.interp(E_GRID[inside], E, s, left=0.0)
    above = ~inside
    out[above] = s[i_end] / E[i_end] * E_GRID[above]
    return out


def extract(path):
    acc = {}
    nsplines = 0
    with open(path, "r", encoding="latin-1") as f:
        key, E, S = None, [], []
        for line in f:
            if "<spline " in line:
                m = name_re.search(line)
                key = classify(m.group(1)) if m else None
                E, S = [], []
            elif key is not None and "<knot>" in line:
                k = knot_re.search(line)
                E.append(float(k.group(1))); S.append(float(k.group(2)) * GEV2_TO_1E38CM2)
            elif "</spline>" in line:
                if key is not None:
                    g = to_grid(E, S)
                    acc[key] = acc.get(key, 0) + g
                    nsplines += 1
                key = None
    return acc, nsplines


def main(out, only=None):
    with h5py.File(out, "w") as h:
        h["E"] = E_GRID
        h.attrs["units"] = "1e-38 cm^2 per nucleus"
        h.attrs["source"] = "GENIE spline lists, " + ROOT
        for label, sub, fname in TUNES if only is None else [t for t in TUNES if t[0] in only]:
            path = os.path.join(ROOT, sub, "data", fname)
            acc, n = extract(path)
            print(f"{label}: {n} splines", flush=True)
            for (flav, tgt, cur, proc), g in acc.items():
                h.create_dataset(f"{label}/{tgt}/{flav}/{cur}/{proc}", data=g.astype(np.float32), compression="gzip", shuffle=True)
            h[label].attrs["source"] = path


if __name__ == "__main__":
    main(sys.argv[1], os.environ.get("ONLY", "").split(",") if os.environ.get("ONLY") else None)
