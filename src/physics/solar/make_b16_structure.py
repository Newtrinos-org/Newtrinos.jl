"""Write the B16 tables b16_<composition>_structure.csv and b16_<composition>_fluxes.dat (B23 formats).

Usage: python3 make_b16_structure.py <directory with the B16 files>

Source: N. Vinyoles et al., "A new generation of standard solar models", ApJ 835, 202 (2017), arXiv:1611.09867;
files of the Barcelona solar-model page (www.ice.csic.es/personal/aldos/Solar_Data.html, offline since 2023):
- struct_b16_{gs98,agss09}.dat: structure (radius, density, H mass fraction), from the Internet Archive (2022-07-01);
- nudistr_b16_{gs98,agss09}.dat: electron density and neutrino production per zone up to 0.5 R_sun, and
  fluxes_b16.dat: fluxes and fractional 1 sigma uncertainties, as redistributed with PEANUTS (Data/).
The B16 flux correlations were published only on that page and are not archived: the correlation matrix of the B23
model of the same composition (b23_<composition>_fluxes.dat, same input physics apart from the nuclear rates) is used.
Radial grid: 0 to 1 R_sun in steps of 1e-3 (as B23). n_e is taken from nudistr up to 0.5 R_sun and computed as
rho (1 + X) / 2 [N_A/cm^3] (full ionisation) further out, where no neutrinos are produced.
"""
import os
import sys

import numpy as np

trapezoid = getattr(np, "trapezoid", None) or np.trapz   # numpy < 2

COMPOSITIONS = {"GS98": ("gs98", 0), "AGSS09": ("agss09", 1)}   # file tag, column pair in fluxes_b16.dat
NAMES = ["pp", "pep", "hep", "be7", "b8", "n13", "o15", "f17"]
SCALE = [1e10, 1e8, 1e3, 1e9, 1e6, 1e8, 1e8, 1e6]                # units of fluxes_b16.dat
HERE = os.path.dirname(os.path.abspath(__file__))


def main(src):
    rows = [l.split() for l in open(os.path.join(src, "fluxes_b16.dat")) if l.strip() and not l.lstrip().startswith("#")]
    rows = [r for r in rows if len(r) == 4]
    assert len(rows) == 8, rows
    r_grid = np.round(np.arange(0, 1.0005, 1e-3), 6)
    for comp, (tag, k) in COMPOSITIONS.items():
        # structure: Mass Radius Temp Rho Pres Lumi H1 ...
        st = np.loadtxt(os.path.join(src, f"struct_b16_{tag}.dat"), comments="#")
        rs, rho, X = st[:, 1], st[:, 3], st[:, 6]
        # production: radius, T, log10 n_e, mass, fractions per zone (pp pep hep 7Be 8B 13N 15O 17F)
        nd = np.loadtxt(os.path.join(src, f"nudistr_b16_{tag}.dat"), comments="#")
        rn, ne_n, frac = nd[:, 0], 10 ** nd[:, 2], nd[:, 4:12]
        dr = np.diff(np.concatenate([[0.0], rn]))
        ne_struct = np.interp(r_grid, rs, rho * (1 + X) / 2)
        ne = np.where(r_grid <= rn[-1], np.interp(r_grid, rn, ne_n), ne_struct)
        jump = abs(np.interp(rn[-1], rs, rho * (1 + X) / 2) / ne_n[-1] - 1)
        table = [r_grid, ne, np.interp(r_grid, rs, rho), np.interp(r_grid, rs, X)]
        for j in range(8):
            d = np.interp(r_grid, np.concatenate([[0.0], rn]), np.concatenate([[0.0], frac[:, j] / dr]), right=0.0)
            table.append(d / trapezoid(d, r_grid))
        out = os.path.join(HERE, f"b16_{comp}_structure.csv")
        with open(out, "w") as f:
            f.write(f"# B16 standard solar model (Vinyoles et al. 2017, arXiv:1611.09867), composition {comp}; "
                    "written by make_b16_structure.py\n")
            f.write("# r [R_sun], n_e [N_A/cm^3], rho [g/cm^3], X (H mass fraction), "
                    "production distributions dN/dr (normalised per R_sun) per component\n")
            f.write(",".join(["r", "ne", "rho", "X"] + NAMES) + "\n")
            for row in np.column_stack(table):
                f.write(",".join(f"{x:.6e}" for x in row) + "\n")
        # fluxes: B23 layout, B16 values and errors, B23 correlation matrix of the same composition
        flux = [float(r[2 * k]) * s for r, s in zip(rows, SCALE)]
        err = [float(r[2 * k + 1]) for r in rows]
        b23 = open(os.path.join(HERE, f"b23_{comp}_fluxes.dat")).read().splitlines()
        corr = [l for l in b23 if l.strip().startswith(("+", "-")) and not l.strip().startswith("--")]
        assert len(corr) == 8
        with open(os.path.join(HERE, f"b16_{comp}_fluxes.dat"), "w") as f:
            f.write(f"Neutrino fluxes - B16 {comp} (arXiv:1611.09867); correlations of B23 {comp} (not archived for B16)\n")
            f.write("   pp        pep       hep       7Be       8B        13N       15O       17F    \n")
            f.write(" ".join(f"{x:.3e}" for x in flux) + "   Flux\n")
            f.write(" ".join(f"{x:.3e}" for x in err) + "  Fract.Err\n\n")
            f.write("-" * 80 + "\n\n")
            f.write("\n".join(corr) + "\n")
        print(out, f"n_e mismatch structure vs nudistr at {rn[-1]} R_sun: {jump:.2%}")


if __name__ == "__main__":
    main(sys.argv[1])
