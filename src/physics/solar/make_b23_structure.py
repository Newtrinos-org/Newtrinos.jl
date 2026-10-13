"""Write the B23 structure tables b23_<composition>_structure.csv from the Zenodo package.

Usage: python3 make_b23_structure.py <path to the unpacked SolarModels_ICE2023_v1.2.zip>

Source: Y. Herrera and A. Serenelli, Standard Solar Models B23 / SF-III, v1.2,
doi:10.5281/zenodo.10822316 (concept doi:10.5281/zenodo.10174170), CC-BY-4.0.
The tables keep every second radial point of the package (step 1e-3 R_sun) from the centre to the surface.
"""
import os
import sys

import numpy as np

COMPOSITIONS = ["GS98", "AGSS09", "C11", "AAG21", "MB22m", "MB22p"]
SOURCES = ["pp", "pep", "hep", "7Be", "8B", "13N", "15O", "17F"]
NAMES = ["pp", "pep", "hep", "be7", "b8", "n13", "o15", "f17"]


def main(package):
    for comp in COMPOSITIONS:
        path = os.path.join(package, "SolarModels", f"struct+nu_SF3_{comp}.dat")
        with open(path) as f:
            cols = f.readline().strip("# \n").split()
        data = np.loadtxt(path)[::2]
        col = lambda name: data[:, cols.index(name)]
        r = col("R_sun")
        assert abs(r[1] - r[0] - 1e-3) < 1e-9 and abs(r[-1] - 1.0) < 1e-9
        table = [r, 10 ** col("log_ne"), 10 ** col("logRho"), col("H1")]
        # production distributions: clip the round-off below zero at the centre, normalise per R_sun
        for s in SOURCES:
            d = np.clip(col("nu_" + s), 0.0, None)
            table.append(d / np.trapezoid(d, r))
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"b23_{comp}_structure.csv")
        with open(out, "w") as f:
            f.write(f"# B23 standard solar model (Herrera & Serenelli 2023, v1.2, doi:10.5281/zenodo.10822316, CC-BY-4.0), "
                    f"composition {comp}; written by make_b23_structure.py\n")
            f.write("# r [R_sun], n_e [N_A/cm^3], rho [g/cm^3], X (H mass fraction), "
                    "production distributions dN/dr (normalised per R_sun) per component\n")
            f.write(",".join(["r", "ne", "rho", "X"] + NAMES) + "\n")
            for row in np.column_stack(table):
                f.write(",".join(f"{x:.6e}" for x in row) + "\n")
        print(out, len(r))


if __name__ == "__main__":
    main(sys.argv[1])
