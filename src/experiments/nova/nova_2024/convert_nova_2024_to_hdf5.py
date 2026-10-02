"""One-time conversion of the NOvA 2024 data release (doi:10.5281/zenodo.17822358, PRL 136, 011802)
ROOT files to a single HDF5 file.

Layout of the output (all names as in the ROOT files, ";1" cycles dropped, spaces -> "_"):
  /data/<hist>                      observed spectra (TH1D)
  /predictions/<directory>/<hist>   prediction components per sample (TH1D)
  /contours/<file stem>/<graph>     official 2D credible-interval contours (TGraph); see README
  /nue_component_to_data_bin        index map for the nue samples (see below)
The nue background components (and Total_pred) use a compressed 21-bin analysis-bin layout,
while the data and the Signal histograms use a 23-bin layout. Total_pred - sum(backgrounds)
equals Signal shifted by +1 bin (low-PID core) or +2 bins (high-PID core, peripheral):
`nue_component_to_data_bin[i]` is the (0-based) 23-bin index of 21-bin component bin i
(-1 for bins that are empty in both layouts).
Each TH1D is a group with datasets `edges`, `values`, `variances` (sum of w^2) and attributes
`title`, `xlabel`, `ylabel`; each TGraph a group with datasets `x`, `y`; a TMarker (best fit) is
stored as attributes `x`, `y` on a group. Every group records `source_file` and `source_key`.

    python convert_nova_2024_to_hdf5.py <dir with the release files> <output.h5>
"""
import glob
import os
import sys

import h5py
import numpy as np
import uproot


def clean(name):
    return name.split(";")[0].replace(" ", "_")


def write_th1(group, obj):
    group.create_dataset("edges", data=obj.axis().edges())
    group.create_dataset("values", data=obj.values())
    group.create_dataset("variances", data=obj.variances())
    group.attrs["title"] = obj.member("fTitle")
    group.attrs["xlabel"] = obj.axis().member("fTitle")
    group.attrs["ylabel"] = obj.member("fYaxis").member("fTitle")


def write_tgraph(group, obj):
    x, y = obj.values()
    group.create_dataset("x", data=np.asarray(x))
    group.create_dataset("y", data=np.asarray(y))
    group.attrs["title"] = obj.member("fTitle")


def convert(indir, outfile):
    with h5py.File(outfile, "w") as out:
        out.attrs["description"] = "NOvA 2024 data release (doi:10.5281/zenodo.17822358), converted from ROOT by convert_nova_2024_to_hdf5.py"
        out.attrs["README"] = open(os.path.join(indir, "README.md")).read()
        cmap = np.full(21, -1, dtype=np.int64)
        cmap[1:7] = np.arange(2, 8)      # low-PID core
        cmap[9:15] = np.arange(11, 17)   # high-PID core
        cmap[18] = 20                    # peripheral
        out.create_dataset("nue_component_to_data_bin", data=cmap)
        for fn in sorted(glob.glob(os.path.join(indir, "*.root"))):
            stem = os.path.basename(fn)[: -len(".root")]
            if "data_histograms" in stem:
                base = out.require_group("data")
            elif "prediction" in stem:
                base = out.require_group("predictions")
            elif stem.startswith("contours_"):
                base = out.require_group("contours").require_group(stem.replace("contours_", ""))
            else:
                raise ValueError(f"unexpected file {fn}")
            f = uproot.open(fn)
            for key, cls in f.classnames().items():
                if cls == "TDirectory":
                    continue
                parts = [clean(p) for p in key.split("/")]
                g = base
                for p in parts[:-1]:
                    g = g.require_group(p)
                obj = f[key]
                if cls.startswith("TH2"):
                    continue  # plot frames of the official figures, no content
                h = g.create_group(parts[-1])
                h.attrs["source_file"] = os.path.basename(fn)
                h.attrs["source_key"] = key
                if cls == "TH1D":
                    write_th1(h, obj)
                elif cls == "TGraph":
                    write_tgraph(h, obj)
                elif cls == "TMarker":
                    h.attrs["x"] = float(obj.member("fX"))
                    h.attrs["y"] = float(obj.member("fY"))
                else:
                    raise ValueError(f"unhandled class {cls} for {key} in {fn}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
