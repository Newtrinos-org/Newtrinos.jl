#!/usr/bin/env python3
"""
Tabulate daemonflux (Yañez & Fedynitch, Phys. Rev. D 107, 123037 (2023); github.com/mceq-project/daemonflux,
BSD-3-Clause) for Newtrinos.jl: the muon-calibrated atmospheric neutrino flux at the "generic" location and its
derivatives with respect to the 24 daemonflux parameters (hadronic-interaction and GSF cosmic-ray knobs).

daemonflux is linear in its parameters: flux(p) = flux_0 * (1 + sum_k p_k R_k), with p_k in units of the calibrated
1-sigma errors and correlated through the calibration covariance. This script stores, per flavour and on a grid of
(log10 E, cos zenith), the calibrated flux flux_0 [1/(GeV cm^2 s sr)] and the relative derivatives R_k, together with
the parameter correlation matrix.

usage: python make_daemonflux_tables.py <output .h5>   (pip install daemonflux h5py)
"""
import sys
import numpy as np
import h5py
import daemonflux
from daemonflux import Flux

out = sys.argv[1]
fl = Flux(location="generic")  # spline revision fl._revision, calibration "default"
pars = fl.params
names = list(pars.known_parameters)
log10e = np.round(np.arange(-0.95, 4.0001, 0.05), 6)   # 0.112 GeV (lower end of the splines) ... 10 TeV
energy = 10.0 ** log10e
zeniths = fl.zenith_angles                                 # cos zenith = 1, 0.95, ..., 0 (strings)
cz = np.array([np.cos(np.radians(float(z))) for z in zeniths])
order = np.argsort(cz)                                     # increasing cos zenith
flavours = {"numu": "numu", "numubar": "antinumu", "nue": "nue", "nuebar": "antinue"}

with h5py.File(out, "w") as h:
    h.attrs["source"] = "daemonflux %s, location generic, spline revision %s, calibration default" % (
        getattr(daemonflux, "__version__", "?"), getattr(fl, "_revision", "?"))
    h.attrs["reference"] = "J. P. Yanez, A. Fedynitch, Phys. Rev. D 107, 123037 (2023)"
    h["log10_energy"] = log10e
    h["coszen"] = cz[order]
    h["parameters"] = np.array(names, dtype="S")
    err = np.asarray(pars.errors)
    cov = np.asarray(pars.cov)
    h["correlation"] = cov / np.outer(err, err)
    for key, q in flavours.items():
        f0 = np.empty((len(log10e), len(cz)))
        rel = np.empty((len(names), len(log10e), len(cz)))
        for j, z in enumerate(zeniths):
            base = fl.flux(energy, z, q) / energy ** 3
            f0[:, j] = base
            for k, name in enumerate(names):
                rel[k, :, j] = fl.flux(energy, z, q, params={name: 1.0}) / energy ** 3 / base - 1.0
        h[key + "/flux"] = f0[:, order]
        h[key + "/relative_derivative"] = rel[:, :, order]
        print(key, "done", flush=True)
