"""Collect the extracted T2K inputs (JSON from the extract_*.py / digitize_*.py scripts) into one HDF5 file.

    /predictions/<sample>/{oscillated,unoscillated}/{edges,values}     zenodo 15701867 (CC-BY 4.0)
    /predictions/<sample>/breakdown/<mode>/{edges,values}              same, oscillated, by NEUT mode
    /data/events/<sample>/{x,theta}  attrs: x = "Erec" [GeV] or "p" [MeV/c], theta [deg]   arXiv:2303.03222 Fig. 'oa:dist'
    /data/fhcnumucc1pi/{lo,hi,count}                                   arXiv:2506.05889 Fig. 1
    /official/dcp/<NO|IO>/{dcp,dchi2}                                  arXiv:2506.05889 Fig. 3 (vector)
    /official/th23dm2/<NO|IO>/<068|090|997>/{ssth23,dm2}               arXiv:2506.05889 Fig. 4 (digitised raster)
"""
import json, sys, h5py, numpy as np
rel = json.load(open("t2k_release_extracted.json")); ev = json.load(open("t2k_data_events.json"))
cc1pi = json.load(open("t2k_cc1pi_data.json")); dcp = json.load(open("t2k_official_dcp.json")); th = json.load(open("t2k_official_th23dm2.json"))
with h5py.File(sys.argv[1], "w") as f:
    f.attrs["description"] = __doc__
    f.attrs["reference_point"] = "sin2th23=0.561 sin2th13=0.022 sin2th12=0.307 dm2_32=2.49e-3 dm2_21=7.53e-5 dCP=-1.601 NO (PDG2021, arXiv:2506.05889)"
    for s, v in rel.items():
        g = f.create_group(f"predictions/{s}")
        for k, name in (("Oscillated", "oscillated"), ("Unoscillated", "unoscillated")):
            g[f"{name}/edges"] = v[k]["edges"]; g[f"{name}/values"] = v[k]["values"]
        for m, h in v["breakdown"].items():
            g[f"breakdown/{m}/edges"] = h["edges"]; g[f"breakdown/{m}/values"] = h["values"]
    names = {"numu1R": "fhc1rmu", "numubar1R": "rhc1rmu", "nue1R": "fhc1re", "nuebar1R": "rhc1re", "nue1RD": "fhc1re1de"}
    for s, v in ev.items():
        e = np.array(v["events"]); g = f.create_group(f"data/events/{names[s]}")
        g["x"] = e[:, 0]; g["theta"] = e[:, 1]; g.attrs["x"] = v["x"]
    g = f.create_group("data/fhcnumucc1pi")
    for k in ("lo", "hi"): g[k] = [r[k] for r in cc1pi]
    g["count"] = [int(round(r["count"])) for r in cc1pi]
    off = min(min(dcp[m]["dchi2"]) for m in dcp)        # axis-label offset: global minimum is 0
    for m, v in dcp.items():
        f[f"official/dcp/{m}/dcp"] = v["dcp"]; f[f"official/dcp/{m}/dchi2"] = np.array(v["dchi2"]) - off
    for m, lv in th.items():
        for l, p in lv.items():
            p = np.array(p); f[f"official/th23dm2/{m}/{l}/ssth23"] = p[:, 0]; f[f"official/th23dm2/{m}/{l}/dm2"] = p[:, 1] * 1e-3
print("written", sys.argv[1])
