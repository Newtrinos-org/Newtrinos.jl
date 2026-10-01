# Extraction of the T2K inputs into `T2K_2025.h5`

T2K publishes no far-detector spectra in machine-readable form. All inputs of `t2k.jl` are taken
from vector PDFs (ROOT output), where histograms are step polylines and data events are individual
filled circles, so values are recovered exactly up to the axis calibration (≲0.1% of the axis range).

| content | source | script |
|---|---|---|
| oscillated / unoscillated predictions, 6 SK samples | data release [zenodo 15701867](https://zenodo.org/records/15701867) (`*_oscunosc.pdf`, CC-BY 4.0) | `extract_release.py` |
| oscillated breakdown by NEUT mode (after ND fit) | same (`*_mcprediction_AL.pdf`); per-mode totals reproduce the legend values to ≤0.3% | `extract_release.py` |
| data events (E_rec, θ) μ-like and (p, θ) e-like | arXiv:2303.03222, Fig. 'oa:dist' (`Data_over_Asimov…_comb.pdf`); event counts 94 / 16 / 14 (e-like) match the paper | `extract_data.py` |
| νμCC1π⁺ data (binned) | arXiv:2506.05889, Fig. 1 (`numucc1pi_twk.pdf`) | `extract_cc1pi.py` |
| official Δχ²(δCP), NO and IO | arXiv:2506.05889, Fig. 3 (`contour_dCP_wRC.pdf`, vector) | `extract_contours.py` |
| official sin²θ₂₃–Δm² regions | arXiv:2506.05889, Fig. 4 (`contour_dm2_32_th23_wRC.pdf`, raster: digitised by colour and line style) | `digitize_th23dm2.py` |

Run (requires `pymupdf`, `numpy`, `scipy`, `pillow`, `h5py`) in a directory containing the unpacked
zenodo archive (`t2k-osc-with-new-sk-cc1pi-samples/`), the arXiv source of 2303.03222 (`arxiv2023/`) and
of 2506.05889 (`arxiv/`):

```bash
python extract_release.py && python extract_data.py && python extract_cc1pi.py
python extract_contours.py && python -c "import pymupdf; d = pymupdf.open('arxiv/contour_dm2_32_th23_wRC.pdf'); x = d.extract_image(d[0].get_images()[0][0]); open('th23dm2.png', 'wb').write(x['image'])"
python digitize_th23dm2.py && python build_hdf5.py T2K_2025.h5
```

The μ-like data figure shows E_rec < 3 GeV only (285 of 318 FHC and 119 of 137 RHC 1Rμ events);
the release predictions for these samples also end at 3 GeV.
