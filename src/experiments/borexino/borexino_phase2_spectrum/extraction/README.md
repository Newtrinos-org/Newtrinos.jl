# Extraction of the Borexino Phase-II fit components

The component curves of Borexino's Phase-II spectral fit are read from the vector graphics of Fig. 6a of
arXiv:1707.09279 (Agostini et al., Phys. Rev. D 100, 082004 (2019); `Figure6a.pdf` in the arXiv source):

```bash
mutool trace Figure6a.pdf > fig6a_trace.xml
python3 extract_fig6a.py fig6a_trace.xml   # stroke paths -> curves in (N_h, rate), axis calibration from the tick marks
python3 build_table.py                     # rate per 1-N_h bin and component (vertex at x ≈ N + 0.248 is bin N + 1)
python3 make_table.py                      # ../data/phase2_fig6a_components.csv
```

The extracted total fit reproduces Borexino's best-fit model of the Nature 2018 Fig. 2a data (data − residual × error)
to 10⁻⁴ relative.
