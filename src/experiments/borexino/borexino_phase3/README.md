# Borexino Phase III: spectral fit of the TFC-subtracted spectrum

Module `borexino_ph3`. Reference: D. Basilico et al. (Borexino), *Final results of Borexino on CNO solar
neutrinos*, Phys. Rev. D 108, 102005 (2023), arXiv:2205.15975.

## Data (`data/`, Borexino open data, https://borex.lngs.infn.it/)

| File | Content |
|---|---|
| `PRL2022_SubtractedSpectrumFit_Data.txt` | TFC-subtracted spectrum, 817 bins (320–2640 keV), counts and residuals of the Borexino fit |
| `*_pdfkeV__*.txt` | Monte Carlo spectral shapes of signals and backgrounds (open data of arXiv:2005.12829) |
| `Phase3Final_BiULLarge_Golden_Energy_Radial_May18_HybridMethod.txt` | Published CNO-rate profile, −2Δln L including systematics (validation only) |
| `calibration.txt` | Output of `calibrate.jl`: energy response and rates fitted to Borexino's best-fit spectrum |

## Model

- Solar ν (pp, ⁷Be, pep, CNO): MC shapes scaled with the zero-threshold interaction rates predicted by
  `solar_flux` × `osc` × `solar_xsec`; ⁸B from the ν–e response with Gaussian resolution.
- Backgrounds ²¹⁰Bi, ¹¹C, ⁸⁵Kr, ²¹⁰Po, external ²¹⁴Bi, ⁴⁰K, ²⁰⁸Tl: free rates `borexino_ph3_*` [cpd/100 t];
  the ²¹⁰Bi rate is normalised at zero threshold, the others to their MC shape above 88 keV.
- Energy response: `E = offset + scale · E_mc` with additional smearing, calibrated on Borexino's best-fit
  spectrum (`calibrate.jl`, deviance 11.4 for 817 bins); nuisances `borexino_ph3_energy_{scale,offset,smear}`
  (1σ = 0.5 %, 5 keV, 5 keV/√MeV) interpolated quadratically between ±1σ templates.
- ²¹⁰Bi upper limit 10.8 ± 1.0 cpd/100 t as a half-Gaussian (`bi210_limit`).
- Exposure 1431.6 d × 71.3 t × 63.97 % (TFC).

## Options

`configure(physics; background_priors = :constrained | :free, background_prior_width = 0.03, bi210_limit = (10.8, 1.0))`

Borexino's fit also uses the TFC-tagged spectrum and the radial distribution, which are not public and constrain
mostly ¹¹C and the external γ's. `:constrained` (default) puts Gaussian priors of relative width
`background_prior_width` on these rates around the calibrated values, which reproduces the published CNO
profile; `:free` leaves them free and gives a wider (conservative) CNO constraint.

## Validation (`test.jl`, oscillations at defaults, ⁷Be and CNO normalisations free)

| | best-fit CNO [cpd/100 t] | −2Δln L at CNO = 4 / 5 / 8.6 / 10 |
|---|---|---|
| Borexino (with systematics) | 6.7 (+2.0/−0.8) | 7.8 / 3.1 / 1.6 / 3.7 |
| `:constrained` | 6.68 | 5.0 / 1.2 / 1.1 / 3.2 |
| `:free` | 6.84 | 2.7 / 0.5 / 0.3 / 1.1 |

Best-fit ⁷Be rate 46.0 cpd/100 t (Phase II: 48.3 ± 1.1 ⁺⁰·⁴₋₀.₇).
