# Daya Bay unfolded reactor antineutrino spectra

Source: Daya Bay Collaboration, "Antineutrino energy spectrum unfolding based on the Daya Bay measurement and its
applications", Chin. Phys. C 45, 073001 (2021), arXiv:2102.04614, ancillary files (v3):

- `UnfoldAntineutrinoSpectra.txt`: unfolded ²³⁵U, Pu combo (²³⁹Pu + 0.183 ²⁴¹Pu) and total antineutrino energy spectra
  (Wiener-SVD, 1958 days, near halls), with and without the IBD cross section, per fission and MeV, 25 bins from 1.8 to
  9.5 MeV, with their smearing matrices A_c and Cov^Wiener.
- `CovMatrix_UnfoldAntineutrinoSpectra.txt`: covariance (75×75, ²³⁵U, Pu combo, total; IBD-weighted) including the
  bias from the unfolding input model.

Average effective fission fractions of the near halls: ²³⁵U : ²³⁸U : ²³⁹Pu : ²⁴¹Pu = 0.564 : 0.076 : 0.304 : 0.056.
Used by `reactor_flux.DayaBayUnfolded` (prediction formula, Eq. 9 of the paper).
