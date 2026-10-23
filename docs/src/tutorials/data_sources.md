# Data sources

Where every external input of Newtrinos.jl comes from. Each resource is catalogued in the
[Newtrinos Index](https://newtrinos-org.github.io/newtrinos-index/), with its licence, all known copies, and a data
card describing its intended use and limitations; the comparisons with the published results are on the
[verifications site](https://newtrinos-org.github.io/newtrinos-verifications/).

"Extracted" means the Newtrinos developers read the numbers off published figures (vector paths where possible, scripts
next to the data); single numbers quoted in papers are cited in the module source only.

## Experiments

| Module | Source | How obtained | Index |
|:-------|:-------|:-------------|:------|
| `deepcore` (8 y) | IceCube oscNext verification sample, [doi:10.7910/DVN/B4RITM](https://doi.org/10.7910/DVN/B4RITM) | official release | [record](https://newtrinos-org.github.io/newtrinos-index/records/icecube-deepcore-8y-verification-sample/) |
| `deepcore_3y` | IceCube 3-year high-statistics sample B, [doi:10.21234/ac23-ra43](https://doi.org/10.21234/ac23-ra43) | official release | [record](https://newtrinos-org.github.io/newtrinos-index/records/icecube-deepcore-3y-highstats/) |
| `ic_upgrade` | IceCube Upgrade neutrino MC, [doi:10.21234/qfz1-yh02](https://doi.org/10.21234/qfz1-yh02) | official release (simulation) | [record](https://newtrinos-org.github.io/newtrinos-index/records/icecube-upgrade-mc-2020/) |
| `super_k` | Super-K I–V atmospheric release, [doi:10.5281/zenodo.8401262](https://doi.org/10.5281/zenodo.8401262); energy-scale priors from T. Wester, PhD thesis, Boston U. (2023), [open.bu.edu/handle/2144/46427](https://open.bu.edu/handle/2144/46427), Table 5.6 | official release + thesis tables | [record](https://newtrinos-org.github.io/newtrinos-index/records/super-k-atm-2023-data-release/) |
| `orca` | KM3NeT/ORCA6 open data, [opendata.km3net.de](https://opendata.km3net.de/dataset.xhtml?persistentId=doi:10.5072/FK2/Y0UXVW) (its identifier is a DataCite test DOI that does not resolve) | official release | [record](https://newtrinos-org.github.io/newtrinos-index/records/km3net-orca6-open-data/) |
| `t2k` | T2K 2025 release figures, [doi:10.5281/zenodo.15701867](https://doi.org/10.5281/zenodo.15701867); events and contours from arXiv:2303.03222 and arXiv:2506.05889 | extracted from the release PDFs and paper figures | [release](https://newtrinos-org.github.io/newtrinos-index/records/t2k-2025-data-release/), [extraction](https://newtrinos-org.github.io/newtrinos-index/records/newtrinos-t2k-figure-data/) |
| `nova` | NOvA 2024 release, [doi:10.5281/zenodo.17822358](https://doi.org/10.5281/zenodo.17822358) | official release | [record](https://newtrinos-org.github.io/newtrinos-index/records/nova-2024-data-release/) |
| `minos` | MINOS/MINOS+ sterile-search release, ancillary files of [arXiv:1710.06488](https://arxiv.org/abs/1710.06488) | official release | [record](https://newtrinos-org.github.io/newtrinos-index/records/minos-sterile-2019-data-release/) |
| `dayabay` | Daya Bay 3158-day supplement, [arXiv:2211.14988](https://arxiv.org/abs/2211.14988); reactor-spectrum correlations from Fig. 29 of [arXiv:1607.05378](https://arxiv.org/abs/1607.05378) | official supplement; correlation matrix read off the figure | [record](https://newtrinos-org.github.io/newtrinos-index/records/daya-bay-3158d-supplemental/) |
| `dayabay_rewrite` | Daya Bay full data release, [doi:10.5281/zenodo.17587229](https://doi.org/10.5281/zenodo.17587229) | official release (shipped in `dayabay_data/`) | [record](https://newtrinos-org.github.io/newtrinos-index/records/daya-bay-full-data-release/) |
| `kamland` | KamLAND PRD 88, 033001 (2013) Figs. 3, 6 and PRD 83, 052002 (2011) Fig. 1; reactors from IAEA PRIS | extracted (vector paths); reactor database compiled | [2013](https://newtrinos-org.github.io/newtrinos-index/records/newtrinos-kamland-2013-periods/), [2011](https://newtrinos-org.github.io/newtrinos-index/records/newtrinos-kamland-2011-spectrum/), [reactors](https://newtrinos-org.github.io/newtrinos-index/records/newtrinos-reactor-database/) |
| `juno`, `tao` | JUNO, Chin. Phys. C 46, 123001 (2022), arXiv:2204.13249 | digitised curves; individual figures not recorded (see `src/experiments/juno/README.md`) | — |
| `chlorine`, `gallex_gno`, `sage`, `gerda`, `katrin` | values quoted in the papers cited in each module | — | — |
| `sno` | SNO, PRC 88, 025501 (2013), Tables XXIV–XXVI | transcribed | [record](https://newtrinos-org.github.io/newtrinos-index/records/sno-three-phase-2013-tables/) |
| `sk1_solar` … `sk4_solar`, `sk_solar` | Super-K solar papers (SK-I…SK-IV), appendix tables | transcribed | [record](https://newtrinos-org.github.io/newtrinos-index/records/super-k-solar-i-iv-spectra/) |
| `sk_solar_dn` | Super-K PRD 109, 092001, A_DN(Δm²₂₁) figures | extracted | [record](https://newtrinos-org.github.io/newtrinos-index/records/newtrinos-sk-solar-dn-amplitude/) |
| `borexino_ph1`, `borexino_ph2` | rates quoted in the Borexino papers; Phase-II correlations as emulated by Super-K (PRD 109, 092001, Sec. VII D) | — | — |
| `borexino_ph1_spectrum` | Borexino PRL 107, 141302 Fig. 1 and PRD 82, 033006 Fig. 7 | extracted | [record](https://newtrinos-org.github.io/newtrinos-index/records/newtrinos-borexino-ph1-spectra/) |
| `borexino_ph2_spectrum` | Borexino open data (Nature 562, 505 Fig. 2a), fit components of PRD 100, 082004 Fig. 6a, MC shapes of arXiv:2005.12829 | collaboration web site + extracted | [data](https://newtrinos-org.github.io/newtrinos-index/records/borexino-phase2-nature2018-data/), [Fig. 6a](https://newtrinos-org.github.io/newtrinos-index/records/newtrinos-borexino-ph2-fig6a-components/) |
| `borexino_ph3` | Borexino open data (PRL 129, 252701) and MC shapes of arXiv:2005.12829 | collaboration web site | [record](https://newtrinos-org.github.io/newtrinos-index/records/borexino-phase3-cno-data/) |
| `coherent_csi` | COHERENT CsI[Na] full-dataset release, ancillary files of [arXiv:2110.07730](https://arxiv.org/abs/2110.07730) (PRL 129, 081801 (2022)) | official release | [record](https://newtrinos-org.github.io/newtrinos-index/records/coherent-csi-2021-data-release/) |
| `coherent_lAr` | COHERENT CENNS-10 Analysis A release, [doi:10.5281/zenodo.3903810](https://doi.org/10.5281/zenodo.3903810) (PRL 126, 012002) | official release | [record](https://newtrinos-org.github.io/newtrinos-index/records/coherent-cenns10-lar-data-release/) |

## Physics modules

| Module | Source | Index |
|:-------|:-------|:------|
| `atm_flux` (HKKM) | Honda et al., PRD 92, 023004 (2015), tables from [icrr.u-tokyo.ac.jp/~mhonda](https://www.icrr.u-tokyo.ac.jp/~mhonda/) | [record](https://newtrinos-org.github.io/newtrinos-index/records/honda-atmospheric-flux-tables/) |
| `atm_flux` (Barr) | Barr et al., PRD 74, 094009 (2006); digitised curves in `src/physics/barr2006/` | [record](https://newtrinos-org.github.io/newtrinos-index/records/barr-2006-atm-flux-uncertainties/) |
| `solar_flux` | B23 solar models, [doi:10.5281/zenodo.10822316](https://doi.org/10.5281/zenodo.10822316) (v1.2); spectra from Bahcall's SNdata pages and Winter et al., PRC 73, 025503 (⁸B default) | [B23](https://newtrinos-org.github.io/newtrinos-index/records/b23-standard-solar-models/), [Bahcall](https://newtrinos-org.github.io/newtrinos-index/records/bahcall-sndata/), [Winter](https://newtrinos-org.github.io/newtrinos-index/records/winter-b8-spectrum/) |
| `solar_xsec` | Cl and Ga cross sections from Bahcall's SNdata pages; ν–e scattering from Bahcall, Kamionkowski & Sirlin, PRD 51, 6146 (formulae) | [record](https://newtrinos-org.github.io/newtrinos-index/records/bahcall-sndata/) |
| `reactor_flux` | Huber, PRC 84, 024617 (+ erratum) and Mueller et al., PRC 83, 054615 (coefficients); Bugey-4 normalisation (Declaïs et al. 1994); energy per fission from Ma et al., PRC 88, 014605; reactors from IAEA PRIS | [HM](https://newtrinos-org.github.io/newtrinos-index/records/huber-mueller-reactor-spectra/), [reactors](https://newtrinos-org.github.io/newtrinos-index/records/newtrinos-reactor-database/) |
| `dayabay_flux` | Daya Bay full data release (reactor parameters, HM spectra, corrections) | [record](https://newtrinos-org.github.io/newtrinos-index/records/daya-bay-full-data-release/) |
| `ibd_xsec` | Strumia & Vissani, PLB 564, 42 (table and Eq. 25); Vogel & Beacom, PRD 60, 053003 (formulae) | [record](https://newtrinos-org.github.io/newtrinos-index/records/strumia-vissani-ibd-xsec/) |
| `sns_flux` | SNS flux tables of the COHERENT CsI release (`snsFlux2D.root`, converted to CSV) | [record](https://newtrinos-org.github.io/newtrinos-index/records/coherent-csi-2021-data-release/) |
| `xsec` | GENIE cross-section splines (`genie_xsec` on CVMFS, extracted with `xsec_data/extract_genie_splines.py`); NEUT 5.4.0 on water digitised from T. Wester's thesis, Fig. 4.7 | [record](https://newtrinos-org.github.io/newtrinos-index/records/genie-cross-section-splines/) |
| `earth_layers` | PREM, Dziewonski & Anderson (1981), via the EarthScope (IRIS) EMC | [record](https://newtrinos-org.github.io/newtrinos-index/records/prem/) |
