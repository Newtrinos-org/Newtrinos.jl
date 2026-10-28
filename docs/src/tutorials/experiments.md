# Experiments

Newtrinos.jl includes likelihoods for more than 30 neutrino data sets across atmospheric, accelerator, reactor, solar,
CEvNS and neutrino-mass experiments. Every experiment exposes the same interface — `configure()`, `forward_model`,
`params`, `priors`, `assets`, and `plot` — so analysis code written for one experiment transfers directly to any other.
See [Configuring Experiments](configure_experiment.md) for a hands-on walkthrough.

!!! tip "Good starting points"
    - **Daya Bay** (`Newtrinos.dayabay`) for reactor neutrinos — the compact module has no detector systematics, so the
      only free parameters are the oscillation parameters. Great for a first likelihood evaluation.
    - **IceCube DeepCore** (`Newtrinos.deepcore`) for atmospheric neutrinos — the most complete example of the full
      physics stack (oscillations, flux, Earth density, cross-sections).

Each experiment defines its own `default_physics()`; pass a NamedTuple of configured physics modules to `configure` to
use a different model (e.g. a BSM oscillation model or a cross-section model shared between experiments).

---

## Atmospheric Neutrinos

Atmospheric neutrino experiments detect neutrinos produced by cosmic-ray interactions in the upper atmosphere. Because
the neutrinos travel through varying amounts of the Earth depending on their arrival direction, these experiments are
simultaneously sensitive to $\theta_{23}$, $\Delta m^2_{31}$, and Earth-matter effects. They require the full physics
stack: oscillations with matter effects (`SI`), the HKKM atmospheric flux, the PREM Earth density model, and a neutrino
cross-section model.

| Experiment | Module | Dataset |
|:-----------|:-------|:--------|
| IceCube DeepCore | `Newtrinos.deepcore` (= `Newtrinos.deepcore_8y`) | 8-year verification sample |
| IceCube DeepCore | `Newtrinos.deepcore_3y` | 3-year high-statistics sample (sample B) |
| Super-Kamiokande | `Newtrinos.super_k` | Atmospheric 2023 analysis (SK-I–V) |
| KM3NeT/ORCA | `Newtrinos.orca` | 6 detection units, 433 kton-years |
| IceCube Upgrade | `Newtrinos.ic_upgrade` | Simulated upgrade detector |

Required physics modules: `osc` (with `interaction = SI()`), `atm_flux`, `earth_layers`, `xsec`.

---

## Accelerator Neutrinos

Accelerator-based experiments use a $\nu_\mu$ ($\bar\nu_\mu$) beam produced from pion decays at a proton target. The
known beam composition and energy spectrum allow precise measurements of $\theta_{23}$, $\Delta m^2_{31}$, $\delta_{CP}$
and sterile-neutrino mixing.

| Experiment | Module | Dataset |
|:-----------|:-------|:--------|
| T2K | `Newtrinos.t2k` | 2025 data release (FHC/RHC μ-like, e-like and νμ CC1π⁺ samples) |
| NOvA | `Newtrinos.nova` | 2024 data release, 10 years (νμ quartiles, νe core/peripheral/low-energy) |
| MINOS | `Newtrinos.minos` | Sterile neutrino search, $16 \times 10^{20}$ POT |

T2K and NOvA reweight the released predictions with oscillation probabilities averaged over an effective, calibrated
energy response. All reference-point quantities are computed with the module's default (standard 3ν) physics, so the
same released predictions are used whatever oscillation model is configured.

Required physics modules: `osc` (with `interaction = SI()` for T2K and NOvA; sterile models supported for MINOS),
`xsec`.

---

## Reactor Neutrinos

Reactor experiments detect electron antineutrinos ($\bar{\nu}_e$) produced by $\beta$-decay of fission products in
nuclear power plants, via inverse beta decay. They provide the cleanest measurements of $\theta_{13}$ and
$\Delta m^2_{31}$ (km baselines) and of $\Delta m^2_{21}$ (KamLAND, ~180 km).

| Experiment | Module | Dataset |
|:-----------|:-------|:--------|
| Daya Bay | `Newtrinos.dayabay` | 3158 days, published spectra, near/far comparison (compact) |
| Daya Bay | `Newtrinos.dayabay_rewrite` | Full data release: 8 ADs × 3 periods, full reactor and detector model |
| KamLAND | `Newtrinos.kamland` | 2002–2012, three data-taking periods (`dataset = :kamland2011` for the 2002–2009 data) |
| JUNO | `Newtrinos.juno` | Simulated (design sensitivity) |
| TAO | `Newtrinos.tao` | Simulated (design sensitivity) |

- `dayabay_rewrite` follows the official reproduction of the Daya Bay analysis: reactor flux model (`dayabay_flux`:
  Huber–Mueller spectra, non-equilibrium and spent-fuel corrections, free spectrum correction), IBD cross section
  (`ibd_xsec`, Vogel–Beacom), IAV, non-linearity and resolution, backgrounds, and per-detector and per-reactor nuisance
  parameters (181 parameters). The data release ships with the package.
- `kamland` computes the reactor prediction from first principles with `reactor_flux` (IAEA PRIS power histories of all
  Japanese and Korean reactors, Huber–Mueller spectra anchored to Bugey-4) and uses matter effects (`SI`).

!!! note
    JUNO and TAO are simulated experiments with no observed data. Their `forward_model` generates expected event
    spectra against an Asimov dataset, making them useful for sensitivity studies and future projections.

Required physics modules: `osc` (`Vacuum()` for Daya Bay, JUNO, TAO; `SI()` for KamLAND), plus `reactor_flux`
(KamLAND) or `dayabay_flux` and `ibd_xsec` (`dayabay_rewrite`).

---

## Solar Neutrinos

Solar neutrino experiments measure the νe flux from the Sun, converted by the MSW effect in the Sun and partially
regenerated in the Earth at night. They constrain $\theta_{12}$ and $\Delta m^2_{21}$ and the solar fluxes. The solar
modules share the standard solar model (`solar_flux`, B23), the ν–e scattering and capture cross sections
(`solar_xsec`), and the solar oscillation probabilities (`osc.solar_prob`, adiabatic or with non-adiabatic level
crossing, with Earth regeneration through the PREM zones).

| Experiment | Module | Dataset |
|:-----------|:-------|:--------|
| Homestake | `Newtrinos.chlorine` | Final chlorine rate (1998) |
| GALLEX/GNO | `Newtrinos.gallex_gno` | Gallium rates (2010 reanalysis / 2005) |
| SAGE | `Newtrinos.sage` | Gallium rate, 1990–2007 |
| SNO | `Newtrinos.sno` | Combined three-phase analysis (2013): ⁸B flux and survival-probability polynomial |
| Super-Kamiokande | `Newtrinos.sk1_solar` … `Newtrinos.sk4_solar` | ⁸B/hep spectra of SK-I–IV (day/night) |
| Super-Kamiokande | `Newtrinos.sk_solar` | SK-I–IV spectra as one experiment (shared probabilities) |
| Super-Kamiokande | `Newtrinos.sk_solar_dn` | SK day/night amplitude fit, A_DN vs Δm²₂₁ |
| Borexino | `Newtrinos.borexino_ph1`, `borexino_ph2` | Phase-I/II interaction rates (⁷Be, pep, pp, ⁸B) |
| Borexino | `Newtrinos.borexino_ph1_spectrum` | Phase-I ⁷Be and ⁸B spectral fits |
| Borexino | `Newtrinos.borexino_ph2_spectrum` | Phase-II TFC-subtracted spectrum (open data) |
| Borexino | `Newtrinos.borexino_ph3` | Phase-III spectrum (CNO) |

Use either the SK day/night spectra (default) or `configure(physics; daynight = :combined)` together with
`sk_solar_dn`, which carries the day/night information as in SK's own analysis.

Required physics modules: `osc` (`SI()`), `solar_flux`, `solar_xsec`, `earth_layers`.

---

## Coherent Elastic Neutrino-Nucleus Scattering (CEvNS)

The COHERENT experiments at the Spallation Neutron Source (SNS) at Oak Ridge measure Coherent Elastic
neutrino-Nucleus Scattering. They include their own SNS flux model (`sns_flux`) and CEvNS cross-section model
(`cevns_xsec`), and do not depend on any oscillation physics. They are primarily used to constrain the weak mixing
angle, nuclear form factors, and BSM couplings such as NSI.

| Experiment | Module | Dataset |
|:-----------|:-------|:--------|
| COHERENT CsI | `Newtrinos.coherent_csi` | CsI[Na] scintillator detector (2022 data release) |
| COHERENT LAr | `Newtrinos.coherent_lAr` | Liquid argon detector |

Required physics modules: none (self-contained).

---

## Neutrino Mass

| Experiment | Module | Dataset |
|:-----------|:-------|:--------|
| KATRIN | `Newtrinos.katrin` | Effective electron-neutrino mass |
| GERDA | `Newtrinos.gerda` | Neutrinoless double-beta decay half-life (⁷⁶Ge) |

These modules translate the oscillation model's mass spectrum into the measured observables, e.g. for models with
towers of neutrino states.

---

## Validation

All implementations are validated against the publications: the
[Newtrinos verifications site](https://newtrinos-org.github.io/newtrinos-verifications/) shows, for each paper, the
published figures next to the Newtrinos reproduction, with the scripts, pinned Newtrinos commits and caveats.

In addition, the subdirectories under `src/experiments/` contain `test.jl` scripts that run the forward model and
reproduce plots from the corresponding published analysis:

```bash
# Run from the Newtrinos.jl root directory
cd src/experiments/icecube/deepcore_8y_verification_sample
julia --project=../../../.. test.jl
```
