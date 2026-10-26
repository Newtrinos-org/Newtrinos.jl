# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Run unit tests (~2500 tests, ~8 min)
julia --project -e 'using Pkg; Pkg.test()'

# Run benchmarks
julia --project benchmark/bench_osc.jl
julia --project benchmark/bench_likelihood.jl --experiments dayabay
julia --project benchmark/bench_likelihood.jl --experiments deepcore super_k

# Run an experiment's validation script (from its directory)
cd src/experiments/icecube/deepcore_8y_verification_sample && julia --project=../../../.. test.jl

# Analysis CLI
julia --project src/analysis/analysis.jl --experiments deepcore dayabay --name myrun --task scan
julia --project src/analysis/distributed_profile.jl --experiments deepcore --name myrun --workers 4
```

## Architecture

Newtrinos.jl is a neutrino physics global analysis framework with three orthogonal layers:

### Physics (`src/physics/`)
Theory predictions with no experiment knowledge. Each module returns a struct `<: Newtrinos.Physics` with `params`, `priors`, and callable functions.

- **`osc.jl`** — Core oscillation probability engine. Configurable via `OscillationConfig` with flavour models (`ThreeFlavour`, `Sterile`, `ADD`, `Darkdim_*`, `NND`), interaction models (`Vacuum`, `SI`, `NSI`), and propagation models (`Basic`, `Spray`, `Decoherent`, `Damping`). Performance-critical: uses `SMatrix`/`SVector` for 3-flavour, `eigen` for matter effects.
- **`earth_layers.jl`** — PREM Earth density model. `compute_layers()` → `compute_paths(coszen, layers)`.
- **`atm_flux.jl`** — HKKM atmospheric neutrino fluxes with Barr systematics. Site-specific flux files in `src/physics/*.d`. Alternatively `Daemonflux()` (nominal) and `Daemon()` (systematics): the muon-calibrated daemonflux model, tabulated with its 24 parameter derivatives in `daemonflux/daemonflux_generic.h5` (`make_daemonflux_tables.py`); one vector parameter `atm_flux_daemon` with the calibration correlation as `MvNormal` prior, applied linearly as relative variations to any nominal flux (generic location: no geomagnetic effects, cos θz mirrored).
- **`xsec.jl`** — Cross-section models with one reweighting interface `xsec.weights(E; flavour, anti, current, target, generator, process)` (closures `params -> weights`), so the model is chosen in the physics configuration: `Nominal` (no systematics; T2K/NOvA defaults, which carry their own post-ND parameters), `SimpleScaling` (NC and ντ norms; IceCube/ORCA/MINOS defaults), `H2O_PCA` (Super-K default), and `CorrelatedXsec` (`xsec_correlated.jl`, for global fits): each experiment is reweighted from its simulation's generator (GENIE tune, `:GENIE2_12`, or `:NEUT5_4_0` on water) to one nominal model (default `:tune_mean`, the mean of the GENIE tunes), with shared parameters: PCA modes of the tune spread around that mean, jointly over targets/flavours/processes (`xsec_pc1..K`), M_A^QE (Llewellyn Smith response), process norms, ν̄/ν (low-energy processes and DIS separately), νe/νμ, NC, ντ. Per-process GENIE splines (10 tunes; H, C, O, Ar, Fe; all flavours) in `xsec_data/genie_splines.h5`, extracted from the CVMFS `genie_xsec` spline lists by `xsec_data/extract_genie_splines.py`. T2K and NOvA apply the weights near-detector-constrained (unoscillated νμ rate per reco bin kept).
- **`cevns_xsec.jl`**, **`sns_flux.jl`** — COHERENT-specific physics.
- **`solar_flux.jl`** — Standard solar model (B23, default MB22-met; data in `src/physics/solar/`): fluxes and spectra per component; flux normalisations as one vector `solar_norms` with the correlated B23 `MvNormal` prior (`CorrelatedSSMPriors`, default) or as scalars `solar_norm_*` (`SSMPriors`, for fits that free single components); `solar_b8_shape` (⁸B spectrum: Winter et al. 2006 by default, `b8_spectrum=:bahcall` for Bahcall et al. 1996; shape variation from Bahcall's ±3σ); production regions with the solar density profile (`production`, input to `osc.solar_prob`; B23 structure from the centre to the surface, tables written by `solar/make_b23_structure.py`); `nadir_exposure(latitude)`.
- **`reactor_flux.jl`** — Reactor ν̄e flux and IBD detection shared by reactor experiments: Huber–Mueller isotope spectra, energy per fission, Strumia–Vissani IBD cross section, Bugey-4 rate anchoring (`bugey4_scale`), and a reactor database (`src/physics/reactors/`: IAEA PRIS units of Japan and Korea with thermal power, annual load factors and site coordinates; `load_units`, `daily_power`, `ecef`/`baseline`).
- **`solar_xsec.jl`** — ν–e elastic scattering (with radiative corrections) and ³⁷Cl/⁷¹Ga capture cross sections.
- **`ibd_xsec.jl`**, **`dayabay_flux.jl`** — IBD cross sections (`StrumiaVissani`, `VogelBeacomO1`) and the reactor flux model of the full Daya Bay analysis (Huber–Mueller, non-equilibrium and spent-fuel corrections, free spectrum correction), used by `dayabay_rewrite`.
- **`src/utils/chi2map.jl`** — published Δχ² tables as experiments or penalty terms (`Chi2Map`, `Chi2Penalty`).
- Solar oscillations live in `osc.jl`: `osc.solar_prob(E, production, params)` (day) and `osc.solar_prob(E, production, paths, layers, params)` (Earth regeneration); MSW in the Sun via `compute_matter_matrices`, eigenstates labelled by eigenvalue order; adiabatic unless Parke's level-crossing estimate is non-negligible, then numerical evolution through the solar profile (`solar_transition_matrix`; LOW/SMA region, slow there).
- `earth_layers.compute_chord_paths` gives each path segment its own chord-averaged PREM density; `PREM_discontinuities(continental=true)` (crust instead of PREM's ocean) is the zoning used for solar day/night.

### Experiments (`src/experiments/`)
Each experiment module has `configure(physics=default_physics())` returning a struct `<: Newtrinos.Experiment` with fields: `physics`, `params`, `priors`, `assets`, `forward_model`, `plot`. Each experiment defines its own `default_physics()` with appropriate oscillation config, flux files, and cross-section models.

Experiment groups and their physics requirements:
- **Atmospheric** (deepcore, ic_upgrade, super_k, orca): `osc` (SI), `atm_flux`, `earth_layers`, `xsec`
- **Reactor** (dayabay, juno, tao): `osc` (Vacuum); **dayabay_rewrite** (full data release, 8 ADs × 3 periods, 181 nuisance parameters; data in `daya_bay_3158days/dayabay_data/`): `osc`, `ibd_xsec`, `dayabay_flux`; **kamland** (2013 data, three periods, PRD 88, 033001; `dataset = :kamland2011` for PRD 83, 052002): `osc` (SI, crust) and `reactor_flux` (first-principles reactor prediction from the PRIS power histories)
- **Accelerator** (t2k, nova, minos): `osc`, `xsec`. T2K and NOvA reweight the released predictions with oscillation probabilities averaged over a calibrated energy response; like Super-K's nominal weights, all reference-point quantities are computed with the module's `default_physics()` (standard 3ν, nominal flux and cross sections), never with the configured physics, so BSM or shared models are compared with the same released predictions.
- **Neutrino mass** (katrin, gerda): `osc` (mass spectrum of the flavour model)
- **COHERENT** (coherent_csi, coherent_lAr): self-contained, no physics input
- **Solar** (chlorine, gallex_gno, sage, sno, sk1_solar–sk4_solar, sk_solar (SK-I–IV as one experiment), sk_solar_dn, borexino_ph1–ph3, borexino_ph1_spectrum, borexino_ph2_spectrum): `osc` (SI), `solar_flux`, `solar_xsec`, `earth_layers` (PREM_discontinuities); shared code in `experiments/solar_common/` (`Site` day/night exposure, ES response, capture rates) and `experiments/super_k/sk_solar_common/` (one `SKPhase` per SK phase). SK day/night: either the day/night spectra (default) or `configure(physics; daynight=:combined)` together with `sk_solar_dn` (SK's amplitude-fit A_DN vs Δm²₂₁). Borexino: `borexino_ph1/ph2` published rates, `borexino_ph3` and `borexino_ph2_spectrum` spectral fits (MC solar shapes with calibrated response; Phase II with Borexino's extracted background components).

### Analysis (`src/analysis/`)
Inference tools treating experiments as black boxes.

- **`analysis_tools.jl`** — `NewtrinosResult` type, `find_mle`, `profile`, `scan`, `generate_likelihood`, `get_params`/`get_priors`, `condition`, `generate_asimov_data`, `Wrapper` for parameter aliasing.
- **`molewhacker.jl`** — Adaptive importance sampling (`whack_a_mole`, `whack_many_moles`).
- **`cli_common.jl`** — Shared `configure_experiments()` for CLI scripts.

## Key Patterns

**Combining experiments into a joint likelihood:**
```julia
experiments = (
    deepcore = Newtrinos.deepcore.configure(),       # uses defaults
    dayabay = Newtrinos.dayabay.configure(physics),   # custom physics override
)
params = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
likelihood = Newtrinos.generate_likelihood(experiments)
```

**Start values for joint fits:** `get_params(experiments; start = :best_fit)` replaces each experiment's own parameters (`x.params`; never physics parameters such as the oscillation, `xsec_*` or flux parameters) by the best fit of that experiment alone in its default setup, stored in `bestfit.toml` next to the experiment's module (e.g. `src/experiments/t2k/t2k_2025/bestfit.toml`) and written by `src/utils/generate_bestfit.jl` (rerun it when an experiment's data or model change). Only start values: results do not depend on them.

**Parameters flow as NamedTuples** throughout the codebase. `get_params`/`get_priors` merge across all physics and experiment modules using `safe_merge` (checks for conflicts). Use `@reset` from Accessors.jl to modify individual fields.

**ForwardDiff compatibility** is critical. The oscillation code runs in the inner loop of gradient-based optimization. Avoid `Float64` literals that would strip Dual numbers; use `zero(T)`, `one(T)`, `promote_type`. Never convert computed values to concrete float types.

## Performance Notes

- `osc_prob` is the hot path: uses `SMatrix`/`SVector` for zero-allocation vacuum oscillations. Matter effects require `eigen` which allocates (~19 allocs/call).
- Response matrix contractions (Super-K) use `contract_R` with pre-flattened Float64 matrices for BLAS-accelerated matrix-vector multiply, avoiding Dual number broadcast over large arrays.
- ForwardDiff chunk size is 12 by default; with N params, gradient costs `ceil(N/12)` passes.
