# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Run unit tests (185 tests, ~20s)
julia --project -e 'using Pkg; Pkg.test()'

# Run benchmarks
julia --project benchmark/bench_osc.jl
julia --project benchmark/bench_likelihood.jl --experiments dayabay
julia --project benchmark/bench_likelihood.jl --experiments deepcore super_k

# Run an experiment's validation script (from its directory)
cd src/experiments/icecube/deepcore_9y_verification_sample && julia --project=../../../.. test.jl

# Analysis CLI
julia --project src/analysis/analysis.jl --experiments deepcore dayabay --name myrun --task scan
julia --project src/analysis/distributed_profile.jl --experiments deepcore --name myrun --workers 4
```

## Architecture

Newtrinos.jl is a neutrino physics global analysis framework with three orthogonal layers:

### Physics (`src/physics/`)
Theory predictions with no experiment knowledge. Each module returns a struct `<: Newtrinos.Physics` with `params`, `priors`, and callable functions.

- **`osc.jl`** — Core oscillation probability engine. Configurable via `OscillationConfig` with flavour models (`ThreeFlavour`, `Sterile`, `ADD`, `Darkdim_*`), interaction models (`Vacuum`, `SI`, `NSI`), and propagation models (`Basic`, `Decoherent`, `Damping`). Performance-critical: uses `SMatrix`/`SVector` for 3-flavour, `eigen` for matter effects.
- **`earth_layers.jl`** — PREM Earth density model. `compute_layers()` → `compute_paths(coszen, layers)`.
- **`atm_flux.jl`** — HKKM atmospheric neutrino fluxes with Barr systematics. Site-specific flux files in `src/physics/*.d`.
- **`xsec.jl`** — Cross-section models: `SimpleScaling` or `Differential_H2O` (for Super-K).
- **`cevns_xsec.jl`**, **`sns_flux.jl`** — COHERENT-specific physics.

### Experiments (`src/experiments/`)
Each experiment module has `configure(physics=default_physics())` returning a struct `<: Newtrinos.Experiment` with fields: `physics`, `params`, `priors`, `assets`, `forward_model`, `plot`. Each experiment defines its own `default_physics()` with appropriate oscillation config, flux files, and cross-section models.

Experiment groups and their physics requirements:
- **Atmospheric** (deepcore, ic_upgrade, super_k, orca): `osc` (SI), `atm_flux`, `earth_layers`, `xsec`
- **Reactor** (dayabay, kamland, juno, tao): `osc` (Vacuum)
- **Accelerator** (minos): `osc`, `xsec`
- **COHERENT** (coherent_csi, coherent_lAr): self-contained, no physics input

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

**Parameters flow as NamedTuples** throughout the codebase. `get_params`/`get_priors` merge across all physics and experiment modules using `safe_merge` (checks for conflicts). Use `@reset` from Accessors.jl to modify individual fields.

**ForwardDiff compatibility** is critical. The oscillation code runs in the inner loop of gradient-based optimization. Avoid `Float64` literals that would strip Dual numbers; use `zero(T)`, `one(T)`, `promote_type`. Never convert computed values to concrete float types.

## Performance Notes

- `osc_prob` is the hot path: uses `SMatrix`/`SVector` for zero-allocation vacuum oscillations. Matter effects require `eigen` which allocates (~19 allocs/call).
- Response matrix contractions (Super-K) use `contract_R` with pre-flattened Float64 matrices for BLAS-accelerated matrix-vector multiply, avoiding Dual number broadcast over large arrays.
- ForwardDiff chunk size is 12 by default; with N params, gradient costs `ceil(N/12)` passes.
