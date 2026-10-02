"""
    t2k

!!! note
    This module is 100% vibe coded: written entirely by an AI coding assistant (Claude) with human
    steering, and validated only against the published results shown in its `test.jl`.

T2K three-flavour oscillation analysis with 19.7 (16.3) × 10²⁰ POT in ν (ν̄) mode
(arXiv:2506.05889), built from public material only (`T2K_2025.h5`, see `extraction/`):

- far-detector predictions (oscillated and unoscillated, plus the oscillated breakdown by NEUT
  interaction mode, after the ND fit) for the six SK samples, extracted from the vector PDFs of the
  official data release (zenodo 15701867, CC-BY 4.0), computed at the reference point
  sin²θ₂₃ = 0.561, sin²θ₁₃ = 0.022, sin²θ₁₂ = 0.307, Δm²₃₂ = 2.49e-3 eV², Δm²₂₁ = 7.53e-5 eV², δCP = −1.601;
- the far-detector data, event by event, from the vector figures of arXiv:2303.03222 (same exposure):
  (E_rec, θ) for the μ-like and (p, θ) for the e-like samples, from which E_rec is computed with the
  T2K formulas; the νμCC1π⁺ sample from the binned figure of arXiv:2506.05889.

The release has no true-energy information. Predictions are built by reweighting with oscillation
probabilities averaged over an effective true-energy response per reconstructed bin, separately for
the correctly reconstructed interaction mode ("A": CCQE, or CC1π⁺ + coherent for the 1π samples) and all
other CC modes ("B", reconstructed at too low energy). The response parameters are calibrated on the
released oscillated/unoscillated pairs and mode breakdowns (`calibrate_response.jl`).

- μ-like (FHC/RHC 1Rμ, FHC νμCC1π⁺): unoscillated CC split into modes A/B, `Σₖ Uₖ ⟨P_μμ⟩ₖ + NC`, with a
  fixed per-bin correction (a few %) such that the released oscillated spectrum is exact at the reference.
- e-like (FHC/RHC 1Re, FHC 1Re1de): the appearance signal at the reference point,
  `osc − NC − (unosc − NC) ⟨P_ee⟩`, is reweighted by `⟨P_μe(θ)⟩ / ⟨P_μe(θ_ref)⟩`; in RHC a wrong-sign νμ → νe
  contribution is included. Intrinsic beam νe oscillate with P_ee; NC is unoscillated.

Matter effects use a constant density (L = 295 km, ρ = 2.6 g/cm³). The e-like samples are fitted in
E_rec only (T2K fits p–θ), and there are no systematic templates, so the official result is not
reproduced exactly.
"""
module t2k

using LinearAlgebra
using Distributions
using HDF5
using StructArrays
using ArraysOfArrays
using CairoMakie
using Printf
using SpecialFunctions: erf
using BAT: distprod
using ..Newtrinos

const datafile = joinpath(@__DIR__, "T2K_2025.h5")

const L_FD = 295.0          # km
const RHO = 2.6             # g/cm³
const YE = 0.5

"Reference point of the released predictions (arXiv:2506.05889)"
const REF = (ss12 = 0.307, ss13 = 0.022, ss23 = 0.561, dm21 = 7.53e-5, dm32 = 2.49e-3, dcp = -1.601)

ref_params() = (θ₁₂ = asin(sqrt(REF.ss12)), θ₁₃ = asin(sqrt(REF.ss13)), θ₂₃ = asin(sqrt(REF.ss23)),
                δCP = mod(REF.dcp, 2π), Δm²₂₁ = REF.dm21, Δm²₃₁ = REF.dm32 + REF.dm21)

"""
Samples: beam mode (`anti` = RHC), μ- or e-like, which NEUT modes count as correctly reconstructed
(mode A), and which μ-like sample provides the response calibration.
"""
const SAMPLES = (
    fhc1rmu      = (anti = false, kind = :mu, A = ("CCQE",), response = :fhc1rmu),
    rhc1rmu      = (anti = true,  kind = :mu, A = ("CCQE",), response = :rhc1rmu),
    fhcnumucc1pi = (anti = false, kind = :mu, A = ("CC1pipm", "CCcoh"), response = :fhcnumucc1pi),
    fhc1re       = (anti = false, kind = :e,  A = ("CCQE",), response = :fhc1rmu),
    rhc1re       = (anti = true,  kind = :e,  A = ("CCQE",), response = :rhc1rmu),
    fhc1re1de    = (anti = false, kind = :e,  A = ("CC1pipm", "CCcoh"), response = :fhcnumucc1pi),
)

"""
Effective response E_rec ~ N((1 + bias) E − shift, σ E) for mode A (`σA`, `biasA`) and mode B
(`σB`, `shiftB` in GeV), calibrated with `calibrate_response.jl`.
"""
const RESPONSE = (
    fhc1rmu      = (σA = 0.142, biasA = 0.001, σB = 0.180, shiftB = 0.182),
    rhc1rmu      = (σA = 0.101, biasA = 0.016, σB = 0.298, shiftB = 0.253),
    fhcnumucc1pi = (σA = 0.101, biasA = 0.200, σB = 0.708, shiftB = 0.359),   # weakly constrained (no events at the dip)
)

"""
Ratio of unoscillated wrong-sign νμ to right-sign ν̄μ CC events in the RHC e-like signal; 0.20 reproduces
T2K's predicted RHC 1Re rates for δCP = −π/2, 0, π/2, π (arXiv:2303.03222, Tab. 'oa:events') to 0.1 events.
"""
const RHC_WRONG_SIGN_RATIO = 0.20

# binning of the νμCC1π⁺ data (arXiv:2506.05889, Fig. 1): 0.1 GeV up to 3 GeV, then 0.5 GeV
const CC1PI_EDGES = vcat(collect(0.5:0.1:3.0), [3.5, 4.0])

@kwdef struct T2K <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function default_physics()
    osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction = Newtrinos.osc.SI(),
                                                                  eigen_method = Newtrinos.BargerEigen()))
    (; osc)
end

function configure(physics = default_physics(); response = RESPONSE, ws_ratio = RHC_WRONG_SIGN_RATIO)
    physics = (; physics.osc)
    assets = get_assets(physics; response, ws_ratio)
    return T2K(
        physics = physics,
        params = get_params(),
        priors = get_priors(),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets),
    )
end

get_params() = (
    t2k_flux_xsec_norm = 1.0,
    t2k_rhc_norm = 1.0,
    t2k_fhc1rmu_norm = 1.0,
    t2k_rhc1rmu_norm = 1.0,
    t2k_cc1pi_norm = 1.0,
    t2k_fhc1re_norm = 1.0,
    t2k_rhc1re_norm = 1.0,
    t2k_fhc1re1de_norm = 1.0,
    t2k_nue_xsec_ratio_nu = 1.0,
    t2k_nue_xsec_ratio_nubar = 1.0,
    t2k_nonqe_norm = 1.0,
    t2k_nc_norm = 1.0,
    t2k_energy_scale = 1.0,
)

# Post-ND-fit uncertainties on the FD event rates (arXiv:2303.03222, Tab. 'percenterrors:postND'):
# flux ⊗ interaction 2–3.5% (common), FD detector + SI + PN per sample; the 1π sample 4.3% total
# (arXiv:2506.05889). The non-QE normalisation changes the mode-B share (shape), NC is loosely
# constrained, the energy scale is SK's ~2%.
get_priors() = (
    t2k_flux_xsec_norm = Normal(1.0, 0.025),
    t2k_rhc_norm = Normal(1.0, 0.025),
    t2k_fhc1rmu_norm = Normal(1.0, 0.021),
    t2k_rhc1rmu_norm = Normal(1.0, 0.019),
    t2k_cc1pi_norm = Normal(1.0, 0.035),
    t2k_fhc1re_norm = Normal(1.0, 0.031),
    t2k_rhc1re_norm = Normal(1.0, 0.039),
    t2k_fhc1re1de_norm = Normal(1.0, 0.134),
    t2k_nue_xsec_ratio_nu = Normal(1.0, 0.03),
    t2k_nue_xsec_ratio_nubar = Normal(1.0, 0.03),
    t2k_nonqe_norm = Normal(1.0, 0.10),
    t2k_nc_norm = Normal(1.0, 0.30),
    t2k_energy_scale = Truncated(Normal(1.0, 0.02), 0.9, 1.1),
)

# ---------------------------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------------------------

Φ(x) = (1 + erf(x / sqrt(2))) / 2

"""
    response_weights(Et, edges, fE, σ; bias = 0.0, shift = 0.0) -> Matrix

Row-normalised weights `W[b, i]` of true energy `Et[i]` in reconstructed bin `b`:
`∝ f(E_i) dE_i · P(E_rec ∈ bin | E_i)` for `E_rec ~ N((1 + bias) E − shift, σ E)` and a true-spectrum
proxy `fE`.
"""
function response_weights(Et, edges, fE, σ; bias = 0.0, shift = 0.0)
    dE = vcat(diff(Et), diff(Et)[end])
    W = zeros(length(edges) - 1, length(Et))
    for b in 1:length(edges)-1, i in eachindex(Et)
        μ = (1 + bias) * Et[i] - shift
        W[b, i] = fE[i] * dE[i] * (Φ((edges[b+1] - μ) / (σ * Et[i])) - Φ((edges[b] - μ) / (σ * Et[i])))
    end
    W ./ max.(sum(W, dims = 2), 1e-300)
end

"Piecewise-constant density of a histogram on `Et` (plus a small floor)"
function spectrum_proxy(Et, edges, values)
    dens = values ./ diff(edges)
    fl = 1e-6 * maximum(dens)
    [(E < edges[1] || E >= edges[end]) ? fl : dens[searchsortedlast(edges, E)] + fl for E in Et]
end

"Values of a histogram on `edges` from one with coarser leading bins (`src`, aligned otherwise)"
function regrid(src_edges, src_values, edges)
    out = zeros(length(edges) - 1)
    for b in 1:length(edges)-1
        c = (edges[b] + edges[b+1]) / 2
        j = searchsortedlast(src_edges, c)
        (1 <= j < length(src_edges)) || continue
        # only take bins of the same width (wider ones are merged empty bins)
        isapprox(src_edges[j+1] - src_edges[j], edges[b+1] - edges[b], atol = 1e-6) && (out[b] = src_values[j])
    end
    out
end

"Sum a histogram on fine `edges` into coarser `target` edges (aligned)"
function rebin(edges, values, target)
    [sum(values[b] for b in eachindex(values) if target[k] - 1e-9 <= (edges[b] + edges[b+1]) / 2 < target[k+1]; init = 0.0)
     for k in 1:length(target)-1]
end

const M_P, M_N, M_E, M_DELTA = 0.938272, 0.939565, 0.000511, 1.232
const E_BIND = 0.027

"CCQE reconstructed energy (struck nucleon at rest, effective mass m − E_b; arXiv:2303.03222)"
function erec_ccqe(p, θ; anti = false, m_l = M_E)
    E = sqrt(p^2 + m_l^2); c = cosd(θ)
    mi, mf = anti ? (M_P - E_BIND, M_N) : (M_N - E_BIND, M_P)
    (m_l^2 + mi^2 - mf^2 - 2E * mi) / (2 * (E - p * c - mi))
end
"Reconstructed energy assuming Δ(1232) production on a nucleon at rest (1Re1de, νμCC1π⁺)"
erec_delta(p, θ; m_l = M_E) = (E = sqrt(p^2 + m_l^2); (2M_P * E + M_DELTA^2 - M_P^2 - m_l^2) / (2 * (M_P - E + p * cosd(θ))))

function observed_counts(f, s, edges)
    if s == :fhcnumucc1pi
        g = f["data/fhcnumucc1pi"]
        lo, cnt = read(g["lo"]), read(g["count"])
        return [sum(cnt[isapprox.(lo, edges[b], atol = 1e-6)]; init = 0) for b in 1:length(edges)-1]
    end
    g = f["data/events/$s"]
    x, θ = read(g["x"]), read(g["theta"])
    E = if read(HDF5.attributes(g)["x"]) == "Erec"
        x
    elseif s == :fhc1re1de
        erec_delta.(x ./ 1000, θ)
    else
        erec_ccqe.(x ./ 1000, θ; anti = SAMPLES[s].anti)
    end
    [count(e -> edges[b] <= e < edges[b+1], E) for b in 1:length(edges)-1]
end

function get_assets(physics; file = datafile, response = RESPONSE, ws_ratio = RHC_WRONG_SIGN_RATIO)
    @info "Loading T2K 2025 inputs"
    layers = StructArray{Newtrinos.Layer}((radius = [6371.0], p_density = [RHO * YE], n_density = [RHO * (1 - YE)]))
    paths = VectorOfVectors{Newtrinos.Path}([[Newtrinos.Path(L_FD, 1)]])
    Et = exp10.(range(log10(0.05), log10(10.0), 500))
    Pref = channel_probabilities(physics, Et, paths, layers, ref_params())

    samples = h5open(file) do f
        hist(path) = (edges = read(f["$path/edges"]), values = read(f["$path/values"]))
        raw = map(keys(SAMPLES)) do s
            g = "predictions/$s"
            osc, unosc = hist("$g/oscillated"), hist("$g/unoscillated")
            edges = osc.edges
            modes = keys(f["$g/breakdown"])
            bd(m) = (h = hist("$g/breakdown/$m"); regrid(h.edges, h.values, edges))
            nc = sum(bd(m) for m in modes if startswith(m, "NC"))
            A = sum(bd(m) for m in modes if m in SAMPLES[s].A)
            B = sum(bd(m) for m in modes if !startswith(m, "NC") && !(m in SAMPLES[s].A))
            # analysis binning: the release binning (0–3 GeV for 1Rμ, 0–1.25 GeV for e-like), the
            # data binning for the 1π sample
            target = s == :fhcnumucc1pi ? CC1PI_EDGES : edges
            (; edges, target, osc = osc.values, unosc = unosc.values, nc, A, B, observed = observed_counts(f, s, target))
        end
        NamedTuple{keys(SAMPLES)}(raw)
    end

    # true-spectrum proxies (unoscillated CC of the μ-like samples) and response matrices
    proxy = map(k -> (s = samples[k]; spectrum_proxy(Et, s.edges, max.(s.unosc .- s.nc, 0.0))), (fhc1rmu = :fhc1rmu, rhc1rmu = :rhc1rmu, fhcnumucc1pi = :fhcnumucc1pi))
    built = map(keys(SAMPLES), values(SAMPLES), values(samples)) do k, cfg, s
        r = response[cfg.response]; fE = proxy[cfg.response]
        WA = response_weights(Et, s.edges, fE, r.σA; bias = r.biasA)
        WB = response_weights(Et, s.edges, fE, r.σB; shift = r.shiftB)
        main, other = cfg.anti ? (Pref.nubar, Pref.nu) : (Pref.nu, Pref.nubar)
        if cfg.kind == :mu
            pA, pB = WA * main.mm, WB * main.mm
            # unoscillated CC per mode, normalised to the released unoscillated CC spectrum
            uA, uB = s.A ./ max.(pA, 0.01), s.B ./ max.(pB, 0.01)
            u_cc = max.(s.unosc .- s.nc, 0.0)
            sc = ifelse.(uA .+ uB .> 0, u_cc ./ max.(uA .+ uB, 1e-12), 0.0)
            # residual per-bin correction so that the released oscillated spectrum is reproduced exactly
            # at the reference point (the calibration matches the unoscillated spectrum to a few %)
            o_model = sc .* (uA .* pA .+ uB .* pB)
            sc = sc .* ifelse.(o_model .> 0, (s.A .+ s.B) ./ max.(o_model, 1e-12), 1.0)
            k => (; kind = :mu, cfg.anti, s.edges, s.target, s.observed, WA, WB, UA = uA .* sc, UB = uB .* sc, s.nc, s.osc, s.unosc, s.A, s.B)
        else
            # appearance signal at the reference point; split into modes by the oscillated CC breakdown
            bkg_cc = max.(s.unosc .- s.nc, 0.0)
            pee = WA * main.ee
            sig = max.(s.osc .- s.nc .- bkg_cc .* pee, 0.0)
            fA = ifelse.(s.A .+ s.B .> 0, s.A ./ max.(s.A .+ s.B, 1e-12), 1.0)
            ref = (A = (me = WA * main.me, me_ws = WA * other.me, ee = WA * main.ee),
                   B = (me = WB * main.me, me_ws = WB * other.me, ee = WB * main.ee))
            # wrong-sign share of the RHC signal at the reference point
            ws = cfg.anti ? (ws_ratio .* ref.A.me_ws) ./ (ref.A.me .+ ws_ratio .* ref.A.me_ws) : zeros(length(sig))
            k => (; kind = :e, cfg.anti, s.edges, s.target, s.observed, WA, WB, sigA = sig .* fA, sigB = sig .* (1 .- fA),
                  ws, bkg_cc, s.nc, ref, s.osc, s.unosc)
        end
    end |> NamedTuple
    observed = map(s -> s.observed, built)
    (; Et, dlnE = log(Et[2] / Et[1]), layers, paths, samples = built, observed)
end

# ---------------------------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------------------------

function channel_probabilities(physics, Et, paths, layers, params)
    p = physics.osc.osc_prob(Et, paths, layers, params)
    pb = physics.osc.osc_prob(Et, paths, layers, params; anti = true)
    (nu = (mm = p[:, 1, 2, 2], me = p[:, 1, 2, 1], ee = p[:, 1, 1, 1]),
     nubar = (mm = pb[:, 1, 2, 2], me = pb[:, 1, 2, 1], ee = pb[:, 1, 1, 1]))
end

"""
    scale_energy(P, dlnE, s)

`P(s E)` from `P` tabulated on a logarithmic grid with spacing `dlnE`, by Catmull-Rom (C¹)
interpolation (exact for `s = 1`; values beyond the grid ends are clamped).
"""
function scale_energy(P::AbstractVector, dlnE, s)
    x = log(s) / dlnE
    k = floor(Int, x)
    t = x - k
    n = length(P)
    w0 = t * (-1 + t * (2 - t)) / 2
    w1 = 1 + t^2 * (-5 + 3t) / 2
    w2 = t * (1 + t * (4 - 3t)) / 2
    w3 = t^2 * (t - 1) / 2
    map(1:n) do i
        w0 * P[clamp(i + k - 1, 1, n)] + w1 * P[clamp(i + k, 1, n)] + w2 * P[clamp(i + k + 1, 1, n)] + w3 * P[clamp(i + k + 2, 1, n)]
    end
end
scale_energy(P::NamedTuple, dlnE, s) = map(p -> scale_energy(p, dlnE, s), P)

_ratio(a, b) = ifelse.(b .> 0, a ./ max.(b, 1e-12), one.(a))

const SAMPLE_NORM = (fhc1rmu = :t2k_fhc1rmu_norm, rhc1rmu = :t2k_rhc1rmu_norm, fhcnumucc1pi = :t2k_cc1pi_norm,
                     fhc1re = :t2k_fhc1re_norm, rhc1re = :t2k_rhc1re_norm, fhc1re1de = :t2k_fhc1re1de_norm)

function sample_expected(k, s, P, params)
    main, other = s.anti ? (P.nubar, P.nu) : (P.nu, P.nubar)
    norm = params.t2k_flux_xsec_norm * getproperty(params, SAMPLE_NORM[k]) * (s.anti ? params.t2k_rhc_norm : one(params.t2k_rhc_norm))
    nqe = params.t2k_nonqe_norm
    nc = params.t2k_nc_norm .* s.nc
    λ = if s.kind == :mu
        norm .* (s.UA .* (s.WA * main.mm) .+ nqe .* s.UB .* (s.WB * main.mm)) .+ nc
    else
        xs = s.anti ? params.t2k_nue_xsec_ratio_nubar : params.t2k_nue_xsec_ratio_nu
        sigA = s.sigA .* ((1 .- s.ws) .* _ratio(s.WA * main.me, s.ref.A.me) .+ s.ws .* _ratio(s.WA * other.me, s.ref.A.me_ws))
        sigB = s.sigB .* ((1 .- s.ws) .* _ratio(s.WB * main.me, s.ref.B.me) .+ s.ws .* _ratio(s.WB * other.me, s.ref.B.me_ws))
        norm .* (xs .* (sigA .+ nqe .* sigB) .+ s.bkg_cc .* (s.WA * main.ee)) .+ nc
    end
    s.target === s.edges ? λ : rebin(s.edges, λ, s.target)
end

function get_expected(params, physics, assets)
    P = scale_energy(channel_probabilities(physics, assets.Et, assets.paths, assets.layers, params), assets.dlnE, params.t2k_energy_scale)
    map(keys(assets.samples), values(assets.samples)) do k, s
        k => sample_expected(k, s, P, params)
    end |> NamedTuple
end

function get_forward_model(physics, assets)
    function forward_model(params)
        expected = get_expected(params, physics, assets)
        distprod(map(λ -> distprod(Poisson.(max.(λ, 1e-9))), expected))
    end
end

# ---------------------------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------------------------

function get_plot(physics, assets)
    titles = (fhc1rmu = "FHC 1Rμ", rhc1rmu = "RHC 1Rμ", fhcnumucc1pi = "FHC νμCC1π⁺",
              fhc1re = "FHC 1Re", rhc1re = "RHC 1Re", fhc1re1de = "FHC 1Re1de")
    function plot(params, data = assets.observed)
        expected = get_expected(params, physics, assets)
        fig = Figure(size = (1200, 650))
        for (i, k) in enumerate(keys(expected))
            e = assets.samples[k].target; w = diff(e); x = (e[1:end-1] .+ e[2:end]) ./ 2
            ax = Axis(fig[(i - 1) ÷ 3 + 1, (i - 1) % 3 + 1], title = titles[k], xlabel = "E_rec (GeV)", ylabel = "events / GeV")
            λ = expected[k] ./ w
            stairs!(ax, e, vcat(λ, λ[end]), step = :post, color = :blue, label = "prediction")
            errorbars!(ax, x, data[k] ./ w, sqrt.(data[k]) ./ w, color = :black)
            scatter!(ax, x, data[k] ./ w, color = :black, markersize = 5, label = "data")
        end
        fig
    end
end

end
