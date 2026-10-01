"""
    nova

NOvA 2024 three-flavour oscillation analysis (26.61e20 ν + 12.5e20 ν̄ POT, PRL 136, 011802 (2026)),
built from the official data release (doi:10.5281/zenodo.17822358), converted once from ROOT to
`NOvA_2024_data_release.h5` by `convert_nova_2024_to_hdf5.py`.

The release provides far-detector data and the predicted signal/background components at a
reference oscillation point (including NOvA's systematic pulls), but no true-energy information
and no systematic templates. The prediction is therefore built by reweighting the released
components with oscillation probabilities averaged over an effective true-energy distribution per
reconstructed bin:

- νμ / ν̄μ disappearance (4 quartiles per beam mode): `NoOscillations_Signal × ⟨P_μμ⟩` plus the
  released beam and cosmic backgrounds. The per-quartile energy resolutions are calibrated so that
  `NoOscillations_Signal × ⟨P_μμ(θ_ref)⟩` reproduces the released `Oscillated_Signal`
  (`calibrate_numu_resolution.jl`).
- νe / ν̄e appearance (low-PID core, high-PID core, peripheral; plus the FHC low-energy sample):
  each component is scaled by `⟨P(θ)⟩ / ⟨P(θ_ref)⟩` for its channel (signal νμ→νe, wrong-sign
  ν̄μ→ν̄e, beam νe νe→νe, νμ CC νμ→νμ, ντ CC νμ→ντ); NC and cosmics are unchanged.

Systematics (priors sized to the rate and parameter uncertainties quoted after the ND
extrapolation, PRL 136, 011802, Tab. I and text): νμ and RHC normalisations, νe signal and
background normalisations, the νe/νμ and ν̄e/ν̄μ cross-section ratios, and a muon and a hadronic
energy scale. The νμ quartiles are bins of hadronic-energy fraction, so each quartile's energy scale
is `1 + (1 - f_q) δ_μ + f_q δ_had` with an estimated mean hadronic fraction `f_q`
(`hadronic_fraction`); the calorimetric νe energy scales with `δ_had`. RHC has an additional
hadronic-scale offset. Energy scales are applied by shifting the oscillation probabilities on the
logarithmic true-energy grid.

Matter effects use a single constant-density layer (L = 810 km, ρ = 2.84 g/cm³, Yₑ = 0.5) through
the standard `osc_prob(E, paths, layers, params)` interface. As the release notes, fits with these
histograms are not expected to reproduce the official NOvA results exactly.
"""
module nova

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

const datafile = joinpath(@__DIR__, "NOvA_2024_data_release.h5")

const L_FD = 810.0          # km
const RHO = 2.84            # g/cm³
const YE = 0.5

"""
Calibrations of the released predictions (`calibrate_numu_resolution.jl`). The reference oscillation
point at which the release's oscillated components were computed is not stated in the release; the
νμ Oscillated/NoOscillations pairs are reproduced equally well by
- `:ref2024`: NOvA's quoted 2024 frequentist best fit (Δm²₃₂ = 2.441e-3 eV², sin²θ₂₃ = 0.55,
  δCP = 0.87π; PRL 136, 011802) with a per-quartile energy resolution σ and reco-energy bias b
  (E_reco ~ N((1+b) E_true, σ E_true));
- `:effective`: the point that reproduces them without a bias, sin²θ₂₃ = 0.5681, Δm²₃₂ = 2.4073e-3 eV²
  (coinciding with NOvA's 2020 best fit; δCP = 0.82π from that fit).
θ₁₂, Δm²₂₁ and θ₁₃ as in the release README.
"""
const CALIBRATIONS = (
    ref2024 = (ref = (ss2th12 = 0.851, dm21 = 7.53e-5, ss2th13 = 0.0851, ssth23 = 0.55, dm32 = 2.441e-3, dcp = 0.87π),
               numu_sigma = (fhc = (0.064, 0.084, 0.100, 0.124), rhc = (0.053, 0.069, 0.083, 0.106)),
               numu_bias = (fhc = (-0.0164, -0.0200, -0.0292, -0.0427), rhc = (-0.0028, -0.0177, -0.0197, -0.0296))),
    effective = (ref = (ss2th12 = 0.851, dm21 = 7.53e-5, ss2th13 = 0.0851, ssth23 = 0.5681, dm32 = 2.4073e-3, dcp = 0.82π),
                 numu_sigma = (fhc = (0.053, 0.077, 0.098, 0.124), rhc = (0.031, 0.060, 0.075, 0.102)),
                 numu_bias = (fhc = (0.0, 0.0, 0.0, 0.0), rhc = (0.0, 0.0, 0.0, 0.0))),
)

ref_params(calibration::Symbol = :ref2024) = (r = getproperty(CALIBRATIONS, calibration).ref;
    (θ₁₂ = asin(sqrt(r.ss2th12)) / 2, θ₁₃ = asin(sqrt(r.ss2th13)) / 2, θ₂₃ = asin(sqrt(r.ssth23)),
     δCP = r.dcp, Δm²₂₁ = r.dm21, Δm²₃₁ = r.dm32 + r.dm21))
"νe / ν̄e energy resolution (PRL 136, 011802)"
const NUE_RESOLUTION = (fhc = 0.11, rhc = 0.09)

"""
    hadronic_fraction(σ; σ_μ = 0.035, σ_had = 0.30)

Mean hadronic-energy fraction of a νμ quartile estimated from its calibrated energy resolution σ,
assuming `σ² = (1 - f)² σ_μ² + f² σ_had²` (the release does not give the fractions; σ_μ and σ_had
are typical NOvA muon-range and hadronic-calorimetry resolutions).
"""
function hadronic_fraction(σ; σ_μ = 0.035, σ_had = 0.30)
    a, b, c = σ_μ^2 + σ_had^2, -2σ_μ^2, σ_μ^2 - σ^2
    (-b + sqrt(b^2 - 4a * c)) / 2a
end

# νe core/peripheral analysis bins (0-based indices of the 23-bin data layout) and their
# reconstructed-energy ranges in GeV: two PID segments of six 0.5 GeV bins from 1 to 4 GeV, and the
# peripheral sample as a single bin (its energy range is not given in the release; 1–4.5 GeV assumed)
const NUE_CORE_BINS = (2:7, 11:16)
const NUE_PERIPHERAL_BIN = 20
const NUE_CORE_EDGES = collect(1.0:0.5:4.0)
const NUE_PERIPHERAL_RANGE = (1.0, 4.5)

@kwdef struct NOvA <: Newtrinos.Experiment
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

function configure(physics = default_physics(); calibration::Symbol = :ref2024)
    physics = (; physics.osc)
    assets = get_assets(physics; calibration)
    return NOvA(
        physics = physics,
        params = get_params(),
        priors = get_priors(),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets),
    )
end

get_params() = (
    nova_numu_norm = 1.0,
    nova_rhc_norm = 1.0,
    nova_nue_signal_norm = 1.0,
    nova_nue_xsec_ratio_nu = 1.0,
    nova_nue_xsec_ratio_nubar = 1.0,
    nova_nue_bkg_norm = 1.0,
    nova_muon_energy_scale = 1.0,
    nova_hadronic_energy_scale = 1.0,
    nova_rhc_hadronic_energy_scale = 1.0,
)

# Far-detector rate uncertainties after the ND extrapolation are 3–6.5% (PRL 136, 011802); the
# energy-scale widths reproduce the lepton-reco (~0.5%) and calibration (~0.7% on the νμ energy)
# contributions to Δm²₃₂ in Tab. I.
get_priors() = (
    nova_numu_norm = Normal(1.0, 0.035),
    nova_rhc_norm = Normal(1.0, 0.03),
    nova_nue_signal_norm = Normal(1.0, 0.04),
    nova_nue_xsec_ratio_nu = Normal(1.0, 0.02),
    nova_nue_xsec_ratio_nubar = Normal(1.0, 0.02),
    nova_nue_bkg_norm = Normal(1.0, 0.10),
    nova_muon_energy_scale = Truncated(Normal(1.0, 0.005), 0.97, 1.03),
    nova_hadronic_energy_scale = Truncated(Normal(1.0, 0.025), 0.9, 1.1),
    nova_rhc_hadronic_energy_scale = Truncated(Normal(1.0, 0.01), 0.95, 1.05),
)

# ---------------------------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------------------------

_hist(g) = (edges = read(g["edges"]), values = read(g["values"]))

Φ(x) = (1 + erf(x / sqrt(2))) / 2

"""
    smearing_weights(Et, ranges, σ, fE) -> Matrix

Row-normalised weights `W[b, i]` of true energy `Et[i]` for a reconstructed-energy range `ranges[b]`:
`∝ f(E_i) dE_i · P(E_reco ∈ range | E_i)` with a Gaussian response `E_reco ~ N((1+bias) E_true, σ E_true)`
and a true-spectrum proxy `fE`. Rows for empty ranges (`nothing`) are zero.
"""
function smearing_weights(Et, ranges, σ, fE; bias = 0.0)
    dE = vcat(diff(Et), diff(Et)[end])
    W = zeros(length(ranges), length(Et))
    for (b, r) in enumerate(ranges)
        r === nothing && continue
        lo, hi = r
        for i in eachindex(Et)
            μ = (1 + bias) * Et[i]
            W[b, i] = fE[i] * dE[i] * (Φ((hi - μ) / (σ * Et[i])) - Φ((lo - μ) / (σ * Et[i])))
        end
        s = sum(W[b, :])
        s > 0 && (W[b, :] ./= s)
    end
    W
end

"Piecewise-constant density of a histogram evaluated on `Et` (plus a small floor)"
function spectrum_proxy(Et, edges, values)
    dens = values ./ diff(edges)
    fl = 1e-6 * maximum(dens)
    [(E < edges[1] || E >= edges[end]) ? fl : dens[searchsortedlast(edges, E)] + fl for E in Et]
end

function get_assets(physics; file = datafile, calibration::Symbol = :ref2024)
    cal = getproperty(CALIBRATIONS, calibration)
    @info "Loading NOvA 2024 data release"
    f = h5open(file)
    pred(path) = _hist(f["predictions/$path"])
    data(name) = round.(Int, _hist(f["data/$name"]).values)

    layers = StructArray{Newtrinos.Layer}((radius = [6371.0], p_density = [RHO * YE], n_density = [RHO * (1 - YE)]))
    paths = VectorOfVectors{Newtrinos.Path}([[Newtrinos.Path(L_FD, 1)]])
    Et = exp10.(range(log10(0.1), log10(10.0), 400))   # true-energy grid [GeV]

    # --- νμ / ν̄μ disappearance: 4 quartiles per beam mode ---
    numu_sample(mode, q) = begin
        σ = getproperty(cal.numu_sigma, Symbol(mode))[q]
        dir = "prediction_components_numu_$(mode)_Quartile$(q)"
        noosc = pred("$dir/NoOscillations_Signal")
        ranges = [(noosc.edges[b], noosc.edges[b+1]) for b in 1:length(noosc.values)]
        fE = spectrum_proxy(Et, noosc.edges, noosc.values)
        (signal_noosc = noosc.values,
         beam_bkg = pred("$dir/Oscillated_Beam_bkg").values,
         cosmic = pred("$dir/Cosmic_bkg").values,
         W = smearing_weights(Et, ranges, σ, fE; bias = getproperty(cal.numu_bias, Symbol(mode))[q]),
         f_had = hadronic_fraction(σ),
         edges = noosc.edges,
         observed = data("$(mode == "fhc" ? "neutrino" : "antineutrino")_mode_numu_quartile$(q)"))
    end
    numu = (fhc_q1 = numu_sample("fhc", 1), fhc_q2 = numu_sample("fhc", 2), fhc_q3 = numu_sample("fhc", 3), fhc_q4 = numu_sample("fhc", 4),
            rhc_q1 = numu_sample("rhc", 1), rhc_q2 = numu_sample("rhc", 2), rhc_q3 = numu_sample("rhc", 3), rhc_q4 = numu_sample("rhc", 4))

    # --- νe / ν̄e appearance ---
    cmap = read(f["nue_component_to_data_bin"])          # 21-bin component layout -> 23-bin data layout (0-based)
    to_data_layout(v) = (out = zeros(23); for (i, j) in enumerate(cmap); j >= 0 && (out[j+1] += v[i]); end; out)
    used = vcat(collect(NUE_CORE_BINS[1]), collect(NUE_CORE_BINS[2]), [NUE_PERIPHERAL_BIN]) .+ 1   # 1-based
    core_ranges = [(NUE_CORE_EDGES[k], NUE_CORE_EDGES[k+1]) for k in 1:6]
    nue_ranges = vcat(core_ranges, core_ranges, [NUE_PERIPHERAL_RANGE])

    nue_sample(mode) = begin
        dir = "prediction_components_nue_$(mode)"
        comp(name) = name == "Signal" ? pred("$dir/Signal").values[used] : to_data_layout(pred("$dir/$name").values)[used]
        proxy = pred("prediction_components_numu_$(mode)_all/NoOscillations_Signal")
        fE = spectrum_proxy(Et, proxy.edges, proxy.values)
        (signal = comp("Signal"), wrong_sign = comp("Wrong_sign_bkg"), beam_nue = comp("Beam_nue_bkg"),
         numu_cc = comp("NumuCC_bkg"), tau_cc = comp("TauCC_bkg"), nc = comp("NC_bkg"), cosmic = comp("Cosmic_bkg"),
         W = smearing_weights(Et, nue_ranges, getproperty(NUE_RESOLUTION, Symbol(mode)), fE),
         observed = data("$(mode == "fhc" ? "neutrino" : "antineutrino")_mode_nue")[used])
    end
    lowe_sample() = begin
        dir = "prediction_components_nue_lowe_fhc"
        sig = pred("$dir/Signal")
        ranges = [(sig.edges[b], sig.edges[b+1]) for b in 1:length(sig.values)]
        proxy = pred("prediction_components_numu_fhc_all/NoOscillations_Signal")
        (signal = sig.values, wrong_sign = pred("$dir/Wrong_sign_bkg").values, beam_nue = pred("$dir/Beam_nue_bkg").values,
         numu_cc = pred("$dir/NumuCC_bkg").values, tau_cc = pred("$dir/TauCC_bkg").values, nc = pred("$dir/NC_bkg").values,
         cosmic = pred("$dir/Cosmic_bkg").values,
         W = smearing_weights(Et, ranges, NUE_RESOLUTION.fhc, spectrum_proxy(Et, proxy.edges, proxy.values)),
         observed = data("neutrino_mode_nueLowE"))
    end
    nue = (fhc = nue_sample("fhc"), rhc = nue_sample("rhc"), lowe_fhc = lowe_sample())
    close(f)

    # oscillation-averaged reference probabilities per νe bin and channel (denominators of the reweighting)
    Pref = channel_probabilities(physics, Et, paths, layers, ref_params(calibration))
    nue = map(nue, (fhc = false, rhc = true, lowe_fhc = false)) do s, anti
        merge(s, (ref = averaged_channels(s.W, Pref, anti),))
    end

    observed = merge(map(s -> s.observed, numu), (nue_fhc = nue.fhc.observed, nue_rhc = nue.rhc.observed, nue_lowe_fhc = nue.lowe_fhc.observed))
    dlnE = log(Et[2] / Et[1])
    (; Et, dlnE, layers, paths, numu, nue, observed, calibration)
end

# ---------------------------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------------------------

"Oscillation probabilities on the true-energy grid for the channels used, for ν and ν̄"
function channel_probabilities(physics, Et, paths, layers, params)
    p = physics.osc.osc_prob(Et, paths, layers, params)
    pb = physics.osc.osc_prob(Et, paths, layers, params; anti = true)
    (nu = (mm = p[:, 1, 2, 2], me = p[:, 1, 2, 1], ee = p[:, 1, 1, 1], mt = p[:, 1, 2, 3]),
     nubar = (mm = pb[:, 1, 2, 2], me = pb[:, 1, 2, 1], ee = pb[:, 1, 1, 1], mt = pb[:, 1, 2, 3]))
end

"""
    scale_energy(P, dlnE, s)

`P(s E)` from `P` tabulated on a logarithmic grid with spacing `dlnE`, by Catmull-Rom (C¹)
interpolation, so that gradients in `s` are continuous across grid nodes (exact for `s = 1`;
values beyond the grid ends are clamped).
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

"Bin-averaged probabilities for a νe sample; `anti` selects the beam mode (signal = ν̄ for RHC)"
function averaged_channels(W, P, anti)
    main, other = anti ? (P.nubar, P.nu) : (P.nu, P.nubar)
    (signal = W * main.me, wrong_sign = W * other.me, beam_nue = W * main.ee, numu_cc = W * main.mm, tau_cc = W * main.mt)
end

_ratio(a, b) = ifelse.(b .> 0, a ./ max.(b, 1e-12), one.(a))

function nue_expected(s, Pavg, params, signal_norm)
    sig = signal_norm .* s.signal .* _ratio(Pavg.signal, s.ref.signal)
    bkg = s.wrong_sign .* _ratio(Pavg.wrong_sign, s.ref.wrong_sign) .+
          s.beam_nue .* _ratio(Pavg.beam_nue, s.ref.beam_nue) .+
          s.numu_cc .* _ratio(Pavg.numu_cc, s.ref.numu_cc) .+
          s.tau_cc .* _ratio(Pavg.tau_cc, s.ref.tau_cc) .+ s.nc
    sig .+ params.nova_nue_bkg_norm .* bkg .+ s.cosmic
end

numu_expected(s, Pmm, norm) = norm .* s.signal_noosc .* (s.W * Pmm) .+ s.beam_bkg .+ s.cosmic

function get_expected(params, physics, assets)
    P = channel_probabilities(physics, assets.Et, assets.paths, assets.layers, params)
    δμ = params.nova_muon_energy_scale - 1
    had = (fhc = params.nova_hadronic_energy_scale,
           rhc = params.nova_hadronic_energy_scale * params.nova_rhc_hadronic_energy_scale)
    numu = map(assets.numu, (fhc_q1 = false, fhc_q2 = false, fhc_q3 = false, fhc_q4 = false,
                             rhc_q1 = true, rhc_q2 = true, rhc_q3 = true, rhc_q4 = true)) do s, anti
        scale = 1 + (1 - s.f_had) * δμ + s.f_had * ((anti ? had.rhc : had.fhc) - 1)
        Pmm = scale_energy(anti ? P.nubar.mm : P.nu.mm, assets.dlnE, scale)
        numu_expected(s, Pmm, params.nova_numu_norm * (anti ? params.nova_rhc_norm : one(params.nova_rhc_norm)))
    end
    P_fhc = scale_energy(P, assets.dlnE, had.fhc)
    P_rhc = scale_energy(P, assets.dlnE, had.rhc)
    sig_fhc = params.nova_nue_signal_norm * params.nova_nue_xsec_ratio_nu
    sig_rhc = params.nova_nue_signal_norm * params.nova_nue_xsec_ratio_nubar * params.nova_rhc_norm
    nue_fhc = nue_expected(assets.nue.fhc, averaged_channels(assets.nue.fhc.W, P_fhc, false), params, sig_fhc)
    nue_rhc = nue_expected(assets.nue.rhc, averaged_channels(assets.nue.rhc.W, P_rhc, true), params, sig_rhc)
    nue_lowe_fhc = nue_expected(assets.nue.lowe_fhc, averaged_channels(assets.nue.lowe_fhc.W, P_fhc, false), params, sig_fhc)
    merge(numu, (; nue_fhc, nue_rhc, nue_lowe_fhc))
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
    titles = (fhc_q1 = "νμ FHC Q1", fhc_q2 = "νμ FHC Q2", fhc_q3 = "νμ FHC Q3", fhc_q4 = "νμ FHC Q4",
              rhc_q1 = "ν̄μ RHC Q1", rhc_q2 = "ν̄μ RHC Q2", rhc_q3 = "ν̄μ RHC Q3", rhc_q4 = "ν̄μ RHC Q4",
              nue_fhc = "νe FHC (low PID | high PID | periph.)", nue_rhc = "ν̄e RHC (low PID | high PID | periph.)",
              nue_lowe_fhc = "νe FHC low-E")
    # bin edges per sample; the νe core/peripheral samples are plotted per analysis bin (unit width)
    edges(k) = haskey(assets.numu, k) ? assets.numu[k].edges :
               k == :nue_lowe_fhc ? collect(0.0:0.5:2.0) : collect(0.5:1.0:(length(assets.observed[k]) + 0.5))
    per_GeV(k) = !(k in (:nue_fhc, :nue_rhc))
    function plot(params, data = assets.observed)
        expected = get_expected(params, physics, assets)
        fig = Figure(size = (1200, 900))
        for (i, k) in enumerate(keys(expected))
            e = edges(k); w = per_GeV(k) ? diff(e) : ones(length(e) - 1); x = (e[1:end-1] .+ e[2:end]) ./ 2
            ax = Axis(fig[(i - 1) ÷ 4 + 1, (i - 1) % 4 + 1], title = titles[k],
                      xlabel = per_GeV(k) ? "E_reco (GeV)" : "analysis bin", ylabel = per_GeV(k) ? "events / GeV" : "events")
            λ = expected[k] ./ w
            stairs!(ax, e, vcat(λ, λ[end]), step = :post, color = :blue, label = "prediction")
            errorbars!(ax, x, data[k] ./ w, sqrt.(data[k]) ./ w, color = :black)
            scatter!(ax, x, data[k] ./ w, color = :black, markersize = 6, label = "data")
        end
        fig
    end
end

end
