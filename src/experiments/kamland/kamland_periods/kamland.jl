"""
KamLAND reactor ν̄e disappearance, 2002–2012 data (PRD 88, 033001, arXiv:1303.4667), fitted per data-taking period:
Period 1 (9 Mar 2002 – May 2007, 1486 live days), Period 2 (after the scintillator purification, until the KamLAND-Zen
inner balloon, 1154 d), Period 3 (Oct 2011 – 20 Nov 2012, post-Fukushima reactor shutdown, 351 d).

`configure(physics; dataset = :kamland2011)` instead fits the 2002–2009 data of PRD 83, 052002 as one spectrum (Fig. 1 of
that paper: `data/kamland2011.csv`, `data/efficiency_2011.csv`, with its fission fractions and systematics).

Data (`data/period*.csv`, from the vector graphics of the paper's Fig. 3, see `extraction/`): prompt-energy spectra in
17 bins of 0.425 MeV (0.9–8.125 MeV) per period, with the accidental, ¹³C(α,n)¹⁶O and best-fit geo-ν̄e contributions
(the remainder of KamLAND's best-fit total is ⁹Li/⁸He) and KamLAND's best-fit reactor spectrum (validation only);
selection efficiencies per period (`data/efficiency.csv`, Fig. 3a).

Reactor prediction from first principles with the common reactor model (`Newtrinos.reactor_flux`): thermal power
histories of all Japanese and Korean units (IAEA PRIS, annual load factors placed within the year by shutdown/restart
dates) integrated over each period (live time spread uniformly over the period), baselines from the site coordinates,
Huber–Mueller spectra with KamLAND's average fission fractions, rate anchored to Bugey-4 (as KamLAND), IBD cross
section of Strumia–Vissani, 5.98×10³¹ target protons, +0.7 % long-lived fission products; other reactors worldwide
1.1 % (paper). 3ν survival probabilities with matter effects in the crust, energy resolution 6.4 %/√E, efficiency of
each period at the reconstructed energy.

Systematics: reactor rate 3.5 % (all periods) ⊕ 1.9 % (Periods 2–3; 4.0 % in total), energy scale 1.8 % (its effect on
Δm²₂₁), ¹³C(α,n) 11 %, ⁹Li/⁸He 6 %, accidentals fixed; geo-ν̄e U and Th normalisations free (relative to KamLAND's
reference Earth model, 109 and 27 events), common to all periods, with the spectral shapes of the paper's Fig. 6. Binned Poisson likelihood (KamLAND: unbinned, time-dependent).
"""
module kamland

using DataFrames, CSV, Distributions, LinearAlgebra, SpecialFunctions, Dates
using BAT: distprod
using CairoMakie
import ..Newtrinos

const DATA = joinpath(@__DIR__, "data")
const KAMLAND_POS = (36.4225, 137.3153, -1.0)     # lat, lon [deg], depth ~1 km
const TARGET_PROTONS = 5.98e31
const LONG_LIVED = 1.007
const WORLD_FRACTION = 0.011
const EDGES = collect(0.9:0.425:8.125)
const ΔE_NU = 0.782 + 0.010                      # E_ν = E_prompt + 0.782 MeV + mean neutron recoil
const RESOLUTION = 0.064                         # σ_E / E = 6.4 % / √(E/MeV)
const BEST_FIT = (θ₁₂ = atan(sqrt(0.481)), θ₁₃ = asin(sqrt(0.010)), θ₂₃ = 0.785, δCP = 0.0, Δm²₂₁ = 7.54e-5, Δm²₃₁ = 2.4e-3)
const ESCALE_UNC = 0.018

"""Data sets: `:periods2013` (PRD 88, 033001: Periods 1–3, default) and `:kamland2011` (PRD 83, 052002: 2002–2009 as one
spectrum, Fig. 1). Per data set: period names, spectrum files (rates per live day), live days, calendar ranges,
efficiency source, fission fractions, rate uncertainties (all periods / additional for periods 2–3), ¹³C(α,n) and ⁹Li
uncertainties, geo-ν̄e reference-model events (U, Th) of the whole data set."""
const DATASETS = (
    periods2013 = (names = (:period1, :period2, :period3), files = ("period1.csv", "period2.csv", "period3.csv"),
                   live_days = (1486.0, 1154.0, 351.0),
                   dates = ((Date(2002, 3, 9), Date(2007, 5, 15)), (Date(2007, 5, 16), Date(2011, 8, 1)), (Date(2011, 10, 12), Date(2012, 11, 20))),
                   efficiency = ("efficiency.csv", (1, 2, 3)), fractions = (U235 = 0.567, U238 = 0.078, Pu239 = 0.298, Pu241 = 0.057),
                   rate_unc = (0.035, sqrt(0.040^2 - 0.035^2)), alpha_n_unc = 0.11, li9_unc = 0.06, geo_reference = (U = 109.0, Th = 27.0)),
    kamland2011 = (names = (:all,), files = ("kamland2011.csv",), live_days = (2135.0,),
                   dates = ((Date(2002, 3, 9), Date(2009, 11, 4)),),
                   efficiency = ("efficiency_2011.csv", nothing), fractions = (U235 = 0.571, U238 = 0.078, Pu239 = 0.295, Pu241 = 0.056),
                   rate_unc = (0.041, 0.0), alpha_n_unc = 23.0 / 198.6, li9_unc = 1.6 / 24.8, geo_reference = (U = 85.0, Th = 21.0)),
)

@kwdef struct KamLAND <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

default_physics() = (osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction = Newtrinos.osc.SI())),
                     reactor_flux = Newtrinos.reactor_flux.configure())

function configure(physics = default_physics(); dataset = :periods2013)
    physics = (; physics.osc, physics.reactor_flux)
    ds = DATASETS[dataset]
    assets = get_assets(physics, ds)
    KamLAND(physics = physics, params = get_params(ds), priors = get_priors(ds), assets = assets,
            forward_model = get_forward_model(physics, assets), plot = get_plot(physics, assets))
end

late(ds) = length(ds.names) > 1
get_params(ds = DATASETS.periods2013) = merge((kamland_flux_scale = 0.0,), late(ds) ? (kamland_flux_scale_late = 0.0,) : NamedTuple(),
    (kamland_energy_scale = 0.0, kamland_alpha_n = 0.0, kamland_li9 = 0.0, kamland_geo_u = 1.0, kamland_geo_th = 1.0))
get_priors(ds = DATASETS.periods2013) = merge((kamland_flux_scale = Truncated(Normal(0, 1), -4, 4),),
    late(ds) ? (kamland_flux_scale_late = Truncated(Normal(0, 1), -4, 4),) : NamedTuple(),
    (kamland_energy_scale = Truncated(Normal(0, 1), -4, 4), kamland_alpha_n = Truncated(Normal(0, 1), -4, 4),
     kamland_li9 = Truncated(Normal(0, 1), -4, 4), kamland_geo_u = Uniform(0, 4), kamland_geo_th = Uniform(0, 8)))

lin(x, xs, ys) = x < xs[1] || x > xs[end] ? zero(eltype(ys)) :
                 (i = clamp(searchsortedlast(xs, x), 1, length(xs) - 1); t = (x - xs[i]) / (xs[i+1] - xs[i]); ys[i] * (1 - t) + ys[i+1] * t)

"Response matrix (analysis bins × true prompt energies): resolution smearing for energy scale 1 + ε, times the period's
efficiency at the reconstructed energy."
response(Et, eff, ε = 0.0) = [(m = e * (1 + ε); s = RESOLUTION * sqrt(m);
                               eff(m) * 0.5 * (erf((EDGES[b+1] - m) / (sqrt(2) * s)) - erf((EDGES[b] - m) / (sqrt(2) * s))))
                              for b in 1:length(EDGES)-1, e in Et]

"""Reactor exposure per site and period: target protons × fissions / (4π L²) [fissions cm⁻²], summed over the units of the
site with their PRIS power histories and the period's live fraction."""
function reactor_exposure(rf, ds)
    units = Newtrinos.reactor_flux.load_units()
    det = Newtrinos.reactor_flux.ecef(KAMLAND_POS...)
    sites = unique(units.site)
    L = [Newtrinos.reactor_flux.baseline(det, Newtrinos.reactor_flux.ecef(units.lat[findfirst(==(s), units.site)], units.lon[findfirst(==(s), units.site)])) for s in sites]
    fps = rf.fission_rate(1.0, ds.fractions)                            # fissions / s / MW
    W = zeros(length(sites), length(ds.names))
    for (p, (a, b)) in enumerate(ds.dates)
        live = ds.live_days[p] / (Dates.value(b - a) + 1)
        for u in eachrow(units)
            MWd = sum(Newtrinos.reactor_flux.daily_power(u, a, b)) * live
            j = findfirst(==(u.site), sites)
            W[j, p] += TARGET_PROTONS * MWd * 86400 * fps / (4π * (L[j] * 1e5)^2)
        end
    end
    (; sites, L, W)
end

function get_assets(physics, ds = DATASETS.periods2013)
    rf = physics.reactor_flux
    np = length(ds.names)
    # selection efficiencies of the data set's periods; geo-ν̄e shapes are those of the 2013 paper (Fig. 6, all periods,
    # efficiency included), with the 2013 efficiencies removed
    e13 = CSV.read(joinpath(DATA, "efficiency.csv"), DataFrame)
    e13p = [(e -> lin(clamp(e, 0.95, 8.45), e13.E_MeV[e13.period .== p], e13.efficiency[e13.period .== p])) for p in 1:3]
    lt13 = DATASETS.periods2013.live_days
    eff_all13(e) = sum(lt13[p] * e13p[p](e) for p in 1:3) / sum(lt13)
    effp = if ds.efficiency[2] === nothing
        e = CSV.read(joinpath(DATA, ds.efficiency[1]), DataFrame; comment = "#")
        [(x -> lin(clamp(x, e.E_MeV[1], e.E_MeV[end]), e.E_MeV, e.efficiency))]
    else
        [e13p[p] for p in ds.efficiency[2]]
    end
    g = CSV.read(joinpath(DATA, "geo_shapes.csv"), DataFrame; comment = "#")
    geo_bins(col, p) = begin
        s = g[!, col] ./ eff_all13.(g.E_MeV)                                  # true-rate shape
        tot = sum(s[k] * effp[q](g.E_MeV[k]) * ds.live_days[q] for k in eachindex(s), q in 1:np) / sum(ds.live_days)
        [sum(s[k] * effp[p](g.E_MeV[k]) for k in eachindex(s) if EDGES[b] <= g.E_MeV[k] < EDGES[b+1]; init = 0.0) for b in 1:length(EDGES)-1] ./ tot .*
            ds.geo_reference[Symbol(col)] * ds.live_days[p] / sum(ds.live_days)
    end
    # true prompt energies and ν̄e flux × IBD cross section per fission [cm² / MeV / fission], Bugey-4 anchored
    Et = collect(1.0:0.01:9.0); dE = 0.01
    Eν = Et .+ ΔE_NU
    norm = Newtrinos.reactor_flux.bugey4_scale(ds.fractions) * LONG_LIVED
    φσ = [rf.spectrum(e, ds.fractions) * rf.ibd_xsec(e) * norm * dE for e in Eν]
    ex = reactor_exposure(rf, ds)
    L = vcat(ex.L, 5000.0)
    periods = map(1:np) do p
        d = CSV.read(joinpath(DATA, ds.files[p]), DataFrame)[1:length(EDGES)-1, :]
        days = ds.live_days[p]
        observed = [isnan(x) ? 0 : round(Int, x * days) for x in d.data]
        acc = d.acc_cum .* days
        alpha_n = max.(d.c13_cum .- d.acc_cum, 0) .* days
        li9 = max.(d.total .- d.osc .- d.geo_cum, 0) .* days
        R = response(Et, effp[p])
        (; observed, acc, alpha_n, li9, geo_u = geo_bins("U", p), geo_th = geo_bins("Th", p), eff = effp[p], R, days,
           best_fit_reactor = d.osc .* days)
    end
    # weights per baseline and period; rest of the world: WORLD_FRACTION of all reactor events, constant in time
    W = ex.W
    world = WORLD_FRACTION / (1 - WORLD_FRACTION) * sum(W) .* [ds.live_days[p] for p in 1:np] ./ sum(ds.live_days)
    W = vcat(W, world')
    layers = Newtrinos.osc.StructArray{Newtrinos.Layer}(([6371.0], [1.3], [1.3]))
    paths = Newtrinos.osc.VectorOfVectors{Newtrinos.osc.Path}([[Newtrinos.osc.Path(l, 1)] for l in L])
    (; observed = NamedTuple{ds.names}(Tuple(pd.observed for pd in periods)), ds, Et, Eν, φσ, periods, W, L,
       sites = vcat(ex.sites, "WORLD"), layers, paths)
end

"Expected reactor events per period (vectors of 17 bins)."
function reactor_events(params, physics, a; ε = zero(eltype(params.Δm²₂₁)))
    P = physics.osc.osc_prob(a.Eν .* 1e-3, a.paths, a.layers, params; anti = true)[:, :, 1, 1]      # energies × baselines
    map(eachindex(a.periods)) do p
        pd = a.periods[p]
        R = iszero(ε) ? pd.R : response(a.Et, pd.eff, ε)
        R * (a.φσ .* (P * a.W[:, p]))
    end
end

"Reactor rate factor of period `p` from the rate nuisance parameters."
rate_factor(params, ds, p) = 1 + ds.rate_unc[1] * params.kamland_flux_scale + (p > 1 ? ds.rate_unc[2] * params.kamland_flux_scale_late : 0.0)

"Expected events per period (vectors of 17 bins)."
function get_expected(params, physics, a)
    ds = a.ds
    reac = reactor_events(params, physics, a; ε = ESCALE_UNC * params.kamland_energy_scale)
    map(eachindex(a.periods)) do p
        pd = a.periods[p]
        reac[p] .* rate_factor(params, ds, p) .+ pd.acc .+ pd.alpha_n .* (1 + ds.alpha_n_unc * params.kamland_alpha_n) .+
            pd.li9 .* (1 + ds.li9_unc * params.kamland_li9) .+ pd.geo_u .* params.kamland_geo_u .+ pd.geo_th .* params.kamland_geo_th
    end
end

function get_forward_model(physics, assets)
    function forward_model(params)
        μ = get_expected(params, physics, assets)
        distprod(; (assets.ds.names[p] => distprod(Poisson.(max.(μ[p], 1e-9))) for p in eachindex(μ))...)
    end
end

function get_plot(physics, a)
    function plot(params, data = a.observed)
        m = get_expected(params, physics, a)
        Ec = 0.5 .* (EDGES[1:end-1] .+ EDGES[2:end]); n = length(Ec)
        np = length(a.periods)
        f = Figure(size = (600, 270 * np))
        for p in 1:np
            ax = Axis(f[p, 1], ylabel = "events / 0.425 MeV", title = "KamLAND $(a.ds.names[p])", xlabel = p == np ? "E_prompt [MeV]" : "")
            scatter!(ax, Ec, data[a.ds.names[p]], color = :black)
            stairs!(ax, Ec, m[p], step = :center, color = :blue)
        end
        f
    end
end

end
