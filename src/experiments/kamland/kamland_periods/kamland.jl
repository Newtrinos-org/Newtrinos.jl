"""
KamLAND reactor ν̄e disappearance, 2002–2012 data (PRD 88, 033001, arXiv:1303.4667), fitted per data-taking period:
Period 1 (9 Mar 2002 – May 2007, 1486 live days), Period 2 (after the scintillator purification, until the KamLAND-Zen
inner balloon, 1154 d), Period 3 (Oct 2011 – 20 Nov 2012, post-Fukushima reactor shutdown, 351 d).

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
const LIVE_DAYS = (1486.0, 1154.0, 351.0)
const PERIODS = ((Date(2002, 3, 9), Date(2007, 5, 15)), (Date(2007, 5, 16), Date(2011, 8, 1)), (Date(2011, 10, 12), Date(2012, 11, 20)))
const GEO_REFERENCE = (U = 109.0, Th = 27.0)      # reference Earth model, all periods (paper)
const KAMLAND_POS = (36.4225, 137.3153, -1.0)     # lat, lon [deg], depth ~1 km
const TARGET_PROTONS = 5.98e31
const FISSION_FRACTIONS = (U235 = 0.567, U238 = 0.078, Pu239 = 0.298, Pu241 = 0.057)
const LONG_LIVED = 1.007
const WORLD_FRACTION = 0.011
const PERIOD_NAMES = (:period1, :period2, :period3)
const EDGES = collect(0.9:0.425:8.125)
const ΔE_NU = 0.782 + 0.010                      # E_ν = E_prompt + 0.782 MeV + mean neutron recoil
const RESOLUTION = 0.064                         # σ_E / E = 6.4 % / √(E/MeV)
const BEST_FIT = (θ₁₂ = atan(sqrt(0.481)), θ₁₃ = asin(sqrt(0.010)), θ₂₃ = 0.785, δCP = 0.0, Δm²₂₁ = 7.54e-5, Δm²₃₁ = 2.4e-3)
const RATE_UNC = 0.035
const RATE_UNC_LATE = sqrt(0.040^2 - 0.035^2)
const ESCALE_UNC = 0.018
const ALPHA_N_UNC = 0.11
const LI9_UNC = 0.06

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

function configure(physics = default_physics())
    physics = (; physics.osc, physics.reactor_flux)
    assets = get_assets(physics)
    KamLAND(physics = physics, params = get_params(), priors = get_priors(), assets = assets,
            forward_model = get_forward_model(physics, assets), plot = get_plot(physics, assets))
end

get_params() = (kamland_flux_scale = 0.0, kamland_flux_scale_late = 0.0, kamland_energy_scale = 0.0, kamland_alpha_n = 0.0,
                kamland_li9 = 0.0, kamland_geo_u = 1.0, kamland_geo_th = 1.0)
get_priors() = (kamland_flux_scale = Truncated(Normal(0, 1), -4, 4), kamland_flux_scale_late = Truncated(Normal(0, 1), -4, 4),
                kamland_energy_scale = Truncated(Normal(0, 1), -4, 4), kamland_alpha_n = Truncated(Normal(0, 1), -4, 4),
                kamland_li9 = Truncated(Normal(0, 1), -4, 4), kamland_geo_u = Uniform(0, 4), kamland_geo_th = Uniform(0, 8))

lin(x, xs, ys) = x < xs[1] || x > xs[end] ? zero(eltype(ys)) :
                 (i = clamp(searchsortedlast(xs, x), 1, length(xs) - 1); t = (x - xs[i]) / (xs[i+1] - xs[i]); ys[i] * (1 - t) + ys[i+1] * t)

"Response matrix (analysis bins × true prompt energies): resolution smearing for energy scale 1 + ε, times the period's
efficiency at the reconstructed energy."
response(Et, eff, ε = 0.0) = [(m = e * (1 + ε); s = RESOLUTION * sqrt(m);
                               eff(m) * 0.5 * (erf((EDGES[b+1] - m) / (sqrt(2) * s)) - erf((EDGES[b] - m) / (sqrt(2) * s))))
                              for b in 1:length(EDGES)-1, e in Et]

"""Reactor exposure per site and period: target protons × fissions / (4π L²) [fissions cm⁻²], summed over the units of the
site with their PRIS power histories and the period's live fraction; plus the rest of the world (`:WORLD`)."""
function reactor_exposure(rf)
    units = Newtrinos.reactor_flux.load_units()
    det = Newtrinos.reactor_flux.ecef(KAMLAND_POS...)
    sites = unique(units.site)
    L = [Newtrinos.reactor_flux.baseline(det, Newtrinos.reactor_flux.ecef(units.lat[findfirst(==(s), units.site)], units.lon[findfirst(==(s), units.site)])) for s in sites]
    fps = rf.fission_rate(1.0, FISSION_FRACTIONS)                       # fissions / s / MW
    W = zeros(length(sites), 3)
    for (p, (a, b)) in enumerate(PERIODS)
        live = LIVE_DAYS[p] / (Dates.value(b - a) + 1)
        for u in eachrow(units)
            MWd = sum(Newtrinos.reactor_flux.daily_power(u, a, b)) * live
            j = findfirst(==(u.site), sites)
            W[j, p] += TARGET_PROTONS * MWd * 86400 * fps / (4π * (L[j] * 1e5)^2)
        end
    end
    (; sites, L, W)
end

function get_assets(physics)
    rf = physics.reactor_flux
    eff = CSV.read(joinpath(DATA, "efficiency.csv"), DataFrame)
    effp = [(e -> lin(clamp(e, 0.95, 8.45), eff.E_MeV[eff.period .== p], eff.efficiency[eff.period .== p])) for p in 1:3]
    eff_all(e) = sum(LIVE_DAYS[p] * effp[p](e) for p in 1:3) / sum(LIVE_DAYS)
    # geo-ν̄e U and Th: shapes of Fig. 6 (all periods, efficiency included) → per period and bin, normalised to the
    # reference Earth model (kamland_geo_u = kamland_geo_th = 1)
    g = CSV.read(joinpath(DATA, "geo_shapes.csv"), DataFrame; comment = "#")
    geo_bins(col, p) = begin
        s = g[!, col] ./ eff_all.(g.E_MeV)
        tot = sum(s .* eff_all.(g.E_MeV))
        [sum(s[k] * effp[p](g.E_MeV[k]) for k in eachindex(s) if EDGES[b] <= g.E_MeV[k] < EDGES[b+1]; init = 0.0) for b in 1:length(EDGES)-1] ./ tot .*
            GEO_REFERENCE[Symbol(col)] * LIVE_DAYS[p] / sum(LIVE_DAYS)
    end
    # true prompt energies and ν̄e flux × IBD cross section per fission [cm² / MeV / fission], Bugey-4 anchored
    Et = collect(1.0:0.01:9.0); dE = 0.01
    Eν = Et .+ ΔE_NU
    norm = Newtrinos.reactor_flux.bugey4_scale(FISSION_FRACTIONS) * LONG_LIVED
    φσ = [rf.spectrum(e, FISSION_FRACTIONS) * rf.ibd_xsec(e) * norm * dE for e in Eν]
    ex = reactor_exposure(rf)
    L = vcat(ex.L, 5000.0)
    periods = map(1:3) do p
        d = CSV.read(joinpath(DATA, "period$p.csv"), DataFrame)[1:length(EDGES)-1, :]
        days = LIVE_DAYS[p]
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
    tot = sum(W)
    world = WORLD_FRACTION / (1 - WORLD_FRACTION) * tot .* [LIVE_DAYS[p] for p in 1:3] ./ sum(LIVE_DAYS)
    W = vcat(W, world')
    layers = Newtrinos.osc.StructArray{Newtrinos.Layer}(([6371.0], [1.3], [1.3]))
    paths = Newtrinos.osc.VectorOfVectors{Newtrinos.osc.Path}([[Newtrinos.osc.Path(l, 1)] for l in L])
    (; observed = NamedTuple{PERIOD_NAMES}(Tuple(pd.observed for pd in periods)), Et, Eν, φσ, periods, W, L, sites = vcat(ex.sites, "WORLD"), layers, paths)
end

"Expected reactor events per period (3 vectors of 17 bins); `scale` multiplies the period's prediction."
function reactor_events(params, physics, a; ε = zero(eltype(params.Δm²₂₁)))
    P = physics.osc.osc_prob(a.Eν .* 1e-3, a.paths, a.layers, params; anti = true)[:, :, 1, 1]      # energies × baselines
    map(1:3) do p
        pd = a.periods[p]
        R = iszero(ε) ? pd.R : response(a.Et, pd.eff, ε)
        R * (a.φσ .* (P * a.W[:, p]))
    end
end

"Expected events per period (vector of 3 vectors of 17 bins)."
function get_expected(params, physics, a)
    reac = reactor_events(params, physics, a; ε = ESCALE_UNC * params.kamland_energy_scale)
    map(1:3) do p
        pd = a.periods[p]
        rate = 1 + RATE_UNC * params.kamland_flux_scale + (p > 1 ? RATE_UNC_LATE * params.kamland_flux_scale_late : 0.0)
        reac[p] .* rate .+ pd.acc .+ pd.alpha_n .* (1 + ALPHA_N_UNC * params.kamland_alpha_n) .+
            pd.li9 .* (1 + LI9_UNC * params.kamland_li9) .+ pd.geo_u .* params.kamland_geo_u .+ pd.geo_th .* params.kamland_geo_th
    end
end

function get_forward_model(physics, assets)
    function forward_model(params)
        μ = get_expected(params, physics, assets)
        distprod(; (PERIOD_NAMES[p] => distprod(Poisson.(max.(μ[p], 1e-9))) for p in 1:3)...)
    end
end

function get_plot(physics, a)
    function plot(params, data = a.observed)
        m = get_expected(params, physics, a)
        Ec = 0.5 .* (EDGES[1:end-1] .+ EDGES[2:end]); n = length(Ec)
        f = Figure(size = (600, 800))
        for p in 1:3
            ax = Axis(f[p, 1], ylabel = "events / 0.425 MeV", title = "KamLAND period $p", xlabel = p == 3 ? "E_prompt [MeV]" : "")
            scatter!(ax, Ec, data[PERIOD_NAMES[p]], color = :black)
            stairs!(ax, Ec, m[p], step = :center, color = :blue)
        end
        f
    end
end

end
