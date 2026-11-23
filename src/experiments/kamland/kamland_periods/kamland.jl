"""
KamLAND reactor ν̄e disappearance, 2002–2012 data (PRD 88, 033001, arXiv:1303.4667), fitted per data-taking period:
Period 1 (9 Mar 2002 – May 2007, 1486 live days), Period 2 (after the scintillator purification, until the KamLAND-Zen
inner balloon, 1154 d), Period 3 (Oct 2011 – 20 Nov 2012, post-Fukushima reactor shutdown, 351 d).

`configure(physics; dataset = :kamland2011)` instead fits the 2002–2009 data of PRD 83, 052002 as one spectrum (Fig. 1 of
that paper: `data/kamland2011.csv`, `data/efficiency_2011.csv`, with its efficiencies and systematics).

Data (`data/period*.csv`, from the vector graphics of the paper's Fig. 3, see `extraction/`): prompt-energy spectra in
17 bins of 0.425 MeV (0.9–8.125 MeV) per period, with the accidental, ¹³C(α,n)¹⁶O and best-fit geo-ν̄e contributions
(the remainder of KamLAND's best-fit total is ⁹Li/⁸He) and KamLAND's best-fit reactor spectrum (validation only);
selection efficiencies per period (`data/efficiency.csv`, Fig. 3a).

Reactor prediction from the common Daya Bay flux model (`Newtrinos.dayabay_flux`, as in JUNO): Huber–Mueller isotope
spectra, non-equilibrium and spent-fuel corrections, a free spectrum correction and a free mean energy per fission,
with fission-fraction, non-equilibrium and spent-fuel scales and priors shared with JUNO/Daya Bay; IBD cross section of
Vogel–Beacom to first order in 1/M (`Newtrinos.ibd_xsec`), folded with the official two-fold Gauss-Legendre integration
over deposited energy and positron angle (`Newtrinos.Helpers.ibd_integral_mesh`). The thermal power histories of all
Japanese and Korean units (IAEA PRIS, annual load factors placed within the year by shutdown/restart dates, see
`Newtrinos.reactor_flux`) are integrated over each period (live time spread uniformly over the period), baselines from
the site coordinates, 5.98×10³¹ target protons; other reactors worldwide 1.1 % (paper). 3ν survival probabilities with
matter effects in the crust, energy resolution 6.4 %/√E, efficiency of each period at the reconstructed energy.

Systematics: reactor rate 3.5 % (all periods) ⊕ 1.9 % (Periods 2–3; 4.0 % in total), energy scale 1.8 % (its effect on
Δm²₂₁), ¹³C(α,n) 11 %, ⁹Li/⁸He 6 %, accidentals fixed; geo-ν̄e U and Th normalisations free (relative to KamLAND's
reference Earth model, 109 and 27 events), common to all periods, with the spectral shapes of the paper's Fig. 6. Binned Poisson likelihood (KamLAND: unbinned, time-dependent).
"""
module kamland

using DataFrames, CSV, Distributions, LinearAlgebra, Dates
using OrderedCollections
using BAT: distprod
using CairoMakie
import YAML
import ..Newtrinos

const DATA = joinpath(@__DIR__, "data")
const KAMLAND_POS = (36.4225, 137.3153, -1.0)     # lat, lon [deg], depth ~1 km
const TARGET_PROTONS = 5.98e31
const WORLD_FRACTION = 0.011
const EDGES = collect(0.9:0.425:8.125)
const ΔE_NU = 0.782 + 0.010 
const RESOLUTION = 0.064                         # σ_E / E = 6.4 % / √(E/MeV)
const BEST_FIT = (θ₁₂ = atan(sqrt(0.481)), θ₁₃ = asin(sqrt(0.010)), θ₂₃ = 0.785, δCP = 0.0, Δm²₂₁ = 7.54e-5, Δm²₃₁ = 2.4e-3)
const ESCALE_UNC = 0.018
const fission_fractions = [0.567, 0.078, 0.298, 0.057]

"""Data sets: `:periods2013` (PRD 88, 033001: Periods 1–3, default) and `:kamland2011` (PRD 83, 052002: 2002–2009 as one
spectrum, Fig. 1). Per data set: period names, spectrum files (rates per live day), live days, calendar ranges,
efficiency source, rate uncertainties (all periods / additional for periods 2–3), ¹³C(α,n) and ⁹Li uncertainties,
geo-ν̄e reference-model events (U, Th) of the whole data set."""
const DATASETS = (
    periods2013 = (names = (:period1, :period2, :period3), files = ("period1.csv", "period2.csv", "period3.csv"),
                   live_days = (1486.0, 1154.0, 351.0),
                   dates = ((Date(2002, 3, 9), Date(2007, 5, 15)), (Date(2007, 5, 16), Date(2011, 8, 1)), (Date(2011, 10, 12), Date(2012, 11, 20))),
                   efficiency = ("efficiency.csv", (1, 2, 3)),
                   rate_unc = (0.035, sqrt(0.040^2 - 0.035^2)), alpha_n_unc = 0.11, li9_unc = 0.06, geo_reference = (U = 109.0, Th = 27.0)),
    kamland2011 = (names = (:all,), files = ("kamland2011.csv",), live_days = (2135.0,),
                   dates = ((Date(2002, 3, 9), Date(2009, 11, 4)),),
                   efficiency = ("efficiency_2011.csv", nothing),
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
                     xsec = Newtrinos.ibd_xsec.configure(Newtrinos.ibd_xsec.VogelBeacomO1()),
                     flux = Newtrinos.dayabay_flux.configure(Newtrinos.dayabay_flux.ReactorFluxConfig(average = true, overwrite_fission_fractions=fission_fractions)))

function configure(physics = default_physics(); dataset = :periods2013)
    # a physics tuple shared with other experiments may lack the flux or the IBD cross section: use the defaults then
    physics = (; physics.osc,
               xsec = haskey(physics, :xsec) ? physics.xsec : Newtrinos.ibd_xsec.configure(Newtrinos.ibd_xsec.VogelBeacomO1()),
               flux = haskey(physics, :flux) ? physics.flux : Newtrinos.dayabay_flux.configure(Newtrinos.dayabay_flux.ReactorFluxConfig(average = true, overwrite_fission_fractions = fission_fractions)))
    ds = DATASETS[dataset]
    assets = get_assets(physics, ds)
    KamLAND(physics = physics, params = get_params(ds), priors = get_priors(ds), assets = assets,
            forward_model = get_forward_model(physics, assets), plot = get_plot(physics, assets))
end

late(ds) = length(ds.names) > 1
get_params(ds = DATASETS.periods2013) = merge((kamland_ffs = ones(4), kamland_neq_scale = ones(4), kamland_snf_scale = 1.0,),
    (kamland_flux_scale = 0.0,), late(ds) ? (kamland_flux_scale_late = 0.0,) : NamedTuple(),
    (kamland_energy_scale = 0.0, kamland_alpha_n = 0.0, kamland_li9 = 0.0, kamland_geo_u = 1.0, kamland_geo_th = 1.0))

function get_priors(ds = DATASETS.periods2013)
    # fission fractions, non-equilibrium and spent fuel: the same model and priors as JUNO/Daya Bay
    data = YAML.load_file(joinpath(Newtrinos.dayabay_flux.datadir, "dayabay_data/parameters/reactor_fission_fractions_scale.yaml"), dicttype = OrderedDict{String, Any})
    fraction_scale = data["parameters"]["fission_fractions_scale"]
    isotopes = ["U235", "U238", "Pu239", "Pu241"]
    scale = zeros((4, 4))
    for i in 1:4
        for j in 1:4
            scale[i, j] = fraction_scale[isotopes[i]][1] * fraction_scale[isotopes[i]][2] * fraction_scale[isotopes[j]][1] * fraction_scale[isotopes[j]][2] * 0.01 * 0.01
        end
    end
    mu = collect([Float64(fraction_scale[name][1]) for name in isotopes])
    corr_mat = hcat(data["correlations"]["fission_fractions_scale"]["matrix"]...)
    ffs = MvNormal(mu, corr_mat .* scale)

    file = YAML.load_file(joinpath(Newtrinos.dayabay_flux.datadir, "dayabay_data/parameters/reactor_snf.yaml"))
    snf_mu = Float64(file["parameters"]["snf_scale"][1])
    snf_sigma = snf_mu * Float64(file["parameters"]["snf_scale"][2]) * 0.01   # percent
    snf = Normal(snf_mu, snf_sigma)

    file = YAML.load_file(joinpath(Newtrinos.dayabay_flux.datadir, "dayabay_data/parameters/reactor_nonequilibrium_correction.yaml"))
    neq_mu = file["parameters"]["nonequilibrium_scale"][1] .* ones(4)
    neq_sigma = neq_mu .* file["parameters"]["nonequilibrium_scale"][2] .* 0.01 # percent
    neq = MvNormal(neq_mu, Diagonal(neq_sigma.^2))

    merge((kamland_ffs = ffs, kamland_neq_scale = neq, kamland_snf_scale = snf),
        (kamland_flux_scale = Truncated(Normal(0, 1), -4, 4),),
        late(ds) ? (kamland_flux_scale_late = Truncated(Normal(0, 1), -4, 4),) : NamedTuple(),
        (kamland_energy_scale = Truncated(Normal(0, 1), -4, 4), kamland_alpha_n = Truncated(Normal(0, 1), -4, 4),
         kamland_li9 = Truncated(Normal(0, 1), -4, 4), kamland_geo_u = Uniform(0, 4), kamland_geo_th = Uniform(0, 8)))
end

lin(x, xs, ys) = x < xs[1] || x > xs[end] ? zero(eltype(ys)) :
                 (i = clamp(searchsortedlast(xs, x), 1, length(xs) - 1); t = (x - xs[i]) / (xs[i+1] - xs[i]); ys[i] * (1 - t) + ys[i+1] * t)

"""Reactor exposure per site and period: total thermal energy [GW s] of the site's units with their PRIS power histories
and the period's live fraction."""
function reactor_exposure(ds)
    units = Newtrinos.reactor_flux.load_units()
    det = Newtrinos.reactor_flux.ecef(KAMLAND_POS...)
    sites = unique(units.site)
    L = [Newtrinos.reactor_flux.baseline(det, Newtrinos.reactor_flux.ecef(units.lat[findfirst(==(s), units.site)], units.lon[findfirst(==(s), units.site)])) for s in sites]
    W = zeros(length(sites), length(ds.names))
    for (p, (a, b)) in enumerate(ds.dates)
        live = ds.live_days[p] / (Dates.value(b - a) + 1)
        for u in eachrow(units)
            MWd = sum(Newtrinos.reactor_flux.daily_power(u, a, b)) * live
            j = findfirst(==(u.site), sites)
            W[j, p] += MWd * 86400 * 1e-3
        end
    end
    (; sites, L, W)
end

function get_assets(physics, ds = DATASETS.periods2013)
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
    # fine deposited-energy mesh (the analysis edges must lie on it for rebin_index) and the IBD integration on it
    E_fine = collect(0.7:0.025:8.45)
    E_fine_c = (E_fine[1:end-1] + E_fine[2:end]) / 2
    mesh = Newtrinos.Helpers.ibd_integral_mesh(physics.xsec, E_fine)
    E_nu_GeV = mesh.E_nu_mesh ./ 1e3
    ex = reactor_exposure(ds)
    L = vcat(ex.L, 5000.0)
    periods = map(1:np) do p
        d = CSV.read(joinpath(DATA, ds.files[p]), DataFrame)[1:length(EDGES)-1, :]
        days = ds.live_days[p]
        observed = [isnan(x) ? 0 : round(Int, x * days) for x in d.data]
        acc = d.acc_cum .* days
        alpha_n = max.(d.c13_cum .- d.acc_cum, 0) .* days
        li9 = max.(d.total .- d.osc .- d.geo_cum, 0) .* days
        (; observed, acc, alpha_n, li9, geo_u = geo_bins("U", p), geo_th = geo_bins("Th", p), eff = effp[p], days,
           best_fit_reactor = d.osc .* days)
    end
    # exposure per site and period [GW s]; rest of the world: WORLD_FRACTION of all reactor events, constant in time
    W = ex.W
    world = WORLD_FRACTION / (1 - WORLD_FRACTION) * sum(W) .* [ds.live_days[p] for p in 1:np] ./ sum(ds.live_days)
    W = vcat(W, world')
    flux_norm = 1e-45 * TARGET_PROTONS * W .* reshape(inv.(4π * (L .* 1e3).^2), :, 1)
    layers = Newtrinos.osc.StructArray{Newtrinos.Layer}(([6371.0], [1.3], [1.3]))
    (; observed = NamedTuple{ds.names}(Tuple(pd.observed for pd in periods)), ds, E_fine, E_fine_c,
       E_nu_mesh = mesh.E_nu_mesh, E_nu_GeV, integral_weights = mesh.integral_weights, NQ = mesh.NQ, NFc = mesh.NFc,
       final_rebin_idx = Newtrinos.Helpers.rebin_index(E_fine, EDGES), periods, flux_norm, L,
       sites = vcat(ex.sites, "WORLD"), layers, mesh_flux = physics.flux.flux_on_mesh(mesh.E_nu_mesh))
end

"Assets with the flux on the IBD integration mesh attached (the setup reads the flux tables, so compute it once)."
# with_mesh_flux(physics, a) = merge(a, (; mesh_flux = physics.flux.flux_on_mesh(assets.E_nu_mesh)))

"Expected reactor events per period (vectors of 17 bins): IBD folding, oscillations, resolution, energy scale and
selection efficiency."
function reactor_events(params, physics, assets; ε = zero(eltype(params.Δm²₂₁)))
    main, neq, snf, correction = assets.mesh_flux(1.0, params.energy_per_fission, params.kamland_ffs, params.spectrum_pulls,
        params.kamland_neq_scale, params.kamland_snf_scale, 1)
    flux_r = main .* correction .+ neq
    Pmat = hcat(Newtrinos.Helpers.survival_probabilities(physics.osc, assets.E_nu_GeV, assets.L, assets.layers, params)...)
    map(eachindex(assets.periods)) do p
        pd = assets.periods[p]
        interm = (flux_r + snf) .* (Pmat * assets.flux_norm[:, p])
        interm .*= assets.integral_weights
        spec = vec(sum(reshape(interm, assets.NQ, assets.NFc), dims = 1))
        iszero(ε) || (spec = Newtrinos.Helpers.project_to_bins(assets.E_fine .* (1 + ε), spec, assets.E_fine))
        spec = Newtrinos.Helpers.smear_erf(spec, RESOLUTION * sqrt.(assets.E_fine_c), assets.E_fine)
        Newtrinos.Helpers.rebin(spec .* pd.eff.(assets.E_fine_c), assets.final_rebin_idx, length(EDGES) - 1)
    end
end

"Reactor rate factor of period `p` from the rate nuisance parameters."
rate_factor(params, ds, p) = 1 + ds.rate_unc[1] * params.kamland_flux_scale + (p > 1 ? ds.rate_unc[2] * params.kamland_flux_scale_late : 0.0)

"Expected events per period (vectors of 17 bins)."
function get_expected(params, physics, assets)
    #ds = dataset
    ds = assets.ds
    reac = reactor_events(params, physics, assets; ε = ESCALE_UNC * params.kamland_energy_scale)
    map(eachindex(assets.periods)) do p
        pd = assets.periods[p]
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

function get_plot(physics, assets)
    function plot(params, data = assets.observed)
        m = get_expected(params, physics, assets)
        Ec = 0.5 .* (EDGES[1:end-1] .+ EDGES[2:end]); n = length(Ec)
        np = length(assets.periods)
        f = Figure(size = (600, 270 * np))
        for p in 1:np
            ax = Axis(f[p, 1], ylabel = "events / 0.425 MeV", title = "KamLAND $(assets.ds.names[p])", xlabel = p == np ? "E_prompt [MeV]" : "")
            scatter!(ax, Ec, data[assets.ds.names[p]], color = :black)
            stairs!(ax, Ec, m[p], step = :center, color = :blue)
        end
        f
    end
end

end
