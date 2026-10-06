module sk_solar_common

using LinearAlgebra
using Distributions
using DelimitedFiles
using CairoMakie
import ..Newtrinos
import ..Newtrinos.solar_common

"""
    SKPhase

Description of one Super-Kamiokande solar data set.

# Fields
- `name::Symbol`: module / parameter prefix, e.g. `:sk4_solar`.
- `title::String`: plot title.
- `datafile::String`: CSV with one row per data sample (energy bin × solar zenith class):
  `E_lo, E_hi, zenith, cz_lo, cz_hi, rate, stat, mc_b8, mc_hep, syst_uncorr_percent`, rates in
  events/kton/year; `zenith` is `all`, `day` or `night` with the bin `[cz_lo, cz_hi]` in SK's
  cos θ_z convention (night > 0).
- `total_energy::Bool`: bins in recoil total (`true`) or kinetic (`false`) energy.
- `resolution::Function`: σ of the reconstructed energy [MeV] vs electron total energy [MeV].
- `scale_unc`, `reso_unc`: fractional energy-scale and resolution uncertainties.
- `norm_unc`: energy-independent rate systematics not contained in the per-bin errors.
- `phi_b8_mc`, `phi_hep_mc`: ⁸B and hep fluxes [cm⁻² s⁻¹] assumed for the expected rates.
- `night_edges`: solar cos θ_z night binning (SK convention, 0 to 1) used to average the
  Earth regeneration; must contain all zenith bin edges of the data samples.
"""
@kwdef struct SKPhase
    name::Symbol
    title::String
    datafile::String
    total_energy::Bool
    resolution::Function
    scale_unc::Float64
    reso_unc::Float64
    norm_unc::Float64
    phi_b8_mc::Float64
    phi_hep_mc::Float64
    night_edges::Vector{Float64} = collect(range(0, 1, length=21))
end

"""
    SKSolar <: Newtrinos.Experiment

Super-Kamiokande solar ⁸B/hep elastic-scattering spectra of one SK phase.

The expected rate of each sample is SK's unoscillated simulation scaled by the ratio of the
oscillated to the unoscillated rate computed here: the ⁸B and hep spectra and fluxes from
`Newtrinos.solar_flux` (including `solar_b8_shape`), ν–e scattering, a Gaussian energy
response with energy-scale and resolution nuisance parameters, and oscillation probabilities
for the day or averaged over the night zenith bins (Earth regeneration). A normalisation
nuisance covers the energy-independent systematics. Statistical and energy-uncorrelated
systematic uncertainties enter a Gaussian likelihood; the latter are fully correlated between
the zenith samples of an energy bin.
"""
@kwdef struct SKSolar <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

param(phase, s) = Symbol(phase.name, :_, s)

get_params(phase::SKPhase) = NamedTuple{(param(phase, :escale), param(phase, :ereso), param(phase, :norm))}((0.0, 0.0, 0.0))
get_priors(phase::SKPhase) = NamedTuple{(param(phase, :escale), param(phase, :ereso), param(phase, :norm))}(
    ntuple(_ -> Truncated(Normal(0.0, 1.0), -3, 3), 3))

"""
    configure(phase::SKPhase, physics) -> SKSolar
"""
function configure(phase::SKPhase, physics)
    physics = (; physics.osc, physics.solar_flux, physics.solar_xsec, physics.earth_layers)
    assets = get_assets(phase, physics)
    SKSolar(
        physics = physics,
        params = get_params(phase),
        priors = get_priors(phase),
        assets = assets,
        forward_model = get_forward_model(phase, physics, assets),
        plot = get_plot(phase, physics, assets),
    )
end

function get_assets(phase::SKPhase, physics)
    @info "Loading $(phase.title) data"
    d, h = readdlm(phase.datafile, ',', Any, '\n'; header=true, comments=true)
    col(name) = d[:, findfirst(==(name), vec(h))]
    num(name) = Float64.(col(name))
    E_lo, E_hi, zen, cz_lo, cz_hi = num("E_lo"), num("E_hi"), String.(col("zenith")), num("cz_lo"), num("cz_hi")
    rate, stat, syst = num("rate"), num("stat"), num("syst_uncorr_percent") ./ 100

    # energy bins and the samples' bin index
    bins = unique(collect(zip(E_lo, E_hi)))
    bin_of = [findfirst(==((E_lo[i], E_hi[i])), bins) for i in eachindex(E_lo)]
    mc_b8 = [num("mc_b8")[findfirst(==(k), bin_of)] for k in eachindex(bins)]
    mc_hep = [num("mc_hep")[findfirst(==(k), bin_of)] for k in eachindex(bins)]

    # zenith weights per sample over [day; night bins]; Site uses cos(zenith) of the Sun = -cos θ_z(SK)
    site = solar_common.Site(physics, 36.43; depth_km=1.0, night_edges=reverse(-phase.night_edges))
    sk_cz = -site.cz_night  # SK convention, night > 0
    W = zeros(length(rate), 1 + length(sk_cz))
    for i in eachindex(rate)
        if zen[i] == "day"
            W[i, 1] = 1
        elseif zen[i] == "all"
            W[i, :] = vcat(site.w_day, site.w_night)
        else
            inbin = (sk_cz .> cz_lo[i]) .& (sk_cz .< cz_hi[i])
            any(inbin) || error("$(phase.name): no night zenith bin inside [$(cz_lo[i]), $(cz_hi[i])]")
            W[i, 2:end] = site.w_night .* inbin
        end
        W[i, :] ./= sum(W[i, :])
    end

    # covariance: statistics, plus energy-uncorrelated systematics correlated within an energy bin
    s = syst .* rate
    cov = Diagonal(stat .^ 2) .+ [bin_of[i] == bin_of[j] ? s[i] * s[j] : 0.0 for i in eachindex(s), j in eachindex(s)]

    E = collect(range(2.0, 18.8, length=120))
    resp = solar_common.ESResponse(physics.solar_xsec, E, bins, phase.resolution;
                                   scale_unc=phase.scale_unc, reso_unc=phase.reso_unc, total_energy=phase.total_energy)
    sf = physics.solar_flux
    rate0_b8 = solar_common.es_rates(resp, sf.spectrum(:b8, E, sf.params) .* (phase.phi_b8_mc / sf.nominal.b8), ones(length(E)))
    rate0_hep = solar_common.es_rates(resp, sf.spectrum(:hep, E, sf.params) .* (phase.phi_hep_mc / sf.nominal.hep), ones(length(E)))

    (
        observed = rate,
        cov = Symmetric(Matrix(cov)),
        samples = (E_lo = E_lo, E_hi = E_hi, zenith = zen, cz_lo = cz_lo, cz_hi = cz_hi),
        bins = bins, bin_of = bin_of, W = W,
        mc_b8 = mc_b8, mc_hep = mc_hep, rate0_b8 = rate0_b8, rate0_hep = rate0_hep,
        resp = resp, site = site,
    )
end

"""
    get_expected(phase, params, physics, assets) -> Vector

Expected rate [events/kton/year] of each data sample.
"""
function get_expected(phase::SKPhase, params, physics, assets)
    sf, resp = physics.solar_flux, assets.resp
    pulls = (scale = params[param(phase, :escale)], reso = params[param(phase, :ereso)])
    # rates per energy bin (rows) and zenith column [day, night bins...] (columns)
    function rates(comp, rate0, mc)
        P = solar_common.survival(physics, assets.site, comp, resp.E, params)
        Pe = hcat(P.day[:, 1], P.night[:, :, 1])
        r = solar_common.es_rates(resp, sf.spectrum(comp, resp.E, params), Pe; pulls...)
        mc .* r ./ rate0
    end
    R = rates(:b8, assets.rate0_b8, assets.mc_b8) .+ rates(:hep, assets.rate0_hep, assets.mc_hep)
    norm = 1 + phase.norm_unc * params[param(phase, :norm)]
    norm .* [sum(assets.W[i, :] .* R[assets.bin_of[i], :]) for i in eachindex(assets.bin_of)]
end

function get_forward_model(phase::SKPhase, physics, assets)
    function forward_model(params)
        MvNormal(get_expected(phase, params, physics, assets), assets.cov)
    end
end

function get_plot(phase::SKPhase, physics, assets)
    function plot(params, data=assets.observed)
        e = solar_common.ForwardDiffValue.(get_expected(phase, params, physics, assets))
        s = assets.samples
        mc = (assets.mc_b8 .+ assets.mc_hep)[assets.bin_of]
        err = sqrt.(diag(assets.cov))
        classes = unique(collect(zip(s.zenith, s.cz_lo, s.cz_hi)))
        label(c) = c[1] == "night" && (c[2], c[3]) != (0.0, 1.0) ? "night $(c[2])–$(c[3])" : c[1]
        f = Figure()
        ax = Axis(f[1, 1], title=phase.title, ylabel="Data / MC (unoscillated)",
                  xlabel=phase.total_energy ? "Recoil electron total energy (MeV)" : "Recoil electron kinetic energy (MeV)")
        for (j, c) in enumerate(classes)
            idx = findall(i -> (s.zenith[i], s.cz_lo[i], s.cz_hi[i]) == c, eachindex(s.zenith))
            x = (s.E_lo[idx] .+ s.E_hi[idx]) ./ 2 .+ 0.04 * (j - (length(classes) + 1) / 2)
            color = Makie.wong_colors()[mod1(j, 7)]
            errorbars!(ax, x, data[idx] ./ mc[idx], err[idx] ./ mc[idx], color=color)
            scatter!(ax, x, data[idx] ./ mc[idx], color=color, label=label(c))
            lines!(ax, x, e[idx] ./ mc[idx], color=color)
        end
        ylims!(ax, 0.2, 0.8)
        axislegend(ax, position=:rt, labelsize=10)
        f
    end
end

end
