module chlorine

using LinearAlgebra
using Distributions
using CairoMakie
import ..Newtrinos
import ..Newtrinos.solar_common

"""
Homestake chlorine experiment, final result: 2.56 ± 0.16 (stat) ± 0.16 (syst) SNU
(Cleveland et al., ApJ 496, 505 (1998)).
"""
@kwdef struct Chlorine <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

default_physics() = solar_common.default_physics()

function configure(physics=default_physics())
    physics = (; physics.osc, physics.solar_flux, physics.solar_xsec, physics.earth_layers)
    assets = get_assets(physics)
    Chlorine(
        physics = physics,
        params = (;),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets),
    )
end

function get_assets(physics)
    @info "Loading chlorine data"
    (
        observed = [2.56],                   # SNU
        sigma = [sqrt(0.16^2 + 0.16^2)],     # stat ⊕ syst
        threshold = Newtrinos.solar_xsec.CL_THRESHOLD,
        site = solar_common.Site(physics, 44.35; depth_km=1.48),  # Homestake mine, 4850 ft level
    )
end

get_expected(params, physics, assets) =
    solar_common.capture_rate(physics, assets.site, :cl37, assets.threshold, params)

function get_forward_model(physics, assets)
    function forward_model(params)
        rate = sum(get_expected(params, physics, assets))
        MvNormal([rate], Diagonal(assets.sigma .^ 2))
    end
end

function get_plot(physics, assets)
    function plot(params, data=assets.observed)
        solar_common.plot_rates(get_expected(params, physics, assets), data, assets.sigma, ["Homestake"], "Chlorine (³⁷Cl)")
    end
end

end
