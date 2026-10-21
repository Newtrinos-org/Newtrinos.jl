module sage

using LinearAlgebra
using Distributions
using CairoMakie
import ..Newtrinos
import ..Newtrinos.solar_common

"""
SAGE, 1990–2007: 65.4 +3.1 −3.0 (stat) +2.6 −2.8 (syst) SNU (Abdurashitov et al.,
Phys. Rev. C 80, 015807 (2009)). The gallium cross-section uncertainty (`ga_xsec_sigma`, from
`Newtrinos.solar_xsec`) is shared with GALLEX/GNO.
"""
@kwdef struct SAGE <: Newtrinos.Experiment
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
    SAGE(
        physics = physics,
        params = (;),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets),
    )
end

function get_assets(physics)
    @info "Loading SAGE data"
    (
        observed = [65.4],                       # SNU
        sigma = [sqrt(3.05^2 + 2.7^2)],          # symmetrised, stat ⊕ syst
        threshold = Newtrinos.solar_xsec.GA_THRESHOLD,
        site = solar_common.Site(physics, 43.27; depth_km=2.1),  # Baksan
    )
end

get_expected(params, physics, assets) =
    solar_common.capture_rate(physics, assets.site, :ga71, assets.threshold, params)

function get_forward_model(physics, assets)
    function forward_model(params)
        rate = sum(get_expected(params, physics, assets))
        MvNormal([rate], Diagonal(assets.sigma .^ 2))
    end
end

function get_plot(physics, assets)
    function plot(params, data=assets.observed)
        solar_common.plot_rates(get_expected(params, physics, assets), data, assets.sigma, ["SAGE"], "Gallium (⁷¹Ga), Baksan")
    end
end

end
