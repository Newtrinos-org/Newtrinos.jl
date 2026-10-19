module gallex_gno

using LinearAlgebra
using Distributions
using CairoMakie
import ..Newtrinos
import ..Newtrinos.solar_common

"""
GALLEX (reanalysed, 73.4 +7.1 −7.3 SNU; Kaether et al., Phys. Lett. B 685, 47 (2010)) and
GNO (62.9 +6.0 −5.9 SNU, stat ⊕ syst; Altmann et al., Phys. Lett. B 616, 174 (2005)) at
Gran Sasso. The gallium cross-section uncertainty (`ga_xsec_sigma`, from
`Newtrinos.solar_xsec`) is shared with SAGE.
"""
@kwdef struct GallexGNO <: Newtrinos.Experiment
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
    GallexGNO(
        physics = physics,
        params = (;),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets),
    )
end

function get_assets(physics)
    @info "Loading GALLEX/GNO data"
    (
        observed = [73.4, 62.9],   # SNU: GALLEX, GNO
        sigma = [7.2, 5.95],       # symmetrised, stat ⊕ syst
        threshold = Newtrinos.solar_xsec.GA_THRESHOLD,
        site = solar_common.Site(physics, 42.45; depth_km=1.4),  # LNGS
    )
end

get_expected(params, physics, assets) =
    solar_common.capture_rate(physics, assets.site, :ga71, assets.threshold, params)

function get_forward_model(physics, assets)
    function forward_model(params)
        rate = sum(get_expected(params, physics, assets))
        MvNormal([rate, rate], Diagonal(assets.sigma .^ 2))
    end
end

function get_plot(physics, assets)
    function plot(params, data=assets.observed)
        solar_common.plot_rates(get_expected(params, physics, assets), data, assets.sigma, ["GALLEX", "GNO"], "Gallium (⁷¹Ga), Gran Sasso")
    end
end

end
