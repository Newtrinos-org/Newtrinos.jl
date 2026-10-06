module borexino_ph3

using LinearAlgebra
using Distributions
import ..Newtrinos
import ..Newtrinos.solar_common

"""
Borexino Phase-III: CNO (¹³N + ¹⁵O + ¹⁷F) neutrino interaction rate 6.7 +2.0 −0.8 cpd/100 t
(D. Basilico et al., Phys. Rev. D 108, 102005 (2023), arXiv:2205.15975), with a Gaussian of
variance linear in the prediction to describe the asymmetric uncertainty.
"""
@kwdef struct BorexinoPh3 <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

const ELECTRONS_PER_100T = 3.307e31
const DAY = 86400.0

default_physics() = solar_common.default_physics()

function configure(physics=default_physics())
    physics = (; physics.osc, physics.solar_flux, physics.solar_xsec, physics.earth_layers)
    assets = get_assets(physics)
    BorexinoPh3(physics = physics, params = (;), priors = (;), assets = assets,
                forward_model = get_forward_model(physics, assets), plot = get_plot(physics, assets))
end

function get_assets(physics)
    @info "Loading Borexino Phase-III data"
    (
        observed = [6.7],
        sigma_up = 2.0, sigma_down = 0.8,
        components = (solar_common.ESTotal(physics, :n13), solar_common.ESTotal(physics, :o15), solar_common.ESTotal(physics, :f17)),
        site = solar_common.Site(physics, 42.45; depth_km=1.4),  # LNGS
    )
end

get_expected(params, physics, assets) =
    [sum(solar_common.es_interaction_rate(physics, assets.site, est, params) for est in assets.components) * ELECTRONS_PER_100T * DAY]

function get_forward_model(physics, assets)
    function forward_model(params)
        pred = get_expected(params, physics, assets)
        V = solar_common.asymmetric_variance(pred[1], assets.observed[1], assets.sigma_up, assets.sigma_down)
        MvNormal(pred, Diagonal([V]))
    end
end

function get_plot(physics, assets)
    plot(params, data=assets.observed) = solar_common.plot_measurements(["CNO"],
        get_expected(params, physics, assets), data, [(assets.sigma_up + assets.sigma_down) / 2], "Borexino Phase-III CNO rate")
end

end
