module borexino_ph2

using LinearAlgebra
using Distributions
import ..Newtrinos
import ..Newtrinos.solar_common

"""
Borexino Phase-II: interaction rates of pp, pep and ⁷Be (both lines) solar neutrinos,
134 ± 10 (stat) +6 −10 (syst), 2.43 ± 0.36 (stat) +0.15 −0.22 (syst) and 48.3 ± 1.1 (stat)
+0.4 −0.7 (syst) cpd/100 t (M. Agostini et al., Phys. Rev. D 100, 082004 (2019),
arXiv:1707.09279), with the correlations of the spectral fit as emulated by the
Super-Kamiokande collaboration (K. Abe et al., Phys. Rev. D 109, 092001 (2024),
arXiv:2312.12907, Sec. VII D: inverse covariance of the (pp, pep, ⁷Be) rates).
"""
@kwdef struct BorexinoPh2 <: Newtrinos.Experiment
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
    BorexinoPh2(physics = physics, params = (;), priors = (;), assets = assets,
                forward_model = get_forward_model(physics, assets), plot = get_plot(physics, assets))
end

function get_assets(physics)
    @info "Loading Borexino Phase-II data"
    Vinv = [0.00647 -0.0126 -0.0170
            -0.0126  6.36   -0.501
            -0.0170 -0.501   0.784]
    (
        observed = [134.0, 2.43, 48.3],
        cov = Symmetric(inv(Vinv)),
        components = (solar_common.ESTotal(physics, :pp), solar_common.ESTotal(physics, :pep), solar_common.ESTotal(physics, :be7)),
        site = solar_common.Site(physics, 42.45; depth_km=1.4),  # LNGS
    )
end

get_expected(params, physics, assets) =
    [solar_common.es_interaction_rate(physics, assets.site, est, params) * ELECTRONS_PER_100T * DAY for est in assets.components]

function get_forward_model(physics, assets)
    forward_model(params) = MvNormal(get_expected(params, physics, assets), assets.cov)
end

function get_plot(physics, assets)
    plot(params, data=assets.observed) = solar_common.plot_measurements(["pp", "pep", "⁷Be"],
        get_expected(params, physics, assets), data, sqrt.(diag(assets.cov)), "Borexino Phase-II rates")
end

end
