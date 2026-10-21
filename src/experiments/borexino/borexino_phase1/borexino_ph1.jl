module borexino_ph1

using LinearAlgebra
using Distributions
import ..Newtrinos
import ..Newtrinos.solar_common

"""
Borexino Phase-I: interaction rates of the 0.862 MeV ⁷Be line, 46.0 ± 1.5 (stat) +1.6 −1.5 (syst)
cpd/100 t (G. Bellini et al., Phys. Rev. Lett. 107, 141302 (2011), arXiv:1104.1816), and of ⁸B
ν–e scattering above 3 MeV, 0.217 ± 0.038 (stat) ± 0.008 (syst) cpd/100 t (G. Bellini et al.,
Phys. Rev. D 82, 033006 (2010), arXiv:0808.2868). Expected rates use 3.307×10³¹ target
electrons per 100 t.
"""
@kwdef struct BorexinoPh1 <: Newtrinos.Experiment
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
    BorexinoPh1(physics = physics, params = (;), priors = (;), assets = assets,
                forward_model = get_forward_model(physics, assets), plot = get_plot(physics, assets))
end

function get_assets(physics)
    @info "Loading Borexino Phase-I data"
    (
        observed = [46.0, 0.217],
        sigma = [sqrt(1.5^2 + 1.55^2), sqrt(0.038^2 + 0.008^2)],
        be7 = solar_common.ESTotal(physics, :be7; line_energies=[0.8613]),
        b8 = solar_common.ESTotal(physics, :b8; T_min=3.0),
        hep = solar_common.ESTotal(physics, :hep; T_min=3.0),
        site = solar_common.Site(physics, 42.45; depth_km=1.4),  # LNGS
    )
end

function get_expected(params, physics, assets)
    r(est) = solar_common.es_interaction_rate(physics, assets.site, est, params) * ELECTRONS_PER_100T * DAY
    [r(assets.be7), r(assets.b8) + r(assets.hep)]
end

function get_forward_model(physics, assets)
    forward_model(params) = MvNormal(get_expected(params, physics, assets), Diagonal(assets.sigma .^ 2))
end

function get_plot(physics, assets)
    plot(params, data=assets.observed) = solar_common.plot_measurements(["⁷Be (862 keV)", "⁸B (T > 3 MeV)"],
        get_expected(params, physics, assets), data, assets.sigma, "Borexino Phase-I rates")
end

end
