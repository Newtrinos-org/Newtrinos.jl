module sk4_solar

import ..Newtrinos
import ..Newtrinos.solar_common
import ..Newtrinos.sk_solar_common
import ..Newtrinos.sk_solar_common: SKPhase

"""
Super-Kamiokande IV solar, 2970 days.

K. Abe et al., Phys. Rev. D 109, 092001 (2024), arXiv:2312.12907: day and night rates, 3.49–19.49 MeV
(46 data points), recoil kinetic energy.

SK-IV energy resolution (Abe et al., PRD 94, 052010 (2016)), energy scale ±0.48%, resolution ±3%,
energy-independent systematics 1.1%.

See `Newtrinos.sk_solar_common.SKSolar` for the likelihood.
"""
const PHASE = SKPhase(
    name = :sk4_solar,
    title = "Super-Kamiokande IV solar, 2970 days",
    datafile = joinpath(@__DIR__, "sk4_solar_spectrum.csv"),
    total_energy = false,
    resolution = E -> -0.0839 + 0.349 * sqrt(E) + 0.0397 * E,
    scale_unc = 0.0048,
    reso_unc = 0.03,
    norm_unc = 0.011,
    phi_b8_mc = 5.25e6,
    phi_hep_mc = 7.88e3,
)

default_physics() = solar_common.default_physics()

configure(physics=default_physics(); kwargs...) = sk_solar_common.configure(PHASE, physics; kwargs...)

get_expected(params, physics, assets) = sk_solar_common.get_expected(PHASE, params, physics, assets)

end
