module sk1_solar

import ..Newtrinos
import ..Newtrinos.solar_common
import ..Newtrinos.sk_solar_common
import ..Newtrinos.sk_solar_common: SKPhase

"""
Super-Kamiokande I solar, 1496 days.

J. Hosaka et al., Phys. Rev. D 73, 112001 (2006), hep-ex/0508053: 5.0–5.5 and 16–20 MeV all-zenith rates and
5.5–16 MeV rates in day, five mantle and one core solar zenith bins (44 data points), recoil total energy.

SK-I energy resolution (hep-ex/0508053), energy scale ±0.64%, resolution ±2.5%, energy-independent
systematics 2.8% (SK-IV paper, arXiv:2312.12907, Table of systematics per phase).

See `Newtrinos.sk_solar_common.SKSolar` for the likelihood.
"""
const PHASE = SKPhase(
    name = :sk1_solar,
    title = "Super-Kamiokande I solar, 1496 days",
    datafile = joinpath(@__DIR__, "sk1_solar_spectrum.csv"),
    total_energy = true,
    resolution = E -> 0.2468 + 0.1492 * sqrt(E) + 0.0690 * E,
    scale_unc = 0.0064,
    reso_unc = 0.025,
    norm_unc = 0.028,
    phi_b8_mc = 5.79e6,
    phi_hep_mc = 7.88e3,
    night_edges = vcat([range(a, b, length=5)[1:end-1] for (a, b) in zip([0, 0.16, 0.33, 0.50, 0.67, 0.84], [0.16, 0.33, 0.50, 0.67, 0.84, 1.0])]..., [1.0]),
)

default_physics() = solar_common.default_physics()

configure(physics=default_physics(); kwargs...) = sk_solar_common.configure(PHASE, physics; kwargs...)

get_expected(params, physics, assets) = sk_solar_common.get_expected(PHASE, params, physics, assets)

end
