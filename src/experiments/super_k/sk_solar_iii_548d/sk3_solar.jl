module sk3_solar

import ..Newtrinos
import ..Newtrinos.solar_common
import ..Newtrinos.sk_solar_common
import ..Newtrinos.sk_solar_common: SKPhase

"""
Super-Kamiokande III solar, 548 days.

K. Abe et al., Phys. Rev. D 83, 052010 (2011), arXiv:1010.0118 (revised spectrum table): day and night rates,
5.0–20 MeV (42 data points), recoil total energy.

SK-III energy resolution (arXiv:1010.0118), energy scale ±0.53%, resolution ±2.5%, energy-independent
systematics 1.6% (SK-IV paper).

See `Newtrinos.sk_solar_common.SKSolar` for the likelihood.
"""
const PHASE = SKPhase(
    name = :sk3_solar,
    title = "Super-Kamiokande III solar, 548 days",
    datafile = joinpath(@__DIR__, "sk3_solar_spectrum.csv"),
    total_energy = true,
    resolution = E -> -0.123 + 0.376 * sqrt(E) + 0.0349 * E,
    scale_unc = 0.0053,
    reso_unc = 0.025,
    norm_unc = 0.016,
    phi_b8_mc = 5.79e6,
    phi_hep_mc = 7.88e3,
)

default_physics() = solar_common.default_physics()

configure(physics=default_physics()) = sk_solar_common.configure(PHASE, physics)

get_expected(params, physics, assets) = sk_solar_common.get_expected(PHASE, params, physics, assets)

end
