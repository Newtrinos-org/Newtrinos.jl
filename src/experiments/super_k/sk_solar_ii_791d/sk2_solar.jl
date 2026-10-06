module sk2_solar

import ..Newtrinos
import ..Newtrinos.solar_common
import ..Newtrinos.sk_solar_common
import ..Newtrinos.sk_solar_common: SKPhase

"""
Super-Kamiokande II solar, 791 days.

J. P. Cravens et al., Phys. Rev. D 78, 032002 (2008), arXiv:0803.4312: day and night rates, 7.5–20 MeV, and the
all-zenith 7.0–7.5 MeV rate (33 data points), recoil total energy.

SK-II energy resolution (arXiv:0803.4312), energy scale ±1.4%, resolution ±2.5%, energy-independent
systematics 4.8% (SK-IV paper); energy-uncorrelated systematics derived from the SK-IV paper appendix.

See `Newtrinos.sk_solar_common.SKSolar` for the likelihood.
"""
const PHASE = SKPhase(
    name = :sk2_solar,
    title = "Super-Kamiokande II solar, 791 days",
    datafile = joinpath(@__DIR__, "sk2_solar_spectrum.csv"),
    total_energy = true,
    resolution = E -> 0.0536 + 0.5200 * sqrt(E) + 0.0458 * E,
    scale_unc = 0.014,
    reso_unc = 0.025,
    norm_unc = 0.048,
    phi_b8_mc = 5.79e6,
    phi_hep_mc = 7.88e3,
)

default_physics() = solar_common.default_physics()

configure(physics=default_physics()) = sk_solar_common.configure(PHASE, physics)

get_expected(params, physics, assets) = sk_solar_common.get_expected(PHASE, params, physics, assets)

end
