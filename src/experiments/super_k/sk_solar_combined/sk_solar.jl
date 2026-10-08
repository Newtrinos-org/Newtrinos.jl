module sk_solar

import ..Newtrinos
import ..Newtrinos.solar_common
import ..Newtrinos.sk_solar_common
import ..Newtrinos: sk1_solar, sk2_solar, sk3_solar, sk4_solar

"""
Super-Kamiokande solar ⁸B/hep spectra of several phases (default SK-I–IV) as one experiment, with the
survival probabilities computed once per evaluation (shared site and energy grid). Each phase keeps its
data, response, covariance and nuisance parameters (`sk1_solar_escale`, …), so the likelihood is the
product of the single-phase likelihoods (`Newtrinos.sk1_solar` … `sk4_solar`), up to the common zenith
binning of the Earth regeneration (that of SK-I).

    configure(physics=default_physics(); phases=(:sk1, :sk2, :sk3, :sk4), daynight=:spectra)

See `Newtrinos.sk_solar_common.configure` for `daynight`.
"""
const PHASES = (sk1 = sk1_solar.PHASE, sk2 = sk2_solar.PHASE, sk3 = sk3_solar.PHASE, sk4 = sk4_solar.PHASE)

default_physics() = solar_common.default_physics()

configure(physics=default_physics(); phases=(:sk1, :sk2, :sk3, :sk4), daynight::Symbol=:spectra) =
    sk_solar_common.configure(Tuple(PHASES[p] for p in phases), physics; daynight)

# phases of a configured experiment, from its assets (keys :sk1_solar, …)
phases_of(assets) = Tuple(only(p for p in values(PHASES) if p.name == k) for k in keys(assets.phases))

get_expected(params, physics, assets) = sk_solar_common.get_expected(phases_of(assets), params, physics, assets)

end
