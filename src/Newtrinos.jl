"""
    Newtrinos

A Julia package for global analysis of neutrino data. Provides modular physics models,
experimental likelihoods, and inference tools (profile likelihood, Bayesian sampling).
"""
module Newtrinos

"""
    Physics

Abstract type for physics modules. Subtypes must have `params::NamedTuple` and `priors::NamedTuple` fields.
"""
abstract type Physics end

"""
    Experiment

Abstract type for experiment modules. Subtypes must have `physics`, `params`, `priors`, `assets`,
`forward_model`, and `plot` fields.
"""
abstract type Experiment end

export Physics, Experiment
export NewtrinosResult, plot
export bin, rebin
export make_init_samples, make_prior_samples, whack_a_moles, whack_many_moles

include("physics/osc.jl")
using .osc
include("physics/barger_eigen.jl")
include("physics/earth_layers.jl")
include("physics/atm_flux.jl")
include("physics/xsec.jl")
include("physics/cevns_xsec.jl")
include("physics/sns_flux.jl")
include("physics/solar_flux.jl")
include("physics/solar_xsec.jl")
include("analysis/analysis_tools.jl")
include("analysis/molewhacker.jl")
include("utils/plotting.jl")
include("utils/plotting_bat.jl")
include("utils/autodiff.jl")
include("utils/helpers.jl")

include("experiments/daya_bay/daya_bay_3158days/dayabay.jl")
include("experiments/minos/minos_sterile_16e20_POT/minos.jl")
include("experiments/icecube/deepcore_3y_highstats_sample_b/deepcore.jl")  # module deepcore_3y
include("experiments/icecube/deepcore_8y_verification_sample/deepcore.jl")  # module deepcore_8y
# default DeepCore dataset: `Newtrinos.deepcore` refers to the current (8-year) sample
const deepcore = deepcore_8y
include("experiments/super_k/sk_atm_2023/super_k.jl")
include("experiments/icecube/upgrade_sim_2020/ic_upgrade.jl")
include("experiments/kamland/kamland_7years/kamland.jl")
include("experiments/km3net/orca6_433kton/orca.jl")
include("experiments/t2k/t2k_2025/t2k.jl")
include("experiments/nova/nova_2024/nova.jl")
#include("experiments/coherent/coherent_2020/coherent_csi.jl")
include("experiments/coherent/coherent_2020/coherent_csi.jl")
include("experiments/coherent/coherent_2020/coherent_lAr.jl")

include("experiments/solar_common/solar_common.jl")
include("experiments/chlorine/homestake_1998/chlorine.jl")
include("experiments/gallium/gallex_gno_2010/gallex_gno.jl")
include("experiments/gallium/sage_2009/sage.jl")
include("experiments/sno/sno_combined_2013/sno.jl")
include("experiments/super_k/sk_solar_common/sk_solar_common.jl")
include("experiments/super_k/sk_solar_i_1496d/sk1_solar.jl")
include("experiments/super_k/sk_solar_ii_791d/sk2_solar.jl")
include("experiments/super_k/sk_solar_iii_548d/sk3_solar.jl")
include("experiments/super_k/sk_solar_iv_2970d/sk4_solar.jl")
include("experiments/super_k/sk_solar_dn_amplitude/sk_solar_dn.jl")
include("experiments/borexino/borexino_phase1/borexino_ph1.jl")
include("experiments/borexino/borexino_phase2/borexino_ph2.jl")
include("experiments/borexino/borexino_phase3/borexino_ph3.jl")

include("experiments/juno/juno.jl")
include("experiments/juno/tao.jl")
include("experiments/gerda/gerda.jl")
include("experiments/katrin/katrin.jl")
end
