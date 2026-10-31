module juno_rewrite

using DataFrames
using CSV
using Distributions
using Statistics
using BAT
using CairoMakie
using Logging
using Interpolations
using LinearAlgebra
import ValueShapes
using DelimitedFiles
using OrderedCollections
using StructArrays
using ArraysOfArrays
import YAML
import ..Newtrinos

const RHO = 2.84            # g/cm³
const YE = 0.5

@kwdef struct JUNO <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function default_physics()
    #osc = Newtrinos.osc.configure(propagation=Neutrinos.osc.Spray(averaging= :uniform, ))
    #osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(propagation=Newtrinos.osc.Damping(0.08)))
    osc = Newtrinos.osc.configure()
    #osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction = Newtrinos.osc.SI(), eigen_method = Newtrinos.BargerEigen()))
    xsec = Newtrinos.ibd_xsec.configure(Newtrinos.ibd_xsec.VogelBeacomO1())
    flux = Newtrinos.dayabay_flux.configure(Newtrinos.dayabay_flux.ReactorFluxConfig(average=true, N_pulls=19))
    (; osc, xsec, flux)
end

function configure(physics=default_physics())
    physics = (;physics.osc, physics.xsec, physics.flux)
    assets = get_assets()
    return JUNO(
        physics = physics,
        params = get_params(),
        priors = get_priors(),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets),
    )
end

function get_params()

    # events per day, backgrounds
    lihe = 1.41
    geo = 2.12
    world_reactors = 0.92
    bipo = 0.57
     # alpha_n, fast_n, double_n, atmo, acc
     # MlE of 2609.37895
    other_rate = 0.05 + 0.02 + 0.13 + 0.11 + 0.07
    # other_rate = 5.40
    IBD_rate = 36.02
    efficiency = 0.7401


    
    # spent nuclear fuel, assumes correlation within plant, uncorrelated between plants
    juno_taishan_snf_scale = 1.0
    juno_yangjiang_snf_scale = 1.0
    juno_dayabay_snf_scale = 1.0
    
    #neq, assumes correlation within plant, uncorrelated between plants
    juno_taishan_neq_scale = ones(4)
    juno_yangjiang_neq_scale = ones(4)
    juno_dayabay_neq_scale = ones(4)
    
    #take fractions from dayabay,
    #use fraction_scale separately? re-use from dayabay model?
    # one entry per isotope
    # one set of parameter for all reactor cores assumed
    juno_taishan_ffs = ones(4)
    juno_yangjiang_ffs = ones(4)
    juno_dayabay_ffs = ones(4)

    reactor_thermal_power_scale = ones(9)
    # reactor_thermal_power_scale = 1.0

    eres_unc_scale = 0.0
    e_scale = 1.0
    e_bias = 0.0

    global_rate = 36.02
    lsnl_pull = 0.0

    params = (
        juno_lihe_rate = lihe,
        juno_geo_rate = geo,
        juno_world_reactor_rate = world_reactors,
        juno_bipo_rate = bipo,
        juno_other_rate = other_rate,
        juno_taishan_snf_scale = juno_taishan_snf_scale,
        juno_yangjiang_snf_scale = juno_yangjiang_snf_scale,
        juno_dayabay_snf_scale = juno_dayabay_snf_scale,
        juno_taishan_neq_scale = juno_taishan_neq_scale,
        juno_yangjiang_neq_scale =juno_yangjiang_neq_scale,
        juno_dayabay_neq_scale = juno_dayabay_neq_scale,        
        #juno_res_a = 0.0261,
        #juno_res_b = 0.0082,
        #juno_res_c = 0.0123,
        #juno_IBD_rate = IBD_rate,
        
        juno_efficiency = efficiency,
        juno_reactor_thermal_power_scale = reactor_thermal_power_scale,
        juno_taishan_ffs = juno_taishan_ffs,
        juno_yangjiang_ffs = juno_yangjiang_ffs,
        juno_dayabay_ffs = juno_dayabay_ffs,
        juno_eres_unc_scale = eres_unc_scale,
        juno_e_scale = e_scale,
        juno_e_bias = e_bias,
        # juno_global_rate = global_rate,
        juno_lsnl_pull = lsnl_pull,
        #juno_lsnl_scale = 0.,
        juno_taishan_tps = 1.0,
        juno_yangjiang_tps = 1.0,
        juno_dayabay_tps = 1.0,
        juno_global_norm = 1.0,
    )

end

function get_priors()

    # events per day, backgrounds
    #lihe_rate = 1.05
    #geo_rate = 1.30
    #world_reactors_rate = 0.92
    #bipo_rate = 0.42
    # alpha_n, fast_n, double_n, atmo, acc
    other_rate = 0.05 + 0.02 + 0.1 + 0.09 + 0.07

    # Gaussian quadrature of other pre-fit background rate uncertainties for combined uncertainty
    other_sigma = sqrt(0.02^2 + 0.02^2 + 0.10^2 + 0.04^2 + 0.01^2 )
    
    lihe = Truncated(Normal(1.05, 0.27), 0.0, Inf)
    # test with fit result of Juno
    #geo = Truncated(Normal(2.21, 0.32), 0.0, Inf)
    # properly use unconstrained value
    geo = Uniform(0, 5)
    world_reactors = Truncated(Normal(0.92, 0.09), 0.0, Inf)
    bipo = Truncated(Normal(0.42, 0.31), 0.0, Inf)
    other = Truncated(Normal(0.33, 0.11), 0.0, Inf)

    efficiency_mu = 0.7401
    efficiency_sigma = efficiency_mu * 0.01 * 1.4
    efficiency = Normal(efficiency_mu, efficiency_sigma)

    IBD_rate = Normal(36.02, 0.81)

    #take fractions from dayabay,
    #use fractions_scale separately? re-use from dayabay model?
    # one entry per isotope
    # one set of parameter for all reactor cores assumed

    ## fission fraction scale
    data = YAML.load_file(joinpath(@__DIR__, "../daya_bay/daya_bay_3158days/dayabay_data/parameters/reactor_fission_fractions_scale.yaml"), dicttype=OrderedDict{String,Any})
    fraction_scale = data["parameters"]["fission_fractions_scale"]

    isotopes = ["U235", "U238", "Pu239", "Pu241"]
    scale = zeros((4, 4))
    for i in 1:4
        for j in 1:4
            scale[i, j] = fraction_scale[isotopes[i]][1] * fraction_scale[isotopes[i]][2] * fraction_scale[isotopes[j]][1] * fraction_scale[isotopes[j]][2] * 0.01 * 0.01   # percent
        end
    end

    mu = collect([Float64(fraction_scale[name][1]) for name in isotopes])
    corr_mat = hcat(data["correlations"]["fission_fractions_scale"]["matrix"]...)

    cov_mat = corr_mat .* scale
    juno_taishan_ffs = MvNormal(mu, cov_mat)
    juno_yangjiang_ffs = MvNormal(mu, cov_mat)
    juno_dayabay_ffs = MvNormal(mu, cov_mat)

    # spent nuclear fuel, assumes correlation within plant, uncorrelated between plants
    # snf correction
    file = YAML.load_file(joinpath(@__DIR__, "../daya_bay/daya_bay_3158days/dayabay_data/parameters/reactor_snf.yaml"))
    snf_mu = Float64(file["parameters"]["snf_scale"][1])
    snf_sigma = snf_mu * Float64(file["parameters"]["snf_scale"][2]) * 0.01   # percent
    juno_taishan_snf_scale = Normal(snf_mu, snf_sigma)
    juno_yangjiang_snf_scale = Normal(snf_mu, snf_sigma)
    juno_dayabay_snf_scale = Normal(snf_mu, snf_sigma)

    #neq, assumes correlation within plant, uncorrelated between plants
    ## non-equilibrium correction
    file = YAML.load_file(joinpath(@__DIR__, "../daya_bay/daya_bay_3158days/dayabay_data/parameters/reactor_nonequilibrium_correction.yaml"))
    mu = file["parameters"]["nonequilibrium_scale"][1] .* ones(4)
    sigma = mu .* file["parameters"]["nonequilibrium_scale"][2] .* 0.01 # percent

    juno_taishan_neq_scale = MvNormal(mu, Diagonal(sigma.^2))
    juno_yangjiang_neq_scale = MvNormal(mu, Diagonal(sigma.^2))
    juno_dayabay_neq_scale = MvNormal(mu, Diagonal(sigma.^2))

    tps_mu = ones(9)
    # assume 10% uncertainty for Taishan cores (outage sometime) and 3 % for all other cores
    tps_sigma = [0.1, 0.1, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    thermal_power_scale = MvNormal(tps_mu, Diagonal(tps_sigma.^2))
    #thermal_power_scale = Normal(1, 0.01)

    #global_rate = Normal(36.02, 0.81)

    eres_unc_scale = Normal(0., 1.)
    # assume 0.5% energy scale uncertainty as per 59day Juno paper
    e_scale = Normal(1.0, 0.005)

    unc = 0.05
    e_bias = Normal(0, unc)

    lsnl_pull = Normal(0.0, 1.0)

    taishan_tps = Normal(1.0, 0.05)
    yangjiang_tps = Normal(1.0, 0.05)
    dayabay_tps = Normal(1.0, 0.05)

    priors = (
        juno_lihe_rate = lihe,
        juno_geo_rate = geo,
        juno_world_reactor_rate = world_reactors,
        juno_bipo_rate = bipo,
        juno_other_rate = other,
        #juno_IBD_rate = IBD_rate,
        #juno_res_a = Truncated(Normal(0.0261, 0.0002), 0.0, Inf),
        #juno_res_b = Truncated(Normal(0.0082, 0.0001), 0.0, Inf),
        #juno_res_c = Truncated(Normal(0.0123, 0.0004), 0.0, Inf),
        juno_efficiency = efficiency,
        juno_taishan_neq_scale = juno_taishan_neq_scale,
        juno_yangjiang_neq_scale = juno_yangjiang_neq_scale,
        juno_dayabay_neq_scale = juno_dayabay_neq_scale,
        juno_taishan_snf_scale = juno_taishan_snf_scale,
        juno_yangjiang_snf_scale = juno_yangjiang_snf_scale,
        juno_dayabay_snf_scale = juno_dayabay_snf_scale,
        juno_taishan_ffs = juno_taishan_ffs,
        juno_yangjiang_ffs = juno_yangjiang_ffs,
        juno_dayabay_ffs = juno_dayabay_ffs,
        juno_taishan_tps = taishan_tps,
        juno_yangjiang_tps = yangjiang_tps,
        juno_dayabay_tps = dayabay_tps,
        juno_reactor_thermal_power_scale = thermal_power_scale,
        juno_e_scale = e_scale,
        juno_e_bias = e_bias,
        juno_eres_unc_scale = eres_unc_scale,
        # juno_global_rate = global_rate,
        juno_lsnl_pull = lsnl_pull,
        #juno_lsnl_scale = Normal(0.0, 0.1),
        juno_global_norm = Uniform(0, 2)
    )
end


function get_lsnl_correction()
    lsnl = readdlm(joinpath(@__DIR__, "207_days/lsnl_207_days.txt"), skipstart=25)
    # TODO: add uncertainty through nuisance param
    E_lsnl = lsnl[:, 1]
    lsnl_nom = lsnl[:, 2]
    errm = lsnl[:, 5]
    errp = lsnl[:, 6]

    # errors are largely symmetric, use mean to create one pull term as in DayaBay model
    m = mean((errm, errp))


    x, nom = refine_lsnl(E_lsnl, lsnl_nom)
    nom_itp = linear_interpolation(x, nom, extrapolation_bc = Line())
    pull_curve = refine_lsnl(E_lsnl, lsnl_nom .+ m)[2] - nom
    pull_itp = linear_interpolation(x, pull_curve, extrapolation_bc = Line())

    function _lsnl(E, pull)

        return @. nom_itp(E) + pull * pull_itp(E)
    end
end


function get_eres()
    data = readdlm(joinpath(@__DIR__, "207_days/eres_207_days.txt"), skipstart=19)
    E = data[:, 1]
    # mul
    rel_eres_nom = data[:, 3] .* 0.01
    # reuse refine_lsnl, it multiplies the nominal curve with energy
    # which is exactly what we want here to get the absolute energy resolution
    #rel_eres_p = data[:, 4] .* 0.01
    x, nom = refine_lsnl(E, rel_eres_nom)
    nom_itp = linear_interpolation(x, nom, extrapolation_bc = Line())
    function _eres(E)
        @. nom_itp(E)
    end
end

function eres_parameterised(E; a = 0.0303, b = 0.010, c = 0.006)
    # default values roughly match the plot provided in the paper
    return @. sqrt(a^2 * E + b^2 * E^2 + c^2)
end


function get_assets(datadir = @__DIR__)
    @info "Loading juno data"

    
    eres_a = 0.0261
    eres_b = 0.0082 
    eres_c = 0.0123 

    nominal_livetime = 207.2
    nominal_livetime_seconds = nominal_livetime * 60 * 60 * 24
    n_protons = 1.442e33

    Delta_E = 20e-3   # 20 keV bins

    #fine_binning = collect(0:Delta_E:12)
 
    reactors = DataFrame(
        Reactor = ["Taishan", "Taishan", "Yangjiang", "Yangjiang", "Yangjiang", "Yangjiang", "Yangjiang", "Yangjiang", "Daya Bay"],
        Core = [1, 2, 1, 2, 3, 4, 5, 6, 0],
        Power_GW_th = [4.6, 4.6, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 17.37],
        Baseline_m = 1e3 .* [52.77, 52.64, 52.74, 52.82, 52.41, 52.49, 52.11, 52.19, 215.0],
        IBD_Rate_per_day = [7.5, 7.6, 4.8, 4.7, 4.8, 4.8, 4.9, 4.9, 3.0], 
        Relative_Flux_percent = [16.0, 16.1, 10.1, 10.1, 10.3, 10.2, 10.4, 10.4, 6.4],
        # thanks claude: compiled from CNEA and IAEA data, translated and compiled into a table by our machine overlords
        load_factor = [0.633, 0.431, 0.921, 0.946, 0.890, 0.886, 0.956, 0.912, 0.918], 
    )

    data = readdlm(joinpath(@__DIR__, "207_days/data_207_days.txt"), skipstart=29)
    Elow = convert(Vector{Float64}, data[:, 1])
    Ehigh = convert(Vector{Float64}, data[:, 2])

    Ebins = unique(stack((Elow, Ehigh)))
    Ec = convert(Vector{Float64}, data[:, 3])

    hist_obs = convert(Vector{Int64}, data[:, 6])
    hist_obs = convert(Vector{Float64}, data[:, 4])
    
    Ewidth = diff(Ebins)
    hist_to_count = Ewidth / 0.04
    observed = convert.(Int64, round.(hist_obs .* hist_to_count, digits=0))

    

    lihe = convert(Vector{Float64}, data[:, 8])
    geo = convert(Vector{Float64}, data[:, 9])
    world_reactor = convert(Vector{Float64}, data[:, 10])
    bipo = convert(Vector{Float64}, data[:, 11])
    other = convert(Vector{Float64}, data[:, 12])
    signal_mle = convert(Vector{Float64}, data[:, 7])
    #observed = round.(Int64, signal_mle .* hist_to_count)
    #observed = signal_mle .* hist_to_count
    lihe_counts = lihe .* hist_to_count
    geo_counts = geo .* hist_to_count
    world_reactor_counts = world_reactor .* hist_to_count
    bipo_counts = bipo .* hist_to_count
    other_counts = other .* hist_to_count

    lihe_shape = lihe_counts ./ sum(lihe_counts)
    geo_shape = geo_counts ./ sum(geo_counts)
    world_reactor_shape = world_reactor_counts ./ sum(world_reactor_counts)
    bipo_shape = bipo_counts ./ sum(bipo_counts)
    other_shape = other_counts ./ sum(other_counts)

    # read in uncoscillated expectation
    data = readdlm(joinpath(@__DIR__, "207_days/unoscillated_207_days.txt"), skipstart=14)
    no_osc = convert.(Float64, data[:, 4]) .* hist_to_count
    #observed = no_osc
    reactor_config = Newtrinos.dayabay_flux.configure()
    energy_per_fission = reactor_config.params.energy_per_fission
    fractions = YAML.load_file(joinpath(@__DIR__, "../daya_bay/daya_bay_3158days", "dayabay_data/parameters/reactor_fission_fractions.yaml"), dicttype=OrderedDict{String,Any})["parameters"]["fission_fractions"]
    fractions = NamedTuple((Symbol(key),value) for (key,value) in fractions)

    file = YAML.load_file(joinpath(@__DIR__, "../daya_bay/daya_bay_3158days", "dayabay_data/parameters/antineutrinos_per_fission_huber_mueller.yaml"))
    nu_per_f = file["parameters"]["antineutrinos_per_fission"]
    nu_per_fission = NamedTuple((Symbol(key), value) for (key, value) in nu_per_f)
    # also convert from MeV to GJ
    elementary_charge = 1.602176634e-19

        #           Giga   Joule -> eV        Mega
    GJ_to_MeV = 1e9 / elementary_charge * 1e-6 # reactor thermal power in GW, energy per fission in MeV

    #multipliy: rate -> thermal power, divide: thermal power -> rate
    nom_conversion_factor = sum([energy_per_fission[i] * fractions[iso] for (i, iso) in enumerate(keys(fractions))]) / sum([fractions[iso] * nu_per_fission[iso] for (i, iso) in enumerate(keys(fractions))]) / GJ_to_MeV
    
    assets = (;
        reactors,
        eres_a,
        eres_b,
        eres_c,
        nominal_livetime,
        nominal_livetime_seconds,
        lihe_shape,
        geo_shape,
        world_reactor_shape,
        bipo_shape,
        other_shape,
        Ebins,
        Ec,
        Ewidth,
        hist_to_count,
        n_protons,
        signal_mle,
        no_osc,
        observed
    )
end

"""
    nue_survival(E, L, params)

Vacuum three-flavour survival probability P(ν̄e → ν̄e) = 1 - 4 Σᵢ<ⱼ |Uₑᵢ|²|Uₑⱼ|² sin²(Δm²ⱼᵢ L / 4E) for energies `E` [GeV]
(rows) and baselines `L` [km] (columns), with the phase convention of `Newtrinos.osc` (`F_units`).
"""
function nue_survival(E, L, params)
    s12², s13² = sin(params.θ₁₂)^2, sin(params.θ₁₃)^2
    ue1, ue2, ue3 = (1 - s12²) * (1 - s13²), s12² * (1 - s13²), s13²
    dm21, dm31 = params.Δm²₂₁, params.Δm²₃₁
    dm32 = dm31 - dm21
    k = Newtrinos.osc.F_units / 2
    @. 1 - 4 * (ue1 * ue2 * sin(k * dm21 * L' / E)^2 + ue1 * ue3 * sin(k * dm31 * L' / E)^2 + ue2 * ue3 * sin(k * dm32 * L' / E)^2)
end


const lsnl_min_energy = 0.7    # MeV, lower end of the refined LSNL curves

"""
    refine_lsnl(E, f; refine_times = 4, newmin = 0.5, newmax = 12.1)

LSNL curve processing of the official model (`refine_lsnl_data`): the relative curve f = Evis/Escint is converted to the
absolute Evis(Escint), interpolated with a cubic spline on a `refine_times` finer grid and extrapolated linearly (with
the outermost fine segments) to [`newmin`, `newmax`]. Returns the fine Escint grid and Evis on it.
"""
function refine_lsnl(E::Vector{Float64}, f::Vector{Float64}; refine_times = 4, newmin = 0.7, newmax = 12.0)
    xfine = collect(range(E[1], E[end], length = (length(E) - 1) * refine_times + 1))
    yfine = Newtrinos.Helpers.not_a_knot_spline(E, f .* E, xfine)
    step = xfine[2] - xfine[1]
    xleft = collect(newmin:step:xfine[1]-step*1e-9)
    stepright = xfine[end] - xfine[end-1]
    xright = collect(xfine[end]:stepright:newmax+stepright*1e-6)[2:end]
    sl = (yfine[2] - yfine[1]) / (xfine[2] - xfine[1])
    sr = (yfine[end] - yfine[end-1]) / (xfine[end] - xfine[end-1])
    yleft = @. yfine[1] + sl * (xleft - xfine[1])
    yright = @. yfine[end] + sr * (xright - xfine[end])
    vcat(xleft, xfine, xright), vcat(yleft, yfine, yright)
end

"""
    project_to_bins(image_edges, counts, target_edges)

Energy scale distortion as a count-conserving bin projection (official `AxisDistortionMatrixPointwise`): the events of
source bin j, assumed uniform within it, are mapped to [image_edges[j], image_edges[j+1]] (monotonic distortion) and
shared among the `target_edges` bins in proportion to the overlap. Events mapped outside the target range are lost.
"""
function project_to_bins(image_edges, counts, target_edges)
    T = promote_type(eltype(image_edges), eltype(counts))
    out = zeros(T, length(target_edges) - 1)
    nt = length(target_edges) - 1
    @inbounds for j in eachindex(counts)
        A, B = image_edges[j], image_edges[j+1]
        B > A || continue
        k0 = max(searchsortedlast(target_edges, A), 1)
        k1 = min(searchsortedlast(target_edges, B), nt)
        for k in k0:k1
            overlap = min(B, target_edges[k+1]) - max(A, target_edges[k])
            overlap > 0 && (out[k] += counts[j] * overlap / (B - A))
        end
    end
    out
end

#function eres(E, a, b, c)
#    sigma_E = @. sqrt(a^2 * E^2 + b^2 * E + c^2)
#end

"""
    smear_bin_centres(counts, σ, edges; nσ = 8)

Energy resolution as in the official model (`HistSmearNormalMatrixBC`): the events of each bin i are smeared with a
Gaussian of width σ[i] around its centre, evaluated at the centres of the bins of the same binning (times the bin
width).
"""
function smear_bin_centres(counts, σ, edges; nσ = 8)
    T = promote_type(eltype(counts), eltype(σ))
    c = (edges[1:end-1] .+ edges[2:end]) ./ 2
    out = zeros(T, length(c))
    @inbounds for i in eachindex(counts, σ)
        iszero(counts[i]) && continue
        lo = searchsortedfirst(c, c[i] - nσ * σ[i])
        hi = searchsortedlast(c, c[i] + nσ * σ[i])
        norm = counts[i] / (sqrt(2π) * σ[i])
        for j in lo:hi
            out[j] += norm * (edges[j+1] - edges[j]) * exp(-0.5 * ((c[j] - c[i]) / σ[i])^2)
        end
    end
    out
end

"""
    rebin_index(fine_edges, coarse_edges)

Index of the coarse bin each fine bin falls into (0 outside); the coarse edges must be a subset of the fine ones.
"""
function rebin_index(fine_edges, coarse_edges)
    @assert all(any(isapprox.(fine_edges, e; atol = 1e-9)) for e in coarse_edges) "coarse edges are not on the fine grid"
    idx = zeros(Int, length(fine_edges) - 1)
    for i in eachindex(idx)
        k = searchsortedlast(coarse_edges, fine_edges[i] + 1e-9)
        1 <= k < length(coarse_edges) && (idx[i] = k)
    end
    idx
end

function rebin(counts, idx, n)
    out = zeros(eltype(counts), n)
    @inbounds for i in eachindex(counts, idx)
        idx[i] > 0 && (out[idx[i]] += counts[i])
    end
    out
end

function get_forward_model(physics, assets; debug = false)

    layers = StructArray{Newtrinos.Layer}((radius = [6371.0], p_density = [RHO * YE], n_density = [RHO * (1 - YE)]))


    E_ana = assets.Ebins
    E_ana_c = assets.Ec
    E_fine = collect(E_ana[1]:0.02:E_ana[end])
    E_fine_c = (E_fine[1:end-1] + E_fine[2:end]) / 2

    final_rebin_idx = rebin_index(E_fine, E_ana)

    load_factor = assets.reactors[!, "load_factor"]
    # IBD integration as in the official model: in each deposited energy bin a two-fold Gauss-Legendre quadrature
    # over Edep (order 5) and the positron angle cosθ ∈ [-1, 1] (order 3), with exact kinematics Eν(Edep, cosθ),
    # the first order cross section dσ/dcosθ and the Jacobian dEν/dEdep
    gl5 = ([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
           [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891])
    gl3 = ([-0.7745966692414834, 0.0, 0.7745966692414834], [0.5555555555555556, 0.8888888888888888, 0.5555555555555556])
    NQ = length(gl5[1]) * length(gl3[1])
    NFc = length(E_fine_c)
    E_nu_mesh = zeros(NQ, NFc)
    integral_weights = zeros(NQ, NFc)
    for i in 1:NFc
        lo, hi = E_fine[i], E_fine[i+1]
        q = 0
        for (ξ, wE) in zip(gl5...), (c, wc) in zip(gl3...)
            q += 1
            Edep = (lo + hi) / 2 + (hi - lo) / 2 * ξ
            Enu = physics.xsec.Enu(Edep, c)
            E_nu_mesh[q, i] = Enu
            integral_weights[q, i] = (hi - lo) / 2 * wE * wc * physics.xsec.xsec(Enu, c) * physics.xsec.jacobian(Enu, Edep, c)
        end
    end
    E_nu_mesh = vec(E_nu_mesh)
    integral_weights = vec(integral_weights)
    E_nu_GeV = E_nu_mesh ./ 1e3
    NF = length(E_nu_mesh)
    lsnl_mask = E_fine[1:end-1] .>= lsnl_min_energy - 1e-9

    lsnl = get_lsnl_correction()
    eres = get_eres()

    # the reactor fluxes on the integration mesh depend only on the reactor parameters: computed once per evaluation
    # and shared by all AD-periods; (main × correction + neq, snf) per reactor
    mesh_flux = physics.flux.flux_on_mesh(E_nu_mesh)
    ffs_symbols = [
        :juno_taishan_ffs,
        :juno_taishan_ffs,
        :juno_yangjiang_ffs,
        :juno_yangjiang_ffs,
        :juno_yangjiang_ffs,
        :juno_yangjiang_ffs,
        :juno_yangjiang_ffs,
        :juno_yangjiang_ffs,
        :juno_dayabay_ffs,
    ]
    neq_symbols = [
        :juno_taishan_neq_scale,
        :juno_taishan_neq_scale,
        :juno_yangjiang_neq_scale,
        :juno_yangjiang_neq_scale,
        :juno_yangjiang_neq_scale,
        :juno_yangjiang_neq_scale,
        :juno_yangjiang_neq_scale,
        :juno_yangjiang_neq_scale,
        :juno_dayabay_neq_scale,
    ]
    snf_symbols = [
        :juno_taishan_snf_scale,
        :juno_taishan_snf_scale,
        :juno_yangjiang_snf_scale,
        :juno_yangjiang_snf_scale,
        :juno_yangjiang_snf_scale,
        :juno_yangjiang_snf_scale,
        :juno_yangjiang_snf_scale,
        :juno_yangjiang_snf_scale,
        :juno_dayabay_snf_scale,
    ]

    tps_symbols = [
        :juno_taishan_tps,
        :juno_taishan_tps,
        :juno_yangjiang_tps,
        :juno_yangjiang_tps,
        :juno_yangjiang_tps,
        :juno_yangjiang_tps,
        :juno_yangjiang_tps,
        :juno_yangjiang_tps,
        :juno_dayabay_tps,
    ]
    function reactor_fluxes(params)
        map(1:9) do r
            main, neq, snf, correction = mesh_flux(params.juno_reactor_thermal_power_scale[r], params.energy_per_fission,
                params[ffs_symbols[r]], params.spectrum_pulls, params[neq_symbols[r]], params[snf_symbols[r]], r)
            (main .* correction .+ neq, snf)
        end
    end
    L = assets.reactors[!, "Baseline_m"]
    nominal_thermal_power = assets.reactors[!, "Power_GW_th"]
    baselines_km = L ./ 1e3
    paths = [VectorOfVectors{Newtrinos.Path}([[Newtrinos.Path(_L, 1) ]]) for _L in baselines_km]
    L2 = 4 * pi .* L.^2;
    invL2 = inv.(L2);
    livetime_days = assets.nominal_livetime
    livetime_seconds = assets.nominal_livetime_seconds
    # for standard three-flavour vacuum oscillations the closed form of P(ν̄e → ν̄e) is used (identical to osc_prob,
    # but much cheaper with dual numbers); any other oscillation model goes through osc_prob
    osc = physics.osc
    oc = osc.cfg
    standard_osc = oc.flavour isa Newtrinos.osc.ThreeFlavour && oc.interaction isa Newtrinos.osc.Vacuum &&
                   oc.propagation isa Newtrinos.osc.Basic && oc.states isa Newtrinos.osc.All
    matter_osc =  oc.flavour isa Newtrinos.osc.ThreeFlavour && oc.interaction isa Newtrinos.osc.SI &&
                   oc.propagation isa Newtrinos.osc.Basic && oc.states isa Newtrinos.osc.All
    function survival_probabilities(params)
        matter_osc && return [osc.osc_prob(E_nu_GeV, path, layers, params, anti=true)[:, 1, 1, 1] for path in paths]
        standard_osc || return [osc.osc_prob(E_nu_GeV, [L], params, anti=true)[:, 1, 1, 1] for L in baselines_km]
        [nue_survival(E_nu_GeV, L, params) for L in baselines_km]
    end

    """
    function osc_probabilities(params)
        standard_osc || return [osc.osc_prob(E_fine_c ./ 1e3, [L], params, anti=true)[:, 1, 1, 1] for L in baselines_km]
        [nue_survival(E_fine_c ./ 1e3, [L], params)[:] for L in baselines_km]
    end
    """

    P_th_seconds = nominal_thermal_power .* livetime_seconds
    n_p = assets.n_protons

    #           e-41cm2 -> m2
    flux_norm = 1e-45 * n_p * P_th_seconds .* invL2 .* load_factor
    spec_prefac_snf = 1e-45 * n_p * P_th_seconds .* invL2

    # background shapes
    lihe_shape = assets.lihe_shape
    world_reactor_shape = assets.world_reactor_shape
    bipo_shape = assets.bipo_shape
    geo_shape = assets.geo_shape
    other_shape = assets.other_shape

    # TODO: properly parameterise and move inside the forward model
    sigma_E = eres(E_fine_c)


    function forward_model(params)
        T = promote_type(
                #eltype(params.juno_res_a),
                #eltype(params.juno_res_b),
                #eltype(params.juno_res_c),
                eltype(params.juno_reactor_thermal_power_scale),
                eltype(params.juno_taishan_ffs),
                eltype(params.juno_yangjiang_ffs),
                eltype(params.juno_dayabay_ffs),
                eltype(params.juno_taishan_neq_scale),
                eltype(params.juno_yangjiang_neq_scale),
                eltype(params.juno_dayabay_neq_scale),
                eltype(params.juno_taishan_snf_scale),
                eltype(params.juno_yangjiang_snf_scale),
                eltype(params.juno_dayabay_snf_scale),
                eltype(params.juno_lihe_rate),
                eltype(params.juno_world_reactor_rate),
                eltype(params.juno_geo_rate),
                eltype(params.juno_bipo_rate),
                eltype(params.juno_other_rate),
                eltype(params.juno_eres_unc_scale),
                eltype(params.juno_e_bias),
                eltype(params.juno_e_scale),
                eltype(params.juno_lsnl_pull),
                eltype(params.spectrum_pulls),
                eltype(params.energy_per_fission),
                eltype(params.θ₁₃),
                eltype(params.θ₁₂),
                eltype(params.θ₂₃),
                eltype(params.Δm²₂₁),
                eltype(params.Δm²₃₁),
                eltype(params.δCP),
            )
        # backgrounds, rates as observed (after efficiency cuts, which are implicit in the rates)
        if debug
            bg = (
                lihe = params.juno_lihe_rate .* lihe_shape .* livetime_days,
                bipo = params.juno_bipo_rate .* bipo_shape .* livetime_days,
                geo = params.juno_geo_rate .* geo_shape .* livetime_days,
                other = params.juno_other_rate * other_shape .* livetime_days,   
                world = params.juno_world_reactor_rate * world_reactor_shape .* livetime_days, 
            )
        else
            bg = @. (
                params.juno_lihe_rate * lihe_shape 
                + params.juno_world_reactor_rate * world_reactor_shape
                + params.juno_bipo_rate * bipo_shape
                + params.juno_geo_rate * geo_shape
                + params.juno_other_rate * other_shape
            ) * livetime_days
        end

        fluxes = reactor_fluxes(params)
        interm_spec = zeros(T, NF)
        P = survival_probabilities(params)
        @inbounds for r in 1:9
            # nominal flux and neq correction scale with sum(daily thermal power * daily eff livetime); the SNF is
            # independent of the actual thermal power (spent fuel) and scales with the eff livetime
            flux_r, snf_r = fluxes[r]
            @. interm_spec += (flux_norm[r] * flux_r + spec_prefac_snf[r] * snf_r)  * P[r]
        end
        interm_spec .*= integral_weights
        spec = vec(sum(reshape(interm_spec, NQ, NFc), dims = 1))

        E_vis_edges = lsnl(E_fine, params.juno_lsnl_pull) .* params.juno_e_scale .+ params.juno_e_bias
        # E_vis_edges = f_lsnl .* (E_fine .+ params.juno_e_bias) .* params.juno_e_scale
        #E_vis_edges = E_fine .* (f_lsnl .+ params.juno_e_bias) .* params.juno_e_scale
        evis_spectrum = project_to_bins(E_vis_edges, spec .* lsnl_mask, E_fine)
        scaled_eres = zeros(T, length(sigma_E))
        scaled_eres .+= sigma_E .* (1 + 0.05 * params.juno_eres_unc_scale)
        erec_spectrum = Newtrinos.Helpers.smear_erf(evis_spectrum, scaled_eres, E_fine)
        coarse_counts = rebin(erec_spectrum, final_rebin_idx, length(E_ana) - 1) .* params.juno_efficiency# .* params.juno_global_norm
        if debug
            return coarse_counts, bg #, E_nu_GeV , survival_probabilities(params)
        end

        #return MvNormal(coarse_counts, Diagonal(assets.observed))# distprod(Poisson.(coarse_counts))# .+ bg))
        return distprod(Poisson.(coarse_counts .+ bg))
    end
end


function get_plot(physics, assets)
    function plot(params; data_to_plot::Vector, title_suffix::String = "")
#
        E_vis = assets.E_bins_visible
        E_nu  = assets.visible_to_neutrino_interp.(E_vis)
        
        L_km  = assets.reactors.Baseline_m ./ 1e3
        P = physics.osc.osc_prob(E_nu ./ 1e3, L_km, params; anti=true)
        Pavg = vec(sum(P[:, :, 1, 1] .* assets.reactors.Flux_Factor', dims=2))

        unosc_counts = assets.no_osc_interp.(E_nu) .* params.junotao_flux_scale .* params.juno_detection_epsilon
        osc_unsmear = unosc_counts .* Pavg
        total_model = get_expected(params, physics, assets)

        bkg = Dict(
            "Geo"      => assets.background_rate_interp.(E_vis) .* params.juno_geo_rate_norm .* assets.LIVETIME_DAYS,
            "Accident" => assets.accidental_bkg_rate_interp.(E_vis) .* params.juno_accidental_norm .* assets.LIVETIME_DAYS,
            "WorldR"   => assets.world_reactor_bkg_rate_interp.(E_vis) .* params.juno_world_reactor_norm .* assets.LIVETIME_DAYS,
            "LiHe"     => assets.lihe_bkg_rate_interp.(E_vis) .* params.juno_lihe_norm .* assets.LIVETIME_DAYS,
            "CO"       => assets.co_bkg_rate_interp.(E_vis) .* params.juno_co_norm .* assets.LIVETIME_DAYS,
            "ATMNC"    => assets.atmnc_bkg_rate_interp.(E_vis) .* params.juno_atmnc_norm .* assets.LIVETIME_DAYS,
            "FastN"    => assets.fast_neutron_bkg_rate_interp.(E_vis) .* params.juno_fast_neutron_norm .* assets.LIVETIME_DAYS,
        )
        
        total_bkg = reduce(+, values(bkg))
        smeared_signal = total_model .- total_bkg

        fig = Figure(resolution=(1200, 1350), fontsize=14)

        ax1 = Axis(fig[1, 1], title="No-osc vs Oscillated $title_suffix",
                   xlabel="E₍vis₎ [MeV]", ylabel="Counts/bin")
        lines!(ax1, E_vis, unosc_counts,  label="No-osc", linestyle=:dash, color=:black)
        lines!(ax1, E_vis, osc_unsmear,  label="Osc unsmeared",  color=:blue)
        axislegend(ax1)

        ax2 = Axis(fig[1, 2], title="Oscillated + Backgrounds + Data",
                   xlabel="E₍vis₎ [MeV]", ylabel="Counts/bin")
        lines!(ax2, E_vis, smeared_signal, label="Signal (smeared)", color=:red, linewidth=2)
        for (name, arr) in bkg
            lines!(ax2, E_vis, arr, label=name, linestyle=:dot)
        end
        lines!(ax2, E_vis, total_model, label="Signal + all bkgs", color=:magenta, linewidth=3)

        scatter!(ax2, E_vis, data_to_plot, color=:black, label="Data being fit", markersize=3)
        axislegend(ax2)

        ax3 = Axis(fig[2, 1], title="Unsmeared vs Smeared",
                   xlabel="E₍vis₎ [MeV]", ylabel="Counts/bin")
        lines!(ax3, E_vis, osc_unsmear, label="Unsmeared (osc)", color=:blue)
        lines!(ax3, E_vis, smeared_signal, label="Smeared", color=:green)
        axislegend(ax3)

        ax4 = Axis(fig[2, 2], title="⟨Pₑₑ⟩ vs Eₙᵤ",
                   xlabel="Eₙᵤ [MeV]", ylabel="⟨Pₑₑ⟩")
        lines!(ax4, E_nu, Pavg, label="⟨Pₑₑ⟩", color=:orange, linewidth=2)
        axislegend(ax4)

        ax5 = Axis(fig[3, 1:2], title="Residuals",
                   xlabel="E₍vis₎ [MeV]", ylabel="(Data-Model)/√Model")

        residuals = (data_to_plot .- total_model) ./ sqrt.(total_model .+ 1e-9)
        hlines!(ax5, 0.0, color=:red, linestyle=:dash)
        ylims!(ax5, -5, 5) 

        errorbars!(ax5, E_vis, residuals, ones(length(residuals)), color=:black)
        scatter!(ax5, E_vis, residuals, color=:black)

        return fig
    end

    return plot
end


end 
