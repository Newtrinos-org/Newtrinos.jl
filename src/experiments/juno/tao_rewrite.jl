module tao_rewrite

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


@kwdef struct TAO <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function default_physics()
    xsec = Newtrinos.ibd_xsec.configure(Newtrinos.ibd_xsec.VogelBeacomO1())
    flux = Newtrinos.dayabay_flux.configure(Newtrinos.dayabay_flux.ReactorFluxConfig(average=true, N_pulls=19))
    (;xsec, flux)
end

function configure(physics=default_physics())
    physics = (;physics.xsec, physics.flux)
    assets = get_assets()
    return TAO(
        physics = physics,
        params = get_params(),
        priors = get_priors(),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets),
    )
end

function get_params()

    tao_efficiency = 0.466
    tao_eres_a = 0.02580
    tao_eres_b = 0.01036
    tao_eres_c = 0.01477

    tao_taishan_ffs = ones(4)

    tao_taishan_snf_scale = 1.0

    tao_taishan_neq_scale = ones(4)

    tao_reactor_thermal_power_scale = 1.0

    tao_eres_unc_scale = 1.0
    tao_e_bias = 0.0
    tao_e_scale = 1.0

    tao_norm = 1.0

    params = (;
        tao_efficiency,
        tao_eres_a,
        tao_eres_b,
        tao_eres_c,
        tao_eres_unc_scale,
        tao_e_bias,
        tao_e_scale,
        tao_taishan_ffs,
        tao_taishan_snf_scale,
        tao_taishan_neq_scale,
        tao_reactor_thermal_power_scale,
        tao_norm,
    )

end

function get_priors()

    # TODO fix errs
    tao_eres_a = Normal(0.02580, 0.02580 * 0.03)
    tao_eres_b = Normal(0.01036, 0.01036 * 0.03)
    tao_eres_c = Normal(0.01477, 0.01477 * 0.03)

    efficiency_mu = 0.466
    efficiency_sigma = 0.050
    tao_efficiency = Normal(efficiency_mu, efficiency_sigma)

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
    tao_taishan_ffs = MvNormal(mu, cov_mat)
    # spent nuclear fuel, assumes correlation within plant, uncorrelated between plants
    # snf correction
    file = YAML.load_file(joinpath(@__DIR__, "../daya_bay/daya_bay_3158days/dayabay_data/parameters/reactor_snf.yaml"))
    snf_mu = Float64(file["parameters"]["snf_scale"][1])
    snf_sigma = snf_mu * Float64(file["parameters"]["snf_scale"][2]) * 0.01   # percent
    tao_taishan_snf_scale = Normal(snf_mu, snf_sigma)

    #neq, assumes correlation within plant, uncorrelated between plants
    ## non-equilibrium correction
    file = YAML.load_file(joinpath(@__DIR__, "../daya_bay/daya_bay_3158days/dayabay_data/parameters/reactor_nonequilibrium_correction.yaml"))
    mu = file["parameters"]["nonequilibrium_scale"][1] .* ones(4)
    sigma = mu .* file["parameters"]["nonequilibrium_scale"][2] .* 0.01 # percent

    tao_taishan_neq_scale = MvNormal(mu, Diagonal(sigma.^2))
   
    tps_mu = 1.0
    # assume 10% uncertainty for Taishan cores (outage sometime) and 3 % for all other cores
    tps_sigma = tps_mu * 0.02
    tao_reactor_thermal_power_scale = Normal(tps_mu, tps_sigma)

    tao_eres_unc_scale = Normal(1.0, 1.0)
    tao_e_bias = Normal(0., 1.)
    tao_e_scale = Normal(1., 1.)

    tao_norm = Uniform(0, Inf)

    priors = (;
        tao_efficiency,
        tao_eres_a,
        tao_eres_b,
        tao_eres_c,
        tao_eres_unc_scale,
        tao_e_bias,
        tao_e_scale,
        tao_taishan_ffs,
        tao_taishan_snf_scale,
        tao_taishan_neq_scale,
        tao_reactor_thermal_power_scale,
        tao_norm,
    )
end


function get_assets(datadir = @__DIR__)
    @info "Loading tao data"


    nominal_thermal_power = 4.6
    Baseline_m = 43.2
    load_factor = 0.633# , 0.431 


    data = readdlm(joinpath(@__DIR__, "tao_data/tao_data.txt"), skipstart=26)
    Elow = convert(Vector{Float64}, data[:, 1])
    Ehigh = convert(Vector{Float64}, data[:, 2])

    Ebins = unique(stack((Elow, Ehigh)))
    Ec = convert(Vector{Float64}, data[:, 3])

    # observed has negative entries bc expected background is subtracted -> stat. fluctuations mean negative entries
    # convert to counts, data is counts / 0.2MeV
    hist_to_count = diff(Ebins)
    Ewidth = diff(Ebins)
    observed = convert(Vector{Float64}, data[:, 4]) .* hist_to_count
    observed_err = convert(Vector{Float64}, data[:, 5]) .* hist_to_count
  

    n_protons = 2e29
    nominal_livetime = 31.3
    nominal_livetime_seconds = nominal_livetime * 60 * 60 * 24

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
        nominal_livetime,
        nominal_livetime_seconds,
        Ebins,
        Ec,
        Ewidth,
        hist_to_count,
        n_protons,
        observed,
        observed_err,
        nominal_thermal_power,
        Baseline_m,
        load_factor,
    )
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

function eres(E, a, b, c)
    sigma_E = @. sqrt(a^2 * E + b^2 * E^2 + c^2)
end

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

    E_ana = assets.Ebins
    E_ana_c = assets.Ec
    E_fine = collect(E_ana[1]:0.02:E_ana[end])
    E_fine_c = (E_fine[1:end-1] + E_fine[2:end]) / 2

    final_rebin_idx = rebin_index(E_fine, E_ana)

    load_factor = assets.load_factor
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
    NF = length(E_nu_mesh)

   
    # the reactor fluxes on the integration mesh depend only on the reactor parameters: computed once per evaluation
    # and shared by all AD-periods; (main × correction + neq, snf) per reactor
    mesh_flux = physics.flux.flux_on_mesh(E_nu_mesh)


    function reactor_fluxes(params)
        main, neq, snf, correction = mesh_flux(params.tao_reactor_thermal_power_scale, params.energy_per_fission,
            params.tao_taishan_ffs, params.spectrum_pulls, params.tao_taishan_neq_scale, params.tao_taishan_snf_scale, 1)
        (main .* correction .+ neq, snf)
    end
    L = assets.Baseline_m
    nominal_thermal_power = assets.nominal_thermal_power
    L2 = 4 * pi .* L.^2;
    invL2 = inv.(L2);
    livetime_seconds = assets.nominal_livetime_seconds
    # for standard three-flavour vacuum oscillations the closed form of P(ν̄e → ν̄e) is used (identical to osc_prob,
    # but much cheaper with dual numbers); any other oscillation model goes through osc_prob
    P_th_seconds = nominal_thermal_power .* livetime_seconds
    n_p = assets.n_protons

    #           e-41cm2 -> m2
    flux_norm = 1e-45 * n_p * P_th_seconds * invL2 * load_factor
    spec_prefac_snf = 1e-45 * n_p * P_th_seconds * invL2

    function forward_model(params)
        T = promote_type(
                eltype(params.tao_eres_a),
                eltype(params.tao_eres_b),
                eltype(params.tao_eres_c),
                eltype(params.tao_reactor_thermal_power_scale),
                eltype(params.tao_taishan_ffs),
                eltype(params.tao_taishan_neq_scale),
                eltype(params.tao_taishan_snf_scale),
                eltype(params.tao_eres_unc_scale),
                eltype(params.tao_e_bias),
                eltype(params.tao_e_scale),
                eltype(params.spectrum_pulls),
                eltype(params.energy_per_fission),
            )

        fluxes = reactor_fluxes(params)
        interm_spec = zeros(T, NF)

        flux_r, snf_r = fluxes
        @. interm_spec += (flux_norm * flux_r + spec_prefac_snf * snf_r)

        interm_spec .*= integral_weights
        spec = vec(sum(reshape(interm_spec, NQ, NFc), dims = 1))
        E_vis_edges = E_fine
        # insert tao leakage here
        # TODO
        evis_spectrum = project_to_bins(E_vis_edges, spec, E_fine)
        
        scaled_eres = eres(E_fine_c, params.tao_eres_a, params.tao_eres_b, params.tao_eres_c) .* (1 + 0.05 * params.tao_eres_unc_scale)
        erec_spectrum = smear_bin_centres(evis_spectrum, scaled_eres, E_fine)
        coarse_counts = rebin(erec_spectrum, final_rebin_idx, length(E_ana) - 1) .* params.tao_norm .* params.tao_efficiency
        if debug
            return coarse_counts, bg
        end

        return MvNormal(coarse_counts, Diagonal(assets.observed_err.^2))
        #return distprod(Poisson.(coarse_counts .+ bg))
    end
end


function get_plot(physics, assets)
    Ec = assets.Ec
    Ebins = assets.Ebins
    obs = assets.observed
    obs_err = assets.observed_err

    
    
    function plot(params)
        m = mean(get_forward_model(physics, assets)(params))

        f = Figure()
        ax = Axis(f[1, 1])

        errorbars!(ax, Ec, obs ./ assets.hist_to_count, obs_err ./ assets.hist_to_count, color=:black, label="observed")  
        scatter!(ax, Ec, obs ./ assets.hist_to_count, markersize = 10, color=:black, label="observed")
        stephist!(ax, Ec, weights=m./assets.hist_to_count, bins=Ebins, label="Expected")

       
        axislegend(ax, merge=true)

        return f
    end

    return plot
end


end 
