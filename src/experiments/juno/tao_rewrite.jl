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

    tao_taishan_ffs = ones(4)
    tao_taishan_snf_scale = 0.0
    tao_taishan_neq_scale = zeros(4)
    tao_reactor_thermal_power_scale = 1.0

    tao_e_bias = 0.0
    tao_e_scale = 1.0

    tao_norm = 1.0
    tao_lsnl_pull = 0.0
    tao_eres_pull = 0.0

    params = (;
        tao_eres_pull,
        tao_e_bias,
        tao_e_scale,
        tao_taishan_ffs,
        tao_taishan_snf_scale,
        tao_taishan_neq_scale,
        tao_reactor_thermal_power_scale,
        tao_lsnl_pull,
        tao_norm,
    )

end

function get_priors()

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
    # snf correction
    file = YAML.load_file(joinpath(@__DIR__, "../daya_bay/daya_bay_3158days/dayabay_data/parameters/reactor_snf.yaml"))
    snf_mu = Float64(file["parameters"]["snf_scale"][1])
    snf_sigma = snf_mu * Float64(file["parameters"]["snf_scale"][2]) * 0.01   # percent
    tao_taishan_snf_scale = 0.

    ## non-equilibrium correction
    file = YAML.load_file(joinpath(@__DIR__, "../daya_bay/daya_bay_3158days/dayabay_data/parameters/reactor_nonequilibrium_correction.yaml"))
    mu = file["parameters"]["nonequilibrium_scale"][1] .* ones(4)
    sigma = mu .* file["parameters"]["nonequilibrium_scale"][2] .* 0.01 # percent

    tao_taishan_neq_scale = zeros(4)
   
    tps_mu = 1.0
    tps_sigma = tps_mu * 0.02
    tao_reactor_thermal_power_scale = Normal(tps_mu, tps_sigma)
    tao_reactor_thermal_power_scale = 1.0

    tao_e_bias = 0.0
    tao_e_scale = Normal(1., 0.006)

    tao_norm = Uniform(0, 10)

    tao_lsnl_pull = Normal(0.0, 1.0)
    tao_eres_pull = Normal(0.0, 1.0)

    priors = (;
        tao_eres_pull,
        tao_e_bias,
        tao_e_scale,
        tao_taishan_ffs,
        tao_taishan_snf_scale,
        tao_taishan_neq_scale,
        tao_reactor_thermal_power_scale,
        tao_lsnl_pull,
        tao_norm,
    )
end


function get_assets(datadir = @__DIR__)
    @info "Loading tao data"

    nominal_thermal_power = 4.6
    Baseline_m = 43.2

    data = readdlm(joinpath(@__DIR__, "tao_data/tao_data.txt"), skipstart=26)
    Elow = convert(Vector{Float64}, data[:, 1])
    Ehigh = convert(Vector{Float64}, data[:, 2])

    Ebins = unique(stack((Elow, Ehigh)))
    Ec = convert(Vector{Float64}, data[:, 3])

    ana_mask = Ec .<= 8.0
    # observed has negative entries bc expected background is subtracted -> stat. fluctuations mean negative entries
    hist_to_count = diff(Ebins)
    Ewidth = diff(Ebins)
    
    observed_err = convert(Vector{Float64}, data[:, 5])
    observed_all = convert(Vector{Float64}, data[:, 4])
    observed = observed_all[ana_mask]
    n_protons = 2e29 # estimate from claude
    nominal_livetime = 31.3
    nominal_livetime_seconds = nominal_livetime * 60 * 60 * 24



    assets = (;
        nominal_livetime,
        nominal_livetime_seconds,
        Ebins,
        Ec,
        Ewidth,
        n_protons,
        observed_all,
        observed,
        observed_err,
        nominal_thermal_power,
        Baseline_m,
        ana_mask,
    )
end

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

function get_eres()
    data = readdlm(joinpath(@__DIR__, "tao_data/tao_energy_resolution.txt"), skipstart=27)
    E = data[:, 1]
    rel_eres_nom = data[:, 3]
    rel_eres_p = data[:, 5]
    rel_eres_m = data[:, 4]
    # use average pull, band is approximately symmetric around nominal curve
    difference = (rel_eres_p .- rel_eres_m) ./ 2
    eres_diff = rel_eres_nom .+ difference
    # reuse refine_lsnl, it multiplies the nominal curve with energy
    # which is exactly what we want here to get the absolute energy resolution
    x, nom = refine_lsnl(E, rel_eres_nom)
    nom_itp = linear_interpolation(x, nom, extrapolation_bc = Line())
    pull_term = refine_lsnl(E, eres_diff)[2] - nom
    pull_itp = linear_interpolation(x, pull_term, extrapolation_bc = Line())

    function _eres(E, pull)
        return @. nom_itp(E) + pull * pull_itp(E)
    end
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

    # load energy leakage and lsnl
    _T = Newtrinos.TAOLeakage.load_leakage_templates(joinpath(@__DIR__, "tao_data/tao_leakage_templates.txt"))
    lsnl = Newtrinos.TAOLeakage.load_tao_positron_nl(joinpath(@__DIR__, "tao_data/TAO_Fig25b_nonlinearity.txt"))
    # do not confuse energy scales:
    # first entry is true prompt energy (not true neutrino energy!), second entry is visible energy
    R = Newtrinos.TAOLeakage.LeakResponse(_T, 1.0:0.02:12.0, 0.7:0.02:13.00; nl=lsnl)

    E_ana = assets.Ebins

    eres = get_eres()

    ana_mask = assets.ana_mask

    #deposited energy, Edep + 0.782 ≈ Enu
    E_fine = R.etrue_edges
    E_fine_c = (E_fine[1:end-1] + E_fine[2:end]) / 2

    E_vis_edges = R.evis_edges
    E_vis_c = (E_vis_edges[1:end-1] + E_vis_edges[2:end]) / 2

    final_rebin_idx = rebin_index(E_vis_edges, E_ana)

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
    flux_norm = 1e-45 * n_p * P_th_seconds * invL2
    spec_prefac_snf = 1e-45 * n_p * P_th_seconds * invL2


    function forward_model(params)
        T = promote_type(
                #eltype(params.tao_eres_a),
                #eltype(params.tao_eres_b),
                #eltype(params.tao_eres_c),
                eltype(params.tao_eres_pull),
                eltype(params.tao_reactor_thermal_power_scale),
                eltype(params.tao_taishan_ffs),
                eltype(params.tao_taishan_neq_scale),
                eltype(params.tao_taishan_snf_scale),
                #eltype(params.tao_eres_unc_scale),
                eltype(params.tao_e_bias),
                eltype(params.tao_e_scale),
                eltype(params.spectrum_pulls),
                eltype(params.energy_per_fission),
            )

        fluxes = reactor_fluxes(params)
        interm_spec = zeros(T, NF)

        flux_r, snf_r = fluxes
        @. interm_spec += (flux_norm * flux_r + spec_prefac_snf * snf_r)

        #bin-wise integrated spec in Eprompt
        interm_spec .*= integral_weights
        spec = vec(sum(reshape(interm_spec, NQ, NFc), dims = 1))

        #fold through energy leakage and lsnl -> visible energy
        #different to juno implementation, for TAO we smear the flux to different bins
        #as for the dayabat IAV matrix (same physical concept)
        #this also applies the lsnl directly, hence the binning stays the same in this step
        evis_spectrum = Newtrinos.TAOLeakage.fold(R, spec; pull=params.tao_lsnl_pull, scale=params.tao_e_scale, bias=params.tao_e_bias, band=:symmetric)  
        
        scaled_eres = eres(E_vis_c, params.tao_eres_pull)
        erec_spectrum = Newtrinos.Helpers.smear_erf(evis_spectrum, scaled_eres, E_vis_edges)
        coarse_counts = rebin(erec_spectrum, final_rebin_idx, length(E_ana) - 1) .* params.tao_norm
        if debug
            return MvNormal(coarse_counts, Diagonal(assets.observed_err.^2))
        end
        return MvNormal(coarse_counts[ana_mask], Diagonal(assets.observed_err[ana_mask].^2))
    end
end


function get_plot(physics, assets)
    Ec = assets.Ec
    Ebins = assets.Ebins
    obs = assets.observed_all
    obs_err = assets.observed_err

    
    
    function plot(params)
        m = mean(get_forward_model(physics, assets; debug=true)(params))
        s = sqrt.(var(get_forward_model(physics, assets; debug=true)(params)))

        f = Figure()
        ax = Axis(f[1, 1])

        errorbars!(ax, Ec, obs, obs_err, color=:black, label="observed")  
        scatter!(ax, Ec, obs, markersize = 10, color=:black, label="observed")
        stephist!(ax, Ec, weights=m, bins=Ebins, label="Expected")
        barplot!(ax, Ec, m.+s, width=diff(Ebins), gap=0, fillto=m .- s, alpha=0.5, label="standard deviation")
        ax.ylabel = "Counts"
        ax.title = "TAO, 31.3 days"
       
        ax.xticksvisible = false
        ax.xticklabelsvisible = false

        ax2 = Axis(f[2,1])
        plot!(ax2, Ec, obs ./ m, color=:black, label="Observed")
        hlines!(ax2, 1, label="Expected")
        barplot!(ax2, Ec, 1 .+ s ./ m, width=diff(Ebins), gap=0, fillto= 1 .- s./m, alpha=0.5, label="Standard Deviation")
        ylims!(ax2, 0.9, 1.1)

        axislegend(ax, merge=true)

        rowsize!(f.layout, 1, Relative(3/4))
        rowgap!(f.layout, 1, 0)
        
        ax2.xlabel="E [MeV]"
        ax2.ylabel="Counts/Expected"
    
        xlims!(ax, minimum(Ebins), maximum(Ebins))
        xlims!(ax2, minimum(Ebins), maximum(Ebins))

        return f
    end

    return plot
end


end 
