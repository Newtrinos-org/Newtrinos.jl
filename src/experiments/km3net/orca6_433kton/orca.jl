module orca

using LinearAlgebra
using Distributions
using DataStructures
using TypedTables
using HDF5
using StatsBase
using CairoMakie
using BAT
using Printf
using ..Newtrinos

@kwdef struct ORCA <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function default_physics()
    osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction=Newtrinos.osc.SI(), eigen_method=Newtrinos.BargerEigen()))
    atm_flux = Newtrinos.atm_flux.configure(Newtrinos.atm_flux.AtmFluxConfig(nominal_model=Newtrinos.atm_flux.HKKM("frj-ally-20-01-mtn-solmin.d")))
    earth_layers = Newtrinos.earth_layers.configure()
    xsec = Newtrinos.xsec.configure()
    (; osc, atm_flux, earth_layers, xsec)
end

function configure(physics=default_physics())
    physics = (;physics.osc, physics.atm_flux, physics.earth_layers, physics.xsec)
    assets = get_assets(physics)
    return ORCA(
        physics = physics,
        params = get_params(),
        priors = get_priors(),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

function get_assets(physics; datadir = @__DIR__)
    h5file = h5open(joinpath(datadir, "ORCA6_433kton_v_0_5.h5"), "r")
    f = read(h5file)
    mc_nu = FlexTable(Dict(Symbol(key) => f["binned_nu_response"][key] for key in keys(f["binned_nu_response"])))
    muons = Table(Dict(Symbol(key) => f["binned_muon"][key] for key in keys(f["binned_muon"])))
    data = Table(Dict(Symbol(key) => f["binned_data"][key] for key in keys(f["binned_data"])))

    binning = (
        e_fine = f["E_true_axis"]["centers"],
        cz_fine = f["Ct_true_axis"]["centers"],
        e_reco = f["E_reco_axis"]["centers"],
        cz_reco = f["Ct_reco_axis"]["centers"],
        e_fine_edges = f["E_true_axis"]["edges"],
        cz_fine_edges = f["Ct_true_axis"]["edges"],
        e_reco_edges = f["E_reco_axis"]["edges"],
        cz_reco_edges = f["Ct_reco_axis"]["edges"]
    )

    #flux = physics.atm_flux.nominal_flux(binning.e_fine, binning.cz_fine)
    layers = physics.earth_layers.compute_layers()
    paths = physics.earth_layers.compute_paths(binning.cz_fine, layers)

    true_shape = (length(binning.e_fine), length(binning.cz_fine))
    reco_shape = (length(binning.e_reco), length(binning.cz_reco), 3, 2)

    mc_nu.he_mask = ((mc_nu.IsCC .== 0) .& (mc_nu.E_true_bin_center .> 100)) .| ((mc_nu.IsCC .== 1) .& (mc_nu.E_true_bin_center .> 500))
    
    mc = (
        nue = Table(mc_nu[mc_nu.Pdg .== 12, :]),
        nuebar = Table(mc_nu[mc_nu.Pdg .== -12, :]),
        numu = Table(mc_nu[mc_nu.Pdg .== 14, :]),
        numubar = Table(mc_nu[mc_nu.Pdg .== -14, :]),
        nutau = Table(mc_nu[mc_nu.Pdg .== 16, :]),
        nutaubar = Table(mc_nu[mc_nu.Pdg .== -16, :])
        )

    rs = [2, 1, 3]
    data_hist = permutedims(reshape(data.W, reco_shape[rs]), rs)
    muon_hist = permutedims(reshape(muons.W, reco_shape[rs]), rs);

    assets = (;mc, muon_hist, observed=cut(data_hist), binning, true_shape, reco_shape, layers, paths)
end

function get_params()
    params = (
        orca_energy_scale = 1.,
        orca_norm_all = 1.,
        orca_norm_hpt = 1.,
        orca_norm_showers = 1.,
        orca_norm_muons = 1.,
        orca_norm_he = 1.,
        )
end

function get_priors()
    priors = (
        orca_energy_scale = Truncated(Normal(1., 0.09), 0.7, 1.3),
        orca_norm_all = Uniform(0.5, 1.5),
        orca_norm_hpt = Uniform(0.5, 1.5),
        orca_norm_showers = Uniform(0.5, 1.5),
        orca_norm_muons = Uniform(0., 2.),
        orca_norm_he = Truncated(Normal(1, 0.5), 0., 3.),
        )
end

# Histogram one MC channel straight from the oscillated flux on the true (E, cosθ) grid: each MC
# row adds lifetime · W · flux[true bin] (× orca_norm_he for high-energy rows) to its reco bin.
# CC rows use the flux oscillated into the channel's flavour j; NC rows (provided only for Pdg ±14,
# standing for NC of all flavours) use the flavour-blind total active flux `p_flux_nc`.
# Gathering and filling in one loop avoids per-event temporary arrays (~600k MC rows, 104 bytes per
# element for Dual{12}), which otherwise dominate allocations and GC time under ForwardDiff.
function make_hist_per_channel(mc, p_flux, p_flux_nc, j, lifetime_seconds, params, assets)
    he_factor = params.orca_norm_he - 1.
    T = typeof(lifetime_seconds * first(mc.W) * first(p_flux) * (first(mc.he_mask) * he_factor + 1.0))
    hist = zeros(T, assets.reco_shape)
    W, ef, cf, he = mc.W, mc.E_true_bin, mc.Ct_true_bin, mc.he_mask
    er, cr, cls, cc = mc.E_reco_bin, mc.Ct_reco_bin, mc.AnaClass, mc.IsCC
    @inbounds for i in eachindex(W)
        f = cc[i] == 1 ? p_flux[ef[i], cf[i], j] : p_flux_nc[ef[i], cf[i]]
        hist[er[i], cr[i], cls[i], cc[i] + 1] += lifetime_seconds * W[i] * f * (he[i] * he_factor + 1.0)
    end
    hist
end


# Oscillated flux on the true grid, indexed [E, cosθ, detected flavour], for ν and ν̄
function reweight(params, physics, assets)

    flux = physics.atm_flux.nominal_flux(assets.binning.e_fine * params.orca_energy_scale, assets.binning.cz_fine)
    
    sys_flux = physics.atm_flux.sys_flux(flux, params)

    s = assets.true_shape

    p = physics.osc.osc_prob(assets.binning.e_fine * params.orca_energy_scale, assets.paths, assets.layers, params)
    nu = @views reshape(sys_flux.nue, s) .* p[:, :, 1, :] .+ reshape(sys_flux.numu, s) .* p[:, :, 2, :]

    p = physics.osc.osc_prob(assets.binning.e_fine * params.orca_energy_scale, assets.paths, assets.layers, params, anti=true)
    nubar = @views reshape(sys_flux.nuebar, s) .* p[:, :, 1, :] .+ reshape(sys_flux.numubar, s) .* p[:, :, 2, :]

    (; nu, nubar)
end

function get_expected(params, physics, assets)

    osc_flux = reweight(params, physics, assets)

    lifetime_seconds = 1.

    # NC is flavour-blind: total active flux after oscillation, Σ_β Φ_osc(β)
    nc_nu = dropdims(sum(osc_flux.nu, dims=3), dims=3)
    nc_nubar = dropdims(sum(osc_flux.nubar, dims=3), dims=3)
    H(ch, f, f_nc, j) = make_hist_per_channel(assets.mc[ch], f, f_nc, j, lifetime_seconds, params, assets)
    hists = (nue = H(:nue, osc_flux.nu, nc_nu, 1), nuebar = H(:nuebar, osc_flux.nubar, nc_nubar, 1),
             numu = H(:numu, osc_flux.nu, nc_nu, 2), numubar = H(:numubar, osc_flux.nubar, nc_nubar, 2),
             nutau = H(:nutau, osc_flux.nu, nc_nu, 3), nutaubar = H(:nutaubar, osc_flux.nubar, nc_nubar, 3))

    hists_nc = sum(h[:, :, :, 1] for h in hists) * physics.xsec.scale(:any, :NC, params)

    hists_cc = hists.nue[:, :, :, 2] .+ hists.nuebar[:, :, :, 2] .+ hists.numu[:, :, :, 2] .+ hists.numubar[:, :, :, 2] .+ (hists.nutau[:, :, :, 2] .+ hists.nutaubar[:, :, :, 2]) * physics.xsec.scale(:nutau, :CC, params)
    expected = (assets.muon_hist * params.orca_norm_muons .+ hists_nc .+ hists_cc) * params.orca_norm_all

    # Poisson > 0
    expected = max.(1e-2, (expected))

    c = cut(expected)
    
    return (
        hpt = c.hpt * params.orca_norm_hpt,
        showers = c.showers * params.orca_norm_showers,
        lpt = c.lpt
    )
        
end

function cut(hist)
    (
    hpt = hist[1:end-1, 1:10, 1],
    showers = hist[1:end, 1:10, 2],
    lpt = hist[1:end-1, 1:10, 3]
        )
end


function get_forward_model(physics, assets)
    function forward_model(params)
        exp_events = get_expected(params, physics, assets)
        #distprod(Poisson.(exp_events))
        distprod(NamedTuple(ch => distprod(Poisson.(exp_events[ch])) for ch in keys(exp_events)))
    end
end

function get_plot(physics, assets)

    function plot(params, data=assets.observed)
        expected = get_expected(params, physics, assets)
    
        fig = Figure(size=(1000, 800))

        channels = [:hpt, :lpt, :showers]

        cz_bin_edges = assets.binning.cz_reco_edges[1:11]
        
        for j in 1:3
            key = channels[j]
            for i in 1:15
                ax = Axis(fig[i,j], yticklabelsize=10)
                if i > size(expected[key])[1] continue end
                stephist!(ax, midpoints(cz_bin_edges), bins=cz_bin_edges, weights=expected[key][i, :])
                scatter!(ax, midpoints(cz_bin_edges), data[key][i, :], color=:black)
                ax.xticksvisible = false
                ax.xticklabelsvisible = false
                ax.xlabel = ""
                up = maximum((maximum(data[key][i, :]), maximum(expected[key][i, :]))) * 1.2
                ylims!(ax, 0, up)
                e_low = assets.binning.e_reco_edges[i]
                e_high = assets.binning.e_reco_edges[i+1]
                text!(ax, 0.5, 0, text=@sprintf("E in [%.1f, %.1f] GeV", e_low, e_high), align = (:center, :bottom), space = :relative)
            end
        end
        for i in [15, 30, 45]
            ax = fig.content[i]
            ax.xticklabelsvisible = true
            ax.xticksvisible = true
            ax.xlabel="cos(zenith)"
        end
        fig.content[1].title = "High-purity Tracks"
        fig.content[16].title = "Low-purity Tracks"
        fig.content[31].title = "Showers"
        rowgap!(fig.layout, 0)
        linkxaxes!(fig.content...)
        fig
    end
end

end

    
    