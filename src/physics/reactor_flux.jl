module reactor_flux

using Interpolations
using DataStructures
using PCHIPInterpolation
using LinearAlgebra
using Integrals
using CSV
using DataFrames
using Distributions
using HDF5
import YAML
using DelimitedFiles
using ..Newtrinos

export ReactorFluxConfig, DayaBayFlux, flux_basis


const datadir = joinpath(@__DIR__, "../experiments/daya_bay/daya_bay_3158days")


abstract type FluxModel end


@kwdef struct ReactorFluxConfig{F<:FluxModel}
    flux_model::F = DayaBayFlux()
    datadir::String = datadir
end


struct DayaBayFlux <: FluxModel
end

@kwdef struct ReactorFlux <: Newtrinos.Physics
    cfg::ReactorFluxConfig
    params::NamedTuple
    priors::NamedTuple
    flux::Function
    datadir::String
end


function configure(cfg::ReactorFluxConfig = ReactorFluxConfig())
    ReactorFlux(
        cfg=cfg,
        params = get_params(cfg.flux_model, datadir=cfg.datadir),
        priors = get_priors(cfg.flux_model, datadir=cfg.datadir),
        flux = get_flux(cfg.flux_model, datadir=cfg.datadir),
        datadir = cfg.datadir
    )
end


function get_params(flux::DayaBayFlux; datadir = datadir)
    # TODO: is this part of DayaBay experiment or the reactor flux itself? not sure where to put
    # currently I define that we need 6 fluxes in here, but this is specific to the experiment
    # re-using for e.g. Juno, we need a different number of reactors, for future Julian...

    ## energy per fission
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_energy_per_fission.yaml"))
    data = file["parameters"]["energy_per_fission"]

    energy_per_fission = [data["U235"][1], data["U238"][1], data["Pu239"][1], data["Pu241"][1]]


    ## reactor thermal power
    data = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_thermal_power_uncertainty.yaml"))
    reactor_thermal_power_scale = data["parameters"]["thermal_power_scale"][1] .* ones(6)

    ## fission fraction scales
    data = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_fission_fractions_scale.yaml"), dicttype=OrderedDict{String,Any})
    fraction_scale = data["parameters"]["fission_fractions_scale"]
    #println(fraction_scale)

    ## neq scale
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_nonequilibrium_correction.yaml"))
    mu = file["parameters"]["nonequilibrium_scale"][1]

    neq_scale_R1 = ones(4) .* mu
    neq_scale_R2 = ones(4) .* mu
    neq_scale_R3 = ones(4) .* mu
    neq_scale_R4 = ones(4) .* mu
    neq_scale_R5 = ones(4) .* mu
    neq_scale_R6 = ones(4) .* mu

    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_snf.yaml"))
    mu = file["parameters"]["snf_scale"][1]

    snf_scale = ones(6) .* mu

    names = data["correlations"]["fission_fractions_scale"]["names"]
    #println(names)
    fission_fractions_scale_R1 = collect([Float64(fraction_scale[name][1]) for name in names])
    fission_fractions_scale_R2 = collect([Float64(fraction_scale[name][1]) for name in names])
    fission_fractions_scale_R3 = collect([Float64(fraction_scale[name][1]) for name in names])
    fission_fractions_scale_R4 = collect([Float64(fraction_scale[name][1]) for name in names])
    fission_fractions_scale_R5 = collect([Float64(fraction_scale[name][1]) for name in names])
    fission_fractions_scale_R6 = collect([Float64(fraction_scale[name][1]) for name in names])
    spectrum_pulls = ones(19)

    params = (;
        energy_per_fission,
        reactor_thermal_power_scale,
        neq_scale_R1,
        neq_scale_R2,
        neq_scale_R3,
        neq_scale_R4,
        neq_scale_R5,
        neq_scale_R6,
        snf_scale,
        fission_fractions_scale_R1,
        fission_fractions_scale_R2,
        fission_fractions_scale_R3,
        fission_fractions_scale_R4,
        fission_fractions_scale_R5,
        fission_fractions_scale_R6,
        spectrum_pulls,
    )
    return params
end


function get_priors(flux::DayaBayFlux; datadir = datadir)

    ## energy per fission
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_energy_per_fission.yaml"))
    data = file["parameters"]["energy_per_fission"]

    mu = [data["U235"][1], data["U238"][1], data["Pu239"][1], data["Pu241"][1]]
    var = [data["U235"][2], data["U238"][2], data["Pu239"][2], data["Pu241"][2]].^2
    energy_per_fission = Distributions.MvNormal(mu, Diagonal(var))
    #energy_per_fission_U235 = Distributions.Normal(data["U235"][1], data["U235"][1])
    #energy_per_fission_U238 = Distributions.Normal(data["U238"][1], data["U238"][1])
    #energy_per_fission_Pu239 = Distributions.Normal(data["Pu239"][1], data["Pu239"][1])
    #energy_per_fission_Pu241 = Distributions.Normal(data["Pu241"][1], data["Pu241"][1])

    #energy_per_fission = (
    #    U235=energy_per_fission_U235,
    #    U238=energy_per_fission_U238,
    #    Pu239=energy_per_fission_Pu239,
    #    Pu241=energy_per_fission_Pu241,
    #    )


    ## reactor thermal power uncertainty
    data = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_thermal_power_uncertainty.yaml"))
    mu = data["parameters"]["thermal_power_scale"][1]
    sigma = data["parameters"]["thermal_power_scale"][2] * mu * 0.01  # percent
    reactor_thermal_power_scale = Distributions.MvNormal(mu .* ones(6), Diagonal(sigma.^2 .* ones(6)))

    ## fission fraction scale

    data = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_fission_fractions_scale.yaml"), dicttype=OrderedDict{String,Any})
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
    fission_fractions_scale_R1 = Distributions.MvNormal(mu, cov_mat)
    fission_fractions_scale_R2 = Distributions.MvNormal(mu, cov_mat)
    fission_fractions_scale_R3 = Distributions.MvNormal(mu, cov_mat)
    fission_fractions_scale_R4 = Distributions.MvNormal(mu, cov_mat)
    fission_fractions_scale_R5 = Distributions.MvNormal(mu, cov_mat)
    fission_fractions_scale_R6 = Distributions.MvNormal(mu, cov_mat)

    spectrum_pulls = Distributions.Uniform.(zeros(19), 2 .* ones(19))

    ## non-equilibrium correction
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_nonequilibrium_correction.yaml"))
    mu = file["parameters"]["nonequilibrium_scale"][1] .* ones(4)
    sigma = mu .* file["parameters"]["nonequilibrium_scale"][2] .* 0.01 # percent


    neq_scale_R1 = Distributions.MvNormal(mu, Diagonal(sigma.^2))
    neq_scale_R2 = Distributions.MvNormal(mu, Diagonal(sigma.^2))
    neq_scale_R3 = Distributions.MvNormal(mu, Diagonal(sigma.^2))
    neq_scale_R4 = Distributions.MvNormal(mu, Diagonal(sigma.^2))
    neq_scale_R5 = Distributions.MvNormal(mu, Diagonal(sigma.^2))
    neq_scale_R6 = Distributions.MvNormal(mu, Diagonal(sigma.^2))

    # snf correction
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_snf.yaml"))
    mu = file["parameters"]["snf_scale"][1] .* ones(6)
    sigma = mu .* file["parameters"]["snf_scale"][2] .* 0.01   # percent
    snf_scale = Distributions.MvNormal(mu, Diagonal(sigma.^2))


    priors = (;
        energy_per_fission,
        reactor_thermal_power_scale,
        neq_scale_R1,
        neq_scale_R2,
        neq_scale_R3,
        neq_scale_R4,
        neq_scale_R5,
        neq_scale_R6,
        snf_scale,
        fission_fractions_scale_R1,
        fission_fractions_scale_R2,
        fission_fractions_scale_R3,
        fission_fractions_scale_R4,
        fission_fractions_scale_R5,
        fission_fractions_scale_R6,
        spectrum_pulls,
    )

end


function extract_reactor_spectra(datadir = datadir)
    file = h5open(joinpath(datadir, "dayabay_data/reactor_antineutrino_spectra_hm.hdf5"))
    spec_Pu239 = Float64[]
    spec_Pu241 = Float64[]
    spec_U235 = Float64[]
    spec_U238 = Float64[]

    E = Float64[]

    foreach(x -> (push!(spec_Pu239, x[2]), push!(E, x[1])), file["Pu239"][1:end])
    foreach(x -> push!(spec_Pu241, x[2]), file["Pu241"][1:end])
    foreach(x -> push!(spec_U235, x[2]), file["U235"][1:end])
    foreach(x -> push!(spec_U238, x[2]), file["U238"][1:end])
    close(file)

    #neq correction
    file = h5open(joinpath(datadir, "dayabay_data/nonequilibrium_correction.hdf5"))

    E_neq = Float64[]
    neq_Pu239 = Float64[]
    neq_Pu241 = Float64[]
    neq_U235 = Float64[]

    foreach(x-> (push!(neq_Pu239, x[2]), push!(E_neq, x[1])), file["Pu239"][1:end])
    foreach(x-> push!(neq_Pu241, x[2]), file["Pu241"][1:end])
    foreach(x-> push!(neq_U235, x[2]), file["U235"][1:end])
    close(file)

    ## spent nuclear fuelfile = h5open("dayabay_data/snf_correction.hdf5")
    file = h5open(joinpath(datadir, "dayabay_data/snf_correction.hdf5"))
    # is per reactor
    E_snf = Float64[]
    R1_snf = Float64[]
    R2_snf = Float64[]
    R3_snf = Float64[]
    R4_snf = Float64[]
    R5_snf = Float64[]
    R6_snf = Float64[]

    foreach(x -> (push!(E_snf, x[1]), push!(R1_snf, x[2])), file["R1"][1:end])
    foreach(x -> push!(R2_snf, x[2]), file["R2"][1:end])
    foreach(x -> push!(R3_snf, x[2]), file["R3"][1:end])
    foreach(x -> push!(R4_snf, x[2]), file["R4"][1:end])
    foreach(x -> push!(R5_snf, x[2]), file["R5"][1:end])
    foreach(x -> push!(R6_snf, x[2]), file["R6"][1:end])
    close(file)

    return (
        Pu239=spec_Pu239,
        Pu241=spec_Pu241,
        U238=spec_U238,
        U235=spec_U235,
        E=E,
        E_neq=E_neq,
        neq_Pu239=neq_Pu239,
        neq_Pu241=neq_Pu241,
        neq_U235=neq_U235,
        E_snf=E_snf,
        R1_snf=R1_snf,
        R2_snf=R2_snf,
        R3_snf=R3_snf,
        R4_snf=R4_snf,
        R5_snf=R5_snf,
        R6_snf=R6_snf,
    )
end


function get_flux(cfg::DayaBayFlux; datadir = datadir)
    
    # fixed ordering of isotopes
    names = (:U235, :U238, :Pu239, :Pu241)
    # get fission fractions for fixed weighting
    fractions = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_fission_fractions.yaml"), dicttype=OrderedDict{String,Any})["parameters"]["fission_fractions"]
    fractions = NamedTuple((Symbol(key),value) for (key,value) in fractions)

    fluxes = extract_reactor_spectra(datadir)
    E = fluxes.E

    f_239_interp = Interpolator(E, fluxes.Pu239)
    Pu_239(E) = isnan(f_239_interp(E)) ? 0.0 : f_239_interp(E)

    f_241_interp = Interpolator(E, fluxes.Pu241)
    Pu_241(E) = isnan(f_241_interp(E)) ? 0.0 : f_241_interp(E)

    f_238_interp = Interpolator(E, fluxes.U238)
    U_238(E) = isnan(f_238_interp(E)) ? 0.0 : f_238_interp(E)

    f_235_interp = Interpolator(E, fluxes.U235)
    U_235(E) = isnan(f_235_interp(E)) ? 0.0 : f_235_interp(E)

    funcs = [U_235, U_238, Pu_239, Pu_241]
    isotope_shapes = NamedTuple{names}(funcs)

    # get nominal thermal power
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_thermal_power_nominal.yaml"))
    nominal_thermal_power = file["parameters"]["nominal_thermal_power"]   # GW

    # neutrinos per fission, is fixed
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/antineutrinos_per_fission_huber_mueller.yaml"))
    nu_per_f = file["parameters"]["antineutrinos_per_fission"]

    nu_per_fission = NamedTuple((Symbol(key), value) for (key, value) in nu_per_f)

    #snf correction
    E_snf = fluxes.E_snf
    snf_R1 = linear_interpolation(E_snf, fluxes.R1_snf, extrapolation_bc=0.)
    snf_R2 = linear_interpolation(E_snf, fluxes.R2_snf, extrapolation_bc=0.)
    snf_R3 = linear_interpolation(E_snf, fluxes.R3_snf, extrapolation_bc=0.)
    snf_R4 = linear_interpolation(E_snf, fluxes.R4_snf, extrapolation_bc=0.)
    snf_R5 = linear_interpolation(E_snf, fluxes.R5_snf, extrapolation_bc=0.)
    snf_R6 = linear_interpolation(E_snf, fluxes.R6_snf, extrapolation_bc=0.)

    snf_rel_corr = [snf_R1, snf_R2, snf_R3, snf_R4, snf_R5, snf_R6]

    # neq correction
    E_neq = fluxes.E_neq

    neq_U235 = linear_interpolation(E_neq, fluxes.neq_U235, extrapolation_bc=0.0)
    neq_Pu239 = linear_interpolation(E_neq, fluxes.neq_Pu239, extrapolation_bc=0.0)
    neq_Pu241 = linear_interpolation(E_neq, fluxes.neq_Pu241, extrapolation_bc=0.0)
    neq_U238(E) = 0.

    neq_funcs = [neq_U235, neq_U238, neq_Pu239, neq_Pu241]
    neq_rel_corr = NamedTuple{names}(neq_funcs)

    elementary_charge = 1.602176634e-19

    #           Giga   Joule -> eV        Mega
    GJ_to_MeV = 1e9 / elementary_charge * 1e-6 # reactor thermal power in GW, energy per fission in MeV
    # all spectra in MeV, hence get rid of the GJ


    # prelim flux shape correction
    spec_edges = convert(Array{Float64, 1}, readdlm(joinpath(datadir, "dayabay_data/parameters/reactor_antineutrino_spectrum_edges.tsv"))[2:end])
    width = diff(spec_edges)
    push!(spec_edges, spec_edges[end] + width[end-3])
    insert!(spec_edges, 1, spec_edges[1] - width[3])

    nodes = Diagonal(ones(length(spec_edges)))
    node_itp = []
    for i in 2:length(spec_edges) - 1
        interp = Interpolator(spec_edges, nodes[i, :])
        if i > 2 && i < length(spec_edges) - 1
            function func_a(E)
                val = interp(E)
                if val > 0.0
                    return val
                else
                    return 0.0
                end
            end
            push!(node_itp, func_a)
        elseif i == 2
            function func_b(E)
                #if E <= spec_edges[2]
                #    return interp(spec_edges[2])
                #end
                val = interp(E)
                if val > 0.0
                    return val
                else
                    return 0.0
                end
            end
            push!(node_itp, func_b)
        elseif i == length(spec_edges) - 1
            function func_c(E)
                #if E >= spec_edges[end-1]
                #    return interp(spec_edges[end-1])
                #end
                val = interp(E)
                if val > 0.0
                    return val
                else
                    return 0.0
                end
            end
            push!(node_itp, func_c)
        else
            println(i)
        end
    end

    # use default params for the snf contribution
    default_params = get_params(cfg, datadir=datadir)
    # get fissions per second for all reactors at nominal values
    fissions_per_second_nom = GJ_to_MeV / sum([fractions[iso] * default_params.energy_per_fission[i] for (i, iso) in enumerate(names)])
    #neutrinos_per_second_nom = fissions_per_second_nom * sum([nu_per_fission[iso] * fractions[iso] for (i, iso) in enumerate(names)])
    #println(neutrinos_per_second_nom)

    function correction(E, pulls)
        return sum([pull * itp.(E) for (pull, itp) in zip(pulls, node_itp)])
    end

    function snf_flux(E, snf_scale, reac_idx)
        snf_scale * fissions_per_second_nom * sum([fractions[iso] * nu_per_fission[iso] * isotope_shapes[iso].(E) for (i, iso) in enumerate(names)]) .* snf_rel_corr[reac_idx].(E)
    end

    function reactor_flux(E, thermal_power_scale, energy_per_fission, fission_fractions_scale, pulls, neq_scale, snf_scale, reac_idx)
        flux = GJ_to_MeV * thermal_power_scale * sum([
            (1.0 .+ neq_scale[i] .* neq_rel_corr[iso].(E)) .*
            nu_per_fission[iso] .* fission_fractions_scale[i] .* fractions[iso] .* isotope_shapes[iso].(E) for (i, iso) in enumerate(names)]
            ) / sum(
                [fission_fractions_scale[i] * fractions[iso] * energy_per_fission[i] for (i, iso) in enumerate(names)]
            )
        return flux, snf_flux(E, snf_scale, reac_idx), correction(E, pulls)
    end

    reactor_flux
end


end