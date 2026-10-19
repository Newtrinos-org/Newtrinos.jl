module dayabay_flux

using Interpolations
using DataStructures
using LinearAlgebra
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
    flux_on_mesh::Function
    datadir::String
end


function configure(cfg::ReactorFluxConfig = ReactorFluxConfig())
    ReactorFlux(
        cfg=cfg,
        params = get_params(cfg.flux_model, datadir=cfg.datadir),
        priors = get_priors(cfg.flux_model, datadir=cfg.datadir),
        flux = get_flux(cfg.flux_model, datadir=cfg.datadir),
        flux_on_mesh = get_flux_on_mesh(cfg.flux_model, datadir=cfg.datadir),
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
    spectrum_pulls = zeros(19)

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

    # free parameters (official: unbounded); exp(±1) is far outside any fitted value
    spectrum_pulls = Distributions.Uniform.(-ones(19), ones(19))

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


"""
    exp_interpolator(x, y)

Exponential interpolation of positive `y(x)`: linear in log(y), extrapolated with the first/last segment.
"""
function exp_interpolator(x::Vector{Float64}, y::Vector{Float64})
    logy = log.(y)
    function f(E)
        k = clamp(searchsortedlast(x, E), 1, length(x) - 1)
        t = (E - x[k]) / (x[k+1] - x[k])
        exp(logy[k] + t * (logy[k+1] - logy[k]))
    end
end

"""
    get_flux_on_mesh(cfg; datadir)

Returns `E -> flux` where `flux(thermal_power_scale, energy_per_fission, fission_fractions_scale, pulls, neq_scale,
snf_scale, reac_idx)` gives `(main, neq, snf, correction)` at the antineutrino energies `E` (see [`get_flux`](@ref)).
Everything that does not depend on the parameters (isotope spectra, NEQ and SNF corrections, interpolation weights of
the free spectrum correction) is tabulated once for the mesh `E`.
"""
function get_flux_on_mesh(cfg::DayaBayFlux; datadir = datadir)

    # fixed ordering of isotopes
    names = (:U235, :U238, :Pu239, :Pu241)
    # get fission fractions for fixed weighting
    fractions = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_fission_fractions.yaml"), dicttype=OrderedDict{String,Any})["parameters"]["fission_fractions"]
    fractions = [Float64(fractions[string(iso)]) for iso in names]

    fluxes = extract_reactor_spectra(datadir)

    # Huber-Mueller spectra [ν/fission/MeV], interpolated exponentially (log-linear) as in the official model
    isotope_shapes = [exp_interpolator(fluxes.E, fluxes.U235), exp_interpolator(fluxes.E, fluxes.U238),
                      exp_interpolator(fluxes.E, fluxes.Pu239), exp_interpolator(fluxes.E, fluxes.Pu241)]
    # note: the Huber-Mueller spectra are per fission already; the number of antineutrinos per fission is only needed
    # to convert the reactor antineutrino rates into thermal power (in the experiment module)

    # get nominal thermal power
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_thermal_power_nominal.yaml"))
    nominal_thermal_power = file["parameters"]["nominal_thermal_power"]   # GW

    # snf correction (relative to the nominal spectrum, per reactor)
    snf_rel_corr = [linear_interpolation(fluxes.E_snf, getproperty(fluxes, Symbol("R$(r)_snf")), extrapolation_bc = 0.0) for r in 1:6]
    # neq correction (relative, per isotope; none for U238)
    neq_rel_corr = [linear_interpolation(fluxes.E_neq, fluxes.neq_U235, extrapolation_bc = 0.0), E -> 0.0,
                    linear_interpolation(fluxes.E_neq, fluxes.neq_Pu239, extrapolation_bc = 0.0),
                    linear_interpolation(fluxes.E_neq, fluxes.neq_Pu241, extrapolation_bc = 0.0)]

    elementary_charge = 1.602176634e-19
    #           Giga   Joule -> eV        Mega
    GJ_to_MeV = 1e9 / elementary_charge * 1e-6 # reactor thermal power in GW, energy per fission in MeV

    # free spectrum correction: one parameter per edge of the official segmentation
    spec_edges = convert(Array{Float64, 1}, readdlm(joinpath(datadir, "dayabay_data/parameters/reactor_antineutrino_spectrum_edges.tsv"))[2:end])

    # SNF: fissions per second of a reactor at nominal thermal power, fission fractions and energies per fission
    default_params = get_params(cfg, datadir=datadir)
    fissions_per_second_nom = nominal_thermal_power * GJ_to_MeV / sum(fractions .* default_params.energy_per_fission)

    function on_mesh(E::AbstractVector)
        S = [f.(E) for f in isotope_shapes]
        NEQ = [neq_rel_corr[i].(E) .* S[i] for i in 1:4]
        SNF = [fissions_per_second_nom .* sum(fractions[i] .* S[i] for i in 1:4) .* snf_rel_corr[r].(E) for r in 1:6]
        # linear interpolation weights of the correction exponent between the edges (extrapolated outside)
        k = [clamp(searchsortedlast(spec_edges, e), 1, length(spec_edges) - 1) for e in E]
        t = [(e - spec_edges[kk]) / (spec_edges[kk+1] - spec_edges[kk]) for (e, kk) in zip(E, k)]

        function flux(thermal_power_scale, energy_per_fission, fission_fractions_scale, pulls, neq_scale, snf_scale, reac_idx)
            w = fission_fractions_scale .* fractions
            fissions_per_GJ = GJ_to_MeV * thermal_power_scale / sum(w .* energy_per_fission)
            main = fissions_per_GJ .* (w[1] .* S[1] .+ w[2] .* S[2] .+ w[3] .* S[3] .+ w[4] .* S[4])
            neq = fissions_per_GJ .* (neq_scale[1] * w[1] .* NEQ[1] .+ neq_scale[3] * w[3] .* NEQ[3] .+ neq_scale[4] * w[4] .* NEQ[4])
            correction = @. exp(pulls[k] + t * (pulls[k+1] - pulls[k]))
            return main, neq, snf_scale .* SNF[reac_idx], correction
        end
    end
end

"""
    get_flux(cfg; datadir)

Returns `reactor_flux(E, thermal_power_scale, energy_per_fission, fission_fractions_scale, pulls, neq_scale,
snf_scale, reac_idx) -> (main, neq, snf, correction)` at antineutrino energies `E` (number or vector): the
antineutrino spectrum per GW thermal power [ν/s/MeV/GW] and the extra flux from the non-equilibrium correction, the
SNF flux [ν/s/MeV] of a reactor at nominal power, and the free spectrum correction, which multiplies only `main` (as
in the official model). For repeated evaluation on a fixed mesh use [`get_flux_on_mesh`](@ref).
"""
function get_flux(cfg::DayaBayFlux; datadir = datadir)
    on_mesh = get_flux_on_mesh(cfg; datadir = datadir)
    function reactor_flux(E, args...)
        E isa Number && return map(first, on_mesh([E])(args...))
        on_mesh(collect(Float64, E))(args...)
    end
end

end