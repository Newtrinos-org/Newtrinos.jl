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
    average::Bool = false
    N_pulls::Int64 = 19
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
    average::Bool
end


function configure(cfg::ReactorFluxConfig = ReactorFluxConfig())
    ReactorFlux(
        cfg=cfg,
        params = get_params(cfg.flux_model, datadir=cfg.datadir, N_pulls=cfg.N_pulls),
        priors = get_priors(cfg.flux_model, datadir=cfg.datadir, N_pulls=cfg.N_pulls),
        flux = get_flux(cfg.flux_model, datadir=cfg.datadir, average=cfg.average),
        flux_on_mesh = get_flux_on_mesh(cfg.flux_model, datadir=cfg.datadir, average=cfg.average, N_pulls=cfg.N_pulls),
        datadir = cfg.datadir,
        average=cfg.average,
    )
end


function get_params(flux::DayaBayFlux; datadir = datadir, N_pulls = 19)
    # TODO: is this part of DayaBay experiment or the reactor flux itself? not sure where to put
    # currently I define that we need 6 fluxes in here, but this is specific to the experiment
    # re-using for e.g. Juno, we need a different number of reactors, for future Julian...

    ## energy per fission
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_energy_per_fission.yaml"))
    data = file["parameters"]["energy_per_fission"]

    energy_per_fission = [data["U235"][1], data["U238"][1], data["Pu239"][1], data["Pu241"][1]]

    spectrum_pulls = zeros(N_pulls)

    params = (;
        energy_per_fission,
        spectrum_pulls,
    )
    return params
end


function get_priors(flux::DayaBayFlux; datadir = datadir, N_pulls = 19)

    ## energy per fission
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_energy_per_fission.yaml"))
    data = file["parameters"]["energy_per_fission"]

    mu = [data["U235"][1], data["U238"][1], data["Pu239"][1], data["Pu241"][1]]
    var = [data["U235"][2], data["U238"][2], data["Pu239"][2], data["Pu241"][2]].^2
    energy_per_fission = Distributions.MvNormal(mu, Diagonal(var))
    spectrum_pulls = Distributions.Uniform.(-ones(N_pulls), ones(N_pulls))

    priors = (;
        energy_per_fission,
        spectrum_pulls,
    )

end


function extract_reactor_spectra(datadir = datadir; average = false)
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

    if average
        av = mean((R1_snf, R2_snf, R3_snf, R4_snf, R5_snf, R6_snf))
        return_tup = (
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
            snf=av
        )

        return return_tup
    end

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
    get_flux_on_mesh(cfg; datadir, average)

Returns `E -> flux` where `flux(thermal_power_scale, energy_per_fission, fission_fractions_scale, pulls, neq_scale,
snf_scale, reac_idx; correction)` gives `(main, neq, snf, correction)` at the antineutrino energies `E` (see
[`get_flux`](@ref)); the free spectrum correction depends only on `pulls` and can be passed in as `correction`.
Everything that does not depend on the parameters (isotope spectra, NEQ and SNF corrections, interpolation weights of
the free spectrum correction) is tabulated once for the mesh `E`.
"""
function get_flux_on_mesh(cfg::DayaBayFlux; datadir = datadir, average = false, N_pulls = 19)

    # fixed ordering of isotopes
    names = (:U235, :U238, :Pu239, :Pu241)
    # get fission fractions for fixed weighting
    fractions = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_fission_fractions.yaml"), dicttype=OrderedDict{String,Any})["parameters"]["fission_fractions"]
    fractions = [Float64(fractions[string(iso)]) for iso in names]

    fluxes = extract_reactor_spectra(datadir; average=average)

    # Huber-Mueller spectra [ν/fission/MeV], interpolated exponentially (log-linear) as in the official model
    isotope_shapes = [exp_interpolator(fluxes.E, fluxes.U235), exp_interpolator(fluxes.E, fluxes.U238),
                      exp_interpolator(fluxes.E, fluxes.Pu239), exp_interpolator(fluxes.E, fluxes.Pu241)]
    # note: the Huber-Mueller spectra are per fission already; the number of antineutrinos per fission is only needed
    # to convert the reactor antineutrino rates into thermal power (in the experiment module)

    # snf correction (relative to the nominal spectrum, per reactor)
    if average
        snf_rel_corr = linear_interpolation(fluxes.E_snf, fluxes.snf, extrapolation_bc = 0.0)
    else
        snf_rel_corr = [linear_interpolation(fluxes.E_snf, getproperty(fluxes, Symbol("R$(r)_snf")), extrapolation_bc = 0.0) for r in 1:6]
    end
    # neq correction (relative, per isotope; none for U238)
    neq_rel_corr = [linear_interpolation(fluxes.E_neq, fluxes.neq_U235, extrapolation_bc = 0.0), E -> 0.0,
                    linear_interpolation(fluxes.E_neq, fluxes.neq_Pu239, extrapolation_bc = 0.0),
                    linear_interpolation(fluxes.E_neq, fluxes.neq_Pu241, extrapolation_bc = 0.0)]

    elementary_charge = 1.602176634e-19
    #           Giga   Joule -> eV        Mega
    GJ_to_MeV = 1e9 / elementary_charge * 1e-6 # reactor thermal power in GW, energy per fission in MeV

    # free spectrum correction: one parameter per edge of the official segmentation
    if N_pulls == 19
        spec_edges = convert(Array{Float64, 1}, readdlm(joinpath(datadir, "dayabay_data/parameters/reactor_antineutrino_spectrum_edges.tsv"))[2:end])
    
    else
        spec_edges = LinRange(1.8, 9.0, N_pulls)
    end

    # SNF: fissions per second of a reactor at nominal thermal power, fission fractions and energies per fission
    default_params = get_params(cfg, datadir=datadir, N_pulls=N_pulls)
    fissions_per_second_nom_bar_nom_th_power = GJ_to_MeV / sum(fractions .* default_params.energy_per_fission)

    function on_mesh(E::AbstractVector)
        #shape for each isotope
        S = [f.(E) for f in isotope_shapes]
        #relative NEQ correction for each isotope
        NEQ = [neq_rel_corr[i].(E) .* S[i] for i in 1:4]
        #relative SNF correction for each isotope for each reactor
        if average
            SNF = fissions_per_second_nom_bar_nom_th_power .* sum(fractions[i] .* S[i] for i in 1:4) .* snf_rel_corr.(E)
        else
            SNF = [fissions_per_second_nom_bar_nom_th_power .* sum(fractions[i] .* S[i] for i in 1:4) .* snf_rel_corr[r].(E) for r in 1:6]
        end
        # linear interpolation weights of the correction exponent between the edges (extrapolated outside)
        k = [clamp(searchsortedlast(spec_edges, e), 1, length(spec_edges) - 1) for e in E]
        t = [(e - spec_edges[kk]) / (spec_edges[kk+1] - spec_edges[kk]) for (e, kk) in zip(E, k)]

        spectrum_correction(pulls) = @. exp(pulls[k] + t * (pulls[k+1] - pulls[k]))
        # `correction` can be passed in when it is shared by several reactors (it depends only on `pulls`)
        function flux(thermal_power_scale, energy_per_fission, fission_fractions_scale, pulls, neq_scale, snf_scale, reac_idx;
            correction = spectrum_correction(pulls))
            w = fission_fractions_scale .* fractions
            fissions_per_GJ = GJ_to_MeV * thermal_power_scale / sum(w .* energy_per_fission)
            main = fissions_per_GJ .* (w[1] .* S[1] .+ w[2] .* S[2] .+ w[3] .* S[3] .+ w[4] .* S[4])
            # skips 2nd entry (U238 has no NEQ correction!)
            neq = fissions_per_GJ .* (neq_scale[1] * w[1] .* NEQ[1] .+ neq_scale[3] * w[3] .* NEQ[3] .+ neq_scale[4] * w[4] .* NEQ[4])
            if average
                SNF_fac = SNF
            else
                SNF_fac = SNF[reac_idx]
            end
            return main, neq, snf_scale .* SNF_fac, correction
        end
    end
end

"""
    get_flux(cfg; datadir, average)

Returns `reactor_flux(E, thermal_power_scale, energy_per_fission, fission_fractions_scale, pulls, neq_scale,
snf_scale, reac_idx) -> (main, neq, snf, correction)` at antineutrino energies `E` (number or vector): the
antineutrino spectrum per GW thermal power [ν/s/MeV/GW] and the extra flux from the non-equilibrium correction, the
SNF flux [ν/s/MeV] of a reactor at nominal power, and the free spectrum correction, which multiplies only `main` (as
in the official model). For repeated evaluation on a fixed mesh use [`get_flux_on_mesh`](@ref).
"""
function get_flux(cfg::DayaBayFlux; datadir = datadir, average = false)
    on_mesh = get_flux_on_mesh(cfg; datadir = datadir, average = average)
    function reactor_flux(E, args...)
        E isa Number && return map(first, on_mesh([E])(args...))
        on_mesh(collect(Float64, E))(args...)
    end
end

end