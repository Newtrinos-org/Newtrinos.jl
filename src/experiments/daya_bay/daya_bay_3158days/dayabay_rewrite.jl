module dayabay_rewrite
using DataFrames
using CSV
using LinearAlgebra
using Distributions
using DataStructures
using CairoMakie
using Accessors
using Logging
using BAT
using HDF5
using Interpolations
using Integrals
using SpecialFunctions: erfc
#using SparseArrays
using DelimitedFiles
using PCHIPInterpolation
import YAML
import ..Newtrinos

@kwdef struct DayaBay <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function default_physics()
    osc = Newtrinos.osc.configure()
    xsec = Newtrinos.ibd_xsec.configure()
    flux = Newtrinos.reactor_flux.configure()
    (; osc, xsec, flux)
end

function configure(physics=default_physics(), datadir = @__DIR__)
    physics = (;physics.osc, physics.xsec, physics.flux)
    assets = get_assets(datadir)
    return DayaBay(
        physics = physics,
        params = get_params(datadir),
        priors = get_priors(datadir),
        assets = assets,
        forward_model = get_forward_model(physics, assets, datadir=datadir),
        plot = get_plot(physics, assets)
    )
end

function retrieve_period(str::String)
    parse(Int, str[1])
end

function retrieve_AD(str::String)
    parse(Int, str[6:7])
end

function retrieve_EH(str::String)
    parse(Int, str[6])
end


period_list = ["6AD", "8AD", "7AD"]

EH_list = [1, 2, 3]

full_setup = ["AD11", "AD12", "AD21", "AD22", "AD31", "AD32", "AD33", "AD34"]
mask_6 =     [true,   true,   true,   false,  true,   true,   true,   false]
mask_8 =     [true,   true,   true,   true,   true,   true,   true,   true]
mask_7 =     [false,  true,   true,   true,   true,   true,   true,   true]

baselines = YAML.load_file(joinpath(@__DIR__, "dayabay_data/parameters/baselines.yaml"))
reactors = ["R$(i)" for i in 1:6]
baselines["parameters"]["baseline"]

distances = OrderedDict()
distances["AD"] = full_setup
for reac in reactors
    distances[reac] = [baselines["parameters"]["baseline"][ad][reac] for ad in full_setup]
end

distances["6AD"] = mask_6
distances["8AD"] = mask_8
distances["7AD"] = mask_7

df_exp = DataFrame(distances)


detectors_6AD = full_setup[mask_6]
detectors_8AD = full_setup[mask_8]
detectors_7AD = full_setup[mask_7]

function get_observed_counts(AD::Int, period::Int, datadir = @__DIR__)
    data_basepath = "dayabay_data/dayabay_dataset"
    spectrum_path = joinpath(datadir, data_basepath, "dayabay_ibd_spectra_$(period)AD.hdf5")
    file = h5open(spectrum_path, "r")

    counts_ibd = Int64[]
    E_bins_MeV = Float64[]

    foreach(x -> push!(counts_ibd, x.N), file["ibd_spectrum_AD$(AD)"][1:end])
    foreach(x -> push!(E_bins_MeV, x.E_min_MeV), file["ibd_spectrum_AD$(AD)"][1:end])
    push!(E_bins_MeV, file["ibd_spectrum_AD$(AD)"][end].E_max_MeV)
    close(file)
    E_center_MeV = (E_bins_MeV[1:end-1] .+ E_bins_MeV[2:end]) ./ 2

    coarse_binning = readdlm(joinpath(datadir, "dayabay_data/parameters/final_erec_bin_edges.tsv"))[2:end];
    coarse_binning_c = (coarse_binning[2:end] + coarse_binning[1:end-1]) / 2
    coarse_bin_width = diff(coarse_binning)

    rebin_idx = searchsortedlast.(Ref(coarse_binning), E_center_MeV)

    rebinned_counts = Int64[]
    for i in eachindex(coarse_binning_c)
        push!(rebinned_counts, sum(counts_ibd[rebin_idx .== i]))
    end
    rebinned_counts

end


function extract_for_AD_period(AD::Int, period::Int, datadir = @__DIR__)
    # extract IBD rate
    data_basepath = "dayabay_data/dayabay_dataset"
    spectrum_path = joinpath(datadir, data_basepath, "dayabay_ibd_spectra_$(period)AD.hdf5")
    file = h5open(spectrum_path, "r")

    E_bins_MeV = Float64[]

    foreach(x -> push!(E_bins_MeV, x.E_min_MeV), file["ibd_spectrum_AD$(AD)"][1:end])
    push!(E_bins_MeV, file["ibd_spectrum_AD$(AD)"][end].E_max_MeV)
    close(file)
    E_center_MeV = (E_bins_MeV[1:end-1] .+ E_bins_MeV[2:end]) ./ 2

    observed = get_observed_counts(AD, period)

    coarse_binning = readdlm(joinpath(datadir, "dayabay_data/parameters/final_erec_bin_edges.tsv"))[2:end];
    coarse_binning_c = (coarse_binning[2:end] + coarse_binning[1:end-1]) / 2
    coarse_bin_width = diff(coarse_binning)

    rebin_idx = searchsortedlast.(Ref(coarse_binning), E_center_MeV)


    # extract background shapes
    bg_shape_path = joinpath(datadir, data_basepath, "dayabay_background_spectra_$(period)AD.hdf5")
    file = h5open(bg_shape_path)
    _shape_accidental = Float64[]
    _shape_alpha_neutron = Float64[]
    _shape_amc = Float64[]
    _shape_fast_neutrons = Float64[]
    _shape_lithium_helium = Float64[]

    foreach(x -> push!(_shape_accidental, x.N), file["spectrum_shape_accidentals_AD$(AD)"][1:end])
    foreach(x -> push!(_shape_alpha_neutron, x.N), file["spectrum_shape_alpha_neutron_AD$(AD)"][1:end])
    foreach(x -> push!(_shape_amc, x.N), file["spectrum_shape_amc_AD$(AD)"][1:end])
    foreach(x -> push!(_shape_fast_neutrons, x.N), file["spectrum_shape_fast_neutrons_AD$(AD)"][1:end])
    foreach(x -> push!(_shape_lithium_helium, x.N), file["spectrum_shape_lithium_helium_AD$(AD)"][1:end])
    close(file)

    shape_accidental = Float64[]
    shape_alpha_neutron = Float64[]
    shape_amc = Float64[]
    shape_fast_neutrons = Float64[]
    shape_lithium_helium = Float64[]
    for i in 1:length(coarse_binning_c)
        push!(shape_accidental, sum(_shape_accidental[rebin_idx.==i]))
        push!(shape_alpha_neutron, sum(_shape_alpha_neutron[rebin_idx.==i]))
        push!(shape_amc, sum(_shape_amc[rebin_idx.==i]))
        push!(shape_fast_neutrons, sum(_shape_fast_neutrons[rebin_idx.==i]))
        push!(shape_lithium_helium, sum(_shape_lithium_helium[rebin_idx.==i]))
    end


    # background rates + uncertainties
    bg_rate_path = joinpath(datadir, data_basepath, "dayabay_background_rates.hdf5")
    data = h5open(bg_rate_path)
    rate_accidental = data["$(period)AD"][1][Symbol("AD$(AD)")]
    uncertainty_accidental = data["$(period)AD"][2][Symbol("AD$(AD)")]
    rate_lithium_helium = data["$(period)AD"][3][Symbol("AD$(AD)")]
    uncertainty_lithium_helium = data["$(period)AD"][4][Symbol("AD$(AD)")]
    rate_fast_neutrons = data["$(period)AD"][5][Symbol("AD$(AD)")]
    uncertainty_fast_neutrons = data["$(period)AD"][6][Symbol("AD$(AD)")]
    rate_amc = data["$(period)AD"][7][Symbol("AD$(AD)")]
    uncertainty_amc = data["$(period)AD"][8][Symbol("AD$(AD)")]
    rate_alpha_neutron = data["$(period)AD"][9][Symbol("AD$(AD)")]
    uncertainty_alpha_neutron = data["$(period)AD"][10][Symbol("AD$(AD)")]
    close(data)

    # daily detector data: lifetime + daily rate for accidentals
    daily_path = joinpath(datadir, data_basepath, "dayabay_daily_detector_data.hdf5")
    data = h5open(daily_path)
    eff_livetime = Float64[]
    acc_rate = Float64[]
    livetime = Float64[]
    day = Int64[]
    foreach(x -> x[:n_det] == period ? push!(eff_livetime, x[:eff_livetime]) : 0, data["AD$(AD)"][1:end])
    foreach(x -> x[:n_det] == period ? push!(acc_rate, x[:rate_accidentals]) : 0, data["AD$(AD)"][1:end])
    foreach(x -> x[:n_det] == period ? push!(livetime, x[:livetime]) : 0, data["AD$(AD)"][1:end])
    foreach(x -> x[:n_det] == period ? push!(day, x[:day]) : 0, data["AD$(AD)"][1:end])
    #foreach(x -> x[:n_det] == 6 ? push!(lt, x[:livetime]) : 0, data["AD11"][1:end])

    close(data)

    eff_livetime_seconds = sum(eff_livetime)
    eff_livetime_days = eff_livetime_seconds / 60 / 60 / 24   # convert from seconds to days

    lihe = (rate=rate_lithium_helium, shape=shape_lithium_helium, uncertainty=uncertainty_lithium_helium)
    amc = (rate=rate_amc, shape=shape_amc, uncertainty=uncertainty_amc)
    fast_neutrons = (rate=rate_fast_neutrons, shape=shape_fast_neutrons, uncertainty=uncertainty_fast_neutrons)
    alpha_neutron = (rate=rate_alpha_neutron, shape=shape_alpha_neutron, uncertainty=uncertainty_alpha_neutron)
    accidentals = (rate=rate_accidental, shape=shape_accidental, uncertainty=uncertainty_accidental)

    bg_dict = Dict()
    bg_dict["accidentals"] = accidentals
    bg_dict["lithium_helium"] = lihe
    bg_dict["alpha_neutron"] = alpha_neutron
    bg_dict["fast_neutrons"] = fast_neutrons
    bg_dict["amc"] = amc

    (;observed, E_bins_MeV, E_center_MeV, bg_dict, eff_livetime_seconds, eff_livetime, eff_livetime_days, day)
end


"""
IAV smearing applier.
`(f)(scale)` returns the scaled dense IAV matrix;
`(f)(scale, spec)` applies it to a spectrum from the r.h.s. without materializing the matrix.
"""
struct IAVApplier
    iav::Matrix{Float64}
    offdiag::Matrix{Float64}
    diag::Vector{Float64}
end

function (f::IAVApplier)(offdiag_scale)
    m = offdiag_scale * f.offdiag
    @inbounds for i in 1:length(f.diag)
        m[i, i] += f.diag[i]
    end
    m
end

(f::IAVApplier)(offdiag_scale, spec) = f.diag .* spec .+ offdiag_scale * (f.offdiag * spec)


function get_iav_matrix(datadir = @__DIR__)
    file = h5open(joinpath(datadir, "dayabay_data/detector_iav_matrix.hdf5"))

    iav = transpose(file["iav_matrix"][])
    close(file)

    IAVApplier(iav, iav .- Diagonal(iav), diag(iav))
end

function read_lsnl_correction(datadir = @__DIR__)
    # gives ratio of visible/true energy, hence multiply true energy to go to visible
    file = h5open(joinpath(datadir, "dayabay_data/detector_lsnl_curves.hdf5"))

    f_nom = Float64[]
    pull0 = Float64[]
    pull1 = Float64[]
    pull2 = Float64[]
    pull3 = Float64[]
    E = Float64[]
    E0 = Float64[]
    E1 = Float64[]
    E2 = Float64[]
    E3 = Float64[]
    foreach(x -> (push!(E, x[1]), push!(f_nom, x[2])), file["nominal"][1:end])
    foreach(x -> (push!(E0, x[1]), push!(pull0, x[2])), file["pull0"][1:end])
    foreach(x -> (push!(E1, x[1]), push!(pull1, x[2])), file["pull1"][1:end])
    foreach(x -> (push!(E2, x[1]), push!(pull2, x[2])), file["pull2"][1:end])
    foreach(x -> (push!(E3, x[1]), push!(pull3, x[2])), file["pull3"][1:end])
    close(file)

    # subtract norm from pull term, can use multiplicative term for pull spectra; factor of zero means nominal
    rel_0 = pull0 .- f_nom
    rel_1 = pull1 .- f_nom
    rel_2 = pull2 .- f_nom
    rel_3 = pull3 .- f_nom

    @assert all(isapprox.(E, E0))
    @assert all(isapprox.(E0, E1))
    @assert all(isapprox.(E1, E2))
    @assert all(isapprox.(E2, E3))

    return (;E, f_nom, rel_0, rel_1, rel_2, rel_3)

end

function get_lsnl_correction(datadir = @__DIR__)
    lsnl = read_lsnl_correction(datadir)
    #interp = Interpolator(lsnl.E, lsnl.f_nom, extrapolate=true)
    #func(E) = isnan(interp(E)) ? 0.0 : interp(E)
    pull1 = linear_interpolation(lsnl.E, lsnl.rel_0, extrapolation_bc=Flat())
    pull2 = linear_interpolation(lsnl.E, lsnl.rel_1, extrapolation_bc=Flat())
    pull3 = linear_interpolation(lsnl.E, lsnl.rel_2, extrapolation_bc=Flat())
    pull4 = linear_interpolation(lsnl.E, lsnl.rel_3, extrapolation_bc=Flat())
    nom = linear_interpolation(lsnl.E, lsnl.f_nom, extrapolation_bc=Flat())

    function _lsnl(E, pull)
        return @. nom(E) + pull[1] * pull1(E) + pull[2] * pull2(E) + pull[3] * pull3(E) + pull[4] * pull4(E)
    end
end


function eres(E, a, b, c)
    sigma_E = @. sqrt(a^2 * E^2 + b^2 * E + c^2)
end

function smear(E_arr_smear_local, smear_arr_in, sigma_arr; width=10, E_scale=1.0, E_bias=0.0)

    l = length(smear_arr_in)
    T_acc = promote_type(eltype(E_arr_smear_local), eltype(sigma_arr), eltype(smear_arr_in), typeof(E_scale))
    out = zeros(T_acc, l)

    for i in 1:l

        e_center = E_arr_smear_local[i] * E_scale + E_bias

        norm_val = zero(T_acc)
        sum_val = zero(T_acc)

        j_min_loop = max(1, i - width)
        j_max_loop = min(l, i + width)

        for j in j_min_loop:j_max_loop
            coeff = (1 / sigma_arr[j]) * exp(-0.5 * ((e_center - E_arr_smear_local[j]) / sigma_arr[j])^2)
            norm_val += coeff
            sum_val += coeff * smear_arr_in[j]
        end

        if norm_val > 1e-10
            out[i] = sum_val / norm_val
        end

    end
    return out
end




function get_proton_number(datadir = @__DIR__)
    data = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/detector_n_protons_nominal.yaml"))
    nom = data["parameters"]["n_protons_nominal_ad"]
    data = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/detector_n_protons_correction.yaml"), dicttype=OrderedDict{String,Any})
    correction = data["parameters"]["n_protons_correction"]
    n_protons = OrderedDict(key => correction[key] * nom for key in keys(correction))
end


function get_params(datadir = @__DIR__)
    ## accidentals
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/background_rate_scale_accidentals.yaml"))
    # multiply for all ADs
    acc_scale_nom = dict["parameters"]["accidentals"][1]  # blow up to vector over all ADs

    len = length(detectors_6AD) + length(detectors_8AD) + length(detectors_7AD)
    acc_scale = acc_scale_nom .* ones(len)


    ## amc
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/background_rate_uncertainty_scale_amc.yaml"))

    amc_unc_scale = dict["parameters"]["amc"][1]


    ## site correlated
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/background_rate_uncertainty_scale_site.yaml"))

    lihe_nom = dict["parameters"]["lithium_helium"][1]
    fast_n_nom = dict["parameters"]["fast_neutrons"][1]

    len = length(period_list) * length(EH_list)
    fast_n_unc_scale = fast_n_nom .* ones(len)
    lihe_unc_scale = lihe_nom .* ones(len)


    ## uncorrelated
    # dicttype keeps the order in which the yaml is written
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/background_rates_uncorrelated.yaml"), dicttype=OrderedDict{String,Any})

    alpha_n_nom = Float64[]

    for k in period_list
        an = dict["parameters"]["alpha_neutron"][k]
        for (k, v) in an
            push!(alpha_n_nom, v[1])
        end
    end

    alpha_n_rate = alpha_n_nom


    ## AD efficiency factor / energy scale: per AD, constant over periods
    # is correlated within AD
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/detector_relative.yaml"))

    eff_nom = Float64(dict["parameters"]["detector_relative"]["energy_scale_factor"][1])
    e_scale_nom =  dict["parameters"]["detector_relative"]["energy_scale_factor"][1]

    eff_eres_nom = Vector([eff_nom, e_scale_nom])
    eff_eres_AD11 = eff_eres_nom
    eff_eres_AD12 = eff_eres_nom
    eff_eres_AD21 = eff_eres_nom
    eff_eres_AD22 = eff_eres_nom
    eff_eres_AD31 = eff_eres_nom
    eff_eres_AD32 = eff_eres_nom
    eff_eres_AD33 = eff_eres_nom
    eff_eres_AD34 = eff_eres_nom


    ## energy resolution, shared across all ADs
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/detector_eres.yaml"))
    a_nom = dict["parameters"]["eres"]["a_nonuniform"][1]
    b_nom = dict["parameters"]["eres"]["b_stat"][1]
    c_nom = dict["parameters"]["eres"]["c_noise"][1]

    eres_a = a_nom
    eres_b = b_nom
    eres_c = c_nom


    ## iav off diagonal scaling, one for each AD, constant over periods
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/detector_iav_offdiag_scale.yaml"))
    iav_scale_nom = dict["parameters"]["iav_offdiag_scale_factor"][1]
    len = length(full_setup)
    iav_offdiag_scale = iav_scale_nom .* ones(len)

    ## lsnl correction, pull parameters
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/detector_lsnl.yaml"))
    lsnl_scale_nom = dict["parameters"]["lsnl_scale_a"][1]
    lsnl_pull = lsnl_scale_nom .* ones(4)


    params = (;
        amc_unc_scale,
        acc_scale,
        lihe_unc_scale,
        fast_n_unc_scale,
        alpha_n_rate,
        eres_a,
        eres_b,
        eres_c,
        eff_eres_AD11,
        eff_eres_AD12,
        eff_eres_AD21,
        eff_eres_AD22,
        eff_eres_AD31,
        eff_eres_AD32,
        eff_eres_AD33,
        eff_eres_AD34,
        lsnl_pull,
        iav_offdiag_scale
    )
    return params
end



function get_priors(datadir = @__DIR__)
    ## accidentals
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/background_rate_scale_accidentals.yaml"))
    # multiply for all ADs
    acc_scale_nom = dict["parameters"]["accidentals"][1]  # blow up to vector over all ADs
    acc_scale_unc = dict["parameters"]["accidentals"][2] * acc_scale_nom * 0.01   # percent error

    len = length(detectors_6AD) + length(detectors_8AD) + length(detectors_7AD)
    acc_scale = Distributions.MvNormal(acc_scale_nom .* ones(len), Diagonal(acc_scale_unc^2 .* ones(len)))


    ## amc
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/background_rate_uncertainty_scale_amc.yaml"))

    amc_scale_nom = dict["parameters"]["amc"][1]
    amc_scale_unc = dict["parameters"]["amc"][2]  # absolute

    amc_unc_scale = Distributions.Normal(amc_scale_nom, amc_scale_unc)


    ## site correlated
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/background_rate_uncertainty_scale_site.yaml"))

    lihe_nom = dict["parameters"]["lithium_helium"][1]
    lihe_unc = dict["parameters"]["lithium_helium"][2]
    fast_n_nom = dict["parameters"]["fast_neutrons"][1]
    fast_n_unc = dict["parameters"]["fast_neutrons"][2]

    len = length(period_list) * length(EH_list)
    fast_n_unc_scale = Distributions.MvNormal(fast_n_nom .* ones(len), Diagonal(fast_n_unc^2 .* ones(len)))
    lihe_unc_scale = Distributions.MvNormal(lihe_nom .* ones(len), Diagonal(lihe_unc^2 .* ones(len)))


    ## uncorrelated
    # dicttype keeps the order in which the yaml is written
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/background_rates_uncorrelated.yaml"), dicttype=OrderedDict{String,Any})

    alpha_n_nom = Float64[]
    alpha_n_unc = Float64[]

    for k in period_list
        an = dict["parameters"]["alpha_neutron"][k]
        for (k, v) in an
            push!(alpha_n_nom, v[1])
            push!(alpha_n_unc, v[2])
        end
    end

    alpha_n_rate = Distributions.MvNormal(alpha_n_nom, Diagonal(alpha_n_unc.^2))


    ## AD efficiency factor / energy scale: per AD, constant over periods
    # is correlated within AD
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/detector_relative.yaml"))

    eff_nom = Float64(dict["parameters"]["detector_relative"]["efficiency_factor"][1])
    eff_unc = eff_nom * Float64(dict["parameters"]["detector_relative"]["efficiency_factor"][2] * 0.01)  # percent
    e_scale_nom =  Float64(dict["parameters"]["detector_relative"]["energy_scale_factor"][1])
    e_scale_unc = e_scale_nom * Float64(dict["parameters"]["detector_relative"]["energy_scale_factor"][2] * 0.01)  # percent

    scale = hcat([[eff_unc^2, eff_unc * e_scale_unc], [eff_unc * e_scale_unc, e_scale_unc^2]]...)
    corr_mat = hcat(dict["correlations"]["detector_relative"]["matrix"]...)
    # correlation(i, j) = covariance(i, j) / sqrt(var_i * var_j)) -> invert to get covariance matrix for MvNormal
    cov_mat = corr_mat .* scale
    eff_eres_nom = Vector([eff_nom, e_scale_nom])
    eff_eres_AD11 = Distributions.MvNormal(eff_eres_nom, cov_mat)
    eff_eres_AD12 = Distributions.MvNormal(eff_eres_nom, cov_mat)
    eff_eres_AD21 = Distributions.MvNormal(eff_eres_nom, cov_mat)
    eff_eres_AD22 = Distributions.MvNormal(eff_eres_nom, cov_mat)
    eff_eres_AD31 = Distributions.MvNormal(eff_eres_nom, cov_mat)
    eff_eres_AD32 = Distributions.MvNormal(eff_eres_nom, cov_mat)
    eff_eres_AD33 = Distributions.MvNormal(eff_eres_nom, cov_mat)
    eff_eres_AD34 = Distributions.MvNormal(eff_eres_nom, cov_mat)


    ## energy resolution, shared across all ADs
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/detector_eres.yaml"))
    a_nom = dict["parameters"]["eres"]["a_nonuniform"][1]
    a_unc = a_nom * dict["parameters"]["eres"]["a_nonuniform"][2] * 0.01   # percent

    b_nom = dict["parameters"]["eres"]["b_stat"][1]
    b_unc = b_nom * dict["parameters"]["eres"]["b_stat"][2] * 0.01   # percent

    c_nom = dict["parameters"]["eres"]["c_noise"][1]
    c_unc = c_nom * dict["parameters"]["eres"]["c_noise"][2] * 0.01   # percent

    eres_a = Distributions.Normal(a_nom, a_unc)
    eres_b = Distributions.Normal(b_nom, b_unc)
    eres_c = Distributions.Normal(c_nom, c_unc)


    ## iav off diagonal scaling, one for each AD, constant over periods
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/detector_iav_offdiag_scale.yaml"))
    iav_scale_nom = dict["parameters"]["iav_offdiag_scale_factor"][1]
    iav_scale_unc = dict["parameters"]["iav_offdiag_scale_factor"][2]
    len = length(full_setup)
    iav_offdiag_scale = Distributions.MvNormal(iav_scale_nom .* ones(len), iav_scale_unc^2 .* Diagonal(ones(len)))

    ## lsnl correction, pull parameters
    dict = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/detector_lsnl.yaml"))
    lsnl_scale_nom = dict["parameters"]["lsnl_scale_a"][1]
    lsnl_scale_unc = dict["parameters"]["lsnl_scale_a"][2]
    lsnl_pull = Distributions.MvNormal(lsnl_scale_nom .* ones(4), lsnl_scale_unc^2 .* ones(4))


    priors = (
        amc_unc_scale=amc_unc_scale,  # done
        acc_scale=acc_scale,   # done
        lihe_unc_scale=lihe_unc_scale,  # done
        fast_n_unc_scale=fast_n_unc_scale,   # done
        alpha_n_rate=alpha_n_rate,   # done
        eres_a=eres_a,  # done
        eres_b=eres_b,  # done
        eres_c=eres_c,  # done
        eff_eres_AD11=eff_eres_AD11,
        eff_eres_AD12=eff_eres_AD12,
        eff_eres_AD21=eff_eres_AD21,
        eff_eres_AD22=eff_eres_AD22,
        eff_eres_AD31=eff_eres_AD31,
        eff_eres_AD32=eff_eres_AD32,
        eff_eres_AD33=eff_eres_AD33,
        eff_eres_AD34=eff_eres_AD34,
        lsnl_pull,
        iav_offdiag_scale=iav_offdiag_scale,   # done
    )
end

function get_assets(datadir = @__DIR__)
    @info "Loading DayaBay data"
    period_list = ["6AD", "8AD", "7AD"]

    EH_list = Int64[1, 2, 3]

    full_setup = String["AD11", "AD12", "AD21", "AD22", "AD31", "AD32", "AD33", "AD34"]
    mask_6 =     Bool[true,   true,   true,   false,  true,   true,   true,   false]
    mask_8 =     Bool[true,   true,   true,   true,   true,   true,   true,   true]
    mask_7 =     Bool[false,  true,   true,   true,   true,   true,   true,   true]

    baselines = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/baselines.yaml"))
    reactors = String["R$(i)" for i in 1:6]

    distances = OrderedDict()
    distances["AD"] = full_setup
    for reac in reactors
        distances[reac] = [baselines["parameters"]["baseline"][ad][reac] for ad in full_setup]
    end

    distances["6AD"] = mask_6
    distances["8AD"] = mask_8
    distances["7AD"] = mask_7

    df_exp = DataFrame(distances)


    detectors_6AD = full_setup[mask_6]
    detectors_8AD = full_setup[mask_8]
    detectors_7AD = full_setup[mask_7]

    detector_dict = OrderedDict("6AD"=>detectors_6AD, "8AD"=>detectors_8AD, "7AD"=>detectors_7AD)

    ## global efficiency
    efficiency = Float64(YAML.load_file(joinpath(datadir, "dayabay_data/parameters/detector_efficiency.yaml"))["parameters"]["efficiency"])

    # detector_list = vcat(full_setup[mask_6], full_setup[mask_8], full_setup[mask_7])

    period_EH_list = String[]
    for p in period_list
        for EH in EH_list
            push!(period_EH_list, "$(p)_$(EH)")
        end
    end

    detector_list = String[]
    for p in period_list
        for AD in detector_dict[p]
            push!(detector_list, "$(p)$(AD)")
        end
    end

    observed = Vector{Vector{Int64}}(undef, 0)
    for p_AD in detector_list
        AD = retrieve_AD(p_AD)
        period = retrieve_period(p_AD)
        push!(observed, get_observed_counts(AD, period))
        # break
    end
    
    observed = vcat(observed...)
    #observed = observed[1]


    ## binning

    data_basepath = "dayabay_data/parameters"
    coarse_binning = readdlm(joinpath(datadir, data_basepath, "final_erec_bin_edges.tsv"))[2:end];
    coarse_binning_c = (coarse_binning[2:end] + coarse_binning[1:end-1]) / 2
    coarse_bin_width = diff(coarse_binning)
    fine_binning_Edep = collect(LinRange(0, 12, 241))
    fine_binning_Enu = fine_binning_Edep .+ 0.782 # approx
    fine_binning_Edep_c = (fine_binning_Edep[1:end-1] + fine_binning_Edep[2:end]) / 2
    fine_binning_Enu_c = (fine_binning_Enu[2:end] + fine_binning_Enu[1:end-1]) ./ 2
    fine_bin_Edep_width = diff(fine_binning_Edep)



    ###read in weekly predicted neutrino counts
    # some reactors may be offline and hence the IBD rate will be reduced for some ADs in some time window
    # take the daily predicted neutrino rate and convert by nominal energy per fission and nominal fission fractions to an averaged thermal power
    # this averaged thermal power is then per reactor and per AD and will be used instead of the nominal thermal power
    # but is globally scaled by the reactor_thermal_power_scale per reactor

    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_thermal_power_nominal.yaml"))
    nominal_thermal_power = file["parameters"]["nominal_thermal_power"]   # GW

    file = h5open(joinpath(datadir, "dayabay_data/neutrino_rate.hdf5"))
    reactor_nu_rates = Dict()
    for r in 1:6
        rates = Float64[]
        n_AD = Int64[]
        foreach(x -> (foreach(y -> (push!(rates, x[:neutrino_rate_per_s]), push!(n_AD, Int(x[:n_det]))), 1:x[:n_days])), file["R$(r)"][1:end])
        reactor_nu_rates["R$(r)"] = (;rates, n_AD)
    end
    close(file)

    reactor_config = Newtrinos.reactor_flux.configure()
    energy_per_fission = reactor_config.params.energy_per_fission
    fractions = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_fission_fractions.yaml"), dicttype=OrderedDict{String,Any})["parameters"]["fission_fractions"]
    fractions = NamedTuple((Symbol(key),value) for (key,value) in fractions)
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/antineutrinos_per_fission_huber_mueller.yaml"))

    nu_per_f = file["parameters"]["antineutrinos_per_fission"]
    nu_per_fission = NamedTuple((Symbol(key), value) for (key, value) in nu_per_f)
    # P_th_av = # nu per second * (sum_iso fission_fraction * energy per fission) / (sum_iso fission_fraction * nu per fission)
    # also convert from MeV to GJ
    elementary_charge = 1.602176634e-19

        #           Giga   Joule -> eV        Mega
    GJ_to_MeV = 1e9 / elementary_charge * 1e-6 # reactor thermal power in GW, energy per fission in MeV


    nom_conversion_factor = sum([energy_per_fission[i] * fractions[iso] for (i, iso) in enumerate(keys(fractions))]) / sum([fractions[iso] * nu_per_fission[iso] for (i, iso) in enumerate(keys(fractions))]) / GJ_to_MeV

    

    assets = (;
        period_list,
        period_EH_list,
        EH_list,
        df_exp,
        detector_dict,
        detectors_6AD,
        detectors_8AD,
        detectors_7AD,
        detector_list,
        coarse_binning,
        coarse_binning_c,
        coarse_bin_width,
        fine_binning_Enu,
        fine_binning_Enu_c,
        fine_binning_Edep,
        fine_binning_Edep_c,
        fine_bin_Edep_width,
        efficiency,
        reactor_nu_rates,
        nom_conversion_factor,
        nominal_thermal_power,
        observed,
    )
end


"""
# claude-generated eres, possibly faster than hand-written one from juno module?
function energy_resolution(E, E_edges, eres_a, eres_b, eres_c; nσ = 6)
    N  = length(E)
    ne = length(E_edges)                 # ne == N + 1 for a square smearing matrix
    T  = promote_type(eltype(E), eltype(eres_a))
    c1 = T(1) / sqrt(T(2))

    Φ = Vector{T}(undef, ne)             # edge CDFs, reused per column
    I = Int[]; J = Int[]; V = T[]
    sizehint!(V, N * 4 * nσ); sizehint!(I, N * 4 * nσ); sizehint!(J, N * 4 * nσ)

    @inbounds for j in 1:N
        Ej   = E[j]
        σ    = sqrt(Ej^2 * eres_a^2 + eres_b^2 * Ej + eres_c^2)
        invσ = 1 / σ

        # edge window within ±nσ of Ej, clamped to the full range.
        # search on values only — the window boundary is a step (zero-gradient)
        # device; the tail mass it drops is ~1e-9, well below fp gradient noise.
        lo  = Ej - nσ * σ
        hi  = Ej + nσ * σ
        klo = clamp(searchsortedlast(E_edges, lo),  1,       ne - 1)
        khi = clamp(searchsortedfirst(E_edges, hi), klo + 1, ne)

        for k in klo:khi
            z = (E_edges[k] - Ej) * invσ
            Φ[k] = T(0.5) * erfc(-z * c1)            # = Φ(z), one erfc per edge
        end

        invZ = 1 / (Φ[khi] - Φ[klo])                 # truncate to the window
        for c in klo:(khi - 1)
            push!(I, c); push!(J, j); push!(V, (Φ[c+1] - Φ[c]) * invZ)
        end
    end

    return sparse(I, J, V, N, N)
end
"""


function get_forward_model(physics, assets; datadir = @__DIR__, debug = false)
    df_exp = assets.df_exp
    detector_list = assets.detector_list

    xsec = physics.xsec.xsec
    n_protons = get_proton_number(datadir)

    iav_func = get_iav_matrix(datadir)
    lsnl = get_lsnl_correction(datadir)

    reactor_flux = physics.flux.flux

    osc = physics.osc
    xsec = physics.xsec.xsec
    
    coarse_binning = assets.coarse_binning
    coarse_binning_c = assets.coarse_binning_c
    coarse_bin_width = assets.coarse_bin_width
    #fine_binning_Enu = assets.fine_binning_Enu
    #fine_binning_Enu_c = assets.fine_binning_Enu_c
    fine_binning_Edep = assets.fine_binning_Edep
    fine_binning_Edep_c = assets.fine_binning_Edep_c
    fine_bin_Edep_width = assets.fine_bin_Edep_width
    fine_binning_Enu = physics.xsec.Enu.(fine_binning_Edep)
    fine_binning_Enu_c = physics.xsec.Enu.(fine_binning_Edep_c)
    fine_bin_Enu_width = diff(fine_binning_Enu)
    
    xsec_eval = xsec.(fine_binning_Enu_c)

    E_nu_GeV = fine_binning_Enu_c ./ 1e3

    NF = length(fine_binning_Enu) - 1
    NFc = length(fine_binning_Edep_c)

    ## create background model functions, taking parameter NamedTuple as arg
    background_models = OrderedDict()

    ## create anti-neutrino forward model
    neutrino_models = OrderedDict()

    for (idx, p_AD) in enumerate(detector_list)
        period = retrieve_period(p_AD)
        AD = retrieve_AD(p_AD)
        EH = retrieve_EH(p_AD)
        output = extract_for_AD_period(AD, period)
        bg_dict = output.bg_dict

        ad_idx = findfirst(df_exp[!, "AD"] .== "AD$(AD)")
        L = collect(df_exp[ad_idx, [:R1, :R2, :R3, :R4, :R5, :R6]])
        L_km = L / 1e3
        L2 = 4 * pi .* L.^2;
        n_p = n_protons["AD$(AD)"]

        lt = output.eff_livetime_days
        accidentals = bg_dict["accidentals"]
        
        ffs_symbols = [Symbol("fission_fractions_scale_R$(i)") for i=1:6]
        neq_symbols = [Symbol("neq_scale_R$(i)") for i=1:6]

        amc = bg_dict["amc"]
        lihe = bg_dict["lithium_helium"]
        fast_n = bg_dict["fast_neutrons"]
        alpha_n = bg_dict["alpha_neutron"]
        period_EH_idx = findfirst(x-> x == "$(period)AD_$(EH)", assets.period_EH_list)
        @assert period_EH_idx !== nothing

        acc_base = lt * accidentals.rate .* accidentals.shape
        amc_base = lt * amc.rate .* amc.shape
        lihe_base = lt * lihe.rate .* lihe.shape
        fast_n_base = lt * fast_n.rate .* fast_n.shape
        alpha_n_base = lt .* alpha_n.shape

        function background_counts(params)
            ## background stuff
            # part of forward model
            _acc = params.acc_scale[idx]
            _amc = 1.0 + amc.uncertainty * params.amc_unc_scale
            _lihe = 1.0 + lihe.uncertainty * params.lihe_unc_scale[period_EH_idx]
            _fast_n = 1.0 + fast_n.uncertainty * params.fast_n_unc_scale[period_EH_idx]
            _alpha_n = params.alpha_n_rate[idx]

            return @. acc_base * _acc + amc_base * _amc + lihe_base * _lihe + fast_n_base * _fast_n + alpha_n_base * _alpha_n
            #return @. alpha_n_base * _alpha_n

            #@return (acc = _acc .* acc_base, amc = _amc .* amc_base, lihe = _lihe .* lihe_base, fast_n = _fast_n .* fast_n_base, alpha_n = _alpha_n .* alpha_n)
            #return @. acc_base * _acc
        end
        background_models[p_AD] = background_counts
  

        

        eff_eres_key = Symbol("eff_eres_AD$(AD)")
        invL2 = inv.(L2)
        # convert the cross section from e-41cm2 to m2 -> multiply by 1e-4
        flux_norm = 1e-45 * output.eff_livetime_seconds * n_p

        integral_dEnu = fine_bin_Enu_width .* xsec_eval

        # get the exposure-averaged per-AD-reactor thermal power
        rates = assets.reactor_nu_rates
        av_thermal_power_scale = zeros(6)

        for r in 1:6
            nu_per_s = rates["R$(r)"].rates
            n_AD = rates["R$(r)"].n_AD

            # check that we are in the correct period
            @assert all(n_AD[output.day .+ 1] .== period)

            exposure_weighted_nu_per_s = sum(nu_per_s[output.day .+ 1] .* output.eff_livetime) ./ sum(output.eff_livetime)
            av_thermal_power_GW = exposure_weighted_nu_per_s * assets.nom_conversion_factor
            av_thermal_power_scale[r] = av_thermal_power_GW / assets.nominal_thermal_power
        end

        # println(av_thermal_power_scale)

        spec_prefac_nom_neq = av_thermal_power_scale .* invL2 .* flux_norm
        spec_prefac_snf = flux_norm .* invL2
        function neutrino_counts(params)
            # value type promoted over all parameter inputs, so intermediate
            # arrays stay dual even when only a subset of the parameters is dual
            T = promote_type(
                eltype(params.eres_a),
                eltype(params.eres_b),
                eltype(params.eres_c),
                eltype(params.lsnl_pull),
                eltype(params.iav_offdiag_scale),
                eltype(params.reactor_thermal_power_scale),
                eltype(params.energy_per_fission),
                eltype(params.fission_fractions_scale_R1),
                eltype(params.fission_fractions_scale_R2),
                eltype(params.fission_fractions_scale_R3),
                eltype(params.fission_fractions_scale_R4),
                eltype(params.fission_fractions_scale_R5),
                eltype(params.fission_fractions_scale_R6),
                eltype(params.neq_scale_R1),
                eltype(params.neq_scale_R2),
                eltype(params.neq_scale_R3),
                eltype(params.neq_scale_R4),
                eltype(params.neq_scale_R5),
                eltype(params.neq_scale_R6),
                eltype(params.snf_scale),
                eltype(params.spectrum_pulls),
                eltype(params.acc_scale),
                eltype(params.alpha_n_rate),
                eltype(params.fast_n_unc_scale),
                eltype(params.lihe_unc_scale),
                eltype(params.amc_unc_scale),
                eltype(params.eff_eres_AD11),
                eltype(params.eff_eres_AD12),
                eltype(params.eff_eres_AD21),
                eltype(params.eff_eres_AD22),
                eltype(params.eff_eres_AD31),
                eltype(params.eff_eres_AD32),
                eltype(params.eff_eres_AD33),
                eltype(params.eff_eres_AD34),
                eltype(params.energy_per_fission),
                eltype(params.θ₁₃),
                eltype(params.θ₁₂),
                eltype(params.θ₂₃),
                eltype(params.Δm²₂₁),
                eltype(params.Δm²₃₁),
                eltype(params.δCP),
            )
            epf = params.energy_per_fission
            pulls = params.spectrum_pulls

            P = osc.osc_prob(E_nu_GeV, L_km, params, anti=true)[:, :, 1, 1]

            spec = zeros(T, NF)
            @inbounds for r in 1:6
                fission_fractions = params[ffs_symbols[r]]
                neq_scale = params[neq_symbols[r]]
                flux_nom, flux_snf, correction = reactor_flux(fine_binning_Enu_c, params.reactor_thermal_power_scale[r], params.energy_per_fission, fission_fractions, params.spectrum_pulls, neq_scale, params.snf_scale[r], r)
                @. spec += (spec_prefac_nom_neq[r] * flux_nom + spec_prefac_snf[r] * flux_snf) * P[:, r] * correction
                #spec .+= reactor_flux(fine_binning_Enu_c, params.reactor_thermal_power_scale[r], params.energy_per_fission, fission_fractions, params.spectrum_pulls, neq_scale, params.snf_scale[r], r) .* invL2[r]
            end
            # multiply with xsec_eval and bin width for integral over neutrino energy
            spec .*= integral_dEnu

            spec_sum = sum(spec)
            smeared_spectrum = iav_func(params.iav_offdiag_scale[ad_idx], spec)

            # transform from Escint to Evis by lsnl and relative energy scale
            E_vis = lsnl(fine_binning_Edep_c, params.lsnl_pull) .* fine_binning_Edep_c
            E_vis_edges = lsnl(fine_binning_Edep, params.lsnl_pull) .* fine_binning_Edep

            # get energy resolution 
            sigma_E = eres(E_vis, params.eres_a, params.eres_b, params.eres_c);
            
            # get relative efficiency and energy scale
            eff_eres = params[eff_eres_key]


            #smeared_sum = sum(smeared_spectrum)
            resolved_spectrum = smear(E_vis, smeared_spectrum, sigma_E, E_scale=eff_eres[2], width=20);
            resolved_sum = sum(resolved_spectrum)

            """
            if idx == 1
                println(smeared_sum)
                println(resolved_sum)
                println(smeared_sum / resolved_sum)
            end
            """
            # Divide by bin width to get approximate pdf for integration over arbitrary bins
            spectrum_pdf = resolved_spectrum ./ diff(E_vis_edges)

            # exact integration of the piecewise-constant pdf over the coarse bins
            NC = length(coarse_binning) - 1
            coarse_counts = zeros(T, NC)
            # loop over coarse bins
            @inbounds for (j, (l, h)) in enumerate(zip(coarse_binning[1:end-1], coarse_binning[2:end]))
                s = zero(T)
                jlo = clamp(searchsortedlast(E_vis_edges, l), 1, NFc)
                jhi = clamp(searchsortedfirst(E_vis_edges, h) - 1, jlo, NFc)
                @inbounds for j2 in jlo:jhi
                    ov = min(h, E_vis_edges[j2 + 1]) - max(l, E_vis_edges[j2])
                    if ov > zero(T)
                        s += spectrum_pdf[j2] * ov
                    end
                end
                coarse_counts[j] = s
            end
           
            eff_eres[1] .* coarse_counts .* assets.efficiency
        end

        neutrino_models[p_AD] = neutrino_counts
        # break
    
    end

    function forward_model(params)
        output = []
        for (c, p_AD) in enumerate(detector_list)
            #push!(output, background_models[p_AD](params) .+ neutrino_models[p_AD](params))
            push!(output, neutrino_models[p_AD](params))
            #push!(output, background_models[p_AD](params))
            # break
        end
        #distprod(Poisson.(background_models[detector_list[1]](params)))
        expected = vcat(output...)
        distprod(Poisson.(expected))
    end

    if !debug
        return forward_model
    else
        return (neutrino_models, background_models)
    end

end

  
 function get_plot(physics, assets)

    function plot(params, data=assets.observed)

        m = mean(get_forward_model(physics, assets)(params))
        v = var(get_forward_model(physics, assets)(params))
        
        detector_list = assets.detector_list
        # number of analysis bins
        n_ana_binning = length(assets.coarse_binning_c)

        EH1_obs = zeros(n_ana_binning)
        EH2_obs = zeros(n_ana_binning)
        EH3_obs = zeros(n_ana_binning)

        EH1_mean = zeros(n_ana_binning)
        EH2_mean = zeros(n_ana_binning)
        EH3_mean = zeros(n_ana_binning)

        EH1_var = zeros(n_ana_binning)
        EH2_var = zeros(n_ana_binning)
        EH3_var = zeros(n_ana_binning)

        _mean = [EH1_mean, EH2_mean, EH3_mean]
        _var = [EH1_var, EH2_var, EH3_var]
        obs = [EH1_obs, EH2_obs, EH3_obs]


        for (c, p_AD) in enumerate(detector_list)
            EH = Newtrinos.dayabay_rewrite.retrieve_EH(p_AD)
            if EH == 1
                EH1_obs .+= data[(c-1) * n_ana_binning + 1:c*n_ana_binning]
                EH1_mean .+= m[(c-1) * n_ana_binning + 1:c*n_ana_binning]
                EH1_var .+= v[(c-1) * n_ana_binning + 1:c*n_ana_binning]
            elseif EH == 2
                EH2_obs .+= data[(c-1) * n_ana_binning + 1:c*n_ana_binning]
                EH2_mean .+= m[(c-1) * n_ana_binning + 1:c*n_ana_binning]
                EH2_var .+= v[(c-1) * n_ana_binning + 1:c*n_ana_binning]
            elseif EH ==3
                EH3_obs .+= data[(c-1) * n_ana_binning + 1:c*n_ana_binning]
                EH3_mean .+= m[(c-1) * n_ana_binning + 1:c*n_ana_binning]
                EH3_var .+= v[(c-1) * n_ana_binning + 1:c*n_ana_binning]
            end

        end
    
        for (c, (m, v, o)) in enumerate(zip(_mean, _var, obs))
            f = Figure()
            ax = Axis(f[1, 1], yminorticksvisible=true, title="EH$(c)")#, yminorticks=IntervalsBetween(9), yscale=log10)
            plot!(ax, assets.coarse_binning_c, o./assets.coarse_bin_width, label="Observed", color=:black)
            stephist!(ax, assets.coarse_binning_c, weights=m./assets.coarse_bin_width, bins=assets.coarse_binning, label="Expected")
            #barplot!(ax, assets.coarse_binning_c, (m .+ sqrt.(v))./assets.coarse_bin_width, width=assets.coarse_bin_width, gap=0, fillto= (m.- sqrt.(v))./assets.coarse_bin_width, alpha=0.5, label="Standard Deviation")
            axislegend(ax, framevisible = false)

            ax.xticksvisible = false
            ax.xticklabelsvisible = false
            ax2 = Axis(f[2, 1])

            ax.ylabel="Counts/MeV"

            plot!(ax2, assets.coarse_binning_c, o ./ m, color=:black, label="Observed")
            hlines!(ax2, 1, label="Expected")
            barplot!(ax2, assets.coarse_binning_c, 1 .+ sqrt.(v) ./ m, width=assets.coarse_bin_width, gap=0, fillto= 1 .- sqrt.(v)./m, alpha=0.5, label="Standard Deviation")


            rowgap!(f.layout, 1, 0)
            rowsize!(f.layout, 1, Relative(3/4))

            ax2.xlabel="Eₚ (MeV)"
            ax2.ylabel="Counts/Expected"

            xlims!(ax2, minimum(assets.coarse_binning), maximum(assets.coarse_binning))
            xlims!(ax, minimum(assets.coarse_binning), maximum(assets.coarse_binning))
            ylims!(ax, 1e1, 1e6)
            ## ylims!(ax2, 0.3, 1.0)
            

            save("EH_$(c).png", f)
            
        end
    end
end

end
