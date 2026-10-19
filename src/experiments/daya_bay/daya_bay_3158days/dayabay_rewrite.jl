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
    xsec = Newtrinos.ibd_xsec.configure(Newtrinos.ibd_xsec.VogelBeacomO1())
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

# (the data-release files are read only in configure(), so the package loads without dayabay_data/)
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
`(f)(scale, spec)` applies it to a spectrum from the l.h.s. without materializing the matrix.
As in the official model (`RenormalizeDiag`, mode "offdiag"), the off-diagonal elements are scaled and each column is
then renormalised to unit sum, so the IAV uncertainty changes the shape but not the number of events.
"""
struct IAVApplier
    iav::Matrix{Float64}
    offdiag::Matrix{Float64}
    diag::Vector{Float64}
    offdiag_colsum::Vector{Float64}
end

# column sums of the scaled matrix (empty columns are left as they are)
_iav_colsum(f::IAVApplier, offdiag_scale) = @. ifelse(f.diag + f.offdiag_colsum == 0, one(offdiag_scale), f.diag + offdiag_scale * f.offdiag_colsum)

function (f::IAVApplier)(offdiag_scale)
    m = offdiag_scale * f.offdiag
    @inbounds for i in 1:length(f.diag)
        m[i, i] += f.diag[i]
    end
    m ./ transpose(_iav_colsum(f, offdiag_scale))
end

function (f::IAVApplier)(offdiag_scale, spec)
    s = spec ./ _iav_colsum(f, offdiag_scale)
    f.diag .* s .+ offdiag_scale * (f.offdiag * s)
end


function get_iav_matrix(datadir = @__DIR__)
    file = h5open(joinpath(datadir, "dayabay_data/detector_iav_matrix.hdf5"))

    iav = transpose(file["iav_matrix"][])
    close(file)

    offdiag = iav .- Diagonal(iav)
    IAVApplier(iav, offdiag, diag(iav), vec(sum(offdiag, dims = 1)))
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

"""
    not_a_knot_spline(x, y, xnew)

Cubic spline through `(x, y)` with not-a-knot end conditions (as scipy's `interp1d(kind="cubic")`), evaluated at `xnew`
(within the range of `x`).
"""
function not_a_knot_spline(x::Vector{Float64}, y::Vector{Float64}, xnew::AbstractVector{Float64})
    n = length(x)
    h = diff(x)
    # second derivatives M: interior continuity of the first derivative, not-a-knot at both ends
    A = zeros(n, n)
    b = zeros(n)
    A[1, 1], A[1, 2], A[1, 3] = h[2], -(h[1] + h[2]), h[1]
    for i in 2:n-1
        A[i, i-1], A[i, i], A[i, i+1] = h[i-1], 2 * (h[i-1] + h[i]), h[i]
        b[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    end
    A[n, n-2], A[n, n-1], A[n, n] = h[n-1], -(h[n-2] + h[n-1]), h[n-2]
    M = A \ b
    map(xnew) do t
        k = clamp(searchsortedlast(x, t), 1, n - 1)
        a, c = x[k+1] - t, t - x[k]
        (M[k] * a^3 + M[k+1] * c^3) / (6h[k]) + (y[k] / h[k] - M[k] * h[k] / 6) * a + (y[k+1] / h[k] - M[k+1] * h[k] / 6) * c
    end
end

"""
    refine_lsnl(E, f; refine_times = 4, newmin = 0.5, newmax = 12.1)

LSNL curve processing of the official model (`refine_lsnl_data`): the relative curve f = Evis/Escint is converted to the
absolute Evis(Escint), interpolated with a cubic spline on a `refine_times` finer grid and extrapolated linearly (with
the outermost fine segments) to [`newmin`, `newmax`]. Returns the fine Escint grid and Evis on it.
"""
function refine_lsnl(E::Vector{Float64}, f::Vector{Float64}; refine_times = 4, newmin = 0.5, newmax = 12.1)
    xfine = collect(range(E[1], E[end], length = (length(E) - 1) * refine_times + 1))
    yfine = not_a_knot_spline(E, f .* E, xfine)
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
    get_lsnl_correction(datadir)

LSNL as in the official model: the nominal and the four nuisance curves are refined and extrapolated with
`refine_lsnl`; returns `(E, pull) -> Evis`, the absolute visible energy for deposited (scintillator) energies `E`,
Evis = nominal(E) + Σᵢ pullᵢ (curveᵢ(E) - nominal(E)). The curves are linear between the refined points and are
defined on [0.5, 12.1] MeV (events below are dropped in the forward model, as in the official model).
"""
function get_lsnl_correction(datadir = @__DIR__)
    lsnl = read_lsnl_correction(datadir)
    x, nom = refine_lsnl(lsnl.E, lsnl.f_nom)
    curves = [refine_lsnl(lsnl.E, lsnl.f_nom .+ rel)[2] .- nom for rel in (lsnl.rel_0, lsnl.rel_1, lsnl.rel_2, lsnl.rel_3)]
    nom_itp = linear_interpolation(x, nom, extrapolation_bc = Line())
    pull_itps = [linear_interpolation(x, c, extrapolation_bc = Line()) for c in curves]

    function _lsnl(E, pull)
        return @. nom_itp(E) + pull[1] * pull_itps[1](E) + pull[2] * pull_itps[2](E) + pull[3] * pull_itps[3](E) + pull[4] * pull_itps[4](E)
    end
end

const lsnl_min_energy = 0.5    # MeV, lower end of the refined LSNL curves


function eres(E, a, b, c)
    sigma_E = @. sqrt(a^2 * E^2 + b^2 * E + c^2)
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
    e_scale_nom =  Float64(dict["parameters"]["detector_relative"]["energy_scale_factor"][1])
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
        iav_offdiag_scale,
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
    end
    
    observed = vcat(observed...)


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
    # also convert from MeV to GJ
    elementary_charge = 1.602176634e-19

        #           Giga   Joule -> eV        Mega
    GJ_to_MeV = 1e9 / elementary_charge * 1e-6 # reactor thermal power in GW, energy per fission in MeV

    #multipliy: rate -> thermal power, divide: thermal power -> rate
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
    fine_binning_Edep = assets.fine_binning_Edep
    fine_binning_Edep_c = assets.fine_binning_Edep_c

    # IBD integration as in the official model: in each deposited energy bin a two-fold Gauss-Legendre quadrature
    # over Edep (order 5) and the positron angle cosθ ∈ [-1, 1] (order 3), with exact kinematics Eν(Edep, cosθ),
    # the first order cross section dσ/dcosθ and the Jacobian dEν/dEdep
    gl5 = ([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
           [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891])
    gl3 = ([-0.7745966692414834, 0.0, 0.7745966692414834], [0.5555555555555556, 0.8888888888888888, 0.5555555555555556])
    NQ = length(gl5[1]) * length(gl3[1])
    NFc = length(fine_binning_Edep_c)
    E_nu_mesh = zeros(NQ, NFc)
    integral_weights = zeros(NQ, NFc)
    for i in 1:NFc
        lo, hi = fine_binning_Edep[i], fine_binning_Edep[i+1]
        q = 0
        for (ξ, wE) in zip(gl5...), (c, wc) in zip(gl3...)
            q += 1
            Edep = (lo + hi) / 2 + (hi - lo) / 2 * ξ
            Enu = physics.xsec.Enu(Edep, c)
            E_nu_mesh[q, i] = Enu
            integral_weights[q, i] = (hi - lo) / 2 * wE * wc * xsec(Enu, c) * physics.xsec.jacobian(Enu, Edep, c)
        end
    end
    E_nu_mesh = vec(E_nu_mesh)
    integral_weights = vec(integral_weights)
    E_nu_GeV = E_nu_mesh ./ 1e3
    NF = length(E_nu_mesh)
    lsnl_mask = fine_binning_Edep[1:end-1] .>= lsnl_min_energy - 1e-9

    # the reactor fluxes on the integration mesh depend only on the reactor parameters: computed once per evaluation
    # and shared by all AD-periods; (main × correction + neq, snf) per reactor
    mesh_flux = physics.flux.flux_on_mesh(E_nu_mesh)
    ffs_symbols = [Symbol("fission_fractions_scale_R$(i)") for i=1:6]
    neq_symbols = [Symbol("neq_scale_R$(i)") for i=1:6]
    function reactor_fluxes(params)
        map(1:6) do r
            main, neq, snf, correction = mesh_flux(params.reactor_thermal_power_scale[r], params.energy_per_fission,
                params[ffs_symbols[r]], params.spectrum_pulls, params[neq_symbols[r]], params.snf_scale[r], r)
            (main .* correction .+ neq, snf)
        end
    end
    # survival probabilities per AD (shared by its periods)
    baselines_km = Dict(ad => collect(df_exp[findfirst(df_exp[!, "AD"] .== ad), [:R1, :R2, :R3, :R4, :R5, :R6]]) ./ 1e3 for ad in unique(df_exp[!, "AD"]))
    # for standard three-flavour vacuum oscillations the closed form of P(ν̄e → ν̄e) is used (identical to osc_prob,
    # but much cheaper with dual numbers); any other oscillation model goes through osc_prob
    oc = osc.cfg
    standard_osc = oc.flavour isa Newtrinos.osc.ThreeFlavour && oc.interaction isa Newtrinos.osc.Vacuum &&
                   oc.propagation isa Newtrinos.osc.Basic && oc.states isa Newtrinos.osc.All
    function survival_probabilities(params)
        standard_osc || return Dict(ad => osc.osc_prob(E_nu_GeV, L, params, anti=true)[:, :, 1, 1] for (ad, L) in baselines_km)
        Dict(ad => nue_survival(E_nu_GeV, L, params) for (ad, L) in baselines_km)
    end
    final_rebin_idx = rebin_index(fine_binning_Edep, coarse_binning)

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
        L_km = L ./ 1e3
        L2 = 4 * pi .* L.^2;
        n_p = n_protons["AD$(AD)"]

        lt = output.eff_livetime_days
        
        accidentals = bg_dict["accidentals"]
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
            # get parameter dependencies and multiply by parameter-indpendent part `_base`
            _acc = params.acc_scale[idx]
            _amc = 1.0 + amc.uncertainty * params.amc_unc_scale
            _lihe = 1.0 + lihe.uncertainty * params.lihe_unc_scale[period_EH_idx]
            _fast_n = 1.0 + fast_n.uncertainty * params.fast_n_unc_scale[period_EH_idx]
            _alpha_n = params.alpha_n_rate[idx]

            return @. acc_base * _acc + amc_base * _amc + lihe_base * _lihe + fast_n_base * _fast_n + alpha_n_base * _alpha_n
        end
        # debug/fancy plotting option to return the dictionary of all background models
        if !debug
            background_models[p_AD] = background_counts
        else
            background_models[p_AD] = (
                alpha_n=alpha_n_base,
                amc=amc_base,
                fast_n=fast_n_base,
                lihe=lihe_base,
                acc=acc_base
            )
        end

        # get the exposure-averaged per-AD-reactor thermal power
        rates = assets.reactor_nu_rates
        P_th_seconds = zeros(6)
        for r in 1:6
            nu_per_s = rates["R$(r)"].rates
            n_AD = rates["R$(r)"].n_AD

            # check that we are in the correct period
            @assert all(n_AD[output.day .+ 1] .== period)

            #convert rates with nominal parameters to daily thermal power
            P_th_daily = nu_per_s .* assets.nom_conversion_factor
            P_th_seconds[r] = sum(P_th_daily[output.day .+ 1] .* output.eff_livetime)
        end

        eff_eres_key = Symbol("eff_eres_AD$(AD)")
        invL2 = inv.(L2)
        # convert the cross section from e-41cm2 to m2 -> multiply by 1e-4
        flux_norm = 1e-45 * n_p .* P_th_seconds .* invL2

        spec_prefac_snf = 1e-45 * n_p * sum(output.eff_livetime_seconds) .* invL2
        function neutrino_counts(params, fluxes, P)
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
            interm_spec = zeros(T, NF)
            @inbounds for r in 1:6
                # nominal flux and neq correction scale with sum(daily thermal power * daily eff livetime); the SNF is
                # independent of the actual thermal power (spent fuel) and scales with the eff livetime
                flux_r, snf_r = fluxes[r]
                @. interm_spec += (flux_norm[r] * flux_r + spec_prefac_snf[r] * snf_r) * P[:, r]
            end
            # integrate over each Edep bin: multiply with cross section × Jacobian × quadrature weights and sum
            interm_spec .*= integral_weights
            spec = vec(sum(reshape(interm_spec, NQ, NFc), dims = 1))
            smeared_spectrum = iav_func(params.iav_offdiag_scale[ad_idx], spec)

            eff_eres = params[eff_eres_key]
            # LSNL and the relative energy scale (eff_eres[2]) as one count-conserving distortion of the Escint bins onto
            # the Evis bins (same 50 keV binning); bins below the LSNL curve range are dropped (as in the official model)
            E_vis_edges = lsnl(fine_binning_Edep, params.lsnl_pull) .* eff_eres[2]
            evis_spectrum = project_to_bins(E_vis_edges, smeared_spectrum .* lsnl_mask, fine_binning_Edep)

            # energy resolution from the Evis bin centres onto the Erec bins (same binning), then rebin to the final bins
            sigma_E = eres(fine_binning_Edep_c, params.eres_a, params.eres_b, params.eres_c)
            erec_spectrum = smear_bin_centres(evis_spectrum, sigma_E, fine_binning_Edep)
            coarse_counts = rebin(erec_spectrum, final_rebin_idx, length(coarse_binning) - 1)            
            return eff_eres[1] .* coarse_counts .* assets.efficiency
        end

        neutrino_counts(params) = neutrino_counts(params, reactor_fluxes(params), survival_probabilities(params)["AD$(AD)"])
        neutrino_models[p_AD] = neutrino_counts    
    end
    if debug
        return (neutrino_models, background_models)
    end

    function forward_model(params)
        fluxes = reactor_fluxes(params)
        P = survival_probabilities(params)
        output = []
        for (c, p_AD) in enumerate(detector_list)
            push!(output, background_models[p_AD](params) .+ neutrino_models[p_AD](params, fluxes, P["AD$(retrieve_AD(p_AD))"]))
        end
        expected = vcat(output...)
        distprod(Poisson.(expected))
    end

    return forward_model

end

  
 function get_plot(physics, assets)

    function plot(params, data=assets.observed; saveto="test_output")

        _m = mean(get_forward_model(physics, assets)(params))
        _v = var(get_forward_model(physics, assets)(params))

        _signal, _bg = get_forward_model(physics, assets, debug=true)

        colors = [
            :green,
            :magenta,
            :grey,
            :yellow,
            :blue,
        ]

        bg_type = (
            :alpha_n,
            :amc,
            :fast_n,
            :lihe,
            :acc,
        )
        
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

        EH1_bg = (
            amc=zeros(n_ana_binning),
            alpha_n=zeros(n_ana_binning),
            acc=zeros(n_ana_binning),
            fast_n=zeros(n_ana_binning),
            lihe=zeros(n_ana_binning)
        )

        EH2_bg = (
            amc=zeros(n_ana_binning),
            alpha_n=zeros(n_ana_binning),
            acc=zeros(n_ana_binning),
            fast_n=zeros(n_ana_binning),
            lihe=zeros(n_ana_binning)
        )

        EH3_bg = (
            amc=zeros(n_ana_binning),
            alpha_n=zeros(n_ana_binning),
            acc=zeros(n_ana_binning),
            fast_n=zeros(n_ana_binning),
            lihe=zeros(n_ana_binning)
        )

        _mean = [EH1_mean, EH2_mean, EH3_mean]
        _var = [EH1_var, EH2_var, EH3_var]
        obs = [EH1_obs, EH2_obs, EH3_obs]
        
        
        for (c, p_AD) in enumerate(detector_list)
            period = retrieve_period(p_AD)
            AD = retrieve_AD(p_AD)
            EH = retrieve_EH(p_AD)
            bg_dict = extract_for_AD_period(AD, period).bg_dict
            accidentals = bg_dict["accidentals"]
            amc = bg_dict["amc"]
            lihe = bg_dict["lithium_helium"]
            fast_n = bg_dict["fast_neutrons"]
            alpha_n = bg_dict["alpha_neutron"]

            EH = Newtrinos.dayabay_rewrite.retrieve_EH(p_AD)
            if EH == 1
                EH1_obs .+= data[(c-1) * n_ana_binning + 1:c*n_ana_binning]
                EH1_mean .+= _m[(c-1) * n_ana_binning + 1:c*n_ana_binning]
                EH1_var .+= _v[(c-1) * n_ana_binning + 1:c*n_ana_binning]
            elseif EH == 2
                EH2_obs .+= data[(c-1) * n_ana_binning + 1:c*n_ana_binning]
                EH2_mean .+= _m[(c-1) * n_ana_binning + 1:c*n_ana_binning]
                EH2_var .+= _v[(c-1) * n_ana_binning + 1:c*n_ana_binning]
            elseif EH ==3
                EH3_obs .+= data[(c-1) * n_ana_binning + 1:c*n_ana_binning]
                EH3_mean .+= _m[(c-1) * n_ana_binning + 1:c*n_ana_binning]
                EH3_var .+= _v[(c-1) * n_ana_binning + 1:c*n_ana_binning]
            end


        #end
        #for (c, (m, v, o)) in enumerate(zip(_mean, _var, obs))
            idx = (c-1) * n_ana_binning + 1 : c * n_ana_binning
            o = data[idx]
            v = _v[idx]
            m = _m[idx]
            f = Figure()
            ax = Axis(f[1, 1], yminorticksvisible=true, title="$(period)AD, AD$(AD)", yminorticks=IntervalsBetween(9), yscale=log10)
            plot!(ax, assets.coarse_binning_c, o./assets.coarse_bin_width, label="Observed", color=:black)
            stephist!(ax, assets.coarse_binning_c, weights=m./assets.coarse_bin_width, bins=assets.coarse_binning, label="Expected")
            prev = zeros(length(assets.coarse_binning_c))


            #plot background
            period_EH_idx = findfirst(x-> x == "$(period)AD_$(EH)", assets.period_EH_list)
            _acc = params.acc_scale[c]
            _amc = 1.0 + amc.uncertainty * params.amc_unc_scale
            _lihe = 1.0 + lihe.uncertainty * params.lihe_unc_scale[period_EH_idx]
            _fast_n = 1.0 + fast_n.uncertainty * params.fast_n_unc_scale[period_EH_idx]
            _alpha_n = params.alpha_n_rate[c]
            
            scales = (acc=_acc, amc=_amc, lihe=_lihe, fast_n=_fast_n, alpha_n=_alpha_n)
            bg_model = _bg[p_AD]
            prev = ones(length(assets.coarse_binning_c)) * 1e-10
            for (c, key) in enumerate(bg_type)
                m_bg = scales[key] .* bg_model[key]
                if EH == 1
                    EH1_bg[key] .+= m_bg
                elseif EH == 2
                    EH2_bg[key] .+= m_bg
                elseif EH == 3
                    EH3_bg[key] .+= m_bg
                end
                barplot!(ax, assets.coarse_binning_c, (m_bg .+ prev)./assets.coarse_bin_width, width=assets.coarse_bin_width, label=string(key), color=colors[c], gap=0, fillto=prev./assets.coarse_bin_width)
                prev .+= m_bg
            end
            barplot!(ax, assets.coarse_binning_c, (m .+ sqrt.(v))./assets.coarse_bin_width, width=assets.coarse_bin_width, gap=0, fillto= (m.- sqrt.(v))./assets.coarse_bin_width, alpha=0.5, label="Standard Deviation")
            axislegend(ax, framevisible = false)

            ax.xticksvisible = false
            ax.xticklabelsvisible = false
            ax2 = Axis(f[2, 1])

            ax.ylabel="Counts/MeV"

            plot!(ax2, assets.coarse_binning_c, o ./ m .-1, color=:black, label="Observed")
            hlines!(ax2, 0, label="Expected")
            barplot!(ax2, assets.coarse_binning_c, sqrt.(v) ./ m, width=assets.coarse_bin_width, gap=0, fillto= .- sqrt.(v)./m, alpha=0.5, label="Standard Deviation")


            rowgap!(f.layout, 1, 0)
            rowsize!(f.layout, 1, Relative(3/4))

            ax2.xlabel="Eₚ (MeV)"
            ax2.ylabel="Counts/Expected"

            xlims!(ax2, minimum(assets.coarse_binning), maximum(assets.coarse_binning))
            xlims!(ax, minimum(assets.coarse_binning), maximum(assets.coarse_binning))
            ylims!(ax, 1e0, 1e4)
            ## ylims!(ax2, 0.3, 1.0)
            

            save(joinpath(saveto, "$(period)ADAD$(AD).png"), f)
            
        end

        # println(EH1_bg)
        for (c, (m, v, o, bg_d)) in enumerate(zip(_mean, _var, obs, [EH1_bg, EH2_bg, EH3_bg]))
            f = Figure()
            ax = Axis(f[1, 1], yminorticksvisible=true, title="EH$(c)", yminorticks=IntervalsBetween(9), yscale=log10)
            plot!(ax, assets.coarse_binning_c, o./assets.coarse_bin_width, label="Observed", color=:black)
            stephist!(ax, assets.coarse_binning_c, weights=m./assets.coarse_bin_width, bins=assets.coarse_binning, label="Expected")
            barplot!(ax, assets.coarse_binning_c, (m .+ sqrt.(v))./assets.coarse_bin_width, width=assets.coarse_bin_width, gap=0, fillto= (m.- sqrt.(v))./assets.coarse_bin_width, alpha=0.5, label="Standard Deviation")
            prev = ones(n_ana_binning) * 1e-10
            for (c, key) in enumerate(bg_type)
                m_bg = bg_d[key]
                barplot!(ax, assets.coarse_binning_c, (m_bg .+ prev)./assets.coarse_bin_width, width=assets.coarse_bin_width, label=string(key), color=colors[c], gap=0, fillto=prev./assets.coarse_bin_width)
                prev .+= m_bg
            end
            axislegend(ax, framevisible = false)

            ax.xticksvisible = false
            ax.xticklabelsvisible = false
            ax2 = Axis(f[2, 1])

            ax.ylabel="Counts/MeV"

            plot!(ax2, assets.coarse_binning_c, o ./ m .-1, color=:black, label="Observed")
            hlines!(ax2, 0, label="Expected")
            barplot!(ax2, assets.coarse_binning_c, sqrt.(v) ./ m, width=assets.coarse_bin_width, gap=0, fillto= .- sqrt.(v)./m, alpha=0.5, label="Standard Deviation")




            rowgap!(f.layout, 1, 0)
            rowsize!(f.layout, 1, Relative(3/4))

            ax2.xlabel="Eₚ (MeV)"
            ax2.ylabel="Counts/Expected"

            xlims!(ax2, minimum(assets.coarse_binning), maximum(assets.coarse_binning))
            xlims!(ax, minimum(assets.coarse_binning), maximum(assets.coarse_binning))
            ylims!(ax, 1e1, 1e6)
            ## ylims!(ax2, 0.3, 1.0)
            

            save(joinpath(saveto, "EH_$(c).png"), f)
            
        end
    end
end

end
