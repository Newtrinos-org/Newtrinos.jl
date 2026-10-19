module borexino_ph2_spectrum

using LinearAlgebra
using Distributions
using BAT: distprod
using DelimitedFiles
using CairoMakie
import ..Newtrinos
import ..Newtrinos.solar_common
import ..Newtrinos.borexino_ph3

"""
Borexino Phase-II spectral analysis (M. Agostini et al., Nature 562, 505 (2018); Phys. Rev. D 100,
082004 (2019), arXiv:1707.09279): the TFC-subtracted energy spectrum in the N_h estimator (Nature
Fig. 2a, Borexino open data), fitted from N_h = 100 (231 keV) to 950 (2630 keV), 851 bins.

Expected spectrum:
- pp, ⁷Be, pep and CNO neutrinos: the Borexino Monte Carlo spectral shapes (open data of
  arXiv:2005.12829, shared with `borexino_ph3`) scaled with the interaction rates predicted by
  `solar_flux`, `solar_xsec` and `osc`; the shapes are mapped to the data energy scale with an
  energy scale, offset and additional Gaussian smearing calibrated on Borexino's best-fit spectrum
  (`calibrate.jl`, result in `data/calibration.txt`), with nuisance parameters around it; ⁸B from
  the ES response.
- Backgrounds (¹⁴C, pile-up, external γ, ²¹⁰Po, ⁸⁵Kr, ²¹⁰Bi, ¹¹C): Borexino's own fitted
  components, extracted from the vector graphics of Fig. 6a of arXiv:1707.09279 (see
  `extraction/`), as fixed shapes per N_h bin with free normalisation factors (1 = Borexino's
  best fit). Their detector response is Borexino's. The bin-to-bin structure of the discrete N_h
  estimator under the ²¹⁰Po peak, present in the data and in Borexino's total fit but not in the
  smooth drawn curves, is added to the ²¹⁰Po shape ((total − Σ components) where ²¹⁰Po makes up
  more than half of the total).

As in Borexino's MC-method fit, the ¹⁴C normalisation has a Gaussian prior of ±5 % (independent
measurement, 40.0 ± 2.0 Bq/100 t) and the pile-up one of ±2.0 % (137.5 ± 2.8 cpd/100 t for the
¹⁴C–¹⁴C contribution) (`c14_prior_width`, `pileup_prior_width`). The radial and pulse-shape
information of Borexino's multivariate fit and the TFC-tagged spectrum are not public.

The shape of each solar component is the MC shape at fixed oscillation probability; only the
rates follow the oscillation parameters (the energy dependence of P_ee within the pp and CNO
spectra is not applied).
"""
@kwdef struct BorexinoPh2Spectrum <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

const ELECTRONS_PER_100T = borexino_ph3.ELECTRONS_PER_100T
const DAY = borexino_ph3.DAY
# exposure [day × 100 t] of the TFC-subtracted spectrum: the data file gives rates per day × 100 t, whose counts
# (rate × exposure) are integers for this exposure (1291.51 d × 71.3 t × 63.25 % TFC-subtracted fraction)
const EXPOSURE = 582.326
const NH_MIN = 100
const DATADIR = joinpath(@__DIR__, "data")

# MC shapes of the solar components (in borexino_ph3's data directory) and their zero-threshold rates
const SOLAR = (pp = ("pp_pdfkeV__16", 131.1), be7 = ("Be7_pdfkeV__2", 47.9), pep = ("pep_pdfkeV__15", 2.74),
               cno = ("CNO_pdfkeV__7", 4.92))
# Borexino background components: column of the component table
const BACKGROUNDS = (c14 = "C14", pileup = "pileup", ext = "ext", po210 = "Po210", kr85 = "Kr85", bi210 = "Bi210", c11 = "C11")
const BKG_UPPER = 3.0
# response nuisances: 1σ of energy scale [fraction], offset [keV], extra smearing [keV/√MeV]
const RESPONSE_SIGMA = (scale = 0.003, offset = 5.0, smear = 10.0)

param_name(k) = Symbol(:borexino_ph2s_, k)

"""
    read_spectrum() -> NamedTuple

Nature 2018 Fig. 2a spectrum for N_h ≥ `NH_MIN`: N_h, bin centres, edges [keV], counts, Borexino's best-fit model
(counts), and Borexino's background components [counts / (day × 100 t) per bin].
"""
function read_spectrum()
    tab = readdlm(joinpath(DATADIR, "phase2_fig6a_components.csv"), ',', Any; comments = true)
    header = String.(vec(tab[1, :])); d = Float64.(tab[2:end, :])
    col(name) = d[:, findfirst(==(name), header)]
    nh = col("N_h")
    sel = nh .>= NH_MIN
    numbers(l) = parse.(Float64, [m.match for m in eachmatch(r"-?inf|-?\d+\.?\d*(?:[eE][-+]?\d+)?", l)])
    raw = reduce(vcat, [permutedims(numbers(l)[1:7]) for l in readlines(joinpath(DATADIR, "Nature2018_Fig2a_DATA.txt"))[5:end]
                        if length(numbers(l)) >= 7])
    rows = [findfirst(==(n), raw[:, 2]) for n in nh[sel]]
    E, W = raw[rows, 3], raw[rows, 4]
    total = col("total")[sel]
    bkg = reduce(hcat, [col(c)[sel] for c in values(BACKGROUNDS)])
    # bin structure of the discrete N_h estimator under the ²¹⁰Po peak
    comps = reduce(hcat, [col(c)[sel] for c in ("C14", "pileup", "Po210", "Kr85", "Bi210", "C11", "ext", "solar")])
    resid = total .- vec(sum(comps, dims = 2))
    ipo = findfirst(==(:po210), keys(BACKGROUNDS))
    podom = bkg[:, ipo] .> 0.5 .* total
    bkg[:, ipo] .+= ifelse.(podom, resid, 0.0)
    (; nh = nh[sel], E, lo = E .- W ./ 2, hi = E .+ W ./ 2, counts = round.(col("data")[sel] .* EXPOSURE),
       bestfit = total .* EXPOSURE, bkg)
end

read_calibration() = Dict(Symbol(r[1]) => Float64(r[2]) for r in eachrow(readdlm(joinpath(DATADIR, "calibration.txt"), comments=true)))

fine_densities() = reduce(hcat, [borexino_ph3.fine_density(f, n) for (f, n) in values(SOLAR)])

default_physics() = solar_common.default_physics()

"""
    configure(physics=default_physics(); c14_prior_width=0.05, pileup_prior_width=0.0204) -> BorexinoPh2Spectrum
"""
function configure(physics=default_physics(); c14_prior_width::Real=0.05, pileup_prior_width::Real=2.8 / 137.5)
    physics = (; physics.osc, physics.solar_flux, physics.solar_xsec, physics.earth_layers)
    assets = get_assets(physics)
    BorexinoPh2Spectrum(physics = physics, params = get_params(assets), priors = get_priors(c14_prior_width, pileup_prior_width),
                        assets = assets, forward_model = get_forward_model(physics, assets), plot = get_plot(physics, assets))
end

get_params(assets) = merge(NamedTuple(param_name(k) => get(assets.calibration, k, 1.0) for k in keys(BACKGROUNDS)),
                           (borexino_ph2s_energy_scale = 0.0, borexino_ph2s_energy_offset = 0.0, borexino_ph2s_energy_smear = 0.0))

function get_priors(c14_width, pileup_width)
    widths = (c14 = c14_width, pileup = pileup_width)
    bkg = map(keys(BACKGROUNDS)) do k
        param_name(k) => haskey(widths, k) ? Truncated(Normal(1.0, widths[k]), 0.0, BKG_UPPER) : Uniform(0.0, BKG_UPPER)
    end
    merge(NamedTuple(bkg), (borexino_ph2s_energy_scale = Truncated(Normal(0, 1), -3, 3),
                            borexino_ph2s_energy_offset = Truncated(Normal(0, 1), -3, 3),
                            borexino_ph2s_energy_smear = Truncated(Normal(0, 1), -3, 3)))
end

function get_assets(physics)
    @info "Loading Borexino Phase-II spectrum"
    spec = read_spectrum()
    @assert maximum(abs.(spec.lo[2:end] .- spec.hi[1:end-1])) < 0.02 "data bins not contiguous"
    cal = read_calibration()
    fine = fine_densities()
    s0, a0, c0 = cal[:scale], cal[:offset], cal[:smear]
    rs = RESPONSE_SIGMA
    T(s, a, c) = borexino_ph3.templates(fine, spec.lo, spec.hi, s, a, c)
    response = (nominal = T(s0, a0, c0),
                scale_up = T(s0 * (1 + rs.scale), a0, c0), scale_down = T(s0 * (1 - rs.scale), a0, c0),
                offset_up = T(s0, a0 + rs.offset, c0), offset_down = T(s0, a0 - rs.offset, c0),
                smear_up = T(s0, a0, c0 + rs.smear), smear_down = T(s0, a0, max(c0 - rs.smear, 0.0)))
    # ⁸B: ES response with the Borexino energy resolution (≈ 5 %/√(E/MeV)), small in this window
    E_b8 = collect(range(0.4, 16.5, length=200))
    resolution(Etot) = 0.05 * sqrt(max(Etot - Newtrinos.solar_xsec.M_E, 1e-3))
    resp_b8 = solar_common.ESResponse(physics.solar_xsec, E_b8, collect(zip(spec.lo .* 1e-3, spec.hi .* 1e-3)), resolution;
                                       scale_unc = 0.01, reso_unc = 0.1)
    (
        observed = spec.counts,
        spectrum = spec,
        calibration = cal,
        response,
        resp_b8,
        components = (pp = (solar_common.ESTotal(physics, :pp),), be7 = (solar_common.ESTotal(physics, :be7),),
                      pep = (solar_common.ESTotal(physics, :pep),),
                      cno = (solar_common.ESTotal(physics, :n13), solar_common.ESTotal(physics, :o15), solar_common.ESTotal(physics, :f17))),
        site = solar_common.Site(physics, 42.45; depth_km=1.4),  # LNGS
    )
end

"""
    solar_rates(params, physics, assets) -> NamedTuple

Zero-threshold interaction rates [cpd/100 t] of pp, ⁷Be, pep and CNO neutrinos.
"""
solar_rates(params, physics, assets) = map(assets.components) do ests
    sum(solar_common.es_interaction_rate(physics, assets.site, est, params) for est in ests) * ELECTRONS_PER_100T * DAY
end

function response_templates(params, assets)
    R = assets.response
    pull(up, down, x) = solar_common.interp_pull(R.nominal, up, down, x) .- R.nominal
    R.nominal .+ pull(R.scale_up, R.scale_down, params.borexino_ph2s_energy_scale) .+
        pull(R.offset_up, R.offset_down, params.borexino_ph2s_energy_offset) .+
        pull(R.smear_up, R.smear_down, params.borexino_ph2s_energy_smear)
end

"""
    get_expected(params, physics, assets) -> NamedTuple

Expected counts per bin: `total`, per component (`components`, columns pp, ⁷Be, pep, CNO, ⁸B, then the backgrounds
in the order of `BACKGROUNDS`) and the solar rates.
"""
function get_expected(params, physics, assets)
    solar = solar_rates(params, physics, assets)
    T = response_templates(params, assets)
    sol = EXPOSURE .* T .* permutedims(collect(values(solar)))
    sf = physics.solar_flux
    Pe = solar_common.averaged_prob(physics, assets.site, :b8, assets.resp_b8.E, params)[:, 1]
    b8 = solar_common.es_rates(assets.resp_b8, sf.spectrum(:b8, assets.resp_b8.E, params), Pe) .* (ELECTRONS_PER_100T * DAY * EXPOSURE)
    norms = collect(promote((params[param_name(k)] for k in keys(BACKGROUNDS))...))   # concrete type (Float or Dual)
    bkg = EXPOSURE .* assets.spectrum.bkg .* permutedims(norms)
    E = promote_type(eltype(sol), eltype(b8), eltype(bkg))
    comps = hcat(E.(sol), E.(b8), E.(bkg))
    (total = vec(sum(comps, dims=2)), components = comps, solar)
end

function get_forward_model(physics, assets)
    function forward_model(params)
        distprod(Poisson.(max.(get_expected(params, physics, assets).total, 1e-9)))
    end
end

const LABELS = ["pp", "⁷Be", "pep", "CNO", "⁸B", "¹⁴C", "pile-up", "ext γ", "²¹⁰Po", "⁸⁵Kr", "²¹⁰Bi", "¹¹C"]

function get_plot(physics, assets)
    function plot(params, data=assets.observed)
        ex = get_expected(params, physics, assets)
        val(x) = solar_common.ForwardDiffValue.(x)
        E, N = assets.spectrum.E, data
        norm = 1 / EXPOSURE                       # counts → rate per day × 100 t × N_h, as in Nature Fig. 2a
        f = Figure(size=(800, 700))
        ax = Axis(f[1, 1], yscale=log10, ylabel="Counts / (day × 100 t × N_h)", title="Borexino Phase II, TFC-subtracted")
        scatter!(ax, E, max.(N, 0.5) .* norm, color=:black, markersize=3, label="Data")
        colors = Makie.to_colormap(:tab20)
        for j in axes(ex.components, 2)
            lines!(ax, E, max.(val(ex.components[:, j]), 1e-3) .* norm, color=colors[j], linestyle=(j <= 5 ? :solid : :dash),
                   label=LABELS[j])
        end
        lines!(ax, E, val(ex.total) .* norm, color=:black, linewidth=1.5, label="Total")
        ylims!(ax, 2e-4, 30)
        axislegend(ax, position=:rt, nbanks=3, labelsize=10)
        ax2 = Axis(f[2, 1], xlabel="Energy (keV)", ylabel="(data − exp)/√exp")
        scatter!(ax2, E, (N .- val(ex.total)) ./ sqrt.(val(ex.total)), color=:black, markersize=3)
        hlines!(ax2, 0, color=:red)
        rowsize!(f.layout, 1, Relative(3/4))
        linkxaxes!(ax, ax2)
        f
    end
end

end
