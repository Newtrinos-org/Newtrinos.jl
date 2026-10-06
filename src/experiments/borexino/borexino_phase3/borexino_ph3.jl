module borexino_ph3

using LinearAlgebra
using Distributions
using BAT: distprod
using DelimitedFiles
using PCHIPInterpolation
using SpecialFunctions: erf
using CairoMakie
import ..Newtrinos
import ..Newtrinos.solar_common

"""
Borexino Phase-III spectral analysis (S. Appel et al., Phys. Rev. Lett. 129, 252701 (2022),
arXiv:2205.15975): the TFC-subtracted (¹¹C-depleted) energy spectrum, 320–2640 keV, 817 bins,
exposure 1431.6 d × 71.3 t × 63.97 % (TFC), from the Borexino open data.

Expected spectrum: the Borexino Monte Carlo spectral shapes (open data of arXiv:2005.12829) of
pp, ⁷Be, pep and CNO neutrinos, scaled with the interaction rates predicted by `solar_flux`,
`solar_xsec` and `osc`, plus a ⁸B component from the ES response, plus the backgrounds ²¹⁰Bi,
¹¹C, ⁸⁵Kr, ²¹⁰Po and external ²¹⁴Bi, ⁴⁰K and ²⁰⁸Tl with free rates. The MC shapes are mapped to
the data energy scale with an energy scale, offset and additional Gaussian smearing, calibrated
on Borexino's best-fit spectrum (`calibrate.jl`, result in `data/calibration.txt`), with
nuisance parameters around the calibration. The ²¹⁰Bi rate has Borexino's upper limit
(10.8 ± 1.0 cpd/100 t, half-Gaussian).

Borexino also fits the TFC-tagged spectrum and the radial distribution, which are not public;
they constrain mostly ¹¹C and the external γ backgrounds. `configure(...; background_priors =
:constrained)` (default) replaces them by Gaussian priors around the calibrated rates with
relative width `background_prior_width`; `:free` leaves those rates unconstrained.
"""
@kwdef struct BorexinoPh3 <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

const ELECTRONS_PER_100T = 3.307e31
const DAY = 86400.0
const EXPOSURE = 1431.6 * 0.713 * 0.6397            # TFC-subtracted exposure [day × 100 t]
const DATADIR = joinpath(@__DIR__, "data")

# MC spectral shapes: file, and the zero-threshold rate [cpd/100 t] they are drawn at
# (2005.12829 Tab. 2); `nothing`: normalised to their integral (rate above 88 keV)
const SOLAR = (pp = ("pp_pdfkeV__16", 131.1), be7 = ("Be7_pdfkeV__2", 47.9),
               pep = ("pep_pdfkeV__15", 2.74), cno = ("CNO_pdfkeV__7", 4.92))
const BACKGROUNDS = (bi210 = ("Bi210_pdfkeV__3", 10.0), c11 = ("C11_pdfkeV__5", nothing), kr85 = ("Kr85_pdfkeV__9", nothing),
                     po210 = ("Po210_pdfkeV__11", nothing), ext_bi214 = ("Ext_Bi214_pdfkeV__12", nothing),
                     ext_k40 = ("Ext_K40_pdfkeV__13", nothing), ext_tl208 = ("Ext_Tl208_pdfkeV__14", nothing))
# backgrounds constrained by the (non-public) TFC-tagged spectrum and radial distribution
const CONSTRAINED = (:c11, :ext_bi214, :ext_k40, :ext_tl208)
const BKG_UPPER = (bi210 = 40.0, c11 = 20.0, kr85 = 60.0, po210 = 200.0, ext_bi214 = 20.0, ext_k40 = 20.0, ext_tl208 = 20.0)

# Response nuisances: 1σ widths of energy scale [fraction], offset [keV], extra smearing [keV/√MeV].
# They cover the difference between calibrating on Borexino's best-fit spectrum and on the data.
const RESPONSE_SIGMA = (scale = 0.005, offset = 5.0, smear = 5.0)

param_name(k) = Symbol(:borexino_ph3_, k)

"""
    read_spectrum() -> NamedTuple

TFC-subtracted spectrum: bin centres and widths [keV], counts, and Borexino's best-fit model
per bin (from the residuals; the 5 empty bins with undefined residuals interpolated).
"""
function read_spectrum()
    numbers(l) = parse.(Float64, [m.match for m in eachmatch(r"-?inf|-?\d+\.?\d*(?:[eE][-+]?\d+)?", l)])
    lines = readlines(joinpath(DATADIR, "PRL2022_SubtractedSpectrumFit_Data.txt"))[3:end]
    d = reduce(vcat, [permutedims(numbers(l)) for l in lines if !isempty(strip(l))])
    E, W, N, σ, res = d[:, 3], d[:, 4], d[:, 5], d[:, 6], d[:, 7]
    bestfit = N .- res .* σ
    for k in findall(!isfinite, bestfit)
        bestfit[k] = (bestfit[k-1] + bestfit[k+1]) / 2
    end
    (; E, lo = E .- W ./ 2, hi = E .+ W ./ 2, counts = N, bestfit)
end

const DE_FINE = 1.0
const E_FINE = collect(DE_FINE/2:DE_FINE:3200)

"""
    fine_density(file, norm) -> Vector

Counts per 1 keV cell per unit rate (cpd/100 t) and day × 100 t, from a monotone (PCHIP)
interpolation of the cumulative distribution of the binned MC shape.
"""
function fine_density(file, norm)
    r = readdlm(joinpath(DATADIR, file * ".txt"), ',')
    l, h, dens = r[:, 1], r[:, 2], max.(r[:, 3], 0.0)
    edges = vcat(l, h[end])
    cdf = vcat(0.0, cumsum(dens .* (h .- l)))
    norm = something(norm, cdf[end])
    itp = Interpolator(edges, cdf)
    F(x) = x <= edges[1] ? 0.0 : x >= edges[end] ? cdf[end] : itp(x)
    max.([F(e + DE_FINE/2) - F(e - DE_FINE/2) for e in E_FINE], 0.0) ./ norm
end

fine_densities() = reduce(hcat, [fine_density(f, n) for (f, n) in (values(SOLAR)..., values(BACKGROUNDS)...)])

"""
    templates(fine, lo, hi, scale, offset, smear) -> Matrix

Counts per data bin (rows) and component (columns) for unit rates and exposure, for
reconstructed energy `offset + scale * E_mc` with additional Gaussian smearing
σ² = smear² · E/MeV [keV²]. Data bins must be contiguous.
"""
function templates(fine, lo, hi, scale, offset, smear)
    T = zeros(promote_type(typeof(scale), typeof(offset), typeof(smear)), length(lo), size(fine, 2))
    w2 = (scale * DE_FINE)^2 / 12
    for i in eachindex(E_FINE)
        any(>(0), @view fine[i, :]) || continue
        μ = offset + scale * E_FINE[i]
        σ = sqrt(w2 + smear^2 * max(μ, 1.0) / 1000)
        klo = searchsortedfirst(hi, μ - 7σ)
        khi = searchsortedlast(lo, μ + 7σ)
        klo > khi && continue
        cprev = erf((lo[klo] - μ) / (sqrt(2) * σ))
        for k in klo:khi
            cnext = erf((hi[k] - μ) / (sqrt(2) * σ))
            p = max(cnext - cprev, zero(cnext)) / 2
            cprev = cnext
            for j in axes(fine, 2)
                T[k, j] += p * fine[i, j]
            end
        end
    end
    T
end

read_calibration() = Dict(Symbol(r[1]) => Float64(r[2]) for r in eachrow(readdlm(joinpath(DATADIR, "calibration.txt"), comments=true)))

default_physics() = solar_common.default_physics()

"""
    configure(physics=default_physics(); background_priors=:constrained, background_prior_width=0.03,
              bi210_limit=(10.8, 1.0)) -> BorexinoPh3

- `background_priors`: `:constrained` for Gaussian priors on the ¹¹C and external γ rates around
  their calibrated values (stand-in for the TFC-tagged spectrum and radial distribution), `:free`
  for flat priors.
- `background_prior_width`: relative 1σ width of these priors.
- `bi210_limit`: `(upper, σ)` of the half-Gaussian ²¹⁰Bi constraint, or `nothing`.
"""
function configure(physics=default_physics(); background_priors::Symbol=:constrained, background_prior_width::Real=0.03,
                   bi210_limit=(10.8, 1.0))
    background_priors in (:constrained, :free) || throw(ArgumentError("background_priors must be :constrained or :free"))
    physics = (; physics.osc, physics.solar_flux, physics.solar_xsec, physics.earth_layers)
    assets = get_assets(physics; bi210_limit)
    BorexinoPh3(physics = physics, params = get_params(assets), priors = get_priors(assets, background_priors, background_prior_width),
                assets = assets, forward_model = get_forward_model(physics, assets), plot = get_plot(physics, assets))
end

get_params(assets) = merge(NamedTuple(param_name(k) => assets.calibration[k] for k in keys(BACKGROUNDS)),
                           (borexino_ph3_energy_scale = 0.0, borexino_ph3_energy_offset = 0.0, borexino_ph3_energy_smear = 0.0))

function get_priors(assets, background_priors, width)
    bkg = map(keys(BACKGROUNDS)) do k
        c = assets.calibration[k]
        param_name(k) => (background_priors == :constrained && k in CONSTRAINED) ?
            Truncated(Normal(c, width * c), 0.0, BKG_UPPER[k]) : Uniform(0.0, BKG_UPPER[k])
    end
    merge(NamedTuple(bkg), (borexino_ph3_energy_scale = Truncated(Normal(0, 1), -3, 3),
                            borexino_ph3_energy_offset = Truncated(Normal(0, 1), -3, 3),
                            borexino_ph3_energy_smear = Truncated(Normal(0, 1), -3, 3)))
end

function get_assets(physics; bi210_limit=(10.8, 1.0))
    @info "Loading Borexino Phase-III spectrum"
    spec = read_spectrum()
    cal = read_calibration()
    fine = fine_densities()
    s0, a0, c0 = cal[:scale], cal[:offset], cal[:smear]
    rs = RESPONSE_SIGMA
    T(s, a, c) = templates(fine, spec.lo, spec.hi, s, a, c)
    response = (nominal = T(s0, a0, c0),
                scale_up = T(s0 * (1 + rs.scale), a0, c0), scale_down = T(s0 * (1 - rs.scale), a0, c0),
                offset_up = T(s0, a0 + rs.offset, c0), offset_down = T(s0, a0 - rs.offset, c0),
                smear_up = T(s0, a0, c0 + rs.smear), smear_down = T(s0, a0, max(c0 - rs.smear, 0.0)))
    # ⁸B (not among the MC shapes): ES response with Gaussian resolution σ = smear · √(T/MeV)
    E_b8 = collect(range(0.4, 16.5, length=200))
    resolution(Etot) = 1e-3 * c0 * sqrt(max(Etot - Newtrinos.solar_xsec.M_E, 1e-3))
    resp_b8 = solar_common.ESResponse(physics.solar_xsec, E_b8, collect(zip(spec.lo .* 1e-3, spec.hi .* 1e-3)), resolution;
                                       scale_unc = 0.01, reso_unc = 0.1)
    (
        observed = (spectrum = spec.counts, bi210_limit = bi210_limit === nothing ? 0.0 : Float64(bi210_limit[1])),
        spectrum = spec,
        calibration = cal,
        response,
        resp_b8,
        bi210_limit,
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
    R.nominal .+ pull(R.scale_up, R.scale_down, params.borexino_ph3_energy_scale) .+
        pull(R.offset_up, R.offset_down, params.borexino_ph3_energy_offset) .+
        pull(R.smear_up, R.smear_down, params.borexino_ph3_energy_smear)
end

"""
    get_expected(params, physics, assets) -> NamedTuple

Expected counts per bin: `total` and per component (`components`, columns in the order pp, ⁷Be,
pep, CNO, backgrounds, ⁸B).
"""
function get_expected(params, physics, assets)
    solar = solar_rates(params, physics, assets)
    bkg = [params[param_name(k)] for k in keys(BACKGROUNDS)]
    rates = vcat(collect(values(solar)), bkg)
    T = response_templates(params, assets)
    comps = EXPOSURE .* T .* permutedims(rates)
    sf = physics.solar_flux
    Pe = solar_common.averaged_prob(physics, assets.site, :b8, assets.resp_b8.E, params)[:, 1]
    b8 = solar_common.es_rates(assets.resp_b8, sf.spectrum(:b8, assets.resp_b8.E, params), Pe) .* (ELECTRONS_PER_100T * DAY * EXPOSURE)
    (total = vec(sum(comps, dims=2)) .+ b8, components = hcat(comps, b8), solar)
end

function get_forward_model(physics, assets)
    function forward_model(params)
        μ = max.(get_expected(params, physics, assets).total, 1e-9)
        lim = assets.bi210_limit
        # half-Gaussian upper limit on ²¹⁰Bi as a pseudo-measurement at the limit
        bi = lim === nothing ? Normal(0.0, 1.0) : Normal(max(params.borexino_ph3_bi210, lim[1]), lim[2])
        distprod((spectrum = distprod(Poisson.(μ)), bi210_limit = bi))
    end
end

const LABELS = ["pp", "⁷Be", "pep", "CNO", "²¹⁰Bi", "¹¹C", "⁸⁵Kr", "²¹⁰Po", "ext ²¹⁴Bi", "ext ⁴⁰K", "ext ²⁰⁸Tl", "⁸B"]

function get_plot(physics, assets)
    function plot(params, data=assets.observed)
        ex = get_expected(params, physics, assets)
        val(x) = solar_common.ForwardDiffValue.(x)
        E, N = assets.spectrum.E, data.spectrum
        f = Figure(size=(800, 700))
        ax = Axis(f[1, 1], yscale=log10, ylabel="Events / bin", title="Borexino Phase-III, TFC-subtracted")
        scatter!(ax, E, max.(N, 0.5), color=:black, markersize=3, label="Data")
        colors = Makie.to_colormap(:tab10)
        for (i, j) in enumerate((2, 3, 4, 5, 6, 7, 8, 12))
            lines!(ax, E, max.(val(ex.components[:, j]), 1e-2), color=colors[i], label=LABELS[j])
        end
        lines!(ax, E, max.(val(vec(sum(ex.components[:, 9:11], dims=2))), 1e-2), color=:grey, linestyle=:dash, label="ext γ")
        lines!(ax, E, val(ex.total), color=:magenta, linewidth=2, label="Total")
        ylims!(ax, 0.5, 2e3)
        axislegend(ax, position=:rt, nbanks=2)
        ax2 = Axis(f[2, 1], xlabel="Energy (keV)", ylabel="(data − exp)/√exp")
        scatter!(ax2, E, (N .- val(ex.total)) ./ sqrt.(val(ex.total)), color=:black, markersize=3)
        hlines!(ax2, 0, color=:magenta)
        rowsize!(f.layout, 1, Relative(3/4))
        linkxaxes!(ax, ax2)
        f
    end
end

end
