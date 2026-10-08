module borexino_ph1_spectrum

using LinearAlgebra
using Distributions
using BAT: distprod
using DelimitedFiles
using CairoMakie
import ..Newtrinos
import ..Newtrinos.solar_common

"""
Borexino Phase-I spectra, the two Phase-I data sets of the NuFIT global analysis:

- Low energy (⁷Be, 740.7 live days): the spectrum of the MC-based fit of G. Bellini et al., Phys. Rev. Lett. 107,
  141302 (2011), arXiv:1104.1816, Fig. 1 (top), 270–1600 keV in 10 keV bins (133 bins, Poisson likelihood; the
  counts follow from the published error bars, exposure 55 172 t·day).
  Solar components pp, ⁷Be, pep, CNO: ν–e elastic scattering of the Newtrinos solar fluxes with the
  energy-dependent survival probability (day/night averaged at LNGS), folded with a Gaussian resolution
  σ = c √(T/MeV) and the energy scale E = s·T + a; (s, a, c) are calibrated on Borexino's fitted ⁷Be component
  (`calibrate.jl`, `data/calibration.txt`), with energy-scale and resolution nuisance parameters around them.
  (The open-data MC shapes used for Phases II and III carry the Phase-III energy scale and resolution and do not
  describe the sharper Phase-I ⁷Be edge.)
  Backgrounds ⁸⁵Kr, ²¹⁰Bi, ¹¹C, ²¹⁰Po and external γ: Borexino's fitted components extracted from the vector
  graphics of the figure (`extraction/`), fixed shapes with free normalisations (1 = Borexino's fit); in the ²¹⁰Po
  peak (where it dominates) the ²¹⁰Po shape is Borexino's total fit minus the other components, since the drawn
  ²¹⁰Po curve has too few vertices for the peak.
- High energy (⁸B, 345.3 live days, 100 t): the background-subtracted ⁸B spectrum of G. Bellini et al., Phys. Rev.
  D 82, 033006 (2010), arXiv:0808.2868, Fig. 7, five 2 MeV bins from 3 to 13 MeV with the published errors
  (Gaussian likelihood), compared with ν–e scattering of ⁸B and hep neutrinos with a Gaussian resolution and an
  energy-scale nuisance.
"""
@kwdef struct BorexinoPh1Spectrum <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

const ELECTRONS_PER_100T = 3.307e31
const DAY = 86400.0
const EXPOSURE_LE = 551.72            # low-energy exposure [day × 100 t] (from the Poisson error bars of the figure)
const EXPOSURE_HE = 345.3             # ⁸B: 345.3 live days × 100 t [day × 100 t]
const BIN_LE = 10.0                   # keV
const DATADIR = joinpath(@__DIR__, "data")

const BACKGROUNDS = (kr85 = "Kr85", bi210 = "Bi210", c11 = "C11", po210 = "Po210", ext = "ext")
const BKG_UPPER = 3.0
const SCALE_UNC = 0.005               # low-energy energy-scale uncertainty (1σ, fraction)
const RESO_UNC = 0.05                 # low-energy resolution uncertainty (1σ, fraction)
const B8_SCALE_UNC = 0.02             # ⁸B energy-scale uncertainty
const B8_RESOLUTION = 0.05            # ⁸B: σ_E = 5 % √(E/MeV)

param_name(k) = Symbol(:borexino_ph1s_, k)

"""
    read_spectra() -> NamedTuple

Low-energy spectrum (energies, edges [keV], counts, Borexino's fit, its ⁷Be component and background components in
counts per bin) and the ⁸B spectrum (bin edges [MeV], counts, symmetric errors).
"""
function read_spectra()
    tab = readdlm(joinpath(DATADIR, "phase1_be7_fig1.csv"), ',', Any; comments = true)
    header = String.(vec(tab[1, :])); d = Float64.(tab[2:end, :])
    col(name) = d[:, findfirst(==(name), header)]
    perbin = BIN_LE * 1e-3 * EXPOSURE_LE * 100          # rate [1 / (MeV t day)] → counts per 10 keV bin
    E = col("energy_keV")
    bkg = reduce(hcat, [col(c) .* perbin for c in values(BACKGROUNDS)])
    total, be7, fixed_solar = col("total") .* perbin, col("Be7") .* perbin, col("pp_pep_cno") .* perbin
    # the ²¹⁰Po curve is drawn with few vertices, too coarse for its peak (~3·10⁴ counts per bin): where ²¹⁰Po dominates,
    # its shape is taken as Borexino's total fit (drawn with ~one vertex per bin) minus all other components
    ipo = findfirst(==(:po210), keys(BACKGROUNDS))
    podom = bkg[:, ipo] .> 0.5 .* total
    others = be7 .+ fixed_solar .+ vec(sum(bkg, dims=2)) .- bkg[:, ipo]
    bkg[podom, ipo] .= max.(total[podom] .- others[podom], 0.0)
    le = (E = E, lo = E .- BIN_LE / 2, hi = E .+ BIN_LE / 2, counts = round.(col("data") .* perbin),
          bestfit = total, be7 = be7, bkg = bkg, fixed_solar = fixed_solar)
    t8 = readdlm(joinpath(DATADIR, "phase1_b8_fig7.csv"), ',', Float64; comments = true, skipstart = 3)
    he = (lo = t8[:, 1], hi = t8[:, 2], counts = t8[:, 3], sigma = (t8[:, 4] .+ t8[:, 5]) ./ 2)
    (; le, he)
end

read_calibration() = Dict(Symbol(r[1]) => Float64(r[2]) for r in eachrow(readdlm(joinpath(DATADIR, "calibration.txt"), comments=true)))

"""
    le_responses(physics, le, scale, offset, smear) -> NamedTuple

ES responses for the low-energy bins with visible energy E = scale·T + offset [keV] and Gaussian resolution
σ = smear·√(T/MeV) [keV]: `cont` on a neutrino-energy grid for the continuous components, `lines` at the line
energies (⁷Be 384 and 862 keV, pep 1442 keV; unit weights).
"""
function le_responses(physics, le, scale, offset, smear)
    bins = [((lo - offset) / scale * 1e-3, (hi - offset) / scale * 1e-3) for (lo, hi) in zip(le.lo, le.hi)]  # in T [MeV]
    resolution(Etot) = smear / scale * 1e-3 * sqrt(max(Etot - Newtrinos.solar_xsec.M_E, 1e-4))
    xs = physics.solar_xsec
    cont = solar_common.ESResponse(xs, collect(range(0.1, 1.75, length=166)), bins, resolution;
                                   scale_unc=SCALE_UNC, reso_unc=RESO_UNC, T_step=0.002)
    El = [0.3843, 0.8613, 1.442]
    r = solar_common.ESResponse(xs, El, bins, resolution; scale_unc=SCALE_UNC, reso_unc=RESO_UNC, T_step=0.002)
    (; cont, lines = solar_common.ESResponse(r.E, ones(length(El)), r.K))
end

default_physics() = solar_common.default_physics()

"""
    configure(physics=default_physics()) -> BorexinoPh1Spectrum
"""
function configure(physics=default_physics())
    physics = (; physics.osc, physics.solar_flux, physics.solar_xsec, physics.earth_layers)
    assets = get_assets(physics)
    BorexinoPh1Spectrum(physics = physics, params = get_params(assets), priors = get_priors(),
                        assets = assets, forward_model = get_forward_model(physics, assets), plot = get_plot(physics, assets))
end

get_params(assets) = merge(NamedTuple(param_name(k) => get(assets.calibration, k, 1.0) for k in keys(BACKGROUNDS)),
                           (borexino_ph1s_energy_scale = 0.0, borexino_ph1s_energy_reso = 0.0, borexino_ph1s_b8_scale = 0.0))

get_priors() = merge(NamedTuple(param_name(k) => Uniform(0.0, BKG_UPPER) for k in keys(BACKGROUNDS)),
                     (borexino_ph1s_energy_scale = Truncated(Normal(0, 1), -3, 3), borexino_ph1s_energy_reso = Truncated(Normal(0, 1), -3, 3),
                      borexino_ph1s_b8_scale = Truncated(Normal(0, 1), -3, 3)))

function get_assets(physics)
    @info "Loading Borexino Phase-I spectra"
    spec = read_spectra()
    cal = read_calibration()
    # ⁸B and hep: ES response in the 2 MeV bins (electron kinetic energy), Gaussian resolution
    E_nu = collect(range(2.5, 18.8, length=160))
    resolution(Etot) = B8_RESOLUTION * sqrt(max(Etot - Newtrinos.solar_xsec.M_E, 1e-3))
    resp_b8 = solar_common.ESResponse(physics.solar_xsec, E_nu, collect(zip(spec.he.lo, spec.he.hi)), resolution;
                                       scale_unc = B8_SCALE_UNC, reso_unc = 0.1)
    (
        observed = (le = spec.le.counts, he = spec.he.counts),
        spectra = spec,
        calibration = cal,
        response = le_responses(physics, spec.le, cal[:scale], cal[:offset], cal[:smear]),
        resp_b8,
        components = (pp = (solar_common.ESTotal(physics, :pp),), be7 = (solar_common.ESTotal(physics, :be7),),
                      pep = (solar_common.ESTotal(physics, :pep),),
                      cno = (solar_common.ESTotal(physics, :n13), solar_common.ESTotal(physics, :o15), solar_common.ESTotal(physics, :f17)),
                      be7_862 = (solar_common.ESTotal(physics, :be7; line_energies=[0.8613]),)),
        site = solar_common.Site(physics, 42.45; depth_km=1.4),  # LNGS
    )
end

"""
    solar_rates(params, physics, assets) -> NamedTuple

Zero-threshold interaction rates [cpd/100 t] of pp, ⁷Be (both lines), pep and CNO neutrinos, and of the 862 keV ⁷Be
line alone (`be7_862`, the quantity quoted by Borexino).
"""
solar_rates(params, physics, assets) = map(assets.components) do ests
    sum(solar_common.es_interaction_rate(physics, assets.site, est, params) for est in ests) * ELECTRONS_PER_100T * DAY
end

"""
    le_solar_counts(params, physics, assets) -> Matrix

Expected low-energy counts per bin of pp, ⁷Be, pep and CNO (columns), with the energy-dependent survival probability.
"""
function le_solar_counts(params, physics, assets)
    sf, R, site = physics.solar_flux, assets.response, assets.site
    pulls = (scale = params.borexino_ph1s_energy_scale, reso = params.borexino_ph1s_energy_reso)
    E = R.cont.E
    cont(c) = solar_common.es_rates(R.cont, sf.spectrum(c, E, params), solar_common.averaged_prob(physics, site, c, E, params)[:, 1]; pulls...)
    l7, lp = sf.lines.be7, sf.lines.pep
    P7 = solar_common.averaged_prob(physics, site, :be7, l7.E, params)[:, 1]
    Pp = solar_common.averaged_prob(physics, site, :pep, lp.E, params)[:, 1]
    # lines in the order of R.lines.E: ⁷Be 384, ⁷Be 862, pep 1442 keV
    i384, i862 = findfirst(e -> abs(e - 0.3843) < 1e-3, l7.E), findfirst(e -> abs(e - 0.8613) < 1e-3, l7.E)
    Φ7, Φp = sf.flux(:be7, params), sf.flux(:pep, params)
    z = zero(Φ7 + P7[1])
    be7 = solar_common.es_rates(R.lines, [Φ7 * l7.br[i384], Φ7 * l7.br[i862], z], [P7[i384], P7[i862], one(z)]; pulls...)
    pep = solar_common.es_rates(R.lines, [z, z, Φp], [one(z), one(z), Pp[1]]; pulls...)
    k = ELECTRONS_PER_100T * DAY * EXPOSURE_LE
    hcat(cont(:pp), be7, pep, cont(:n13) .+ cont(:o15) .+ cont(:f17)) .* k
end

"""
    get_expected(params, physics, assets) -> NamedTuple

Expected counts: `le` (low-energy spectrum, with `le_components`: pp, ⁷Be, pep, CNO, then the backgrounds in the
order of `BACKGROUNDS`), `he` (⁸B + hep counts per 2 MeV bin), and the solar rates.
"""
function get_expected(params, physics, assets)
    solar = solar_rates(params, physics, assets)
    sol = le_solar_counts(params, physics, assets)
    norms = collect(promote((params[param_name(k)] for k in keys(BACKGROUNDS))...))
    bkg = assets.spectra.le.bkg .* permutedims(norms)
    El = promote_type(eltype(sol), eltype(bkg))
    comps = hcat(El.(sol), El.(bkg))
    sf, r8 = physics.solar_flux, assets.resp_b8
    pull = (scale = params.borexino_ph1s_b8_scale, reso = zero(params.borexino_ph1s_b8_scale))
    he = sum(solar_common.es_rates(r8, sf.spectrum(c, r8.E, params),
                                   solar_common.averaged_prob(physics, assets.site, c, r8.E, params)[:, 1]; pull...)
             for c in (:b8, :hep)) .* (ELECTRONS_PER_100T * DAY * EXPOSURE_HE)
    (le = vec(sum(comps, dims=2)), le_components = comps, he = he, solar)
end

function get_forward_model(physics, assets)
    σ = assets.spectra.he.sigma
    function forward_model(params)
        e = get_expected(params, physics, assets)
        distprod((le = distprod(Poisson.(max.(e.le, 1e-9))), he = MvNormal(e.he, Diagonal(σ .^ 2))))
    end
end

const LABELS = ["pp", "⁷Be", "pep", "CNO", "⁸⁵Kr", "²¹⁰Bi", "¹¹C", "²¹⁰Po", "ext γ"]

function get_plot(physics, assets)
    function plot(params, data=assets.observed)
        ex = get_expected(params, physics, assets)
        val(x) = solar_common.ForwardDiffValue.(x)
        le = assets.spectra.le
        norm = 1 / (BIN_LE * 1e-3 * EXPOSURE_LE * 100)      # counts → events / (1000 keV × t × day), as in the paper
        f = Figure(size=(1100, 650))
        ax = Axis(f[1, 1], yscale=log10, ylabel="Event rate [evt / (1000 keV × t × day)]", title="Borexino Phase I, ⁷Be")
        scatter!(ax, le.E, max.(data.le, 0.5) .* norm, color=:black, markersize=4, label="Data")
        colors = Makie.to_colormap(:tab10)
        for j in axes(ex.le_components, 2)
            lines!(ax, le.E, max.(val(ex.le_components[:, j]), 1e-3) .* norm, color=colors[j], label=LABELS[j])
        end
        lines!(ax, le.E, val(ex.le) .* norm, color=:black, linewidth=1.5, label="Total")
        ylims!(ax, 1e-2, 1e2); axislegend(ax, position=:rt, nbanks=2, labelsize=10)
        ax2 = Axis(f[2, 1], xlabel="Energy (keV)", ylabel="(data − exp)/√exp")
        scatter!(ax2, le.E, (data.le .- val(ex.le)) ./ sqrt.(val(ex.le)), color=:black, markersize=4)
        hlines!(ax2, 0, color=:red)
        rowsize!(f.layout, 1, Relative(3/4))
        he = assets.spectra.he
        ax3 = Axis(f[1:2, 2], xlabel="Energy (MeV)", ylabel="Counts / 2 MeV / 345.3 days", title="Borexino Phase I, ⁸B")
        x = (he.lo .+ he.hi) ./ 2
        errorbars!(ax3, x, data.he, he.sigma, color=:red); scatter!(ax3, x, data.he, color=:red, label="Data")
        stairs!(ax3, vcat(he.lo, he.hi[end]), vcat(val(ex.he), val(ex.he[end])), step=:post, color=:blue, label="Newtrinos")
        axislegend(ax3, position=:rt)
        colsize!(f.layout, 2, Relative(1/3))
        f
    end
end

end
