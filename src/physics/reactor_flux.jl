"""
Reactor ν̄e flux and inverse-β-decay (IBD) detection, common to reactor experiments.

- Isotope spectra (ν̄e per fission per MeV) of ²³⁵U, ²³⁹Pu, ²⁴¹Pu (Huber, PRC 84, 024617 (2011)) and ²³⁸U (Mueller et
  al., PRC 83, 054615 (2011)), exponential-of-polynomial fits valid for 2 ≲ E_ν ≲ 8 MeV (the Huber–Mueller model,
  `HuberMueller()`, default), or the data-driven spectra unfolded from Daya Bay (`DayaBayUnfolded()`, Chin. Phys. C 45,
  073001 (2021)): measured total, ²³⁵U and Pu-combo spectra, corrected to other fission fractions with Huber–Mueller ²³⁸U
  and ²⁴¹Pu.
- Energy released per fission (Ma et al., PRC 88, 014605 (2013)).
- IBD cross section of Strumia & Vissani (PLB 564, 42 (2003)), Eq. (25).
- Reactor power → fission rate for given fission fractions.

Experiments combine these with their reactor list (thermal powers, load factors, baselines), detector response and
oscillation probabilities; the physics module itself has no parameters (flux normalisation and shape uncertainties are
experiment nuisance parameters).
"""
module reactor_flux

using ..Newtrinos

const ISOTOPES = (:U235, :U238, :Pu239, :Pu241)

# exp(Σₖ aₖ E^(k-1)), E in MeV → ν̄e / fission / MeV
const HM_COEFFS = (U235  = (4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3),      # Huber
                   U238  = (4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4),  # Mueller
                   Pu239 = (4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3),     # Huber
                   Pu241 = (2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3))     # Huber

const ENERGY_PER_FISSION = (U235 = 202.36, U238 = 205.99, Pu239 = 211.12, Pu241 = 214.26)   # MeV

const M_N = 939.56542      # MeV
const M_P = 938.27209
const M_E = 0.51099895
const DELTA_NP = M_N - M_P
const IBD_THRESHOLD = ((M_N + M_E)^2 - M_P^2) / (2 * M_P)        # 1.806 MeV
const MEV_TO_J = 1.602176634e-13

"""
    ReactorFlux <: Newtrinos.Physics

# Fields
- `model::SpectrumModel`: [`HuberMueller`](@ref) or [`DayaBayUnfolded`](@ref).
- `isotope_spectrum(iso, E)`: ν̄e per fission per MeV of isotope `iso` ∈ `ISOTOPES` at energy `E` [MeV].
- `spectrum(E, fractions)`: ν̄e per fission per MeV for fission fractions `fractions` (NamedTuple over `ISOTOPES`).
- `rate_scale(fractions)`: normalisation applied to `spectrum` by experiments: the Bugey-4 anchoring
  ([`bugey4_scale`](@ref)) for the Huber–Mueller model, 1 for the measured Daya Bay spectra.
- `ibd_xsec(E)`: IBD cross section [cm²].
- `fission_rate(P_MW, fractions)`: fissions per second at thermal power `P_MW` [MW].
"""
@kwdef struct ReactorFlux <: Newtrinos.Physics
    params::NamedTuple = NamedTuple()
    priors::NamedTuple = NamedTuple()
    model = HuberMueller()
    isotope_spectrum::Function
    spectrum::Function
    rate_scale::Function
    ibd_xsec::Function
    fission_rate::Function
end

"Reactor antineutrino spectrum model of [`ReactorFlux`](@ref)."
abstract type SpectrumModel end

"""
    HuberMueller <: SpectrumModel

Huber (²³⁵U, ²³⁹Pu, ²⁴¹Pu) and Mueller (²³⁸U) conversion spectra; rates anchored to Bugey-4 ([`bugey4_scale`](@ref)).
"""
struct HuberMueller <: SpectrumModel end

"""
    DayaBayUnfolded <: SpectrumModel

Data-driven spectra from Daya Bay (Chin. Phys. C 45, 073001 (2021); tables in `reactors/dayabay_unfolded/`): the
unfolded total, ²³⁵U and Pu-combo (²³⁹Pu + 0.183 ²⁴¹Pu) antineutrino spectra of the near halls, at average fission
fractions f_DB = 0.564 : 0.076 : 0.304 : 0.056 (²³⁵U : ²³⁸U : ²³⁹Pu : ²⁴¹Pu). For fission fractions f, the paper's
prediction (Eq. 9) is

``S(f) = S_\\mathrm{total} + Δf_{235} S_{235} + Δf_{239} S_\\mathrm{combo} + Δf_{238} S^\\mathrm{HM}_{238} + (Δf_{241} - 0.183 Δf_{239}) S^\\mathrm{HM}_{241}``,

Δf = f − f_DB. Each unfolded spectrum is used as the Huber–Mueller spectrum times a ratio that is linear between the bin centres
(1.925–8.65 MeV) and constant beyond, with node values chosen such that the bin means reproduce the unfolded bin
contents exactly; this keeps the fine shape within the 0.25 MeV bins. The measured absolute normalisation is kept
(`rate_scale = 1`). The Wiener-SVD regularisation smears the unfolded spectra (matrices A_c of the data release), which is
not undone here; it is small compared with the energy resolution of liquid-scintillator detectors.
"""
struct DayaBayUnfolded <: SpectrumModel end

const DYB_FRACTIONS = (U235 = 0.564, U238 = 0.076, Pu239 = 0.304, Pu241 = 0.056)

isotope_spectrum(iso::Symbol, E) = (a = HM_COEFFS[iso]; exp(a[1] + E * (a[2] + E * (a[3] + E * (a[4] + E * (a[5] + E * a[6]))))))

spectrum(E, fractions) = sum(fractions[i] * isotope_spectrum(i, E) for i in ISOTOPES)

# ───────────────────────────── Daya Bay unfolded spectra ─────────────────────────────

# unfolded spectra (ν̄e / fission / MeV, IBD cross section removed) of the data release: (centres, edges, U235, combo, total)
function read_dyb_unfolded()
    lines = readlines(joinpath(@__DIR__, "reactors", "dayabay_unfolded", "UnfoldAntineutrinoSpectra.txt"))
    blocks = Vector{Vector{Float64}}[]
    current = nothing
    for l in lines
        if occursin("Bin_Center[MeV]", l)
            current = Vector{Float64}[]; push!(blocks, current)
        elseif startswith(l, "#") || isempty(strip(l))
            current = nothing
        elseif current !== nothing
            push!(current, parse.(Float64, split(l)))
        end
    end
    length(blocks) == 3 || error("unexpected Daya Bay unfolded-spectrum file format")
    centres = [r[1] for r in blocks[1]]
    edges = vcat(collect(1.8:0.25:7.8), 9.5)
    (centres = centres, edges = edges, U235 = [r[3] for r in blocks[1]], combo = [r[3] for r in blocks[2]], total = [r[3] for r in blocks[3]])
end

# mean of f over [a, b] (Simpson, 64 intervals)
function bin_mean(f, a, b; n = 64)
    h = (b - a) / n
    (f(a) + f(b) + sum((isodd(k) ? 4 : 2) * f(a + k * h) for k in 1:n-1)) * h / 3 / (b - a)
end

# piecewise-linear interpolation through (x, y), constant beyond the end points
function _interp_const(x, y, xi)
    xi <= x[1] && return y[1] * one(xi)
    xi >= x[end] && return y[end] * one(xi)
    j = searchsortedlast(x, xi)
    y[j] + (y[j+1] - y[j]) * (xi - x[j]) / (x[j+1] - x[j])
end

function dyb_unfolded_model()
    t = read_dyb_unfolded()
    hm_total(E) = spectrum(E, DYB_FRACTIONS)
    hm_combo(E) = isotope_spectrum(:Pu239, E) + 0.183 * isotope_spectrum(:Pu241, E)
    # node values r at the bin centres such that the bin means of hm(E) × interp(r)(E) equal the unfolded bin contents
    # (a linear system; interpolating the plain bin ratios would miss the bin contents by up to 5 % around the 5 MeV bump)
    hat(j, E) = _interp_const(t.centres, [Float64(k == j) for k in eachindex(t.centres)], E)
    function ratio(data, hm)
        M = [bin_mean(E -> hm(E) * hat(j, E), t.edges[b], t.edges[b+1]) for b in eachindex(data), j in eachindex(t.centres)]
        M \ data
    end
    r235, rcombo, rtotal = ratio(t.U235, E -> isotope_spectrum(:U235, E)), ratio(t.combo, hm_combo), ratio(t.total, hm_total)
    S235(E) = isotope_spectrum(:U235, E) * _interp_const(t.centres, r235, E)
    Scombo(E) = hm_combo(E) * _interp_const(t.centres, rcombo, E)
    Stotal(E) = hm_total(E) * _interp_const(t.centres, rtotal, E)
    function spec(E, f)
        d = NamedTuple{ISOTOPES}(Tuple(f[i] - DYB_FRACTIONS[i] for i in ISOTOPES))
        Stotal(E) + d.U235 * S235(E) + d.Pu239 * Scombo(E) + d.U238 * isotope_spectrum(:U238, E) +
            (d.Pu241 - 0.183 * d.Pu239) * isotope_spectrum(:Pu241, E)
    end
    # per isotope, consistent with Eq. 9 for the isotopic spectra (²³⁹Pu = combo − 0.183 ²⁴¹Pu)
    iso(i::Symbol, E) = i === :U235 ? S235(E) : i === :Pu239 ? Scombo(E) - 0.183 * isotope_spectrum(:Pu241, E) : isotope_spectrum(i, E)
    (; spec, iso, ratios = (centres = t.centres, U235 = r235, combo = rcombo, total = rtotal))
end

"Mean energy released per fission [MeV]."
energy_per_fission(fractions) = sum(fractions[i] * ENERGY_PER_FISSION[i] for i in ISOTOPES)

"Fissions per second at thermal power `P_MW`."
fission_rate(P_MW, fractions) = P_MW * 1e6 / (energy_per_fission(fractions) * MEV_TO_J)

"IBD cross section [cm²], Strumia & Vissani (2003), Eq. (25)."
function ibd_xsec(E)
    E <= IBD_THRESHOLD && return zero(E)
    Ee = E - DELTA_NP
    pe = sqrt(max(Ee^2 - M_E^2, zero(E)))
    lE = log(E)
    1e-43 * pe * Ee * E^(-0.07056 + 0.02018 * lE - 0.001953 * lE^3)
end

"""
    positron_energy(E) -> Float64

Mean total positron energy [MeV] of IBD at antineutrino energy `E` [MeV], including the neutron recoil (Strumia & Vissani
(2003), Eq. (16): ⟨E_e⟩ ≈ (E₁ + E₂)/2 = E − δ − E^CM_ν E^CM_e / m_p, better than 1 % below 100 MeV). The prompt (visible)
energy of the event is `positron_energy(E) + M_E`.
"""
function positron_energy(E)
    s = M_P^2 + 2 * M_P * E; rs = sqrt(s)
    E - (M_N^2 - M_P^2 - M_E^2) / (2 * M_P) - (s - M_P^2) / (2rs) * (s - M_N^2 + M_E^2) / (2rs) / M_P
end

"""
    neutrino_energy(E_prompt) -> Float64

Antineutrino energy [MeV] whose mean prompt energy `positron_energy(E) + M_E` is `E_prompt` [MeV] (inverse of
[`positron_energy`](@ref); E − E_prompt grows from 0.784 MeV at threshold to 0.84 MeV at E = 8 MeV).
"""
function neutrino_energy(E_prompt)
    E = E_prompt + DELTA_NP - M_E
    for _ in 1:5                               # Newton with dE_e/dE ≈ 1 − 2E/m_p
        E -= (positron_energy(E) + M_E - E_prompt) / (1 - 2E / M_P)
    end
    E
end

"""
    configure(model::SpectrumModel = HuberMueller()) -> ReactorFlux

Reactor flux module with the Huber–Mueller ([`HuberMueller`](@ref), default) or the Daya Bay data-driven
([`DayaBayUnfolded`](@ref)) antineutrino spectra.
"""
configure(model::HuberMueller = HuberMueller()) =
    ReactorFlux(model = model, isotope_spectrum = isotope_spectrum, spectrum = spectrum, rate_scale = bugey4_scale,
                ibd_xsec = ibd_xsec, fission_rate = fission_rate)

function configure(model::DayaBayUnfolded)
    m = dyb_unfolded_model()
    ReactorFlux(model = model, isotope_spectrum = m.iso, spectrum = m.spec, rate_scale = f -> 1.0,
                ibd_xsec = ibd_xsec, fission_rate = fission_rate)
end

"IBD yield per fission ∫ S(E) σ(E) dE [cm²/fission] of the Huber–Mueller model."
function ibd_yield(fractions; Emax = 12.0, n = 4000)
    E = range(IBD_THRESHOLD, Emax, length = n + 1); h = step(E)
    h * sum((k == 1 || k == n + 1 ? 0.5 : 1.0) * spectrum(e, fractions) * ibd_xsec(e) for (k, e) in enumerate(E))
end

"""
    bugey4_scale(fractions) -> Float64

Rate normalisation anchored to the Bugey-4 measurement (Declais et al. 1994: ⟨σ_f⟩ = 5.752×10⁻⁴³ cm²/fission at fission
fractions 0.538 : 0.078 : 0.328 : 0.056), corrected for the fuel difference with the Huber–Mueller isotope yields, relative
to the plain Huber–Mueller yield (KamLAND, PRD 88, 033001, Eq. 1).
"""
function bugey4_scale(fractions)
    b4 = (U235 = 0.538, U238 = 0.078, Pu239 = 0.328, Pu241 = 0.056)
    yi = Dict(i => ibd_yield(NamedTuple{ISOTOPES}(Tuple(Float64(j == i) for j in ISOTOPES))) for i in ISOTOPES)
    anchored = 5.752e-43 + sum((fractions[i] - b4[i]) * yi[i] for i in ISOTOPES)
    anchored / ibd_yield(fractions)
end

# ───────────────────────────── reactor database (IAEA PRIS: Japan, Korea) ─────────────────────────────

using CSV, DataFrames, Dates

const DBDIR = joinpath(@__DIR__, "reactors")

"Earth-centred coordinates [km] of a point at latitude/longitude [deg] and height `h` [km] (WGS84)."
function ecef(lat, lon, h = 0.0)
    a, f = 6378.137, 1 / 298.257223563; e2 = f * (2 - f)
    φ, λ = deg2rad(lat), deg2rad(lon); N = a / sqrt(1 - e2 * sin(φ)^2)
    ((N + h) * cos(φ) * cos(λ), (N + h) * cos(φ) * sin(λ), (N * (1 - e2) + h) * sin(φ))
end
baseline(p, q) = sqrt(sum(abs2, p .- q))

"""
    load_units(; countries) -> DataFrame

Power reactor units (PRIS) with thermal power, site coordinates and annual operating history (hours online, load factor).
"""
function load_units(; countries = ("JP", "KR"))
    units = CSV.read(joinpath(DBDIR, "units.csv"), DataFrame; comment = "#")
    sites = CSV.read(joinpath(DBDIR, "sites.csv"), DataFrame; comment = "#")
    units = innerjoin(units[in.(units.country, Ref(countries)), :], sites[:, [:site, :lat, :lon]], on = :site)
    hist = CSV.read(joinpath(DBDIR, "operating_history.csv"), DataFrame; comment = "#")
    units.history = [hist[hist.id .== id, :] for id in units.id]
    units
end

parse_date(s) = ismissing(s) || isempty(string(s)) ? nothing : Date(string(s)[1:10])

"""
    daily_power(unit, from::Date, to::Date) -> Vector{Float64}

Daily mean thermal power [MW] of a unit, from its annual load factor (PRIS): within a year the operating time is
placed contiguously before a suspension/shutdown date in that year, at the end of the year after a year without
operation (restart), at the start of the year before a year without operation, and uniformly otherwise.
"""
function daily_power(u, from::Date, to::Date)
    days = from:Day(1):to
    out = zeros(length(days))
    h = u.history
    stop = filter(!isnothing, [parse_date(u.suspended_date), parse_date(u.shutdown_date)])
    for y in year(from):year(to)
        r = h[h.year .== y, :]
        nrow(r) == 0 && continue
        lf = r.load_factor[1] / 100
        lf <= 0 && continue
        y0, y1 = Date(y, 1, 1), Date(y, 12, 31); ndays = Dates.value(y1 - y0) + 1
        energy_days = lf * ndays                                        # full-power days in year y
        prev = h[h.year .== y - 1, :]; next = h[h.year .== y + 1, :]
        off_prev = nrow(prev) > 0 && prev.load_factor[1] <= 0
        off_next = nrow(next) > 0 && next.load_factor[1] <= 0
        s = filter(d -> year(d) == y, stop)
        n_on = clamp(round(Int, r.hours_online[1] / 24), 1, ndays)
        if !isempty(s)
            last_day = minimum(s); first_day = max(y0, last_day - Day(n_on - 1))
        elseif off_prev
            first_day = y1 - Day(n_on - 1); last_day = y1
        elseif off_next
            first_day = y0; last_day = y0 + Day(n_on - 1)
        else
            first_day, last_day = y0, y1
        end
        on = first_day:Day(1):last_day
        p = u.thermal_power_MW * energy_days / length(on)
        for d in on
            from <= d <= to && (out[Dates.value(d - from) + 1] += p)
        end
    end
    out
end

end
