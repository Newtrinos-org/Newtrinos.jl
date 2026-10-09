"""
Reactor ν̄e flux and inverse-β-decay (IBD) detection, common to reactor experiments.

- Isotope spectra (ν̄e per fission per MeV) of ²³⁵U, ²³⁹Pu, ²⁴¹Pu (Huber, PRC 84, 024617 (2011)) and ²³⁸U (Mueller et
  al., PRC 83, 054615 (2011)), exponential-of-polynomial fits valid for 2 ≲ E_ν ≲ 8 MeV (the Huber–Mueller model).
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
- `isotope_spectrum(iso, E)`: ν̄e per fission per MeV of isotope `iso` ∈ `ISOTOPES` at energy `E` [MeV].
- `spectrum(E, fractions)`: ν̄e per fission per MeV for fission fractions `fractions` (NamedTuple over `ISOTOPES`).
- `ibd_xsec(E)`: IBD cross section [cm²].
- `fission_rate(P_MW, fractions)`: fissions per second at thermal power `P_MW` [MW].
"""
@kwdef struct ReactorFlux <: Newtrinos.Physics
    params::NamedTuple = NamedTuple()
    priors::NamedTuple = NamedTuple()
    isotope_spectrum::Function
    spectrum::Function
    ibd_xsec::Function
    fission_rate::Function
end

isotope_spectrum(iso::Symbol, E) = (a = HM_COEFFS[iso]; exp(a[1] + E * (a[2] + E * (a[3] + E * (a[4] + E * (a[5] + E * a[6]))))))

spectrum(E, fractions) = sum(fractions[i] * isotope_spectrum(i, E) for i in ISOTOPES)

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

configure() = ReactorFlux(isotope_spectrum = isotope_spectrum, spectrum = spectrum, ibd_xsec = ibd_xsec, fission_rate = fission_rate)

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
