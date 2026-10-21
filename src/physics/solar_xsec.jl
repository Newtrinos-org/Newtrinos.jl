module solar_xsec

using DelimitedFiles
using Interpolations
using Distributions
using ..Newtrinos

const datadir = joinpath(@__DIR__, "solar")

const M_E = 0.51099895        # electron mass [MeV]
const G_F = 1.1663788e-11     # Fermi constant [MeV⁻²]
const HBARC2 = 3.893794e-22   # (ħc)² [MeV² cm²]
const SIN2_W = 0.23129        # sin²θ_W (MS-bar, M_Z)
const RHO_NC = 1.0126         # ρ_NC (Bahcall, Kamionkowski & Sirlin 1995)
const SIGMA_0 = 2 * G_F^2 * M_E / π * HBARC2  # [cm² MeV⁻¹]

const CL_THRESHOLD = 0.814    # [MeV]
const GA_THRESHOLD = 0.2332   # [MeV]
const CL_XSEC_UNC = 0.033     # 1σ, from the ⁸B cross section (Bahcall et al. 1996)

"""
    SolarXsec <: Newtrinos.Physics

Detection cross sections for solar neutrino experiments.

# Fields
- `params`, `priors`: `cl_xsec_sigma` and `ga_xsec_sigma`, cross-section uncertainties of the
  radiochemical experiments in units of standard deviations (prior `Normal(0, 1)`).
- `dσdT_es::Function`: `dσdT_es(flavour, E, T)`, differential ν–e elastic scattering cross
  section [cm² MeV⁻¹] for neutrino energy `E` and electron recoil kinetic energy `T` [MeV];
  `flavour` is `:e` or `:x` (νμ or ντ).
- `capture::Function`: `capture(target, E, params)`, absorption cross section [cm²] on
  `target` `:cl37` or `:ga71` at neutrino energy `E` [MeV].
"""
@kwdef struct SolarXsec <: Newtrinos.Physics
    params::NamedTuple
    priors::NamedTuple
    dσdT_es::Function
    capture::Function
end

"""
    configure() -> SolarXsec

Create the solar detection cross-section module.
"""
function configure()
    SolarXsec(
        params = (cl_xsec_sigma = 0.0, ga_xsec_sigma = 0.0),
        priors = (cl_xsec_sigma = Truncated(Normal(0.0, 1.0), -3, 3), ga_xsec_sigma = Truncated(Normal(0.0, 1.0), -3, 3)),
        dσdT_es = dσdT_es,
        capture = get_capture(),
    )
end

"""
    t_max(E) -> Real

Maximum electron recoil kinetic energy [MeV] in ν–e scattering at neutrino energy `E` [MeV].
"""
t_max(E) = 2E^2 / (M_E + 2E)

# Bahcall, Kamionkowski & Sirlin, Phys. Rev. D 51, 6146 (1995) (BKS), Eqs. (A4), (A5) and (A7):
# κ(T) of νe–e and of νμ,τ–e scattering
function κ_ES(flavour::Symbol, T)
    x = sqrt(1 + 2M_E / T)
    I = (1 / 3 + (3 - x^2) * (x / 2 * log((x + 1) / (x - 1)) - 1)) / 6
    flavour == :e ? 0.9791 + 0.0097 * I : 0.9970 - 0.00037 * I
end

const ALPHA = 1 / 137.035999

# dilogarithm Li₂(x) for 0 ≤ x ≤ 1 (BKS's Spence function is L(x) = -Li₂(x))
function li2(x)
    iszero(x) && return zero(x)
    x == 1 && return oftype(x, π^2 / 6)
    x > 0.5 && return π^2 / 6 - log(x) * log1p(-x) - li2(1 - x)
    s = zero(x); p = one(x)
    for k in 1:60
        p *= x; s += p / k^2
    end
    s
end

# QED corrections of BKS, Eqs. (B3)–(B5): f₋(z), (1 - z)² f₊(z) and f₊₋(z) for neutrino energy E and recoil
# kinetic energy T (z = T/E). The log divergence at the end point is integrable.
function qed_es(E, T)
    Ee = T + M_E; ℓ = sqrt(T * (T + 2M_E)); β = ℓ / Ee; z = T / E
    c = Ee / ℓ * log1p((T + ℓ) / M_E) - 1
    a = 1 - z - M_E / (Ee + ℓ)
    a <= 0 && return (zero(z), zero(z), zero(z))
    la, l1, lz = log(a), log1p(-z), log(z)
    L(x) = -li2(x)
    fm = c * (2la - l1 - lz / 2 - 5 / 12) + (L(z) - L(β)) / 2 - l1^2 / 2 - (11 / 12 + z / 2) * l1 +
         z * (lz + log(2E / M_E) / 2) - (31 / 18 + lz / 12) * β - 11z / 12 + z^2 / 24
    fp = c * ((1 - z)^2 * (2la - l1 - lz / 2 - 2 / 3) - (z^2 * lz + 1 - z) / 2) -
         (1 - z)^2 / 2 * (l1^2 + β * (L(1 - z) - lz * l1)) +
         l1 * (z^2 / 2 * lz + (1 - z) / 3 * (2z - 1 / 2)) -
         z^2 / 2 * L(1 - z) - z * (1 - 2z) / 3 * lz - z * (1 - z) / 6 -
         β / 12 * (lz + (1 - z) * (115 - 109z) / 6)
    fm, fp, c * 2la
end

"""
    dσdT_es(flavour, E, T) -> Real

Differential ν–e elastic scattering cross section [cm² MeV⁻¹].

Includes the electroweak (ρ_NC, κ(T)) and QED radiative corrections of Bahcall, Kamionkowski &
Sirlin, Phys. Rev. D 51, 6146 (1995), Eq. (A1). `flavour` is `:e` (charged + neutral current) or
`:x` (νμ, ντ; neutral current only).
"""
function dσdT_es(flavour::Symbol, E, T)
    (T <= 0 || T >= t_max(E)) && return zero(promote_type(typeof(E), typeof(T)))
    κs = κ_ES(flavour, T) * SIN2_W
    gL = RHO_NC * (0.5 - κs) - (flavour == :e ? 1 : 0)
    gR = -RHO_NC * κs
    z = T / E
    fm, fp, fpm = qed_es(E, T)
    r = ALPHA / π
    SIGMA_0 * (gL^2 * (1 + r * fm) + gR^2 * ((1 - z)^2 + r * fp) - gL * gR * M_E * T / E^2 * (1 + r * fpm))
end

function read_table(name)
    data, header = readdlm(joinpath(datadir, name), ',', Float64, '\n'; header=true, comments=true)
    data
end

# Linear interpolation that is zero below threshold; between threshold and the first tabulated
# point (if above threshold) the first value is used. Above the table (30 MeV, beyond all solar
# neutrinos) it is zero.
function threshold_interpolation(E, σ, threshold)
    E[1] <= threshold && return linear_interpolation(E, σ, extrapolation_bc=0.0)
    linear_interpolation(vcat(threshold, E), vcat(σ[1], σ), extrapolation_bc=0.0)
end

# Interpolation of log σ in log(E - threshold): σ rises as a power of the kinetic energy above
# threshold, which linear interpolation of a coarse table overestimates. Below the first tabulated
# point the first power law is extrapolated down to threshold; zero above the table.
function powerlaw_interpolation(E, σ, threshold)
    itp = linear_interpolation(log.(E .- threshold), log.(σ), extrapolation_bc=Line())
    E_max = E[end]
    e -> (e <= threshold || e > E_max) ? zero(float(e)) : exp(itp(log(e - threshold)))
end

"""
    get_capture() -> Function

Absorption cross sections on ³⁷Cl (Bahcall et al. 1996) and ⁷¹Ga (Bahcall 1997), with
uncertainties steered by `cl_xsec_sigma` (fully correlated 3.3% scale) and `ga_xsec_sigma`
(interpolation towards the energy-dependent ±3σ tables).
"""
function get_capture()
    cl = read_table("xsec_cl37.csv")
    ga = read_table("xsec_ga71.csv")
    σ_cl = powerlaw_interpolation(cl[2:end, 1], cl[2:end, 2] .* 1e-46, CL_THRESHOLD)
    σ_ga = threshold_interpolation(ga[:, 1], ga[:, 2] .* 1e-46, GA_THRESHOLD)
    σ_ga_m3 = threshold_interpolation(ga[:, 1], ga[:, 3] .* 1e-46, GA_THRESHOLD)
    σ_ga_p3 = threshold_interpolation(ga[:, 1], ga[:, 4] .* 1e-46, GA_THRESHOLD)
    function capture(target::Symbol, E, params)
        if target == :cl37
            return σ_cl(E) * (1 + CL_XSEC_UNC * params.cl_xsec_sigma)
        elseif target == :ga71
            s = params.ga_xsec_sigma
            σ = σ_ga(E)
            return σ + s / 3 * (s >= 0 ? σ_ga_p3(E) - σ : σ - σ_ga_m3(E))
        end
        throw(ArgumentError("unknown target $target, expected :cl37 or :ga71"))
    end
end

end
