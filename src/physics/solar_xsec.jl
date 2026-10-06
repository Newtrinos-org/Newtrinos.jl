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

# Bahcall, Kamionkowski & Sirlin, Phys. Rev. D 51, 6146 (1995), eq. (B3)
function κ_ES(T)
    x = sqrt(1 + 2M_E / T)
    I = (1 / 3 + (3 - x^2) * (x / 2 * log((x + 1) / (x - 1)) - 1)) / 6
    0.9791 + 0.0097 * I
end

"""
    dσdT_es(flavour, E, T) -> Real

Differential ν–e elastic scattering cross section [cm² MeV⁻¹].

Includes the electroweak radiative corrections of Bahcall, Kamionkowski & Sirlin (1995)
through ρ_NC and κ(T); the QED corrections (≲ 2% on the recoil spectrum) are neglected.
`flavour` is `:e` (charged + neutral current) or `:x` (νμ, ντ; neutral current only).
"""
function dσdT_es(flavour::Symbol, E, T)
    (T <= 0 || T >= t_max(E)) && return zero(promote_type(typeof(E), typeof(T)))
    κs = κ_ES(T) * SIN2_W
    gL = RHO_NC * (0.5 - κs) - (flavour == :e ? 1 : 0)
    gR = -RHO_NC * κs
    z = T / E
    SIGMA_0 * (gL^2 + gR^2 * (1 - z)^2 - gL * gR * M_E * T / E^2)
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
