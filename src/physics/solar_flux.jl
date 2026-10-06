module solar_flux

using DelimitedFiles
using Interpolations
using Distributions
using LinearAlgebra
using ..Newtrinos

export SolarFluxConfig, B23, SSMPriors

const datadir = joinpath(@__DIR__, "solar")

"""
Solar neutrino flux components, in the order used by the B23 tables.
"""
const COMPONENTS = (:pp, :pep, :hep, :be7, :b8, :n13, :o15, :f17)

"""
Mono-energetic components: line energies [MeV] and branching ratios.
"""
const LINES = (
    be7 = (E = [0.8613, 0.3843], br = [0.8956, 0.1044]),
    pep = (E = [1.442], br = [1.0]),
)

"""
    SolarModel

Abstract type for standard solar models, providing total neutrino fluxes per component,
their uncertainties and correlations, and the solar structure (electron density and
neutrino production distributions).
"""
abstract type SolarModel end

"""
    B23 <: SolarModel

B23 standard solar models (Herrera & Serenelli 2023, doi:10.5281/zenodo.10174170), computed
with SF-III nuclear rates. This is the solar model used in NuFIT 6.0.

# Fields
- `composition::Symbol = :MB22m`: solar surface composition, one of `:MB22m` (Magg et al.
  2022, meteoritic; NuFIT 6.0 reference), `:MB22p`, `:AAG21`, `:GS98`, `:AGSS09`, `:C11`.
"""
@kwdef struct B23 <: SolarModel
    composition::Symbol = :MB22m
end

"""
    SolarFluxSystematics

Abstract type for the treatment of solar flux uncertainties.
"""
abstract type SolarFluxSystematics end

"""
    SSMPriors <: SolarFluxSystematics

One normalisation parameter per flux component (`solar_norm_pp`, ..., nominal 1) with a
Gaussian prior of the solar-model fractional uncertainty, truncated to positive values, plus
`solar_b8_shape` (prior `Normal(0, 1)`) shifting the ⁸B spectral shape by its ±1σ variation
(positive values give a harder spectrum).

The priors are independent; the solar-model correlations are available via
[`ssm_prior`](@ref).
"""
struct SSMPriors <: SolarFluxSystematics end

"""
    SolarFluxConfig{M<:SolarModel, S<:SolarFluxSystematics}

# Fields
- `model::M = B23()`: standard solar model.
- `systematics::S = SSMPriors()`: flux uncertainty treatment.
- `n_production::Int = 40`: number of radial sample points per component used to average the
  oscillation probability over the production region.
"""
@kwdef struct SolarFluxConfig{M<:SolarModel, S<:SolarFluxSystematics}
    model::M = B23()
    systematics::S = SSMPriors()
    n_production::Int = 40
end

"""
    SolarFlux <: Newtrinos.Physics

Configured solar neutrino flux model.

# Fields
- `cfg::SolarFluxConfig`: configuration.
- `params`, `priors`: flux nuisance parameters and their priors.
- `nominal::NamedTuple`: total flux per component [cm⁻² s⁻¹].
- `fractional_error::NamedTuple`: solar-model fractional uncertainty per component.
- `correlation::Matrix{Float64}`: solar-model correlation matrix, in [`COMPONENTS`](@ref) order.
- `production::NamedTuple`: per component, the production region `(r, ne, nn, w)` with `r` in
  R☉, electron and neutron densities in mol/cm³ and weights summing to 1; the format expected
  by `Newtrinos.osc` `solar_prob`.
- `flux::Function`: `flux(comp, params)`, total flux of a component [cm⁻² s⁻¹].
- `spectrum::Function`: `spectrum(comp, E, params)`, differential flux dΦ/dE
  [cm⁻² s⁻¹ MeV⁻¹] of a continuous component at energies `E` [MeV].
- `lines::NamedTuple`: energies [MeV] and branching ratios of the mono-energetic components.
"""
@kwdef struct SolarFlux <: Newtrinos.Physics
    cfg::SolarFluxConfig
    params::NamedTuple
    priors::NamedTuple
    nominal::NamedTuple
    fractional_error::NamedTuple
    correlation::Matrix{Float64}
    production::NamedTuple
    flux::Function
    spectrum::Function
    lines::NamedTuple
end

"""
    configure(cfg::SolarFluxConfig=SolarFluxConfig()) -> SolarFlux

Create a configured solar flux physics module.

# Examples
```julia
sf = Newtrinos.solar_flux.configure()
sf.spectrum(:b8, [5.0, 10.0], sf.params)          # dΦ/dE [cm⁻² s⁻¹ MeV⁻¹]
osc.solar_prob(E_GeV, sf.production.b8, osc_params) # oscillation probability
```
"""
function configure(cfg::SolarFluxConfig=SolarFluxConfig())
    nominal, frac_err, corr = read_fluxes(cfg.model)
    shapes = read_spectra()
    SolarFlux(
        cfg = cfg,
        params = get_params(cfg.systematics),
        priors = get_priors(cfg.systematics, frac_err),
        nominal = nominal,
        fractional_error = frac_err,
        correlation = corr,
        production = read_production(cfg.model, cfg.n_production),
        flux = get_flux(cfg.systematics, nominal),
        spectrum = get_spectrum(cfg.systematics, nominal, shapes),
        lines = LINES,
    )
end

norm_param(comp::Symbol) = Symbol(:solar_norm_, comp)

get_params(::SSMPriors) = merge(NamedTuple(norm_param(c) => 1.0 for c in COMPONENTS), (solar_b8_shape = 0.0,))

function get_priors(::SSMPriors, frac_err)
    merge(NamedTuple(norm_param(c) => truncated(Normal(1.0, frac_err[c]), 0.0, 1.0 + 5 * frac_err[c]) for c in COMPONENTS),
          (solar_b8_shape = Truncated(Normal(0.0, 1.0), -3, 3),))
end

"""
    ssm_prior(sf::SolarFlux) -> MvNormal

Correlated solar-model prior on the flux normalisations `solar_norm_*` (in
[`COMPONENTS`](@ref) order), for use with `Newtrinos.correlated_priors_vars`.
"""
function ssm_prior(sf::SolarFlux)
    σ = [sf.fractional_error[c] for c in COMPONENTS]
    MvNormal(ones(length(σ)), Symmetric(Diagonal(σ) * sf.correlation * Diagonal(σ)))
end

"""
    read_fluxes(model::B23) -> (nominal, fractional_error, correlation)

Read total fluxes [cm⁻² s⁻¹], fractional uncertainties and the correlation matrix.
"""
function read_fluxes(model::B23)
    lines = filter(!isempty, strip.(readlines(joinpath(datadir, "b23_$(model.composition)_fluxes.dat"))))
    n = length(COMPONENTS)
    flux = parse.(Float64, split(lines[3])[1:n])
    err = parse.(Float64, split(lines[4])[1:n])
    corr_rows = filter(l -> startswith(l, "+") || (startswith(l, "-") && !startswith(l, "--")), lines)
    corr = reduce(vcat, [permutedims(parse.(Float64, split(l)[1:n])) for l in corr_rows])
    size(corr) == (n, n) || error("unexpected B23 flux file format")
    NamedTuple{COMPONENTS}(flux), NamedTuple{COMPONENTS}(err), corr
end

"""
    read_production(model::B23, n) -> NamedTuple

Production regions per component, compressed to `n` sample points of equal production
probability (quantiles of the radial distribution), each carrying the production-weighted mean
electron and neutron densities of its quantile bin.
"""
function read_production(model::B23, n)
    data, header = readdlm(joinpath(datadir, "b23_$(model.composition)_structure.csv"), ',', Float64, '\n'; header=true, comments=true)
    col(name) = data[:, findfirst(==(name), vec(header))]
    r, ne, ρ, X = col("r"), col("ne"), col("rho"), col("X")
    nn = ρ .* (1 .- X) ./ 2  # neutrons per nucleon ≈ ½ for everything but hydrogen
    NamedTuple{COMPONENTS}(compress_production(r, ne, nn, col(String(c)), n) for c in COMPONENTS)
end

function compress_production(r, ne, nn, dndr, n)
    w = dndr ./ sum(dndr)
    cdf = cumsum(w)
    bin = clamp.(ceil.(Int, cdf .* n .- 1e-9), 1, n)
    out = (r = zeros(n), ne = zeros(n), nn = zeros(n), w = zeros(n))
    for (i, b) in enumerate(bin)
        out.w[b] += w[i]
        out.r[b] += w[i] * r[i]
        out.ne[b] += w[i] * ne[i]
        out.nn[b] += w[i] * nn[i]
    end
    keep = out.w .> 0
    (r = out.r[keep] ./ out.w[keep], ne = out.ne[keep] ./ out.w[keep], nn = out.nn[keep] ./ out.w[keep], w = out.w[keep])
end

function read_shape(name)
    data, header = readdlm(joinpath(datadir, "spectrum_$(name).csv"), ',', Float64, '\n'; header=true, comments=true)
    E = data[:, 1]
    norm = sum(diff(E) .* (data[1:end-1, 2] .+ data[2:end, 2]) ./ 2)
    [linear_interpolation(E, data[:, k] ./ norm, extrapolation_bc=0.0) for k in 2:size(data, 2)]
end

"""
    read_spectra() -> NamedTuple

Normalised spectral shapes [MeV⁻¹] of the continuous components as interpolators. For ⁸B,
the best estimate and the ±3σ shape variations (Bahcall et al. 1996).
"""
function read_spectra()
    (pp = only(read_shape("pp")), hep = only(read_shape("hep")), n13 = only(read_shape("n13")),
     o15 = only(read_shape("o15")), f17 = only(read_shape("f17")), b8 = read_shape("b8"))
end

function get_flux(::SSMPriors, nominal)
    flux(comp::Symbol, params) = nominal[comp] * params[norm_param(comp)]
end

function get_spectrum(::SSMPriors, nominal, shapes)
    function spectrum(comp::Symbol, E, params)
        Φ = nominal[comp] * params[norm_param(comp)]
        if comp == :b8
            # positive shifts towards the harder spectrum (Bahcall's "-3σ" column)
            best, soft3, hard3 = shapes.b8
            s = params.solar_b8_shape
            return Φ .* (best.(E) .+ s .* (hard3.(E) .- soft3.(E)) ./ 6)
        end
        Φ .* shapes[comp].(E)
    end
end

"""
    nadir_exposure(latitude_deg; cz_edges=range(-1, 1, length=41), n_days=730, n_hours=720)
        -> NamedTuple

Yearly distribution of the Sun's direction at a detector at geographic latitude
`latitude_deg`, weighted by the solar neutrino flux (∝ 1/d², with d the Earth–Sun distance).

Returns `(cz_edges, cz, w)` where `w[k]` is the fraction of the yearly flux arriving with
cos(zenith) in bin `k`; `cz` is the cosine of the zenith angle of the Sun, i.e. of the
direction the neutrinos come from, so `cz < 0` is night (Earth crossing), in the convention
of `Newtrinos.earth_layers.compute_paths`.
"""
function nadir_exposure(latitude_deg; cz_edges=range(-1, 1, length=41), n_days=730, n_hours=720)
    φ = deg2rad(latitude_deg)
    ε = deg2rad(23.44)
    w = zeros(length(cz_edges) - 1)
    for t in range(0, 1, length=n_days + 1)[1:end-1]
        g = 2π * t  # mean anomaly
        λ = deg2rad(282.94) + g + deg2rad(1.915) * sin(g) + deg2rad(0.020) * sin(2g)
        δ = asin(sin(ε) * sin(λ))
        d = 1.00014 - 0.01671 * cos(g) - 0.00014 * cos(2g)
        for H in range(0, 2π, length=n_hours + 1)[1:end-1]
            cz = sin(φ) * sin(δ) + cos(φ) * cos(δ) * cos(H)
            k = clamp(searchsortedlast(cz_edges, cz), 1, length(w))
            w[k] += 1 / d^2
        end
    end
    (cz_edges = collect(cz_edges), cz = (cz_edges[1:end-1] .+ cz_edges[2:end]) ./ 2, w = w ./ sum(w))
end

end
