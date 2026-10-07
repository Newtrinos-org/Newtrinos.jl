module solar_common

using LinearAlgebra
using StructArrays
using ArraysOfArrays
using CairoMakie
using SpecialFunctions: erf
import ..Newtrinos

"""
    default_physics() -> NamedTuple

Default physics for solar neutrino experiments: three-flavour oscillations with matter
effects (MSW in the Sun and Earth regeneration), the B23 MB22-met solar model with
solar-model flux priors, solar detection cross sections and the PREM Earth with zones at its
density discontinuities (`earth_layers.PREM_discontinuities`), used with chord-averaged
densities by [`Site`](@ref). All solar detectors are in continental rock, so PREM's ocean layer
is replaced by upper crust (`continental = true`).
"""
function default_physics()
    osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction=Newtrinos.osc.SI(), eigen_method=Newtrinos.BargerEigen()))
    solar_flux = Newtrinos.solar_flux.configure()
    solar_xsec = Newtrinos.solar_xsec.configure()
    earth_layers = Newtrinos.earth_layers.configure(Newtrinos.earth_layers.PREM_discontinuities(continental=true))
    (; osc, solar_flux, solar_xsec, earth_layers)
end

"""
    Site

Day/night exposure of a detector: the night is binned in cos(zenith) of the Sun, with Earth
paths to the detector precomputed; the day is a single weight (no Earth matter crossed).

# Fields
- `w_day::Float64`: fraction of the yearly flux arriving during the day.
- `cz_night`, `w_night`: night bin centres and flux fractions (summing to `1 - w_day`).
- `paths`: Earth paths for the night bins.
- `layers`: Earth layers.
- `E_night_min::Float64`: Earth regeneration is computed only for neutrino energies at or above
  this value [MeV]; below, the night probabilities equal the day ones. Regeneration grows
  roughly ∝ E and is ≲10⁻³ in P_ee below 2 MeV (Borexino: A_DN(⁷Be) = 0.001 ± 0.013).
"""
struct Site{P, L}
    w_day::Float64
    cz_night::Vector{Float64}
    w_night::Vector{Float64}
    paths::P
    layers::L
    E_night_min::Float64
end
Site(w_day, cz_night, w_night, paths, layers) = Site(w_day, cz_night, w_night, paths, layers, 2.0)

"""
    Site(physics, latitude_deg; depth_km=1.0, n_night=20, night_edges=range(-1, 0, length=n_night + 1)) -> Site

Exposure of a detector at geographic latitude `latitude_deg` and depth `depth_km` below the
surface, using `physics.solar_flux` for the Sun's yearly path and `physics.earth_layers` for
the Earth model. The night is binned with `night_edges` in cos(zenith) of the Sun (from -1 to
0).
"""
function Site(physics, latitude_deg; depth_km=1.0, n_night=20, night_edges=range(-1, 0, length=n_night + 1), E_night_min=2.0)
    n_night = length(night_edges) - 1
    ex = Newtrinos.solar_flux.nadir_exposure(latitude_deg; cz_edges=vcat(night_edges, [1.0]))
    layers = physics.earth_layers.compute_layers()
    cz_night = ex.cz[1:n_night]
    # densities averaged along each chord within each PREM zone, see `earth_layers.compute_chord_paths`
    paths, chord_layers = physics.earth_layers.compute_chord_paths(cz_night, layers; r_detector=layers.radius[2] - depth_km)
    Site(ex.w[end], cz_night, ex.w[1:n_night], paths, chord_layers, E_night_min)
end

"""
    survival(physics, site, comp, E, params; night=true) -> NamedTuple

Oscillation probabilities P(νe → νβ) for solar component `comp` at energies `E` [MeV],
returned as `(day, night)` with shapes `(n_E, n_flav)` (day) and `(n_E, n_night, n_flav)`
(night, only if `night=true`; otherwise `nothing`). Earth regeneration is computed for
energies of at least `site.E_night_min`.
"""
function survival(physics, site::Site, comp::Symbol, E, params; night=true)
    production = physics.solar_flux.production[comp]
    E_GeV = E .* 1e-3
    day = physics.osc.solar_prob(E_GeV, production, params)
    night || return (; day, night = nothing)
    high = findall(>=(site.E_night_min), E)
    isempty(high) && return (; day, night = repeat(reshape(day, size(day, 1), 1, size(day, 2)), 1, length(site.paths), 1))
    P_high = physics.osc.solar_prob(E_GeV[high], production, site.paths, site.layers, params)
    T = promote_type(eltype(day), eltype(P_high))
    n = Array{T}(undef, length(E), length(site.paths), size(day, 2))
    for ip in axes(n, 2)
        n[:, ip, :] .= day
    end
    n[high, :, :] .= P_high
    (; day, night = n)
end

"""
    averaged_prob(physics, site, comp, E, params) -> Matrix

Yearly (day and night) averaged P(νe → νβ) of shape `(n_E, n_flav)`.
"""
function averaged_prob(physics, site::Site, comp::Symbol, E, params)
    P = survival(physics, site, comp, E, params)
    P.day .* site.w_day .+ dropdims(sum(P.night .* reshape(site.w_night, 1, :, 1), dims=2), dims=2)
end

"""
    integration_grid(comp, E_min; n=60) -> (E, weights)

Trapezoidal integration grid [MeV] for a continuous solar component above `E_min`, up to the
spectral endpoint.
"""
function integration_grid(comp::Symbol, E_min; n=60)
    E_max = ENDPOINTS[comp]
    E_min >= E_max && return (Float64[], Float64[])
    E = collect(range(E_min, E_max, length=n))
    w = fill(step(range(E_min, E_max, length=n)), n)
    w[1] /= 2
    w[end] /= 2
    E, w
end

const ENDPOINTS = (pp = 0.4234, hep = 18.79, b8 = 16.56, n13 = 1.199, o15 = 1.732, f17 = 1.740)

"""
    capture_rate(physics, site, target, threshold, params; components=...) -> NamedTuple

Expected capture rate in SNU (10⁻³⁶ captures per target atom per second) in a radiochemical
experiment, per solar component, including oscillations averaged over day and night.

`target` is `:cl37` or `:ga71` (see `Newtrinos.solar_xsec`), `threshold` the capture
threshold [MeV].
"""
function capture_rate(physics, site::Site, target::Symbol, threshold, params;
                      components=Newtrinos.solar_flux.COMPONENTS)
    sf, xs = physics.solar_flux, physics.solar_xsec
    rates = map(components) do comp
        if haskey(sf.lines, comp)
            line = sf.lines[comp]
            P = averaged_prob(physics, site, comp, line.E, params)[:, 1]
            σ = [xs.capture(target, e, params) for e in line.E]
            sf.flux(comp, params) * sum(line.br .* P .* σ) / 1e-36
        else
            E, w = integration_grid(comp, threshold; n = comp in (:b8, :hep) ? 120 : 40)
            isempty(E) && return zero(eltype(sf.spectrum(comp, [1.0], params)))
            P = averaged_prob(physics, site, comp, E, params)[:, 1]
            σ = [xs.capture(target, e, params) for e in E]
            sum(w .* sf.spectrum(comp, E, params) .* P .* σ) / 1e-36
        end
    end
    NamedTuple{components}(rates)
end

"""
    ESResponse

Precomputed response of an elastic-scattering detector with Gaussian energy resolution.

For neutrino energies `E` (with trapezoidal weights `wE`) and reconstructed recoil kinetic
energy bins, `K[f][k, j] = ∫ dT dσ_f/dT(E_j, T) R_k(T)` [cm²] for `f = :e` (νe) and `:x`
(νμ,τ), where `R_k(T)` is the probability to reconstruct a recoil of kinetic energy `T` in bin
`k`. The response is tabulated at the nominal energy scale and resolution and at ±1σ of each
(`:scale_up`, `:scale_down`, `:reso_up`, `:reso_down`), and interpolated quadratically in the
corresponding nuisance parameters by [`es_rates`](@ref).
"""
struct ESResponse
    E::Vector{Float64}
    wE::Vector{Float64}
    K::Dict{Symbol, NamedTuple{(:e, :x), Tuple{Matrix{Float64}, Matrix{Float64}}}}
end

"""
    ESResponse(xs, E, bins, resolution; scale_unc, reso_unc, total_energy=false, T_step=0.02) -> ESResponse

Build the response for neutrino energies `E` [MeV] and reconstructed energy bins, given as a
vector of bin edges or of `(lo, hi)` tuples [MeV] in electron kinetic energy (or total energy
if `total_energy`), with resolution `resolution(E_total)` [MeV] (σ of the reconstructed
energy) and fractional energy-scale and resolution uncertainties `scale_unc`, `reso_unc`.
`xs` is a `Newtrinos.solar_xsec` module providing `dσdT_es`.
"""
function ESResponse(xs, E, bins, resolution; scale_unc, reso_unc, total_energy=false, T_step=0.02)
    me = Newtrinos.solar_xsec.M_E
    bins = eltype(bins) <: Tuple ? collect(bins) : [(bins[k], bins[k+1]) for k in 1:length(bins)-1]
    offset = total_energy ? me : 0.0  # reconstructed quantity = kinetic + offset
    wE = trapezoid_weights(E)
    lo_min = minimum(first.(bins)) - offset
    T_min = max(T_step, lo_min - 6 * resolution(lo_min + me))
    T = collect(range(T_min, Newtrinos.solar_xsec.t_max(maximum(E)), step=T_step))
    wT = trapezoid_weights(T)
    nbins = length(bins)
    function build(scale, reso)
        R = zeros(nbins, length(T))
        for (i, t) in enumerate(T)
            μ = (t + offset) * scale
            σ = resolution(t + me) * reso
            for (k, (lo, hi)) in enumerate(bins)
                R[k, i] = (erf((hi - μ) / (sqrt(2) * σ)) - erf((lo - μ) / (sqrt(2) * σ))) / 2
            end
        end
        dσe = [xs.dσdT_es(:e, e, t) for t in T, e in E]
        dσx = [xs.dσdT_es(:x, e, t) for t in T, e in E]
        (e = R * (wT .* dσe), x = R * (wT .* dσx))
    end
    K = Dict(:nominal => build(1.0, 1.0),
             :scale_up => build(1 + scale_unc, 1.0), :scale_down => build(1 - scale_unc, 1.0),
             :reso_up => build(1.0, 1 + reso_unc), :reso_down => build(1.0, 1 - reso_unc))
    ESResponse(collect(E), wE, K)
end

function trapezoid_weights(x)
    w = zeros(length(x))
    for i in 1:length(x)-1
        h = x[i+1] - x[i]
        w[i] += h / 2
        w[i+1] += h / 2
    end
    w
end

# quadratic interpolation through the -1σ, nominal and +1σ responses
interp_pull(K0, Kup, Kdown, s) = K0 .+ s .* (Kup .- Kdown) ./ 2 .+ s^2 .* (Kup .+ Kdown .- 2 .* K0) ./ 2

"""
    es_rates(resp, flux, P_e; scale=0, reso=0) -> Vector or Matrix

Elastic-scattering rate per reconstructed energy bin, `Σ_j wE_j flux_j [P_e,j K_e[k,j] +
(1 - P_e,j) K_x[k,j]]`, for differential flux `flux` and νe survival probability `P_e` at the
response energies, with energy-scale and resolution pulls `scale`, `reso` (in σ). `P_e` may be
a matrix with one column per set of probabilities (e.g. zenith bins), giving one column of rates
each. The units are those of `flux` times cm² (per target electron).
"""
function es_rates(resp::ESResponse, flux, P_e; scale=0.0, reso=0.0)
    K = resp.K
    Ke = K[:nominal].e .+ (interp_pull(K[:nominal].e, K[:scale_up].e, K[:scale_down].e, scale) .- K[:nominal].e) .+
         (interp_pull(K[:nominal].e, K[:reso_up].e, K[:reso_down].e, reso) .- K[:nominal].e)
    Kx = K[:nominal].x .+ (interp_pull(K[:nominal].x, K[:scale_up].x, K[:scale_down].x, scale) .- K[:nominal].x) .+
         (interp_pull(K[:nominal].x, K[:reso_up].x, K[:reso_down].x, reso) .- K[:nominal].x)
    w = resp.wE .* flux
    Ke * (w .* P_e) .+ Kx * (w .* (1 .- P_e))
end

"""
    ESTotal

Precomputed ν–e elastic-scattering cross sections, integrated over electron recoil kinetic
energies `T ≥ T_min`, for one solar component: at its line energies (weighted by branching
ratio) or on an integration grid of its continuous spectrum.
"""
struct ESTotal
    comp::Symbol
    line::Bool
    E::Vector{Float64}
    w::Vector{Float64}    # branching ratios (lines) or trapezoidal weights (continuum)
    σe::Vector{Float64}   # [cm²]
    σx::Vector{Float64}
end

"""
    ESTotal(physics, comp; T_min=0.0, line_energies=nothing, n=80) -> ESTotal

`line_energies` selects a subset of the lines of a mono-energetic component (e.g. only the
862 keV ⁷Be line).
"""
function ESTotal(physics, comp::Symbol; T_min=0.0, line_energies=nothing, n=80)
    sf, xs = physics.solar_flux, physics.solar_xsec
    function σ(flavour, e)
        Tmax = Newtrinos.solar_xsec.t_max(e)
        Tmax <= T_min && return 0.0
        T = range(T_min, Tmax, length=2001)
        sum(trapezoid_weights(T) .* xs.dσdT_es.(flavour, e, T))
    end
    if haskey(sf.lines, comp)
        sel = line_energies === nothing ? eachindex(sf.lines[comp].E) : findall(e -> any(isapprox.(e, line_energies, atol=1e-3)), sf.lines[comp].E)
        E, w = sf.lines[comp].E[sel], sf.lines[comp].br[sel]
        return ESTotal(comp, true, E, w, σ.(:e, E), σ.(:x, E))
    end
    # minimum neutrino energy for a recoil of kinetic energy T_min
    E_min = (T_min + sqrt(T_min * (T_min + 2 * Newtrinos.solar_xsec.M_E))) / 2
    E = collect(range(max(E_min, 1e-3), ENDPOINTS[comp], length=n))
    ESTotal(comp, false, E, trapezoid_weights(E), σ.(:e, E), σ.(:x, E))
end

"""
    es_interaction_rate(physics, site, est::ESTotal, params) -> Real

Yearly averaged (day and night) ν–e elastic-scattering rate per target electron [s⁻¹] of the
component of `est`, including oscillations.
"""
function es_interaction_rate(physics, site::Site, est::ESTotal, params)
    sf = physics.solar_flux
    Pe = averaged_prob(physics, site, est.comp, est.E, params)[:, 1]
    σ = Pe .* est.σe .+ (1 .- Pe) .* est.σx
    if est.line
        sf.flux(est.comp, params) * sum(est.w .* σ)
    else
        sum(est.w .* sf.spectrum(est.comp, est.E, params) .* σ)
    end
end

"""
    plot_measurements(labels, expected, observed, sigma, title; ylabel) -> Figure

Measured values with uncertainties against the expectation, one panel entry per measurement,
shown as ratio to the measured value.
"""
function plot_measurements(labels, expected, observed, sigma, title; ylabel="Expected / measured")
    expected = ForwardDiffValue.(expected)
    n = length(observed)
    f = Figure()
    ax = Axis(f[1, 1], xticks=(1:n, labels), ylabel=ylabel, title=title)
    errorbars!(ax, 1:n, ones(n), sigma ./ observed, whiskerwidth=10, color=:black)
    scatter!(ax, 1:n, ones(n), color=:black, label="Measured")
    scatter!(ax, 1:n, expected ./ observed, color=:red, marker=:diamond, markersize=14, label="Expected")
    axislegend(ax, position=:rt)
    f
end

"""
    asymmetric_variance(pred, obs, σ_up, σ_down) -> Real

Variance of a measurement `obs` with asymmetric errors as a linear function of the prediction
(R. Barlow, "Asymmetric errors", PHYSTAT 2003): `σ₊σ₋ + (σ₊ − σ₋)(pred − obs)`, bounded below.
"""
asymmetric_variance(pred, obs, σ_up, σ_down) = max(σ_up * σ_down + (σ_up - σ_down) * (pred - obs), 0.01 * σ_up * σ_down)

"""
    plot_rates(rates, observed, sigma, labels, title) -> Figure

Predicted rate per solar component (stacked) against the measured rates with uncertainties.
"""
function plot_rates(rates::NamedTuple, observed, sigma, labels, title)
    comps = collect(keys(rates))
    vals = Float64[ForwardDiffValue(v) for v in values(rates)]
    n = length(observed)
    f = Figure()
    ax = Axis(f[1, 1], xticks=(1:n, labels), ylabel="Rate (SNU)", title=title)
    xs = repeat(1:n, inner=length(comps))
    barplot!(ax, xs .- 0.15, repeat(vals, n), stack=repeat(eachindex(comps), n), color=repeat(eachindex(comps), n),
             width=0.3, colormap=:tab10, colorrange=(1, 10))
    errorbars!(ax, (1:n) .+ 0.15, observed, sigma, whiskerwidth=10)
    scatter!(ax, (1:n) .+ 0.15, observed, color=:black, label="Observed")
    elems = [PolyElement(color=Makie.to_colormap(:tab10)[i]) for i in eachindex(comps)]
    Legend(f[1, 2], elems, String.(comps), "Expected")
    ylims!(ax, 0, nothing)
    f
end

ForwardDiffValue(x::Real) = Float64(x)
ForwardDiffValue(x) = Float64(Newtrinos.osc.ForwardDiff.value(x))

end
