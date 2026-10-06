module earth_layers

using CSV, DataFrames
using StatsBase
using StaticArrays, ArraysOfArrays, StructArrays
using DataStructures
using Distributions

using ..Newtrinos
export configure
export PREM, VariableDensity, PREM_discontinuities

const datadir = @__DIR__

"""
    DensityModel

Abstract type for Earth density profile models.

Each subtype provides a recipe for dividing the Earth into concentric shells of constant
density. Currently the only implementations are [`PREM`](@ref) and [`VariableDensity`](@ref).
"""
abstract type DensityModel end

"""
    PREM <: DensityModel

Preliminary Reference Earth Model (Dziewonski & Anderson, 1981).

Reads the tabulated PREM density profile from `PREM_1s.csv` and groups radial shells into
zones defined by density boundaries.

# Fields
- `zones::Array{Float64} = [0., 4., 7.5, 12.5, 13.1]`: density boundaries [g/cm³]
  defining the constant-density zones. Adjacent PREM rows whose density falls within the
  same bin are averaged into a single [`Newtrinos.osc.Layer`](@ref).
- `p_fractions::Vector{Float64} = [0.496, 0.494, 0.468, 0.466]`: proton number fraction
  ``Y_p = N_p / (N_p + N_n)`` for each density zone (one value per interval defined by
  `zones`), rather than a single value assumed constant across all layers.
- `atm_heihgt::Float64 = 20.`: atmospheric shell thickness [km] added above the Earth's
  surface (density = 0).
"""
@kwdef struct PREM <: DensityModel
    zones::Array{Float64} = [0., 4., 7.5, 12.5, 13.1]
    p_fractions::Vector{Float64} = [0.496, 0.494, 0.468, 0.466]  # Ye per density zone
    atm_heihgt::Float64 = 20.
end

"""
    PREM_discontinuities() -> PREM

[`PREM`](@ref) zoning that follows the density discontinuities of the PREM table (ocean,
upper and lower crust, LID/low-velocity zone, transition zone, lower mantle, outer and inner
core), with the mantle gradient split into a few zones. Together with chord-averaged densities
([`get_compute_chord_paths`](@ref)) this reproduces the continuous PREM profile for Earth
regeneration of solar neutrinos, whose oscillation length in the Earth (~300 km) resolves the
crust and upper mantle.
"""
PREM_discontinuities() = PREM(
    zones = [0.0, 2.0, 2.75, 3.0, 3.45, 3.75, 4.0, 4.6, 5.0, 5.3, 5.6, 10.5, 11.5, 12.2, 13.1],
    p_fractions = [0.555, 0.495, 0.495, 0.495, 0.495, 0.495, 0.495, 0.495, 0.495, 0.495, 0.467, 0.467, 0.467, 0.467],
)

"""
    VariableDensity <: DensityModel

PREM density model exposing an overall Earth-density normalization nuisance parameter.

Wraps a [`PREM`](@ref) profile unchanged, but its [`configure`](@ref) method additionally
registers an `electron_density_scale` parameter and prior. Unlike [`PREM`](@ref)'s
`compute_layers`/`compute_paths` closures, `VariableDensity` does not itself rescale the
layers — the caller is responsible for applying [`scale_densities`](@ref) to the nominal
layers with `params.electron_density_scale` inside the forward model (see `calc_weights`
in `src/experiments/super_k/sk_atm_2023/super_k.jl`). This lets a
global fit float the overall normalization of the Earth's matter density (relevant to
matter-effect systematics in atmospheric neutrino oscillations) without refitting the
underlying PREM shell structure.

# Fields
- `prem::PREM = PREM()`: the underlying PREM density profile to be scaled.
"""
@kwdef struct VariableDensity <: DensityModel
    prem::PREM = PREM()
end

"""
    EarthLayers <: Newtrinos.Physics

Configured Earth density model, returned by [`configure`](@ref).

Provides the layer structure and path-computation functions needed by the oscillation
module's matter-effect calculations (see [`Newtrinos.osc.SI`](@ref)).

# Fields
- `cfg::DensityModel`: the density model used to build this module.
- `params::NamedTuple`: oscillation parameters.
- `priors::NamedTuple`: prior distributions.
- `compute_layers::Function`: closure `compute_layers() -> StructVector{Layer}` returning
  the concentric density shells.
- `compute_paths::Function`: function
  `compute_paths(cz, layers; r_detector) -> VectorOfVectors{Path}` computing the layer
  traversal for each cosine-zenith value.
"""
@kwdef struct EarthLayers <: Newtrinos.Physics
    cfg::DensityModel
    params::NamedTuple
    priors::NamedTuple
    compute_layers::Function
    compute_paths::Function
    compute_chord_paths::Function
end

"""
    configure(cfg::PREM=PREM()) -> EarthLayers

Create a fully configured Earth density physics module.

# Arguments
- `cfg::PREM`: density model (defaults to [`PREM`](@ref)).

# Returns
An [`EarthLayers`](@ref) instance with `compute_layers` and `compute_paths` closures and empty parameters and priors.

# Examples
```julia
using Newtrinos

earth = Newtrinos.earth_layers.configure()
layers = earth.compute_layers()
paths = earth.compute_paths([-1.0, -0.5, 0.0], layers)
```
"""
function configure(cfg::PREM=PREM())
    EarthLayers(
        cfg=cfg,
        params = (;),
        priors = (;),
        compute_layers = get_compute_layers(cfg),
        compute_paths = compute_paths,
        compute_chord_paths = get_compute_chord_paths(cfg)
        )
end

"""
    configure(cfg::VariableDensity) -> EarthLayers

Create an Earth density physics module with a floatable overall density normalization.

Uses the same `compute_layers`/`compute_paths` closures as [`configure(::PREM)`](@ref)
(built from `cfg.prem`), but additionally registers an `electron_density_scale`
parameter (nominal `1.0`, prior `Normal(1.0, 0.068)`). The scale is not applied by
`compute_layers` itself — callers that want it applied must pass the parameter to
[`scale_densities`](@ref) explicitly.

# Arguments
- `cfg::VariableDensity`: density model wrapping the [`PREM`](@ref) profile to be scaled.

# Returns
An [`EarthLayers`](@ref) instance with the `electron_density_scale` parameter and prior.
"""
function configure(cfg::VariableDensity)
    EarthLayers(
        cfg=cfg,
        params = (electron_density_scale = 1.0,),
        priors = (electron_density_scale = Normal(1.0, 0.068),),
        compute_layers = get_compute_layers(cfg.prem),
        compute_paths = compute_paths,
        compute_chord_paths = get_compute_chord_paths(cfg.prem)
        )
end


# Thickness-weighted (trapezoidal) mean density of a radial zone. The PREM table is not evenly
# spaced (1 km steps near the surface, up to 100 km deeper), so a plain mean over rows would
# over-weight the thin shallow layers. Zones end at density discontinuities, where the table lists
# the boundary radius twice, so the integral covers the zone's full radial extent.
function _radial_mean(radius, density)
    order = sortperm(radius)
    r, ρ = radius[order], density[order]
    ∫ρ = sum((r[k+1] - r[k]) * (ρ[k+1] + ρ[k]) / 2 for k in 1:length(r)-1)
    ∫ρ / (r[end] - r[1])
end

"""
    get_compute_layers(cfg::PREM) -> Function

Construct a closure that builds the Earth layer structure from the PREM density profile.

The returned function `compute_layers()` reads `PREM_1s.csv`, bins rows by the density
boundaries in `cfg.zones`, and returns a `StructVector{Layer}` with one entry per zone
plus an atmospheric shell.

# Arguments
- `cfg::PREM`: PREM configuration with zone boundaries, per-zone proton fractions, and
  atmosphere height.

# Returns
A zero-argument closure `compute_layers() -> StructVector{Layer}`.
"""
function get_compute_layers(cfg::PREM)
    function compute_layers()

        PREM = CSV.read(joinpath(datadir, "PREM_1s.csv"), DataFrame, header=["radius","depth","density","Vpv","Vph","Vsv","Vsh","eta","Q-mu","Q-kappa"])
        # density boundaries to define the constant density zones

        radii = Float64[]
        ave_densities = Float64[]

        push!(radii, 6371+cfg.atm_heihgt)
        push!(ave_densities, 0.)

        for i in 1:length(cfg.zones)-1
            mask = (PREM.density .< cfg.zones[i+1]) .& (PREM.density .>= cfg.zones[i])
            push!(radii, maximum(PREM.radius[mask]))
            push!(ave_densities, _radial_mean(PREM.radius[mask], PREM.density[mask]))
        end

        ye = vcat([0.5], cfg.p_fractions)  # prepend atmosphere Ye (density=0, so value irrelevant)
        layers = StructArray{Newtrinos.Layer}((radii, ave_densities .* ye, ave_densities .* (1 .- ye)))
    end
end

"""
    scale_densities(layers, scale) -> StructVector{Layer}

Rescale the proton (electron) density of every layer by `scale`, holding total density
(proton + neutron) fixed.

Used to apply the [`VariableDensity`](@ref) `electron_density_scale` nuisance parameter:
`p_density` is multiplied by `scale`, and `n_density` is adjusted (``\\Delta n = -\\Delta
p``) so `p_density + n_density` is unchanged. This shifts the electron number density
that drives the matter-effect potential without altering the total (PREM) mass density
profile.

# Arguments
- `layers::StructVector{Layer}`: nominal Earth density layers from `compute_layers()`.
- `scale`: multiplicative factor applied to each layer's proton (electron) density.

# Returns
A new `StructVector{Layer}` with rescaled `p_density`/`n_density`.
"""
function scale_densities(layers, scale)
    p_old = layers.p_density
    p = p_old .* scale
    n = layers.n_density .+ p_old .- p  # conserve total density: Δn = -Δp
    StructArray{Newtrinos.Layer}((layers.radius, p, n))
end

"""
    ray_circle_path_length(r, y, cz) -> Real

Compute the chord length of a ray through a circle of radius `r`.

The ray originates at radial distance `y` from the Earth's centre with direction
cosine `cz` (cosine of the zenith angle). Returns zero if the ray does not intersect
the circle or if the chord is shorter than 1 km (numerical noise filter).

# Arguments
- `r`: radius of the spherical shell [km].
- `y`: radial position of the detector [km].
- `cz`: cosine of the zenith angle (−1 = vertically upgoing through the core).

# Returns
Path length through the shell [km], or zero if no intersection.
"""
function ray_circle_path_length(r, y, cz)
    # Compute the discriminant
    disc = r^2 - y^2 + (y * cz)^2
    T = typeof(disc)

    if disc < 0
        return zero(T)  # No intersection
    end

    sqrt_disc = sqrt(disc)

    # Compute intersection points
    t1 = - y * cz - sqrt_disc
    t2 = - y * cz + sqrt_disc

    # Compute path length, ensuring we only count positive t-values
    L = max(zero(T), t2 - max(zero(T), t1))

    if L < 1
        return zero(T)
    end
    L
end

# ToDo: could probably skip layers smaller than few km and "absorb" those into the next outer layer

"""
    compute_paths(cz::Number, layers, r_detector) -> StructArray{Path}
    compute_paths(cz::AbstractArray, layers; r_detector=6369) -> VectorOfVectors{Path}

Compute the sequence of [`Newtrinos.osc.Path`](@ref) segments a neutrino traverses through the Earth.

For a single cosine-zenith value, determines which [`Newtrinos.osc.Layer`](@ref) shells are crossed
using [`ray_circle_path_length`](@ref), then builds an ordered list of segments. Layers
below the detector are traversed twice (entry and exit), while layers above the detector
are traversed once.

The array method broadcasts over multiple cosine-zenith values and returns a
`VectorOfVectors{Path}`.

# Arguments
- `cz`: cosine of the zenith angle (scalar or array). −1 is vertically upgoing.
- `layers::StructVector{Layer}`: Earth density layers from `compute_layers()`.
- `r_detector::Real`: radial position of the detector [km] (default 6369, approximate
  IceCube depth).

# Returns
- Scalar method: `StructArray{Path}` with `length` and `layer_idx` columns.
- Array method: `VectorOfVectors{Path}`, one `Vector{Path}` per zenith angle.
"""
function compute_paths(cz::Number, layers, r_detector)
    radii = layers.radius
    intersections = ray_circle_path_length.(radii, r_detector, cz)
    for i in 1:length(intersections) - 1
        intersections[i] -= intersections[i+1]
    end
    mask = intersections .> 0.
    rs = radii[mask]
    intersections = intersections[mask]

    n_layers_outside = sum(radii .>= r_detector)

    n_layers = 2 * (length(intersections) - n_layers_outside) + n_layers_outside

    lengths_traversed = zeros(n_layers)
    layer_idx_traversed = zeros(Int, n_layers)

    for i in 1:length(intersections)
        if (i < n_layers_outside) | (i == length(intersections))
            lengths_traversed[i] = intersections[i]
            layer_idx_traversed[i] = i
        elseif i == n_layers_outside
            len_det = -cz * (rs[i] - r_detector)
            inter = intersections[i] - len_det
            lengths_traversed[i] = inter/2 + len_det
            layer_idx_traversed[i] = i
            lengths_traversed[end-i+n_layers_outside] = inter/2
            layer_idx_traversed[end-i+n_layers_outside] = i
        else
            lengths_traversed[i] = intersections[i]/2
            layer_idx_traversed[i] = i
            lengths_traversed[end-i+n_layers_outside] = intersections[i]/2
            layer_idx_traversed[end-i+n_layers_outside] = i
        end
    end

    la = StructArray{Newtrinos.Path}((lengths_traversed, layer_idx_traversed))

end

function compute_paths(cz::AbstractArray, layers; r_detector = 6369)
    VectorOfVectors{Newtrinos.Path}(compute_paths.(cz, Ref(layers), r_detector));
end

"""
    get_compute_chord_paths(cfg::PREM) -> Function

Construct `compute_chord_paths(cz, layers; r_detector=6369, n_samples=20) -> (paths, chord_layers)`.

The constant-density zones of [`get_compute_layers`](@ref) carry the radially averaged PREM
density of each zone, but a chord through a zone samples a path-dependent part of the density
gradient (a shallow chord sees mostly the top of a zone). `compute_chord_paths` keeps the zone
geometry of [`compute_paths`](@ref), but gives every segment of every path its own
[`Newtrinos.osc.Layer`](@ref), with the PREM density averaged along that chord segment
(midpoint rule with `n_samples` points). The atmosphere layer (index 1) is shared.

The returned `chord_layers` and `paths` can be passed to the oscillation probability in place
of `layers` and `compute_paths(cz, layers)`; `layers` must come from `compute_layers()` of the
same configuration. With finite `max_length` [km], each segment is further split into equal
pieces no longer than `max_length`, each with its own chord-averaged density, which resolves
density gradients within a zone along the chord.
"""
function get_compute_chord_paths(cfg::PREM)
    profiles = _zone_profiles(cfg)
    function compute_chord_paths(cz::AbstractArray, layers; r_detector = 6369, n_samples = 20, max_length = Inf)
        paths = compute_paths(cz, layers; r_detector)
        radius = [layers.radius[1]]
        p_density = [layers.p_density[1]]
        n_density = [layers.n_density[1]]
        chord_paths = map(zip(cz, paths)) do (c, path)
            L = sum(seg.length for seg in path)
            t = 0.0  # distance from the entry point
            out = Newtrinos.Path[]
            for seg in path
                if seg.layer_idx == 1
                    push!(out, seg)
                else
                    r_zone, ρ_zone = profiles[seg.layer_idx - 1]
                    ye = cfg.p_fractions[seg.layer_idx - 1]
                    n_pieces = max(1, ceil(Int, seg.length / max_length))
                    ℓ = seg.length / n_pieces
                    for j in 1:n_pieces
                        ρ = 0.0
                        for k in 1:n_samples
                            s = L - (t + (j - 1 + (k - 0.5) / n_samples) * ℓ)  # distance from the detector
                            r = sqrt(max(r_detector^2 + s^2 + 2 * r_detector * s * c, 0.0))
                            ρ += _interp_flat(r_zone, ρ_zone, r) / n_samples
                        end
                        push!(radius, layers.radius[seg.layer_idx])
                        push!(p_density, ρ * ye)
                        push!(n_density, ρ * (1 - ye))
                        push!(out, Newtrinos.Path(ℓ, length(radius)))
                    end
                end
                t += seg.length
            end
            out
        end
        VectorOfVectors{Newtrinos.Path}(chord_paths), StructArray{Newtrinos.Layer}((radius, p_density, n_density))
    end
end

# PREM radial density profile restricted to each density zone, sorted by radius
function _zone_profiles(cfg::PREM)
    PREM = CSV.read(joinpath(datadir, "PREM_1s.csv"), DataFrame, header=["radius","depth","density","Vpv","Vph","Vsv","Vsh","eta","Q-mu","Q-kappa"])
    map(1:length(cfg.zones)-1) do i
        mask = (PREM.density .< cfg.zones[i+1]) .& (PREM.density .>= cfg.zones[i])
        order = sortperm(PREM.radius[mask])
        Float64.(PREM.radius[mask][order]), Float64.(PREM.density[mask][order])
    end
end

# linear interpolation, constant beyond the end points
function _interp_flat(x, y, xi)
    xi <= x[1] && return y[1]
    xi >= x[end] && return y[end]
    j = searchsortedlast(x, xi)
    x[j+1] == x[j] && return y[j]
    y[j] + (y[j+1] - y[j]) * (xi - x[j]) / (x[j+1] - x[j])
end

"""
    compute_dldcz(cz_values, layers; r_detector=6369, eps=1e-5)

Compute dL/d(cosθ) for each section of each path by centered finite differences.
Returns a vector of vectors matching the structure of compute_paths output.

Computes dL_total/dcz from the total path length (which is smooth across layer
boundaries), then distributes to each section proportional to its fraction of
the total path length. This avoids divergent per-section derivatives near layer
boundaries where thin sections shrink rapidly.
"""
function compute_dldcz(cz_values, layers; r_detector=6369, eps=1e-5)
    map(cz_values) do cz
        path = compute_paths(cz, layers, r_detector)
        path_plus = compute_paths(cz + eps, layers, r_detector)
        path_minus = compute_paths(cz - eps, layers, r_detector)
        L_total = sum(s.length for s in path)
        L_plus = sum(s.length for s in path_plus)
        L_minus = sum(s.length for s in path_minus)
        dLdcz = (L_plus - L_minus) / (2eps)
        [s.length / L_total * dLdcz for s in path]
    end
end

end
