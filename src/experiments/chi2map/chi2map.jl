module chi2map

using Distributions
using Interpolations
using CairoMakie
import ..Newtrinos

"""
A tabulated χ² as a likelihood term: χ²(x₁, …, x_n) on a regular grid, where each coordinate x_k is a
function of the parameters (e.g. `p -> sin(p.θ₁₂)^2`), interpolated with cubic splines (ForwardDiff-safe,
linear extrapolation outside the grid). Used to add the profiled likelihood of a whole sector (e.g. the
solar data, or published χ² maps such as Super-Kamiokande's atmospheric one) to a fit:

    configure(coordinates, axes, chi2; title="χ² map") -> Chi2Map

- `coordinates`: tuple of functions `params -> x_k`;
- `axes`: tuple of `AbstractRange`s (uniform grid along each coordinate);
- `chi2`: array of size `length.(axes)` (an offset is irrelevant for profiles and is removed).

The term contributes −χ²/2 to the log-likelihood and has no parameters of its own.
"""
@kwdef struct Chi2Map <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

"""
    Chi2Penalty(χ²)

Pseudo-distribution whose log-density is −χ²/2 for any observation (the χ² term of a [`Chi2Map`](@ref)).
"""
struct Chi2Penalty{T<:Real} <: ContinuousUnivariateDistribution
    chi2::T
end
Distributions.logpdf(d::Chi2Penalty, ::Real) = -d.chi2 / 2
Distributions.pdf(d::Chi2Penalty, x::Real) = exp(logpdf(d, x))
Distributions.minimum(::Chi2Penalty) = -Inf
Distributions.maximum(::Chi2Penalty) = Inf
Distributions.insupport(::Chi2Penalty, ::Real) = true
Base.rand(::Distributions.AbstractRNG, ::Chi2Penalty) = 0.0

function configure(coordinates::Tuple, axes::Tuple, chi2::AbstractArray; title::AbstractString="χ² map")
    length(coordinates) == length(axes) == ndims(chi2) || throw(ArgumentError("coordinates, axes and chi2 dimensions differ"))
    size(chi2) == Tuple(length.(axes)) || throw(ArgumentError("chi2 has size $(size(chi2)), axes $(length.(axes))"))
    c = chi2 .- minimum(chi2)
    itp = cubic_spline_interpolation(axes, c; extrapolation_bc=Line())
    assets = (observed = 0.0, axes = axes, chi2 = c, itp = itp, coordinates = coordinates, title = title)
    Chi2Map(physics = (;), params = (;), priors = (;), assets = assets,
            forward_model = params -> Chi2Penalty(itp(map(f -> f(params), coordinates)...)),
            plot = get_plot(assets))
end

"""
    value(experiment, params) -> χ²

The interpolated χ² (relative to the minimum of the table) at `params`.
"""
value(e::Chi2Map, params) = e.assets.itp(map(f -> f(params), e.assets.coordinates)...)

function get_plot(assets)
    function plot(params, data=assets.observed)
        # Δχ² of the first two coordinates, minimised over the others
        c = ndims(assets.chi2) > 2 ? dropdims(minimum(assets.chi2, dims=Tuple(3:ndims(assets.chi2))), dims=Tuple(3:ndims(assets.chi2))) : assets.chi2
        f = Figure()
        ax = Axis(f[1, 1], title=assets.title, xlabel="coordinate 1", ylabel="coordinate 2")
        contourf!(ax, collect(assets.axes[1]), collect(assets.axes[2]), c .- minimum(c), levels=[0, 2.3, 6.18, 11.83])
        f
    end
end

end
