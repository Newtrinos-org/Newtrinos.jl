module Helpers

using ..Newtrinos

using SpecialFunctions: erf

"""
    bin(df, out_centers; col=1)

Bins a list of events in DataFrame or Matrix `df` into `out_centers` using the specified column (`col`) for the observable.
Returns a vector of counts per bin.
"""
function bin(df, out_edges; col=1)
    counts = zeros(length(out_edges)-1)
    for row in eachrow(df)
        val = row[col]
        # Only bin if val is within the window
        if out_edges[1] <= val < out_edges[end]
            idx = searchsortedlast(out_edges, val)
            if 1 <= idx < length(out_edges)
                counts[idx] += 1
            end
        end
    end
    return counts
end

"""
    project_to_bins(image_edges, counts, target_edges)

Energy scale distortion as a count-conserving bin projection (official `AxisDistortionMatrixPointwise`): the events of
source bin j, assumed uniform within it, are mapped to [image_edges[j], image_edges[j+1]] (monotonic distortion) and
shared among the `target_edges` bins in proportion to the overlap. Events mapped outside the target range are lost.
"""
function project_to_bins(image_edges, counts, target_edges)
    T = promote_type(eltype(image_edges), eltype(counts))
    out = zeros(T, length(target_edges) - 1)
    nt = length(target_edges) - 1
    @inbounds for j in eachindex(counts)
        A, B = image_edges[j], image_edges[j+1]
        B > A || continue
        k0 = max(searchsortedlast(target_edges, A), 1)
        k1 = min(searchsortedlast(target_edges, B), nt)
        for k in k0:k1
            overlap = min(B, target_edges[k+1]) - max(A, target_edges[k])
            overlap > 0 && (out[k] += counts[j] * overlap / (B - A))
        end
    end
    out
end

"""
    rebin(df, out_centers; var_col=1, count_col=2)

Rebins a table of (observable, count) pairs in DataFrame or Matrix `df` into `out_centers`.
`var_col` specifies which column contains the observable to bin in.
`count_col` specifies which column contains the counts.
Returns a vector of summed counts per bin.
"""
function rebin(df, out_edges; var_col=1, count_col=2)
    counts = zeros(length(out_edges)-1)
    for row in eachrow(df)
        val = row[var_col]
        count = row[count_col]
        idx = searchsortedlast(out_edges, val)
        if 1 <= idx < length(out_edges)
            counts[idx] += count
        end
    end
    return counts
end

"""
    rebin_index(fine_edges, coarse_edges)

Index of the coarse bin each fine bin falls into (0 outside); the coarse edges must be a subset of the fine ones.
"""
function rebin_index(fine_edges, coarse_edges)
    @assert all(any(isapprox.(fine_edges, e; atol = 1e-9)) for e in coarse_edges) "coarse edges are not on the fine grid"
    idx = zeros(Int, length(fine_edges) - 1)
    for i in eachindex(idx)
        k = searchsortedlast(coarse_edges, fine_edges[i] + 1e-9)
        1 <= k < length(coarse_edges) && (idx[i] = k)
    end
    idx
end

function rebin(counts, idx, n)
    out = zeros(eltype(counts), n)
    @inbounds for i in eachindex(counts, idx)
        idx[i] > 0 && (out[idx[i]] += counts[i])
    end
    out
end

"""
    not_a_knot_spline(x, y, xnew)

Cubic spline through `(x, y)` with not-a-knot end conditions (as scipy's `interp1d(kind="cubic")`), evaluated at `xnew`
(within the range of `x`).
"""
function not_a_knot_spline(x::Vector{Float64}, y::Vector{Float64}, xnew::AbstractVector{Float64})
    n = length(x)
    h = diff(x)
    # second derivatives M: interior continuity of the first derivative, not-a-knot at both ends
    A = zeros(n, n)
    b = zeros(n)
    A[1, 1], A[1, 2], A[1, 3] = h[2], -(h[1] + h[2]), h[1]
    for i in 2:n-1
        A[i, i-1], A[i, i], A[i, i+1] = h[i-1], 2 * (h[i-1] + h[i]), h[i]
        b[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    end
    A[n, n-2], A[n, n-1], A[n, n] = h[n-1], -(h[n-2] + h[n-1]), h[n-2]
    M = A \ b
    map(xnew) do t
        k = clamp(searchsortedlast(x, t), 1, n - 1)
        a, c = x[k+1] - t, t - x[k]
        (M[k] * a^3 + M[k+1] * c^3) / (6h[k]) + (y[k] / h[k] - M[k] * h[k] / 6) * a + (y[k+1] / h[k] - M[k+1] * h[k] / 6) * c
    end
end

"""
    smear_bin_centres(counts, σ, edges; nσ = 8)

Energy resolution as in the official model (`HistSmearNormalMatrixBC`): the events of each bin i are smeared with a
Gaussian of width σ[i] around its centre, evaluated at the centres of the bins of the same binning (times the bin
width).
"""
function smear_bin_centres(counts, σ, edges; nσ = 8)
    T = promote_type(eltype(counts), eltype(σ))
    c = (edges[1:end-1] .+ edges[2:end]) ./ 2
    out = zeros(T, length(c))
    @inbounds for i in eachindex(counts, σ)
        iszero(counts[i]) && continue
        lo = searchsortedfirst(c, c[i] - nσ * σ[i])
        hi = searchsortedlast(c, c[i] + nσ * σ[i])
        norm = counts[i] / (sqrt(2π) * σ[i])
        for j in lo:hi
            out[j] += norm * (edges[j+1] - edges[j]) * exp(-0.5 * ((c[j] - c[i]) / σ[i])^2)
        end
    end
    out
end
 
# standard normal CDF and the antiderivative of the CDF, Ψ(z) = ∫_{-∞}^{z} Φ(t) dt = z Φ(z) + φ(z)
@inline _Φ(z) = (1 + erf(z / sqrt(oftype(float(z), 2)))) / 2
@inline _φ(z) = exp(-z * z / 2) / sqrt(2 * oftype(float(z), π))
@inline _Ψ(z) = z * _Φ(z) + _φ(z)
 
"""
    smear_erf(counts, σ, edges; nσ = 8, integrate_source = true)
 
Gaussian energy-resolution smearing of a binned spectrum, conserving events.
 
`counts[i]` are the events in bin i of `edges` (length n+1) and `σ[i]` the absolute resolution (same units as
`edges`) for that bin. Returns the smeared counts on the same bins.
 
Each source bin is spread over the target bins with probabilities computed from the Gaussian CDF (error function)
integrated over each target bin, instead of sampling the Gaussian density at the target bin centres:
 
- `integrate_source = false`: source events sit at the bin centre c_i,
      P_ij = Φ((e_{j+1} - c_i)/σ_i) - Φ((e_j - c_i)/σ_i)
- `integrate_source = true` (default): source events are flat across [a_i, b_i] (the usual assumption for a
  binned spectrum); averaging the above over the source bin gives, with Ψ(z) = zΦ(z) + φ(z),
      P_ij = σ_i / (b_i - a_i) * [K_i(e_{j+1}) - K_i(e_j)],   K_i(e) = Ψ((e - a_i)/σ_i) - Ψ((e - b_i)/σ_i)
  This removes the residual bin-centre ripple and stays exact when σ is much smaller than the bin width
  (P_ii -> 1).
 
Both versions sum to 1 over target bins (up to the part falling outside `edges`, which is lost, as for a selection
cut), for any σ, unlike sampling the density at bin centres, which mis-normalises when σ is comparable to or
smaller than the bin width. Only target bins within `nσ` σ of the source bin are visited.
 
Generic in the number types of `counts` and `σ`, so ForwardDiff.Dual propagates (erf has derivative rules via
SpecialFunctions/DiffRules).
"""
function smear_erf(counts::AbstractVector, σ::AbstractVector, edges::AbstractVector;
                   nσ::Real = 8, integrate_source::Bool = true)
    n = length(counts)
    length(σ) == n && length(edges) == n + 1 ||
        throw(DimensionMismatch("need length(σ) == length(counts) == length(edges) - 1"))
    T = promote_type(eltype(counts), typeof(float(one(eltype(σ)))), eltype(edges))
    out = zeros(T, n)
    @inbounds for i in 1:n
        s = counts[i]
        iszero(s) && continue
        a = edges[i]; b = edges[i+1]; σi = σ[i]
        jlo = max(searchsortedlast(edges, a - nσ * σi), 1)
        jhi = min(searchsortedfirst(edges, b + nσ * σi) - 1, n)
        if integrate_source
            w = b - a
            pre = σi / w
            Kprev = _Ψ((edges[jlo] - a) / σi) - _Ψ((edges[jlo] - b) / σi)
            for j in jlo:jhi
                Knext = _Ψ((edges[j+1] - a) / σi) - _Ψ((edges[j+1] - b) / σi)
                out[j] += s * pre * (Knext - Kprev)
                Kprev = Knext
            end
        else
            c = (a + b) / 2
            Fprev = _Φ((edges[jlo] - c) / σi)
            for j in jlo:jhi
                Fnext = _Φ((edges[j+1] - c) / σi)
                out[j] += s * (Fnext - Fprev)
                Fprev = Fnext
            end
        end
    end
    return out
end

"""
    refine_lsnl(E, f; refine_times = 4, newmin = 0.7, newmax = 12.1)

LSNL curve processing of the official model (`refine_lsnl_data`): the relative curve f = Evis/Escint is converted to the
absolute Evis(Escint), interpolated with a cubic spline on a `refine_times` finer grid and extrapolated linearly (with
the outermost fine segments) to [`newmin`, `newmax`]. Returns the fine Escint grid and Evis on it.
"""
function refine_lsnl(E::Vector{Float64}, f::Vector{Float64}, refine_times::Int64; newmin = 0.7, newmax = 12.0)
    xfine = collect(range(E[1], E[end], length = (length(E) - 1) * refine_times + 1))
    refine_lsnl(E, f, xfine, newmin=newmin, newmax=newmax)
end

function refine_lsnl(E::Vector{Float64}, f::Vector{Float64}, xfine::Vector{Float64}; newmin = 0.7, newmax = 12.0)
    yfine = not_a_knot_spline(E, f .* E, xfine)
    step = xfine[2] - xfine[1]
    xleft = collect(newmin:step:xfine[1]-step*1e-9)
    stepright = xfine[end] - xfine[end-1]
    xright = collect(xfine[end]:stepright:newmax+stepright*1e-6)[2:end]
    sl = (yfine[2] - yfine[1]) / (xfine[2] - xfine[1])
    sr = (yfine[end] - yfine[end-1]) / (xfine[end] - xfine[end-1])
    yleft = @. yfine[1] + sl * (xleft - xfine[1])
    yright = @. yfine[end] + sr * (xright - xfine[end])
    vcat(xleft, xfine, xright), vcat(yleft, yfine, yright)
end
"""
    wrap_func(func, pars, keep)

Wrap a function `func` s.t. it only takes arguments provided by `keep` of all arguments `pars`.
Returns the wrapped function, the kept arguments and the skipped arguments
"""
function wrap_func(func, pars, keep)
    skipped_pars = []
    for n in keys(pars)
        #println(n)
        if !(n in keep)
            push!(skipped_pars, n)
        end
    end
    skip = NamedTuple((k, pars[k]) for k in skipped_pars)
    kept_pars = NamedTuple((key, pars[key]) for key in keep)
    println(kept_pars)
    @assert length(kept_pars) + length(skip) == length(pars)
    function wrapped_func(p)
        arg = Newtrinos.safe_merge(p, skip)
        return func(arg)
    end

    return wrapped_func, kept_pars, skip
end


end # module Helpers