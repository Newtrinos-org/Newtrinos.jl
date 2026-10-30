module Helpers

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


end # module Helpers