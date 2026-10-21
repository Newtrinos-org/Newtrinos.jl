module Helpers

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

end # module Helpers