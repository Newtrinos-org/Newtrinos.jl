"""
    TAOLeakage

Energy-leakage + non-linearity response for the TAO detector (JUNO-TAO, arXiv:2610.23331, Fig. 26), rebuilt
for an arbitrary positron non-linearity curve and usable inside automatic differentiation.

Model: for a positron with true energy E_true = T_e+ + 1.022 MeV, the visible energy is
    E_vis = E_peak(E_true) + delta,   E_peak = scale * E_true * f_e+(E_true) + bias,
where delta <= 0 is the energy lost by leakage, `scale` is a multiplicative energy scale and `bias` an additive
energy offset (MeV). The bias shifts the whole distribution of each E_true bin (peak and leakage tail) by the
same amount, which is exact for an additive offset. The scale multiplies the peak position; the leakage offsets
delta are not rescaled (for a 0.6% scale this is a <~6 keV effect on the tail). The distribution of delta was extracted from Fig. 26 as six
templates in E_true bands (`TAO_leakage_templates.txt`), interpolated linearly in E_true between band centres.
Changing the non-linearity curve moves the full-containment peak and the whole leakage tail with it.

Design for speed and ForwardDiff.jl:
  * Everything that does not depend on fit parameters (template mixing per E_true bin, the non-linearity table
    at each bin centre) is precomputed once in a `LeakResponse`.
  * Parameters enter only through the peak position of each E_true column (one scalar per column), so the
    hot loop evaluates a piecewise-linear CDF at shifted bin edges. Only the ~155 E_vis bins inside each
    column's leakage window are touched (the template spans 3.06 MeV), never the full dense matrix.
  * `fold`/`fold!` apply the response directly to a spectrum without materialising a matrix, which is what
    keeps gradients cheap: no 615x565 matrix of dual numbers is allocated.
  * Bin integration over E_true (default, `integrate=true`): each E_true bin is treated as a flat distribution
    between its edges, so its full-containment peak is spread uniformly between E_peak(E_lo) and E_peak(E_hi)
    instead of being placed at the image of the bin centre. Because the template contains a spike narrower
    than the bins, point evaluation at bin centres maps each bin to a single E_vis bin; with a non-linearity
    slope dE_peak/dE_true != 1 some E_vis bins then receive two peaks and others none, giving a periodic,
    grid-dependent ripple. Bin integration tiles the E_vis axis exactly and removes it. It is done
    analytically with the second cumulative of the template (piecewise quadratic), at the same cost.
  * All code is generic in the number type: `ForwardDiff.Dual` (in the non-linearity parameters, in the input
    spectrum, or both) propagates. No dependency on ForwardDiff itself.

No external packages are required.
"""
module TAOLeakage

export LeakageTemplates, load_leakage_templates,
       PositronNL, load_tao_positron_nl, positron_nl,
       LeakResponse, peak_positions, fold, fold!, response_matrix,
       leak_nl_matrix

# =========================================================================================================
# Leakage templates
# =========================================================================================================

"""
    LeakageTemplates

Templates of delta = E_vis - E_peak on a uniform grid.
`δ0`, `dδ`, `n`: first edge, bin width and number of bins; `cdf`: cumulative probability at the n+1 edges,
one column per E_true band; `centres`: count-weighted E_true centres of the bands (MeV).
"""
struct LeakageTemplates
    δ0::Float64
    dδ::Float64
    n::Int
    cdf::Matrix{Float64}
    centres::Vector{Float64}
end

"""
    load_leakage_templates(path) -> LeakageTemplates

Read `TAO_leakage_templates.txt`.
"""
function load_leakage_templates(path::AbstractString)
    centres = Float64[]
    lo = Float64[]; hi = Float64[]
    dens = Vector{Vector{Float64}}()
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        if startswith(s, "#")
            if occursin("BAND_CENTRES_MEV", s)
                centres = parse.(Float64, split(strip(split(s, "BAND_CENTRES_MEV")[2])))
            end
            continue
        end
        v = parse.(Float64, split(s))
        push!(lo, v[1]); push!(hi, v[2]); push!(dens, v[3:end])
    end
    isempty(centres) && error("BAND_CENTRES_MEV line not found in $path")
    n = length(lo); K = length(centres)
    dδ = (hi[end] - lo[1]) / n
    maximum(abs.(hi .- lo .- dδ)) < 1e-9 || error("template grid is not uniform")
    cdf = zeros(n + 1, K)
    for k in 1:K, i in 1:n
        cdf[i+1, k] = cdf[i, k] + dens[i][k] * (hi[i] - lo[i])
    end
    cdf ./= cdf[end:end, :]          # exact normalisation
    return LeakageTemplates(lo[1], dδ, n, cdf, centres)
end

# =========================================================================================================
# Positron non-linearity (TAO, Fig. 25b)
# =========================================================================================================

"""
    PositronNL

Tabulated positron non-linearity E_rec/E_true: nominal curve and lower/upper edges of the 68% C.L. band.
"""
struct PositronNL
    E::Vector{Float64}
    nom::Vector{Float64}
    lo::Vector{Float64}
    hi::Vector{Float64}
end

"""
    load_tao_positron_nl(path) -> PositronNL

Read the positron columns of `TAO_Fig25b_nonlinearity.txt`, skipping rows where the curve is undefined.
"""
function load_tao_positron_nl(path::AbstractString)
    E = Float64[]; nom = Float64[]; lo = Float64[]; hi = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        x = parse.(Float64, split(s)[1:4])
        any(isnan, x) && continue
        push!(E, x[1]); push!(nom, x[2]); push!(lo, x[3]); push!(hi, x[4])
    end
    return PositronNL(E, nom, lo, hi)
end

# piecewise-linear interpolation with flat extrapolation (generic in x)
@inline function _interp(xs::AbstractVector{Float64}, ys::AbstractVector{Float64}, x::Real)
    x <= xs[1]   && return ys[1] + zero(x)
    x >= xs[end] && return ys[end] + zero(x)
    i = searchsortedlast(xs, x)
    t = (x - xs[i]) / (xs[i+1] - xs[i])
    return ys[i] + t * (ys[i+1] - ys[i])
end

# one-parameter pull along the band.
#  asymmetric: +1 -> upper edge, -1 -> lower edge (derivative jumps at pull = 0)
#  symmetric:  shift by pull * (average half-width), smooth everywhere
@inline _pull_shift(pull, up, dn) = ifelse(pull >= 0, pull * up, pull * dn)
@inline _band_shift(pull, up, dn, sym::Bool) = sym ? pull * ((up + dn) / 2) : _pull_shift(pull, up, dn)

function _is_symmetric(band::Symbol)
    band === :symmetric  && return true
    band === :asymmetric && return false
    throw(ArgumentError("band must be :asymmetric or :symmetric, got :$band"))
end

"""
    positron_nl(nl::PositronNL, E; pull=0, scale=1, band=:asymmetric)

Positron non-linearity at true energy `E`, moved along its 68% band by `pull` and multiplied by a global
energy-scale factor `scale`.

`band` selects how the (asymmetric) band is used:
- `:asymmetric` (default): pull = +1 gives the upper edge, -1 the lower edge, linear in between and beyond.
  The derivative jumps at pull = 0; exactly at 0 it is taken from the upper side.
- `:symmetric`: shift by pull x (average of the upper and lower half-widths), so pull = +-1 moves by the
  mean band width in both directions. Smooth everywhere (continuous derivative at pull = 0).
"""
function positron_nl(nl::PositronNL, E::Real; pull::Real=0, scale::Real=1, band::Symbol=:asymmetric)
    sym = _is_symmetric(band)
    n = _interp(nl.E, nl.nom, E)
    up = _interp(nl.E, nl.hi, E) - n
    dn = n - _interp(nl.E, nl.lo, E)
    return scale * (n + _band_shift(pull, up, dn, sym))
end

# =========================================================================================================
# Precomputed response on fixed grids
# =========================================================================================================

"""
    LeakResponse(T, etrue_edges, evis_edges; nl=nothing, Emin=1.022, integrate=true)

Precompute everything that does not depend on fit parameters for fixed E_true / E_vis grids:
the template mixture (CDF and its integral) for every active E_true bin, and, if a `PositronNL` table is given,
its nominal value and band widths at the evaluation points (enabling the fast `pull`/`scale` path).

`integrate=true` (default) averages the response over each E_true bin (flat spectrum within the bin), which
removes the grid-dependent ripple described in the module docstring. `integrate=false` reproduces the previous
behaviour (evaluation at bin centres).

`evis_edges` must be increasing; a uniform grid enables O(1) bin lookup. Make its upper edge large enough
for E_true*f(E_true) (about 13 MeV for E_true up to 12 MeV). Columns with bin centre below `Emin` are empty.
"""
struct LeakResponse
    etrue_edges::Vector{Float64}
    evis_edges::Vector{Float64}
    Ec::Vector{Float64}             # E_true bin centres
    jfirst::Int                     # first active E_true bin (centre >= Emin)
    δ0::Float64
    dδ::Float64
    n::Int
    cdf::Matrix{Float64}            # (n+1) x ntrue mixed template CDF per E_true bin
    uniform_vis::Bool
    v0::Float64
    dv::Float64
    nom::Vector{Float64}            # non-linearity at the evaluation points (empty if no table given):
    up::Vector{Float64}             #   E_true bin edges if `integrate`, else bin centres
    dn::Vector{Float64}
    integrate::Bool                 # average over each E_true bin (true) or evaluate at its centre (false)
    G::Matrix{Float64}              # (n+1) x ntrue integral of the template CDF, G(x) = int_{δ0}^{x} F
end

function LeakResponse(T::LeakageTemplates, etrue_edges::AbstractVector{<:Real},
                      evis_edges::AbstractVector{<:Real}; nl::Union{Nothing,PositronNL}=nothing,
                      Emin::Real=1.022, integrate::Bool=true)
    te = collect(Float64, etrue_edges); ve = collect(Float64, evis_edges)
    (issorted(te) && issorted(ve)) || error("bin edges must be increasing")
    ntrue = length(te) - 1
    Ec = [(te[j] + te[j+1]) / 2 for j in 1:ntrue]
    jfirst = something(findfirst(>=(Emin), Ec), ntrue + 1)
    c = T.centres; K = length(c)
    cdf = zeros(T.n + 1, ntrue)
    for j in jfirst:ntrue
        k = clamp(searchsortedlast(c, Ec[j]), 1, K - 1)
        w = clamp((Ec[j] - c[k]) / (c[k+1] - c[k]), 0.0, 1.0)
        @views cdf[:, j] .= (1 - w) .* T.cdf[:, k] .+ w .* T.cdf[:, k+1]
    end
    G = zeros(T.n + 1, ntrue)
    for j in jfirst:ntrue, k in 1:T.n
        G[k+1, j] = G[k, j] + T.dδ * (cdf[k, j] + cdf[k+1, j]) / 2
    end
    dvs = diff(ve)
    uniform = maximum(abs.(dvs .- dvs[1])) < 1e-9 * max(1.0, abs(dvs[1]))
    Ep = integrate ? te : Ec                     # where the non-linearity is evaluated
    if nl === nothing
        nom = Float64[]; up = Float64[]; dn = Float64[]
    else
        nom = [_interp(nl.E, nl.nom, E) for E in Ep]
        up  = [_interp(nl.E, nl.hi, E) for E in Ep] .- nom
        dn  = nom .- [_interp(nl.E, nl.lo, E) for E in Ep]
    end
    return LeakResponse(te, ve, Ec, jfirst, T.δ0, T.dδ, T.n, cdf, uniform, ve[1], dvs[1], nom, up, dn,
                        integrate, G)
end

ntrue(R::LeakResponse) = length(R.Ec)
nvis(R::LeakResponse) = length(R.evis_edges) - 1

# ---------------------------------------------------------------------------------------------------------
# peak positions
# ---------------------------------------------------------------------------------------------------------

"""
    peak_positions(R, f; bias=0) -> Vector
    peak_positions(R; pull=0, scale=1, bias=0, band=:asymmetric) -> Vector

Full-containment peak E_true*f(E_true) + bias (bias in MeV) at the E_true bin edges (`R.integrate == true`, length ntrue+1) or bin
centres (`R.integrate == false`, length ntrue). Entries for inactive bins (centre below `Emin`) are unused.
The keyword form uses the precomputed TAO table (`LeakResponse(...; nl=...)`); `f` can be any callable.
`band` is `:asymmetric` or `:symmetric`, as in `positron_nl`.
"""
_evalpoints(R::LeakResponse) = R.integrate ? R.etrue_edges : R.Ec
_firstpoint(R::LeakResponse) = R.jfirst            # same index for edges (lower edge of bin jfirst) and centres

function peak_positions(R::LeakResponse, f; bias::Real=0)
    X = _evalpoints(R)
    P = typeof(float(X[end] * f(X[end]) + bias))
    p = zeros(P, length(X))
    @inbounds for j in _firstpoint(R):length(X)
        p[j] = X[j] * f(X[j]) + bias
    end
    return p
end

function peak_positions(R::LeakResponse; pull::Real=0, scale::Real=1, bias::Real=0, band::Symbol=:asymmetric)
    isempty(R.nom) && error("LeakResponse was built without a PositronNL table; pass `nl=` or use peak_positions(R, f)")
    sym = _is_symmetric(band)
    X = _evalpoints(R)
    P = typeof(float(scale * (1.0 + _pull_shift(pull, 1.0, 1.0)) + bias))
    p = zeros(P, length(X))
    if sym
        @inbounds for j in _firstpoint(R):length(X)
            p[j] = X[j] * scale * (R.nom[j] + pull * ((R.up[j] + R.dn[j]) / 2)) + bias
        end
    else
        @inbounds for j in _firstpoint(R):length(X)
            p[j] = X[j] * scale * (R.nom[j] + _pull_shift(pull, R.up[j], R.dn[j])) + bias
        end
    end
    return p
end

# ---------------------------------------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------------------------------------

# template CDF of column j at x = E_vis - E_peak (piecewise linear, generic in x)
@inline function _F(R::LeakResponse, j::Int, x)
    u = (x - R.δ0) / R.dδ
    u <= 0   && return zero(u)
    u >= R.n && return one(u)
    k = floor(Int, u)                       # 0-based bin index (works for ForwardDiff.Dual)
    @inbounds c0 = R.cdf[k+1, j]
    @inbounds c1 = R.cdf[k+2, j]
    return c0 + (u - k) * (c1 - c0)
end

# integral of the template CDF of column j, G(x) = int_{δ0}^{x} F (piecewise quadratic, generic in x)
@inline function _G(R::LeakResponse, j::Int, x)
    u = (x - R.δ0) / R.dδ
    u <= 0 && return zero(u)
    @inbounds if u >= R.n
        return R.G[R.n+1, j] + (u - R.n) * R.dδ
    end
    k = floor(Int, u)
    t = u - k
    @inbounds c0 = R.cdf[k+1, j]
    @inbounds c1 = R.cdf[k+2, j]
    @inbounds g0 = R.G[k+1, j]
    return g0 + R.dδ * (c0 * t + (c1 - c0) * t * t / 2)
end

# bin-averaged CDF: mean of F(v - p) for p uniform in [plo, phi]
@inline function _H(R::LeakResponse, j::Int, v, plo, phi, inv_dp)
    return (_G(R, j, v - plo) - _G(R, j, v - phi)) * inv_dp
end

# range of E_vis bins overlapping the leakage window [plo + δ0, phi + δ0 + n dδ]
@inline _vis_window(R::LeakResponse, peak) = _vis_window(R, peak, peak)
@inline function _vis_window(R::LeakResponse, plo, phi)
    lo = plo + R.δ0
    hi = phi + R.δ0 + R.n * R.dδ
    nv = nvis(R)
    if R.uniform_vis
        (hi <= R.v0 || lo >= R.v0 + nv * R.dv) && return 1, 0
        ilo = floor(Int, (lo - R.v0) / R.dv) + 1
        ihi = floor(Int, (hi - R.v0) / R.dv) + 1
    else
        ilo = searchsortedlast(R.evis_edges, lo)
        ihi = searchsortedlast(R.evis_edges, hi)
    end
    return max(ilo, 1), min(ihi, nv)
end

# apply column j: calls sink(i, value) for every E_vis bin i in the column's window
@inline function _column(sink, R::LeakResponse, j::Int, peaks)
    ve = R.evis_edges
    if R.integrate
        @inbounds plo = peaks[j]
        @inbounds phi = peaks[j+1]
        if phi < plo
            plo, phi = phi, plo
        end
        dp = phi - plo
        if dp > 1e-12
            ilo, ihi = _vis_window(R, plo, phi)
            ilo > ihi && return nothing
            inv_dp = 1 / dp
            Hprev = _H(R, j, ve[ilo], plo, phi, inv_dp)
            @inbounds for i in ilo:ihi
                Hnext = _H(R, j, ve[i+1], plo, phi, inv_dp)
                sink(i, Hnext - Hprev)
                Hprev = Hnext
            end
            return nothing
        end
        pk = (plo + phi) / 2                      # degenerate bin: fall back to point evaluation
    else
        @inbounds pk = peaks[j]
    end
    ilo, ihi = _vis_window(R, pk)
    ilo > ihi && return nothing
    Fprev = _F(R, j, ve[ilo] - pk)
    @inbounds for i in ilo:ihi
        Fnext = _F(R, j, ve[i+1] - pk)
        sink(i, Fnext - Fprev)
        Fprev = Fnext
    end
    return nothing
end

"""
    fold!(out, R, spectrum, peaks) -> out

Add the response applied to `spectrum` (events per E_true bin) into `out` (length = number of E_vis bins):
out[i] += sum_j P(E_vis bin i | E_true bin j) * spectrum[j], with column peaks `peaks`
(from `peak_positions`). Non-allocating; `out` must be able to hold the promoted number type.
"""
function fold!(out::AbstractVector, R::LeakResponse, spectrum::AbstractVector, peaks::AbstractVector)
    length(spectrum) == ntrue(R) || throw(DimensionMismatch("spectrum has length $(length(spectrum)), expected $(ntrue(R))"))
    length(out) == nvis(R) || throw(DimensionMismatch("out has length $(length(out)), expected $(nvis(R))"))
    length(peaks) == length(_evalpoints(R)) || throw(DimensionMismatch("peaks has the wrong length; use peak_positions(R, ...)"))
    @inbounds for j in R.jfirst:ntrue(R)
        s = spectrum[j]
        iszero(s) && continue
        _column((i, m) -> (@inbounds out[i] += m * s), R, j, peaks)
    end
    return out
end

"""
    fold(R, spectrum; pull=0, scale=1, bias=0, band=:asymmetric) -> Vector
    fold(R, spectrum, f; bias=0) -> Vector

Visible-energy spectrum from a true-energy spectrum (events per E_true bin), using the TAO positron
non-linearity moved by `pull`/`scale` (needs `LeakResponse(...; nl=...)`) or any curve `f(E) = E_vis/E_true`.
`scale` multiplies and `bias` (MeV) is added to the peak position: E_peak = scale * E_true * f(E_true) + bias.
`band = :symmetric` uses the average band width for the pull (smooth at pull = 0); see `positron_nl`.
Events falling outside the E_vis grid (e.g. below a 0.7-MeV threshold) are dropped, as in the selection.
"""
function fold(R::LeakResponse, spectrum::AbstractVector; pull::Real=0, scale::Real=1, bias::Real=0,
              band::Symbol=:asymmetric)
    peaks = peak_positions(R; pull=pull, scale=scale, bias=bias, band=band)
    out = zeros(promote_type(eltype(peaks), eltype(spectrum)), nvis(R))
    return fold!(out, R, spectrum, peaks)
end

function fold(R::LeakResponse, spectrum::AbstractVector, f; bias::Real=0)
    peaks = peak_positions(R, f; bias=bias)
    out = zeros(promote_type(eltype(peaks), eltype(spectrum)), nvis(R))
    return fold!(out, R, spectrum, peaks)
end

"""
    response_matrix(R; pull=0, scale=1, bias=0, band=:asymmetric) -> Matrix
    response_matrix(R, f; bias=0) -> Matrix

Dense matrix M[i, j] = P(E_vis bin i | E_true bin j). Use `fold` inside fits; build the matrix only when it
is needed explicitly (it is ~(nvis x ntrue) numbers, and with dual numbers much larger).
"""
response_matrix(R::LeakResponse; pull::Real=0, scale::Real=1, bias::Real=0, band::Symbol=:asymmetric) =
    _matrix(R, peak_positions(R; pull=pull, scale=scale, bias=bias, band=band))
response_matrix(R::LeakResponse, f; bias::Real=0) = _matrix(R, peak_positions(R, f; bias=bias))

function _matrix(R::LeakResponse, peaks::AbstractVector)
    M = zeros(eltype(peaks), nvis(R), ntrue(R))
    for j in R.jfirst:ntrue(R)
        _column((i, m) -> (@inbounds M[i, j] = m), R, j, peaks)
    end
    return M
end

# ---------------------------------------------------------------------------------------------------------
# backwards-compatible wrappers (previous version of this module)
# ---------------------------------------------------------------------------------------------------------

"""
    leak_nl_matrix(T, etrue_edges, evis_edges, f; bias=0, Emin=1.022, integrate=true)
    leak_nl_matrix(T, nl, etrue_edges, evis_edges; pull=0, scale=1, bias=0, band=:asymmetric, Emin=1.022,
                   integrate=true)

Dense response matrix, as in the previous version. Builds a `LeakResponse` on every call; inside a fit,
construct the `LeakResponse` once and use `fold`.
"""
leak_nl_matrix(T::LeakageTemplates, etrue_edges, evis_edges, f; bias::Real=0, Emin::Real=1.022,
               integrate::Bool=true) =
    response_matrix(LeakResponse(T, etrue_edges, evis_edges; Emin=Emin, integrate=integrate), f; bias=bias)

leak_nl_matrix(T::LeakageTemplates, nl::PositronNL, etrue_edges, evis_edges;
               pull::Real=0, scale::Real=1, bias::Real=0, band::Symbol=:asymmetric, Emin::Real=1.022,
               integrate::Bool=true) =
    response_matrix(LeakResponse(T, etrue_edges, evis_edges; nl=nl, Emin=Emin, integrate=integrate);
                    pull=pull, scale=scale, bias=bias, band=band)

end # module

# =========================================================================================================
# Example
# =========================================================================================================
# include("tao_leakage.jl"); using .TAOLeakage
# T  = load_leakage_templates("TAO_leakage_templates.txt")
# nl = load_tao_positron_nl("TAO_Fig25b_nonlinearity.txt")
# R  = LeakResponse(T, 0.70:0.02:12.00, 0.70:0.02:13.00; nl=nl)   # build once
#
# vis = fold(R, spectrum_true; pull=θ.tao_nl_pull, scale=θ.tao_escale)  # inside the likelihood
# vis = fold(R, spectrum_true; pull=θ.tao_nl_pull, band=:symmetric)    # pull by the average band width
# vis = fold(R, spectrum_true; scale=θ.tao_escale, bias=θ.tao_ebias)   # E_peak = scale*E*f(E) + bias (MeV)
# vis = fold(R, spectrum_true, E -> my_nl(E, θ))                       # any custom curve
#
# using ForwardDiff
# g = ForwardDiff.gradient(θ -> chi2(fold(R, model_true(θ); pull=θ[1], scale=θ[2]), data), θ0)