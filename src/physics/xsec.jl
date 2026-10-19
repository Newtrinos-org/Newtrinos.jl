module xsec

using LinearAlgebra
using Distributions
using CSV, DataFrames
using Interpolations
using FunctionChains
using JLD2
using ..Newtrinos

"""
    XsecModel

Abstract type for neutrino cross-section models.

Each subtype defines how cross-section normalization factors are applied to event rates.
All models provide `weights` (see [`get_weights`](@ref)), so that an experiment written against
`physics.xsec.weights` runs with any model and the model is chosen when configuring the physics.
Subtypes:
- [`Nominal`](@ref): no cross-section systematics (unit weights, no parameters), for experiments whose
  own module carries the cross-section uncertainties of the publication.
- [`SimpleScaling`](@ref): global normalization for NC and ``\\nu_\\tau`` CC channels.
- [`Differential_H2O`](@ref): per-interaction-mode differential scaling for water
  Cherenkov detectors.
- [`H2O_PCA`](@ref): GENIE-tune-based per-channel PCA shape and normalization
  systematics for water Cherenkov detectors.
- [`CorrelatedXsec`](@ref): one shared, correlated parameter set for all experiments (targets,
  flavours, processes), reweighting each experiment from its simulation's generator to a common
  reference model; for global fits.
"""
abstract type XsecModel end

"""
    Nominal <: XsecModel

No cross-section systematics: unit weights and no parameters.
"""
struct Nominal <: XsecModel end

"""
    SimpleScaling <: XsecModel

Simple cross-section model applying global normalization factors.

Provides two nuisance parameters:
- `nc_norm`: neutral-current cross-section scale factor.
- `nutau_cc_norm`: ``\\nu_\\tau`` charged-current cross-section scale factor.

All other flavour/interaction combinations return unit scaling.
"""
struct SimpleScaling <: XsecModel end

"""
    Differential_H2O <: XsecModel

Differential cross-section model for water (H₂O) targets.

Provides per-interaction-mode normalization parameters (`cc1p1h_norm`, `cc2p2h_norm`,
`cc1pi_norm`, `ccother_norm`, `ccdis_norm`) in addition to `nc_norm` and `nutau_cc_norm`.
Energy-dependent CC interaction fractions are interpolated from digitized data
(T. Wester, Super-K PhD thesis, Figure 4.7) for ``\\nu_e`` and ``\\bar{\\nu}_e``
separately.

Additional systematic parameters adjust the underlying shape of the interpolated
curves: `xsec_MA_QE`/`xsec_MA_Res` (axial mass form-factor scaling for CCQE and
resonance interactions), `xsec_I12` (isospin-1/2 background normalization for
CCother), `xsec_fsi` (final-state interaction strength, shifting weight between
CC1p1h and CC1pi), an overall `cc_norm`, and `nubar_ratio` (the ``\\bar{\\nu}/\\nu``
cross-section ratio, applied in a normalization-conserving way).
"""
struct Differential_H2O <: XsecModel end

"""
    H2O_PCA <: XsecModel

GENIE-based cross-section model for water (H₂O) targets with per-channel PCA shape and
normalization systematics.

Cross-section curves for six interaction channels (CC1p1h, CC2p2h, CC1π, CCDIS, CCother,
NC) are read from a fixed grid of GENIE generator tunes (`G18_10a`, `G21_11a`, `G18_02a`)
plus the NEUT5.4.0 reference, stored in `xsec_genie_data.jld2`. For each channel, a
principal-component decomposition of the spread across tunes yields a single leading
shape mode; the `xsec_*_shape` nuisance parameters move the nominal curve along that
mode, while `xsec_*_norm` parameters independently scale each channel's overall rate.
`xsec_*_nubar_ratio` parameters allow the ``\\bar{\\nu}/\\nu`` cross-section ratio to
float per channel (normalization-conserving), and `xsec_cc1p1h_nue_numu_ratio` allows an
independent ``\\nu_e/\\nu_\\mu`` CC1p1h ratio (nuclear-model uncertainty, e.g. RFG vs LFG).
Reweighting can be applied per-energy-bin (`scale`/`get_scale`), per-event using GENIE
interaction-mode codes (`scale_event`/`get_scale_event`, with a precomputed
`event_weights`/`get_event_weights` fast path), or on an energy grid without per-event
channel information (`grid_weights`/`get_grid_weights`).

# Fields
- `nominal::Symbol = :NEUT5_4_0`: reference generator tune that defines unit scaling
  (`xsec_*_norm = 1`, `xsec_*_shape = 0`). One of `:NEUT5_4_0`, `:G18_10a`, `:G21_11a`,
  `:G18_02a`.
- `mc_nominal::Symbol = :NEUT5_4_0`: the cross-section model actually used to generate
  the Monte Carlo being reweighted; per-event/grid weights are computed relative to
  this baseline rather than `nominal`.
"""
@kwdef struct H2O_PCA <: XsecModel
    nominal::Symbol = :NEUT5_4_0   # reference curve (norm=1): :NEUT5_4_0, :G18_10a, :G21_11a, :G18_02a
    mc_nominal::Symbol = :NEUT5_4_0 # cross-section model used to generate the MC being reweighted
end

const GENIE_H2O = H2O_PCA  # backward compat alias

"""
    Xsec <: Newtrinos.Physics

Configured cross-section physics module, returned by [`configure`](@ref).

# Fields
- `cfg::XsecModel`: the cross-section model used to build this module.
- `params::NamedTuple`: default normalization parameter values.
- `priors::NamedTuple`: prior distributions for each parameter.
- `scale::Function`: closure that computes the cross-section scale factor.
  See [`get_scale`](@ref) for the call signatures.
- `dσdE::Union{Function, Nothing}`: optional closure returning the absolute
  differential cross-section, when supported by `cfg`; `nothing` otherwise.
- `scale_event::Union{Function, Nothing}`: optional per-event reweighting closure
  using generator interaction codes, when supported by `cfg`; `nothing` otherwise.
- `event_weights::Union{Function, Nothing}`: optional precomputed fast-path
  per-event reweighting closure, when supported by `cfg`; `nothing` otherwise.
- `grid_weights::Union{Function, Nothing}`: optional precomputed fast-path
  per-energy-grid reweighting closure, when supported by `cfg`; `nothing` otherwise.
- `weights::Function`: the common reweighting interface of all models, see [`get_weights`](@ref).
- `sigma::Union{Function, Nothing}`: absolute σ/E with uncertainties, when supported by `cfg`.
"""
@kwdef struct Xsec <: Newtrinos.Physics
    cfg::XsecModel
    params::NamedTuple
    priors::NamedTuple
    scale::Function
    dσdE::Union{Function, Nothing} = nothing
    scale_event::Union{Function, Nothing} = nothing  # per-event reweighting with GENIE interaction codes
    event_weights::Union{Function, Nothing} = nothing # fast path: precompute(E,codes,flav,ia,anti) -> (params)->weights
    grid_weights::Union{Function, Nothing} = nothing  # fast path: precompute(E_grid,flav,ia,anti) -> (params)->weights
    weights::Function
    sigma::Union{Function, Nothing} = nothing
end


"""
    configure(cfg::XsecModel=SimpleScaling()) -> Xsec

Create a fully configured cross-section physics module.

Assembles default parameters, priors, and the scaling closure (plus any optional
`dσdE`, per-event, or grid-reweighting closures supported by `cfg`, falling back to
`nothing` when a model does not implement them) into an [`Xsec`](@ref) struct ready
for use in experiment forward models.

# Arguments
- `cfg::XsecModel`: cross-section model (defaults to [`SimpleScaling`](@ref)).

# Returns
An [`Xsec`](@ref) instance.

# Examples
```julia
using Newtrinos

# Default simple scaling
xsec_physics = Newtrinos.xsec.configure()

# Differential water cross-sections for Super-K
xsec_physics = Newtrinos.xsec.configure(Differential_H2O())
```
"""
function configure(cfg::XsecModel=SimpleScaling())
    weights = get_weights(cfg)
    Xsec(
        cfg=cfg,
        params = get_params(cfg),
        priors = get_priors(cfg),
        scale = get_scale(cfg),
        dσdE = get_dσdE(cfg),
        scale_event = get_scale_event(cfg),
        event_weights = get_event_weights(cfg),
        grid_weights = get_grid_weights(cfg),
        weights = weights,
        sigma = cfg isa CorrelatedXsec ? get_sigma(cfg, weights) : get_sigma(cfg),
        )
end

get_dσdE(::XsecModel) = nothing
get_scale_event(::XsecModel) = nothing
get_event_weights(::XsecModel) = nothing
get_grid_weights(::XsecModel) = nothing
get_sigma(::XsecModel) = nothing

get_params(::Nominal) = (;)
get_priors(::Nominal) = (;)
get_scale(::Nominal) = (args...) -> 1.0
get_weights(::Nominal) = (E; kwargs...) -> (params -> 1.0)

"""
    get_weights(cfg::SimpleScaling)

`weights(E; flavour, anti, current, ...)` returns `params -> nc_norm` for NC, `params -> nutau_cc_norm`
for ντ CC and `params -> 1` otherwise (scalars; target, generator and process are ignored).
"""
function get_weights(cfg::SimpleScaling)
    function weights(E; flavour::Symbol, anti::Bool = false, current::Symbol, kwargs...)
        current === :NC && return params -> params.nc_norm
        flavour === :nutau && return params -> params.nutau_cc_norm
        return params -> one(params.nc_norm)
    end
end

"""
    get_params(cfg::XsecModel) -> NamedTuple

Return the default cross-section normalization parameter values for the given model.

# Arguments
- `cfg::XsecModel`: a [`SimpleScaling`](@ref) or [`Differential_H2O`](@ref) instance.

# Returns
A `NamedTuple` mapping parameter names to their nominal values. Most normalization
parameters default to `1.0`; a few [`Differential_H2O`](@ref) systematic parameters
(`xsec_MA_QE`, `xsec_MA_Res`, `xsec_I12`) default to their central physical values
instead.
"""
function get_params(cfg::SimpleScaling)
    (
        nc_norm = 1.,
        nutau_cc_norm = 1.,
    )
end

"""
    get_priors(cfg::XsecModel) -> NamedTuple

Return prior distributions for each cross-section normalization parameter.

Most priors are truncated normal distributions centred near 1.0; a few
[`Differential_H2O`](@ref) systematic parameters (`xsec_MA_QE`, `xsec_MA_Res`,
`xsec_I12`, `xsec_fsi`) use untruncated `Normal` priors centred on their nominal
physical values instead.

# Arguments
- `cfg::XsecModel`: a [`SimpleScaling`](@ref) or [`Differential_H2O`](@ref) instance.

# Returns
A `NamedTuple` mapping parameter names to `Distributions.UnivariateDistribution` priors.
"""
function get_priors(cfg::SimpleScaling)
    (
        nc_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        nutau_cc_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
    )
end

function get_params(cfg::Differential_H2O)
    (
        nc_norm = 1.,
        nutau_cc_norm = 1.,
        cc1p1h_norm = 1.,
        cc2p2h_norm = 1.,
        cc1pi_norm = 1.,
        ccother_norm = 1.,
        ccdis_norm = 1.,
        xsec_MA_QE = 1.05,
        xsec_MA_Res = 0.95,
        xsec_I12 = 1.30,
        xsec_fsi = 1.,
        cc_norm = 1.,
        nubar_ratio = 1.,
    )
end

function get_priors(cfg::Differential_H2O)
    (
        nc_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        nutau_cc_norm = Truncated(Normal(1, 0.25), 0.3, 1.7),
        cc1p1h_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        cc2p2h_norm = Truncated(Normal(1, 1.0), 0, 3),
        cc1pi_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        ccother_norm = Truncated(Normal(1, 0.4), 0.2, 1.8),
        ccdis_norm = Truncated(Normal(1, 0.1), 0.7, 1.3),
        xsec_MA_QE = Normal(1.05, 0.16),
        xsec_MA_Res = Normal(0.95, 0.15),
        xsec_I12 = Normal(1.30, 0.20),
        xsec_fsi = Normal(1, 0.1),
        cc_norm = Truncated(Normal(1, 0.15), 0.5, 1.5),
        nubar_ratio = Truncated(Normal(1, 0.05), 0.7, 1.3),
    )
end

"""
    get_scale(cfg::XsecModel) -> Function

Construct the cross-section scaling closure for the given model.

**[`SimpleScaling`](@ref)** returns a function with signature:
```julia
scale(flav::Symbol, interaction::Symbol, params::NamedTuple) -> Real
```
Returns `params.nc_norm` for NC interactions, `params.nutau_cc_norm` for ``\\nu_\\tau`` CC,
and `1.0` for everything else.

**[`Differential_H2O`](@ref)** returns a function with signature:
```julia
scale(E::AbstractArray, flav::Symbol, interaction::Symbol, anti::Bool, params::NamedTuple) -> Real or AbstractArray
```
For NC interactions returns `params.nc_norm`. For CC interactions computes an
energy-dependent weighted sum of per-mode normalizations using interpolated
interaction fractions, further reweighted by axial-mass form-factor ratios
(`xsec_MA_QE`, `xsec_MA_Res`), an FSI shift (`xsec_fsi`) between CC1p1h/CC1pi,
an isospin-1/2 background scale (`xsec_I12`) for CCother, an overall `cc_norm`,
and a normalization-conserving ``\\bar{\\nu}/\\nu`` ratio (`nubar_ratio`), with an
additional ``\\nu_\\tau`` CC factor when applicable.

# Arguments
- `cfg::XsecModel`: cross-section model instance.

# Returns
A closure computing the cross-section scale factor.
"""
function get_scale(cfg::SimpleScaling)
    function scale(flav::Symbol, interaction::Symbol, params::NamedTuple)
        if interaction == :NC
            return params.nc_norm
        elseif flav == :nutau
            return params.nutau_cc_norm
        else
            return one(params.nc_norm)
        end
    end
end


function ma_qe_ratio(E, MA)
    # Approximate CCQE σ ratio from dipole form factor
    MA_nom = 1.05
    Q2_typ = 0.5 * E
    r_nom = 1 / (1 + Q2_typ / MA_nom^2)^2
    r_new = 1 / (1 + Q2_typ / MA^2)^2
    return r_new / r_nom
end

function ma_res_ratio(E, MA)
    # Approximate resonance σ ratio from dipole form factor
    MA_nom = 0.95
    Q2_typ = 0.3 * E
    r_nom = 1 / (1 + Q2_typ / MA_nom^2)^2
    r_new = 1 / (1 + Q2_typ / MA^2)^2
    return r_new / r_nom
end

function _load_h2o_interpolations()
    # σ/E per channel on water, digitized from T. Wester Super-K PhD thesis Figure 4.7
    df_nue = CSV.read(joinpath(@__DIR__, "xsec_nue_water.csv"), DataFrame, skipto=3);
    df_nuebar = CSV.read(joinpath(@__DIR__, "xsec_nuebar_water.csv"), DataFrame, skipto=3);

    function make_interpolation(name, df)
        idx = findfirst(==(name), names(df))
        x = collect(skipmissing(df[:,idx]))
        y = collect(skipmissing(df[:,idx+1]))
        itp = interpolate((x,), y, Gridded(Linear()))
        m(x) = max.(0, x)
        return m ∘ extrapolate(itp, Flat())
    end

    nue = (
        CC1p1h = make_interpolation("CC1p1h", df_nue),
        CC2p2h = make_interpolation("CC2p2h", df_nue),
        CC1pi = make_interpolation("CC1pi", df_nue),
        CCother = make_interpolation("CCother", df_nue),
        CCDIS = make_interpolation("CCDIS", df_nue),
        NC = make_interpolation("NC", df_nue),
    )

    nuebar = (
        CC1p1h = make_interpolation("CC1p1h", df_nuebar),
        CC2p2h = make_interpolation("CC2p2h", df_nuebar),
        CC1pi = make_interpolation("CC1pi", df_nuebar),
        CCother = make_interpolation("CCother", df_nuebar),
        CCDIS = make_interpolation("CCDIS", df_nuebar),
        NC = make_interpolation("NC", df_nuebar),
    )

    return nue, nuebar
end

"""Apply MA_QE, MA_Res, FSI, channel norms, cc_norm, nubar_ratio to per-channel σ/E values."""
function _apply_h2o_systematics(E, funs, flav, anti, params)
    ma_qe = ma_qe_ratio.(E, params.xsec_MA_QE)
    ma_res = ma_res_ratio.(E, params.xsec_MA_Res)
    fsi_1p1h = 1 .- 0.1 .* (params.xsec_fsi - 1)
    fsi_1pi = 1 .+ 0.1 .* (params.xsec_fsi - 1)

    s = (funs.CC1p1h.(E) .* params.cc1p1h_norm .* ma_qe .* fsi_1p1h .+
         funs.CC2p2h.(E) .* params.cc2p2h_norm .+
         funs.CC1pi.(E) .* params.cc1pi_norm .* ma_res .* fsi_1pi .+
         funs.CCother.(E) .* params.ccother_norm .* params.xsec_I12 .+
         funs.CCDIS.(E) .* params.ccdis_norm) .* params.cc_norm

    r = params.nubar_ratio
    s = anti ? s .* (2 * r / (1 + r)) : s .* (2 / (1 + r))

    # TODO: nutau CC cross-sections are not yet handled correctly — they use numu
    # curves as proxy, but nutau has a kinematic threshold at ~3.5 GeV (tau mass)
    # and different channel fractions at low energy.
    if flav == :nutau
        return s .* params.nutau_cc_norm
    end
    return s
end

function get_scale(cfg::Differential_H2O)
    nue, nuebar = _load_h2o_interpolations()

    function ratios(funs, E)
        cc_funs = (CC1p1h=funs.CC1p1h, CC2p2h=funs.CC2p2h, CC1pi=funs.CC1pi, CCother=funs.CCother, CCDIS=funs.CCDIS)
        x = map(f -> f.(E), cc_funs)
        total_CC = reduce(+, values(x))
        return map(v -> v ./ total_CC, x)
    end

    function scale(E::AbstractArray, flav::Symbol, interaction::Symbol, anti::Bool, params::NamedTuple)
        if interaction == :NC
            return params.nc_norm
        end

        funs = anti ? nuebar : nue
        rs = ratios(funs, E)

        ma_qe = ma_qe_ratio.(E, params.xsec_MA_QE)
        ma_res = ma_res_ratio.(E, params.xsec_MA_Res)
        fsi_1p1h = 1 .- 0.1 .* (params.xsec_fsi - 1)
        fsi_1pi = 1 .+ 0.1 .* (params.xsec_fsi - 1)

        s = (rs.CC1p1h .* params.cc1p1h_norm .* ma_qe .* fsi_1p1h .+ rs.CC2p2h * params.cc2p2h_norm .+ rs.CC1pi .* params.cc1pi_norm .* ma_res .* fsi_1pi .+ rs.CCother * params.ccother_norm * params.xsec_I12 .+ rs.CCDIS * params.ccdis_norm) .* params.cc_norm

        r = params.nubar_ratio
        s = anti ? s .* (2 * r / (1 + r)) : s .* (2 / (1 + r))

        if flav == :nutau
            return s * params.nutau_cc_norm
        end
        return s
    end
end

function get_dσdE(cfg::Differential_H2O)
    nue, nuebar = _load_h2o_interpolations()

    function dσdE(E::AbstractArray, flav::Symbol, interaction::Symbol, anti::Bool, params::NamedTuple)
        funs = anti ? nuebar : nue

        if interaction == :NC
            return funs.NC.(E) .* params.nc_norm
        end

        return _apply_h2o_systematics(E, funs, flav, anti, params)
    end
end

function get_params(cfg::H2O_PCA)
    (
        xsec_cc1p1h_subgev_norm = 1.,
        xsec_cc1p1h_multigev_norm = 1.,
        xsec_cc2p2h_norm = 1.,
        xsec_cc1pi_norm = 1.,
        xsec_ccdis_norm = 1.,
        xsec_ccother_norm = 1.,
        xsec_nc_norm = 1.,
        xsec_nutau_cc_norm = 1.,
        xsec_cc1p1h_shape = 0.,
        xsec_cc2p2h_shape = 0.,
        xsec_cc1pi_shape = 0.,
        xsec_ccdis_shape = 0.,
        xsec_ccother_shape = 0.,
        xsec_nc_shape = 0.,
        xsec_cc1p1h_nubar_ratio = 1.,
        xsec_cc2p2h_nubar_ratio = 1.,
        xsec_cc1pi_nubar_ratio = 1.,
        xsec_ccdis_nubar_ratio = 1.,
        xsec_ccother_nubar_ratio = 1.,
        xsec_nc_nubar_ratio = 1.,
        xsec_cc1p1h_nue_numu_ratio = 1.,
    )
end

function get_priors(cfg::H2O_PCA)
    (
        xsec_cc1p1h_subgev_norm = Truncated(Normal(1, 0.05), 0.7, 1.3),
        xsec_cc1p1h_multigev_norm = Truncated(Normal(1, 0.25), 0.3, 1.7),
        xsec_cc2p2h_norm = Uniform(0, 2),
        xsec_cc1pi_norm = Truncated(Normal(1, 0.2), 0.4, 1.6),
        xsec_ccdis_norm = Truncated(Normal(1, 0.10), 0.5, 1.5),
        xsec_ccother_norm = Truncated(Normal(1, 0.30), 0.2, 1.8),
        xsec_nc_norm = Truncated(Normal(1, 0.20), 0.4, 1.6),
        xsec_nutau_cc_norm = Truncated(Normal(1, 0.25), 0.3, 1.7),
        xsec_cc1p1h_shape = Normal(0, 1),
        xsec_cc2p2h_shape = Normal(0, 1),
        xsec_cc1pi_shape = Normal(0, 1),
        xsec_ccdis_shape = Normal(0, 1),
        xsec_ccother_shape = Normal(0, 1),
        xsec_nc_shape = Normal(0, 1),
        # ν̄/ν ratio priors: σ from GENIE tune spread of per-process ν̄/ν xsec ratio
        xsec_cc1p1h_nubar_ratio = Truncated(Normal(1, 0.17), 0.3, 1.7),
        xsec_cc2p2h_nubar_ratio = Truncated(Normal(1, 0.17), 0.3, 1.7),
        xsec_cc1pi_nubar_ratio = Truncated(Normal(1, 0.05), 0.7, 1.3),
        xsec_ccdis_nubar_ratio = Truncated(Normal(1, 0.05), 0.7, 1.3),
        xsec_ccother_nubar_ratio = Truncated(Normal(1, 0.10), 0.5, 1.5),
        xsec_nc_nubar_ratio = Truncated(Normal(1, 0.05), 0.7, 1.3),
        # νe/νμ CCQE ratio: nuclear model uncertainty (RFG vs LFG)
        xsec_cc1p1h_nue_numu_ratio = Truncated(Normal(1, 0.05), 0.7, 1.3),
    )
end

const _genie_data_cache = Ref{Union{Nothing, Tuple}}(nothing)
const _genie_data_lock = ReentrantLock()

function _load_genie_data()
    return lock(_genie_data_lock) do
        if !isnothing(_genie_data_cache[])
            return _genie_data_cache[]
        end
    data = load(joinpath(@__DIR__, "xsec_genie_data.jld2"))
    E_grid = data["E_grid"]
    wester_xsec = data["wester_xsec"]  # NEUT5.4.0 cross-sections (key kept for backward compat)
    all_xsec = data["all_xsec"]

    # Trim last 2 E_grid points: GENIE has boundary artifacts there
    # (CCDIS drops ~3% in the final grid step vs ~0.1%/step trend)
    n_trim = findlast(E_grid .<= 950.0)
    E_grid = E_grid[1:n_trim]
    for (_, flav_dict) in wester_xsec
        for (ch, vals) in flav_dict
            flav_dict[ch] = vals[1:n_trim]
        end
    end
    for (_, tune_dict) in all_xsec
        for (_, flav_dict) in tune_dict
            for (ch, vals) in flav_dict
                flav_dict[ch] = vals[1:n_trim]
            end
        end
    end

        result = (E_grid, wester_xsec, all_xsec)
        _genie_data_cache[] = result
        result
    end
end

"""Build a flat-extrapolated linear interpolation of `vals` over `E_grid`."""
function make_itp(vals, E_grid)
    itp = interpolate((E_grid,), vals, Gridded(Linear()))
    return extrapolate(itp, Flat())
end

"""Map `(flav, anti)` to the `wester_xsec`/`all_xsec` flavor key."""
function get_flavor_key(flav::Symbol, anti::Bool)
    if flav == :nue
        return anti ? "nuebar" : "nue"
    elseif flav == :numu || flav == :nutau
        return anti ? "numubar" : "numu"
    else
        return anti ? "nuebar" : "nue"
    end
end

"""Extend a NEUT/Wester-digitized curve above `i_last` using the GENIE shape,
scaled to match at the boundary (no discontinuity)."""
function extend_neut(neut_vals, genie_vals, i_last)
    extended = copy(neut_vals)
    g = genie_vals[i_last]
    n = neut_vals[i_last]
    if g > 0 && n > 0
        sc = n / g
        extended[i_last+1:end] .= genie_vals[i_last+1:end] .* sc
    end
    return extended
end

"""σ/E curve for `(key, flav, ch)`. `NEUT5_4_0` is the digitized Wester curve
extended above its valid range using the G18_10a GENIE shape."""
function get_curve(wester_xsec, all_xsec, key, flav, ch, i_last)
    key == "NEUT5_4_0" ? extend_neut(wester_xsec[flav][ch], all_xsec["G18_10a"][flav][ch], i_last) : all_xsec[key][flav][ch]
end

"""ν+ν̄-averaged σ/E curve for `(source, ch)`, used as PCA input (shape is
flavor-symmetric)."""
function get_ch_curve(wester_xsec, all_xsec, source, ch, i_last)
    if source == "NEUT5_4_0"
        gn = all_xsec["G18_10a"]["nue"][ch]; gnb = all_xsec["G18_10a"]["nuebar"][ch]
        return (extend_neut(wester_xsec["nue"][ch], gn, i_last) .+ extend_neut(wester_xsec["nuebar"][ch], gnb, i_last)) ./ 2
    else
        return (all_xsec[source]["nue"][ch] .+ all_xsec[source]["nuebar"][ch]) ./ 2
    end
end

"""Fit the leading shape-PCA mode of `alt_curves`' fractional deviation from
`nom` (masked to the valid `E_pca_mask` range and tapered), plus the norm-spread
std across nom/alt means over the full `E_grid` range. Returns `(itp, norm_sigma)`."""
function compute_shape_pca(nom, alt_curves, E_grid, E_pca_mask, taper)
    n_E = length(E_grid)
    nom_mean = sum(nom) / n_E
    all_means = Float64[nom_mean]
    for alt in alt_curves
        push!(all_means, sum(alt) / n_E)
    end
    norm_sigma = std(all_means ./ nom_mean)

    n_pca = sum(E_pca_mask)
    Delta = zeros(n_pca, length(alt_curves))
    nom_pca = nom[E_pca_mask]
    nom_pca_mean = sum(nom_pca) / n_pca
    for (i, alt) in enumerate(alt_curves)
        alt_pca = alt[E_pca_mask]
        alt_pca_mean = sum(alt_pca) / n_pca
        alt_rescaled = alt_pca .* (nom_pca_mean / alt_pca_mean)
        frac_dev = (alt_rescaled .- nom_pca) ./ nom_pca
        frac_dev[isnan.(frac_dev) .| isinf.(frac_dev)] .= 0.0
        Delta[:, i] = frac_dev
    end

    U, S, _ = svd(Delta)
    pc_full = zeros(n_E)
    pc_full[E_pca_mask] = U[:, 1] .* S[1]
    pc_full .*= taper

    return make_itp(pc_full, E_grid), norm_sigma
end

const H2O_CC_CHANNELS = ("CC1p1h", "CC2p2h", "CC1pi", "CCDIS", "CCother")
const H2O_FLAV_KEYS = ("nue", "nuebar", "numu", "numubar")
const H2O_NORM = (CC2p2h = :xsec_cc2p2h_norm, CC1pi = :xsec_cc1pi_norm, CCDIS = :xsec_ccdis_norm, CCother = :xsec_ccother_norm)
const H2O_SHAPE = (CC1p1h = :xsec_cc1p1h_shape, CC2p2h = :xsec_cc2p2h_shape, CC1pi = :xsec_cc1pi_shape,
                   CCDIS = :xsec_ccdis_shape, CCother = :xsec_ccother_shape)
const H2O_NUBAR = (CC1p1h = :xsec_cc1p1h_nubar_ratio, CC2p2h = :xsec_cc2p2h_nubar_ratio, CC1pi = :xsec_cc1pi_nubar_ratio,
                   CCDIS = :xsec_ccdis_nubar_ratio, CCother = :xsec_ccother_nubar_ratio)
# GENIE interaction codes (DeepCore data release convention) → channel; NC events are passed with interaction = :NC
const H2O_GENIE_CODE = Dict(0 => "CC1p1h", 1 => "CC1pi", 2 => "CCDIS", 3 => "CCother", -1 => "CCother")

"Normalisation-conserving ratio factor: `2r/(1+r)` for the scaled side, `2/(1+r)` for the other"
ratio_factor(r, scaled::Bool) = scaled ? 2r / (1 + r) : 2 / (1 + r)

"""
Tables shared by all H2O_PCA closures (computed once per closure): the energy grid, the leading
shape PC per channel (and NC per ν/ν̄), the nominal σ/E and channel fractions per flavour, and the
nominal/MC-generator ratios per channel and the MC total CC for reweighting simulations.
"""
function _h2o_pca_setup(cfg::H2O_PCA)
    E_grid, wester_xsec, all_xsec = _load_genie_data()
    nominal_key = string(cfg.nominal)
    mc_key = string(cfg.mc_nominal)

    # NEUT (Wester) curves are only valid up to ~26 GeV and are continued with the GENIE G18_10a shape;
    # shape corrections fade out between 20 and 30 GeV, where the curves are extrapolated
    i_neut_last = findlast(E_grid .<= 26.0)
    E_valid_max, E_fade_start = 30.0, 20.0
    taper = [E < E_fade_start ? 1.0 : E > E_valid_max ? 0.0 : 1.0 - (E - E_fade_start) / (E_valid_max - E_fade_start) for E in E_grid]
    E_pca_mask = E_grid .<= E_valid_max

    alt_keys = [t for t in ("G18_10a", "G21_11a", "G18_02a") if t != nominal_key]
    nominal_key != "NEUT5_4_0" && push!(alt_keys, "NEUT5_4_0")
    curve(key, fk, ch) = get_curve(wester_xsec, all_xsec, key, fk, ch, i_neut_last)

    # shape PCs: per channel from the ν+ν̄ average; NC separately for ν and ν̄
    shape = Dict{String, Any}()
    norm_sigma = Dict{String, Float64}()
    for ch in (H2O_CC_CHANNELS..., "NC")
        nom = get_ch_curve(wester_xsec, all_xsec, nominal_key, ch, i_neut_last)
        alts = [get_ch_curve(wester_xsec, all_xsec, k, ch, i_neut_last) for k in alt_keys]
        shape[ch], norm_sigma[ch] = compute_shape_pca(nom, alts, E_grid, E_pca_mask, taper)
    end
    for (label, fk) in (("NC_nu", "nue"), ("NC_nubar", "nuebar"))
        shape[label], _ = compute_shape_pca(curve(nominal_key, fk, "NC"), [curve(k, fk, "NC") for k in alt_keys], E_grid, E_pca_mask, taper)
    end

    σ_nom = Dict(fk => Dict(ch => make_itp(curve(nominal_key, fk, ch), E_grid) for ch in (H2O_CC_CHANNELS..., "NC")) for fk in H2O_FLAV_KEYS)
    frac = Dict(fk => begin
        total = sum(curve(nominal_key, fk, ch) for ch in H2O_CC_CHANNELS)
        Dict(ch => (f = curve(nominal_key, fk, ch) ./ total; f[isnan.(f) .| isinf.(f)] .= 0.0; make_itp(f, E_grid)) for ch in H2O_CC_CHANNELS)
    end for fk in H2O_FLAV_KEYS)
    # σ_nominal / σ_mc per channel (1 where the MC has none)
    ratio = Dict(ch => Dict(fk => begin
        nom, mc = curve(nominal_key, fk, ch), curve(mc_key, fk, ch)
        make_itp(map(i -> mc[i] > 0 ? nom[i] / mc[i] : 1.0, eachindex(nom)), E_grid)
    end for fk in H2O_FLAV_KEYS) for ch in (H2O_CC_CHANNELS..., "NC"))
    # MC total CC without 2p2h (the reweighted simulations have no 2p2h)
    mc_total_cc = Dict(fk => make_itp(max.(sum(curve(mc_key, fk, ch) for ch in H2O_CC_CHANNELS if ch != "CC2p2h"), 1e-30), E_grid)
                       for fk in H2O_FLAV_KEYS)

    (; norm_sigma, shape, σ_nom, frac, ratio, mc_total_cc)
end

h2o_qe_norm(E, params) = ifelse.(E .< 1.33, params.xsec_cc1p1h_subgev_norm, params.xsec_cc1p1h_multigev_norm)

"""
Reweighted contribution of one CC channel: `max(0, base × norm × (1 + ε shape)) × ν̄/ν factor`
(× νe/νμ factor for CC1p1h); `base` is the channel's fraction, σ/E or σ ratio, `shape` its PC at E.
"""
function h2o_cc_channel(ch, base, shape, E, flav, anti, params)
    T = promote_type(eltype(base), typeof(params.xsec_nc_norm))
    norm = ch == "CC1p1h" ? h2o_qe_norm(E, params) : getproperty(params, H2O_NORM[Symbol(ch)])
    w = max.(zero(T), base .* norm .* (one(T) .+ getproperty(params, H2O_SHAPE[Symbol(ch)]) .* shape)) .*
        ratio_factor(getproperty(params, H2O_NUBAR[Symbol(ch)]), anti)
    ch == "CC1p1h" ? w .* ratio_factor(params.xsec_cc1p1h_nue_numu_ratio, flav == :nue) : w
end

"Reweighted NC: `max(0, base × nc_norm × (1 + ε shape)) × ν̄/ν factor`"
function h2o_nc(base, shape, anti, params)
    T = promote_type(eltype(base), typeof(params.xsec_nc_norm))
    max.(zero(T), base .* params.xsec_nc_norm .* (one(T) .+ params.xsec_nc_shape .* shape)) .* ratio_factor(params.xsec_nc_nubar_ratio, anti)
end

h2o_nutau(w, flav, params) = flav == :nutau ? w .* params.xsec_nutau_cc_norm : w
h2o_nc_shape(t, anti) = anti ? t.shape["NC_nubar"] : t.shape["NC_nu"]

function get_scale(cfg::H2O_PCA)
    t = _h2o_pca_setup(cfg)
    @info "H2O_PCA xsec configured" nominal=cfg.nominal mc_nominal=cfg.mc_nominal norm_sigmas=t.norm_sigma
    # CC: Σ_ch f_ch(E) × norm_ch × (1 + ε_ch PC_ch(E)) with the nominal channel fractions f_ch; NC: nc_norm × shape
    function scale(E::AbstractArray, flav::Symbol, interaction::Symbol, anti::Bool, params::NamedTuple)
        interaction == :NC && return h2o_nc(one(eltype(E)), h2o_nc_shape(t, anti).(E), anti, params)
        fk = get_flavor_key(flav, anti)
        h2o_nutau(sum(h2o_cc_channel(ch, t.frac[fk][ch].(E), t.shape[ch].(E), E, flav, anti, params) for ch in H2O_CC_CHANNELS), flav, params)
    end
end

function get_dσdE(cfg::H2O_PCA)
    t = _h2o_pca_setup(cfg)
    # absolute σ/E: the nominal curves reweighted as in `scale`
    function dσdE(E::AbstractArray, flav::Symbol, interaction::Symbol, anti::Bool, params::NamedTuple)
        fk = get_flavor_key(flav, anti)
        interaction == :NC && return h2o_nc(t.σ_nom[fk]["NC"].(E), h2o_nc_shape(t, anti).(E), anti, params)
        h2o_nutau(sum(h2o_cc_channel(ch, t.σ_nom[fk][ch].(E), t.shape[ch].(E), E, flav, anti, params) for ch in H2O_CC_CHANNELS), flav, params)
    end
end

# Per-event reweighting of a simulation (`mc_nominal`) with GENIE interaction codes: an event of channel ch
# gets σ_nominal,ch / σ_mc,ch × norm × shape × ratios (at nominal parameters σ_nominal/σ_mc, which moves the
# MC to the nominal model). The simulations have no 2p2h, so CC2p2h is not reweighted here.
function h2o_event_tables(t, E, codes, flav, anti)
    fk = get_flavor_key(flav, anti)
    ch_of = [H2O_GENIE_CODE[c] for c in vec(codes)]
    Ev = vec(E)
    map(("CC1p1h", "CC1pi", "CCDIS", "CCother")) do ch
        m = ch_of .== ch
        ch => (base = ifelse.(m, t.ratio[ch][fk].(Ev), 0.0), shape = ifelse.(m, t.shape[ch].(Ev), 0.0))
    end
end

function h2o_event_eval(t, tabs, E, flav, anti, params)
    h2o_nutau(sum(h2o_cc_channel(ch, tab.base, tab.shape, E, flav, anti, params) for (ch, tab) in tabs), flav, params)
end

function get_scale_event(cfg::H2O_PCA)
    t = _h2o_pca_setup(cfg)
    @info "H2O_PCA scale_event configured" nominal=cfg.nominal mc_nominal=cfg.mc_nominal
    function scale_event(E::AbstractArray, genie_codes, flav::Symbol, interaction::Symbol, anti::Bool, params::NamedTuple)
        fk = get_flavor_key(flav, anti)
        interaction == :NC && return h2o_nc(t.ratio["NC"][fk].(vec(E)), h2o_nc_shape(t, anti).(vec(E)), anti, params)
        h2o_event_eval(t, h2o_event_tables(t, E, genie_codes, flav, anti), vec(E), flav, anti, params)
    end
end

# Fast path of `scale_event`: the per-event tables are precomputed, evaluation is vectorised arithmetic.
function get_event_weights(cfg::H2O_PCA)
    t = _h2o_pca_setup(cfg)
    function event_weights(E, genie_codes, flav::Symbol, interaction::Symbol, anti::Bool)
        fk = get_flavor_key(flav, anti)
        Ev = collect(vec(E))
        if interaction == :NC
            base, sh = t.ratio["NC"][fk].(Ev), h2o_nc_shape(t, anti).(Ev)
            return params -> h2o_nc(base, sh, anti, params)
        end
        tabs = h2o_event_tables(t, Ev, genie_codes, flav, anti)
        return params -> h2o_event_eval(t, tabs, Ev, flav, anti, params)
    end
end

# Reweighting of a simulation without interaction codes on a true-energy grid: CC weight
# Σ_ch σ_nominal,ch(E) × norm_ch × shape_ch / σ_mc,CC(E); NC σ_nominal,NC × norm × shape / σ_mc,NC.
function get_grid_weights(cfg::H2O_PCA)
    t = _h2o_pca_setup(cfg)
    function grid_weights(E_eval, flav::Symbol, interaction::Symbol, anti::Bool)
        fk = get_flavor_key(flav, anti)
        Ev = collect(vec(E_eval))
        if interaction == :NC
            base, sh = t.ratio["NC"][fk].(Ev), h2o_nc_shape(t, anti).(Ev)
            return params -> h2o_nc(base, sh, anti, params)
        end
        denom = t.mc_total_cc[fk].(Ev)
        tabs = map(ch -> ch => (base = t.σ_nom[fk][ch].(Ev) ./ denom, shape = t.shape[ch].(Ev)), H2O_CC_CHANNELS)
        return params -> h2o_event_eval(t, tabs, Ev, flav, anti, params)
    end
end

"""
    get_weights(cfg::Union{H2O_PCA, Differential_H2O})

`weights(E; flavour, anti, current, ...)` = `dσdE(params) / dσdE(nominal params)`: the model's change
relative to its nominal curves (target and generator are ignored: the model is for water, and its
nominal is the generator). Multiply the experiment's nominal cross section by it.
"""
function get_weights(cfg::Union{H2O_PCA, Differential_H2O})
    dσ = get_dσdE(cfg)
    p0 = get_params(cfg)
    function weights(E; flavour::Symbol, anti::Bool, current::Symbol, kwargs...)
        Ev = collect(Float64, vec(E))
        d0 = dσ(Ev, flavour, current, anti, p0)
        inv0 = ifelse.(d0 .> 0, 1 ./ max.(d0, 1e-300), 0.0)
        params -> dσ(Ev, flavour, current, anti, params) .* inv0
    end
end

"""
    event_weights(xsec, E, current; flavour, anti, target, generator, process = :all) -> (params -> weights)

Per-event cross-section weights for events of mixed currents (`current[i] ∈ (:CC, :NC)`), built from
`xsec.weights` per current; `process` is `:all` or one process per event (ignored for NC events).
"""
function event_weights(xs, E, current::AbstractArray{Symbol}; flavour::Symbol, anti::Bool, target::Symbol,
                       generator::Symbol, process = :all)
    current = vec(current)
    process = process isa AbstractArray ? vec(process) : process
    idx_nc = findall(current .=== :NC)
    idx_cc = findall(current .!== :NC)
    Ev = collect(Float64, vec(E))
    wnc = xs.weights(Ev[idx_nc]; flavour, anti, current = :NC, target, generator)
    wcc = xs.weights(Ev[idx_cc]; flavour, anti, current = :CC, target, generator,
                     process = process isa AbstractVector ? process[idx_cc] : process)
    n = length(Ev)
    function (params)
        a = wnc(params); b = wcc(params)
        w = Vector{promote_type(eltype(a), eltype(b))}(undef, n)
        w[idx_nc] .= a
        w[idx_cc] .= b
        w
    end
end

include("xsec_correlated.jl")

end
