module cevns_xsec

using LinearAlgebra
using Distributions
using SpecialFunctions
using ..Newtrinos

"""
    CevnsXsec <: Newtrinos.Physics

Configured Coherent Elastic neutrino-Nucleus Scattering (CEvNS) cross-section module.

Computes the differential cross section ``d\\sigma/dE_r`` for neutrino scattering off
atomic nuclei, including optional beyond-Standard-Model deformations parameterized by
`cevns_xsec_a` through `cevns_xsec_d`.

# Fields
- `params::NamedTuple`: nominal values for ``\\sin^2\\theta_W``, the nuclear radii
  `Rn_*` (one per isotope), and the four BSM deformation parameters.
- `priors::NamedTuple`: prior distributions for each parameter.
- `diff_xsec::Function`: closure returning the differential cross section. See
  [`get_diff_xsec`](@ref) for the call signature.
"""
@kwdef struct CevnsXsec <: Newtrinos.Physics
    params::NamedTuple
    priors::NamedTuple
    diff_xsec::Function
end

const gf=1.1663787e-11
const me = 0.510998
const mmu = 105.6
const mtau=1776.86
const mpi = 139.57
const alph = 1/137
const ep= (mpi^2-mmu^2)/(2*mpi)
const hcut_c = 197.326963 # MeV*fm

"""
    configure(isotopes, er_centers, enu_centers; ff_model=:helm, ff_kwargs=(;)) -> CevnsXsec

Create a [`CevnsXsec`](@ref) module from a list of isotope descriptors and energy grids.

Builds cross-section assets and assembles per-isotope parameters and priors via
[`build_params_and_priors`](@ref) and [`get_assets`](@ref).

# Arguments
- `isotopes`: collection of isotope descriptors. Each entry must have the fields:
  - `Rn_key::Symbol`: parameter name for this isotope's nuclear radius (e.g. `:Rn_Cs133`).
  - `Rn_nom::Float64`: nominal nuclear radius ``R_n`` [fm].
  - `mass::Float64`: nuclear mass [MeV/c²].
  - `Z::Int`: proton number.
  - `N::Int`: neutron number.
  - `fraction::Float64`: isotopic abundance fraction.
- `er_centers`: nuclear recoil energy grid [MeV].
- `enu_centers`: neutrino energy grid [MeV].
- `ff_model::Symbol`: nuclear form-factor model, one of `:helm`, `:klein_nystrand`,
  `:sym_fermi` (default `:helm`). See [`ffsq`](@ref).
- `ff_kwargs::NamedTuple`: extra keyword arguments forwarded to the chosen form-factor
  model (e.g. `(s_fm=0.9,)`).

# Returns
A [`CevnsXsec`](@ref) instance.
"""
function configure(isotopes, er_centers, enu_centers; ff_model::Symbol = :helm, ff_kwargs::NamedTuple = (;))
    # Build assets from isotopes (and store FF choice in assets)
    assets = get_assets(isotopes, er_centers, enu_centers; ff_model = ff_model, ff_kwargs = ff_kwargs)
    params, priors = build_params_and_priors(isotopes)
    CevnsXsec(
        params = params,
        priors = priors,
        diff_xsec = get_diff_xsec(assets),
    )
end

"""
    configure(params::NamedTuple, priors::NamedTuple) -> CevnsXsec

Create a [`CevnsXsec`](@ref) from externally supplied parameter and prior NamedTuples.

Use this constructor when the parameter set has already been assembled (e.g. by a previous
call to [`build_params_and_priors`](@ref)) and no isotope list is available.

# Arguments
- `params::NamedTuple`: parameter values (must include `sin2thetaW`, `cevns_xsec_a`–`d`,
  and one `Rn_*` entry per isotope).
- `priors::NamedTuple`: corresponding prior distributions.

# Returns
A [`CevnsXsec`](@ref) instance.
"""
function configure(params::NamedTuple, priors::NamedTuple)
    CevnsXsec(
        params = params,
        priors = priors,
        diff_xsec_lar = get_diff_xsec_lar(),
        diff_xsec_csi = get_diff_xsec_csi(),
    )
end

"""
    build_params_and_priors(isotopes) -> (NamedTuple, NamedTuple)

Assemble default parameter values and prior distributions for a given set of isotopes.

Includes the weak mixing angle ``\\sin^2\\theta_W`` and four BSM deformation parameters
`cevns_xsec_a` through `cevns_xsec_d` that modify the SM differential cross section.

Per-isotope nuclear radius parameters `Rn_*` are appended automatically using the
`Rn_key` field of each isotope descriptor.

# Arguments
- `isotopes`: collection of isotope descriptors; each must have `Rn_key::Symbol` and
  `Rn_nom::Float64`.

# Returns
A 2-tuple `(params::NamedTuple, priors::NamedTuple)`.
"""
function build_params_and_priors(isotopes)
    # Dynamic parameter/prior builder for isotope-specific Rn keys, using isotope list
    param_dict = Dict(
        :cevns_xsec_a => 1.0,
        :cevns_xsec_b => -0.5,
        :cevns_xsec_c => -1.0,
        :cevns_xsec_d => 1.0,
        :sin2thetaW => 0.231,
    )
    prior_dict = Dict{Symbol, Distributions.Distribution}(
        :cevns_xsec_a => Uniform(0, 2),
        :cevns_xsec_b => Uniform(-2.0, 1.0),
        :cevns_xsec_c => Uniform(-3000, 3000),
        :cevns_xsec_d => Uniform(-1e6, 1e6),
        #:sin2thetaW => truncated(Normal(0.2382, 0.0011), 0.2, 0.26),
        :sin2thetaW => Uniform(0.2, 0.26),
    )
    for iso in isotopes
        param_dict[iso.Rn_key] = iso.Rn_nom
        prior_dict[iso.Rn_key] = truncated(Normal(iso.Rn_nom, iso.Rn_nom*0.05), 0.8*iso.Rn_nom, 1.2*iso.Rn_nom)
        #prior_dict[iso.Rn_key] = Uniform(0.5*iso.Rn_nom, 1.5*iso.Rn_nom) # allow up to ±50% expansion from nominal Rn
    end
    return ((; param_dict...), (; prior_dict...))
end

"""
    get_assets(isotopes, er_centers, enu_centers; ff_model=:helm, ff_kwargs=(;)) -> NamedTuple

Package isotope data, energy grids, and the chosen form-factor model into an assets
NamedTuple for use by [`get_diff_xsec`](@ref).

# Arguments
- `isotopes`: collection of isotope descriptors (same format as [`configure`](@ref)).
- `er_centers`: nuclear recoil energy grid [MeV].
- `enu_centers`: neutrino energy grid [MeV].
- `ff_model::Symbol`: nuclear form-factor model (default `:helm`). See [`ffsq`](@ref).
- `ff_kwargs::NamedTuple`: extra keyword arguments forwarded to the form-factor model.

# Returns
A `NamedTuple` with fields:
- `isotopes`: `Dict{Symbol, NamedTuple}` mapping each `Rn_key` to `(mass, Z, N, fraction, Rn_nom)`.
- `er_centers`: the recoil energy grid.
- `enu_centers`: the neutrino energy grid.
- `ff_model`, `ff_kwargs`: the form-factor model choice and its extra keyword arguments.
"""
function get_assets(isotopes, er_centers, enu_centers; ff_model::Symbol = :helm, ff_kwargs::NamedTuple = (;))
    # Extract isotope data into a structured format
    @info "Configuring CEvNS cross-section assets"
    isotope_data = Dict(iso.Rn_key => (
        mass = iso.mass,
        Z = iso.Z,
        N = iso.N,
        fraction = iso.fraction,
        Rn_nom = iso.Rn_nom
    ) for iso in isotopes)

    return (
        isotopes = isotope_data,
        er_centers = er_centers,
        enu_centers = enu_centers,
        ff_model = ff_model,
        ff_kwargs = ff_kwargs,
    )
end

# ---- Nuclear form factors ----------------------------------------------------
# ffsq(er,mn,rn) remains the public entry point; model selection is optional.

@inline function _q_from_er(er, mn)
    arg = 2 * mn * er
    return sqrt(max(arg, zero(arg))) # typed zero
end

@inline function _three_j1_over_x(x)
    j1 = sphericalbesselj(1, x)
    return iszero(x) ? one(j1) : (3 * j1) / x
end

"""
    ffsq(assets) -> (er,mn,rn) -> F(q)^2

Returns a closure that uses `assets.ff_model` and `assets.ff_kwargs`.
"""
function ffsq(assets)
    model = getproperty(assets, :ff_model)
    kwargs = getproperty(assets, :ff_kwargs)
    return (er, mn, rn) -> ffsq(er, mn, rn; model = model, kwargs...)
end

"""
    ffsq(er, mn, rn; model=:helm, ...)

Supported models (ASCII symbols):
- :helm
- :klein_nystrand
- :sym_fermi
"""
function ffsq(er, mn, rn; model::Symbol = :helm, s_fm::Real = 0.9, a_fm::Real = 0.7, sf_a_fm::Real = 0.523)
    if model === :helm
        # Helm-like: F(q) = 3 j1(q r0)/(q r0) * exp(-(q s)^2/4), so F^2 uses exp(-(q s)^2/2)
        r0_fm = sqrt((5/3)*(rn^2-(3*s_fm^2)))
        r0 = r0_fm / hcut_c
        q = _q_from_er(er, mn)
        denom = q * r0
        ratio = _three_j1_over_x(denom)
        exp_factor = exp(-((q * (s_fm / hcut_c))^2) / 2)
        return (ratio * exp_factor)^2

    elseif model === :klein_nystrand
        # Klein-Nystrand form factor:
        #   F(q) = [3 j1(q R)/(q R)] * 1/(1 + (q a)^2)
        #
        # Citation for this parameterization:
        #   S. R. Klein and J. Nystrand, Phys. Rev. C 60, 014903 (1999).
        q = _q_from_er(er, mn)
        r0_fm = sqrt((rn^2 - 6*a_fm^2)*(5/3))
        R = r0_fm / hcut_c
        a = a_fm / hcut_c
        x = q * R
        hard = _three_j1_over_x(x)
        yuk = inv(one(x) + (q * a)^2)
        return (hard * yuk)^2

    elseif model === :sym_fermi
        # Symmetrized Fermi analytic approximation:
        #   F(q) = (3/(q c)^3) * (sin(qc) - qc cos(qc)) * (pi q a)/sinh(pi q a)
        # with rms relation:
        #   <r^2> = 3/5 c^2 + 7/5 (pi a)^2  =>  c^2 = (5/3) rn^2 - (7/3)(pi a)^2
        #
        # Citation for this closed form used in recoil calculations:
        #   J. D. Lewin and P. F. Smith, Astropart. Phys. 6 (1996) 87-112.
        q = _q_from_er(er, mn)

        a = sf_a_fm
        c2 = (5 / 3) * rn^2 - (7 / 3) * (pi * a)^2
        if c2 <= 0
            return zero(er)
        end
        c_fm = sqrt(c2)

        aM = a / hcut_c
        cM = c_fm / hcut_c

        qc = q * cM
        qa = q * aM

        num = sin(qc) - qc * cos(qc)
        shape = iszero(qc) ? (one(qc) / 3) : (num / (qc^3))

        pqa = pi * qa
        skin = iszero(pqa) ? one(pqa) : (pqa / sinh(pqa))

        F = 3 * shape * skin
        return F^2

    else
        throw(ArgumentError("Unknown model=$(model). Use :helm, :klein_nystrand, or :sym_fermi."))
    end
end

"""
    ds(er, enu, params, nupar, Rn_key; ffsq_fn=ffsq) -> Matrix

Vectorized differential CEvNS cross section ``d\\sigma/dE_r`` on an
``(N_{E_r} \\times N_{E_\\nu})`` grid.

Implements the BSM-extended SM formula (with ``x = E_r/E_\\nu``):

```math
\\frac{d\\sigma}{dE_r} = \\frac{G_F^2}{4\\pi}\\,M_N\\,Q_W^2\\,F^2(q^2)\\;
    \\max\\!\\Bigl(0,\\; a + b\\,\\tfrac{M_N E_r}{E_\\nu^2} + c\\,x + d\\,x^2\\Bigr)
```

where ``Q_W = N - (1 - 4\\sin^2\\theta_W)\\,Z`` and `a`–`d` are the `cevns_xsec_a`–`d`
deformation parameters (the SM cross section is recovered at their nominal values from
[`build_params_and_priors`](@ref)). The bracket is additionally zeroed outside the
kinematically allowed region ``1 - M_N E_r / (2 E_\\nu^2) \\geq 0``.

# Arguments
- `er`: recoil energy grid [MeV], shape `(n_er,)`.
- `enu`: neutrino energy grid [MeV], shape `(n_enu,)`.
- `params::NamedTuple`: must include `sin2thetaW`, `cevns_xsec_a`–`d`, and `params[Rn_key]`.
- `nupar`: 3-tuple `(mass, Z, N)` — nuclear mass [MeV/c²], proton number, neutron number.
- `Rn_key::Symbol`: key used to look up the nuclear radius ``R_n`` in `params`.
- `ffsq_fn::Function`: form-factor function, `(er, mn, rn) -> F^2` (default [`ffsq`](@ref)).

# Returns
Matrix of shape `(n_er, n_enu)` with units [MeV⁻¹].
"""
function ds(er, enu, params, nupar, Rn_key; ffsq_fn::Function = ffsq)
    mN = nupar[1]
    Z  = nupar[2]
    N  = nupar[3]
    rn = params[Rn_key]
    sw2 = params.sin2thetaW
    qwsq = (N - (1 - 4 * sw2) * Z)^2

    # Per-Er prefactor (n_er,) using the provided FF function
    C1d = (gf^2 / (4 * pi)) * qwsq .* ffsq_fn.(er, mN, rn)

    c1 = params.cevns_xsec_a # nominal = 1.0
    c2 = params.cevns_xsec_b #nominal = -0.5
    c3 = params.cevns_xsec_c # nominal = -1.0
    c4 = params.cevns_xsec_d # nominal = 1.0

    er_grid  = reshape(er, :, 1)
    enu_grid = reshape(enu, 1, :)

    base2 = mN .* er_grid ./ (enu_grid.^2)
    base3 = er_grid ./ enu_grid
    base4 = (er_grid.^2) ./ (enu_grid.^2)

    # Kinematic hard cut mask (0/1)
    T = eltype(base2)
    kin = one(T) .- 0.5 .* base2
    kin_mask = ifelse.(kin .>= zero(T), one(T), zero(T))

    xf = c1 .+ c2 .* base2 .+ c3 .* base3 .+ c4 .* base4
    xf_masked = xf .* kin_mask

    # Zero Clamp
    zxf = zero(eltype(xf_masked))
    heav = max.(xf_masked, zxf)

    return (C1d .* mN) .* heav
end

"""
    get_diff_xsec_lar() -> Function

Return a closure that calls [`ds`](@ref) for the liquid-argon (LAr) target.

The returned function has signature:
```julia
diff_xsec_lar(er_centers, enu_centers, params, nupar, Rn_key) -> Matrix
```
"""
function get_diff_xsec_lar()
    function diff_xsec_lar(er_centers, enu_centers, params, nupar, Rn_key)
        ds(er_centers, enu_centers, params, nupar, Rn_key)
    end
end

"""
    get_diff_xsec_csi() -> Function

Return a closure that calls [`ds`](@ref) for the CsI target.

The returned function has signature:
```julia
diff_xsec_csi(er_centers, enu_centers, params, nupar, Rn_key) -> Matrix
```
"""
function get_diff_xsec_csi()
    function diff_xsec_csi(er_centers, enu_centers, params, nupar, Rn_key)
        ds(er_centers, enu_centers, params, nupar, Rn_key)
    end
end

"""
    get_diff_xsec(assets) -> Function

Build the differential cross-section closure over all isotopes from pre-computed assets.

The energy grids and isotope data are captured from `assets` at construction time.
The returned function has signature:
```julia
diff_xsec(params::NamedTuple) -> Dict{Symbol, Matrix}
```
It iterates over all isotopes, calling [`ds`](@ref) for each, and returns a `Dict`
mapping each `Rn_key` to its ``(n_{E_r} \\times n_{E_\\nu})`` cross-section matrix.

# Arguments
- `assets`: NamedTuple produced by [`get_assets`](@ref).

# Returns
A callable closure `params -> Dict{Symbol, Matrix}`.
"""
function get_diff_xsec(assets)
    er_centers = assets.er_centers
    enu_centers = assets.enu_centers
    isotopes = assets.isotopes

    # Configure the form-factor function from assets (model+kwargs live here)
    ffsq_fn = ffsq(assets)

    return function (params)
        #param_type = eltype(params[:cevns_xsec_a])
        param_type = promote_type(typeof(params.cevns_xsec_a), typeof(params.sin2thetaW))
        xsec_dict = Dict{Symbol, Matrix{param_type}}()

        for (Rn_key, iso) in isotopes
            mass = iso.mass
            Z = iso.Z
            N = iso.N
            xsec_dict[Rn_key] = ds(er_centers, enu_centers, params, (mass, Z, N), Rn_key; ffsq_fn = ffsq_fn)
        end

        return xsec_dict
    end
end
end # module cevns_xsec
