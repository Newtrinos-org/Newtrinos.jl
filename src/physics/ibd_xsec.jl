module ibd_xsec

using DelimitedFiles
using Interpolations
using PCHIPInterpolation
using DataFrames
using CSV
using ..Newtrinos



"""
    Abstract type for inverse beta decay cross-section models.
"""
abstract type IBDModel end


"""
    StrumiaVissani <: IBDModel

Tabulated cross-section of Strumia and Vissani (https://arxiv.org/abs/astro-ph/0302055).
"""
struct StrumiaVissani <: IBDModel end

"""
    VogelBeacomO1 <: IBDModel

Differential IBD cross-section dσ/dcosθ to first order in 1/M (Vogel & Beacom, https://arxiv.org/abs/hep-ph/9903554),
with the exact kinematic relation Eν(Ee, cosθ) and the Jacobian dEν/dEe, as used in the official Daya Bay
full-data analysis (dgm_reactor_neutrino). Here the functions take the deposited energy Edep = Ee + mₑ:
`xsec(Enu, cosθ)` [1e-41 cm²], `Enu(Edep, cosθ)` [MeV], `jacobian(Enu, Edep, cosθ)` (dEν/dEdep).
Constants: PDG 2024 masses and neutron lifetime, f = 1, g = 1.2701, f₂ = 3.706, phase space factor 1.71465.
"""
@kwdef struct VogelBeacomO1 <: IBDModel
    ElectronMass::Float64 = 0.5109989500
    ProtonMass::Float64 = 938.27208816
    NeutronMass::Float64 = 939.5654205
    NeutronLifeTime::Float64 = 878.4
    PhaseSpaceFactor::Float64 = 1.71465
    g::Float64 = 1.2701
    f::Float64 = 1.0
    f2::Float64 = 3.706
end

@kwdef struct IBDXsec <: Newtrinos.Physics
    cfg::IBDModel
    params::NamedTuple
    priors::NamedTuple
    xsec::Function
    Epos::Function
    Enu::Function
    jacobian::Function = (x...) -> 1.0
end


function configure(cfg::IBDModel=StrumiaVissani())
    IBDXsec(
        cfg = cfg,
        params = get_params(cfg),
        priors = get_priors(cfg),
        xsec = get_xsec(cfg)[1],
        Epos = get_xsec(cfg)[2],
        Enu = get_xsec(cfg)[3],
    )
end


function configure(cfg::VogelBeacomO1)
    xsec, Enu, jacobian = get_xsec(cfg)
    IBDXsec(cfg = cfg, params = (;), priors = (;), xsec = xsec, Epos = (Edep, cosθ) -> Edep - cfg.ElectronMass,
            Enu = Enu, jacobian = jacobian)
end


function get_params(cfg::StrumiaVissani)
    return (;)
end


function get_priors(cfg::StrumiaVissani)
    return (;)
end


function get_xsec(cfg::StrumiaVissani)
    # create spline of cross section from table
    xsec = CSV.read(joinpath(@__DIR__, "ibd_xsec.csv"), DataFrame, header=0, delim=",")
    energies = reshape(Matrix(xsec[!, [1, 5, 9]]), (45))
    xsection = reshape(Matrix(xsec[!, [2, 6, 10]]), (45))
    #Epos1 = [parse(Float64, i) for i in collect(xsec[!, 3])[2:end]]
    #Epos2 = reshape(Matrix(xsec[!, [7, 11]]), (30))
    #println(Epos1)
    #println(Epos2)
    #Epos = vcat(Epos1, Epos2)
    Epos = reshape(Matrix(xsec[!, [3, 7, 11]]), (45))
    _Epos_itp = Interpolator(energies, Epos, extrapolate=true);
    _x_sec_itp = Interpolator(energies, xsection, extrapolate=true);
    _Enu_itp = Interpolator(Epos, energies, extrapolate=true);

    # impose zero as lower bound
    function x_sec_itp(x)
        out = _x_sec_itp(x)
        if out < 0
            return 0.
        else return out
        end
    end

    function E_pos_interp(x)
        out = _Epos_itp(x)
        #if x < energies[2]
        if x < energies[1]
            return 0.
        else
            return out
        end
    end

    function E_nu_interp(x)
        # maps from deposited to neutrino,
        # hence subtract electron mass
        out = _Enu_itp.(x) - 0.511
    end


    return x_sec_itp, E_pos_interp, E_nu_interp
end


function get_xsec(cfg::VogelBeacomO1)
    me, mp, mn = cfg.ElectronMass, cfg.ProtonMass, cfg.NeutronMass
    me2 = me^2
    M = 0.5 * (mn + mp)
    Δ = mn - mp
    y2 = 0.5 * (Δ^2 - me2)
    Enu_threshold = 0.5 * (mn^2 / (mp - me) - mp + me)
    f, g, f2 = cfg.f, cfg.g, cfg.f2
    fsq, gsq = f^2, g^2
    # σ₀ from the neutron lifetime; ħ/e converts the lifetime from s to MeV⁻¹, (ħc)² MeV⁻² to cm²
    ħ = 1.054571817e-34
    qe = 1.602176634e-19
    c = 299792458.0
    σ0 = 2π^2 / (cfg.PhaseSpaceFactor * (fsq + 3gsq) * me2^2 * me * cfg.NeutronLifeTime * 1e6 * qe / ħ)
    MeV2cm = (ħ * c / (1e6 * qe))^2 * 1e4
    # result in units of 1e-41 cm²
    norm = MeV2cm * 1e41

    function xsec(Enu, cosθ)
        Enu < Enu_threshold && return zero(Enu)
        Ee0 = Enu - Δ
        Ee0 <= me && return zero(Enu)
        pe0 = sqrt(Ee0^2 - me2)
        ve0 = pe0 / Ee0
        Ee1 = Ee0 * (1 - Enu / M * (1 - ve0 * cosθ)) - y2 / M
        Ee1 <= me && return zero(Enu)
        pe1 = sqrt(Ee1^2 - me2)
        ve1 = pe1 / Ee1
        σ1a = σ0 * 0.5 * ((fsq + 3gsq) + (fsq - gsq) * ve1 * cosθ) * Ee1 * pe1
        Γ1 = 2g * (f + f2) * ((2Ee0 + Δ) * (1 - ve0 * cosθ) - me2 / Ee0)
        Γ2 = (fsq + gsq) * (Δ * (1 + ve0 * cosθ) + me2 / Ee0)
        A = (Ee0 + Δ) * (1 - cosθ / ve0) - Δ
        Γ3 = (fsq + 3gsq) * A
        Γ4 = (fsq - gsq) * A * ve0 * cosθ
        σ1b = -0.5 * σ0 * Ee0 * pe0 * (Γ1 + Γ2 + Γ3 + Γ4) / M
        norm * (σ1a + σ1b)
    end

    δ = 0.5 * (mn^2 - mp^2 - me2) / mp
    # neutrino energy from the deposited energy and the positron angle (exact two-body kinematics, proton at rest)
    function Enu(Edep, cosθ)
        Ee = Edep - me
        ve = Ee > me ? sqrt(1 - me2 / Ee^2) : zero(Ee)
        (Ee + δ) / (1 - Ee / mp * (1 - ve * cosθ))
    end

    function jacobian(Enu, Edep, cosθ)
        Ee = Edep - me
        Ee <= me && return zero(Enu)
        ve = sqrt(1 - me2 / Ee^2)
        den = mp - Ee * (1 - ve * cosθ)
        den <= 0 && return zero(Enu)
        (mp + Enu * (1 - cosθ / ve)) / den
    end

    return xsec, Enu, jacobian
end

end
