module sno

using LinearAlgebra
using Distributions
using DelimitedFiles
using CairoMakie
import ..Newtrinos
import ..Newtrinos.solar_common

"""
SNO combined analysis of all three phases (B. Aharmim et al., Phys. Rev. C 88, 025501 (2013),
arXiv:1109.0763).

SNO summarises its data by the total active ⁸B flux Φ_B8 (from NC), a quadratic day-time νe
survival probability P_ee^d(E_ν) = c₀ + c₁(E_ν − 10) + c₂(E_ν − 10)² and a linear day/night
asymmetry A_ee(E_ν) = a₀ + a₁(E_ν − 10), with their covariance (Tables XXIV, XXV).
Following the SNO prescription (Sec. VI B), a model is compared by weighting the detectable
E_ν spectra S(E_ν) of CC, ES_e and ES_μτ interactions (Table XXVI) with the model's
oscillation probabilities, fitting the polynomials to the resulting spectra, and comparing the
fitted coefficients to the measured ones.
"""
@kwdef struct SNO <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

default_physics() = solar_common.default_physics()

function configure(physics=default_physics())
    physics = (; physics.osc, physics.solar_flux, physics.solar_xsec, physics.earth_layers)
    assets = get_assets(physics)
    SNO(
        physics = physics,
        params = (;),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets),
    )
end

function get_assets(physics; datadir=@__DIR__)
    @info "Loading SNO data"
    S = readdlm(joinpath(datadir, "sno_sensitivity.csv"), ',', Float64, '\n'; header=true, comments=true)[1]
    E = (S[:, 1] .+ S[:, 2]) ./ 2
    CC_d, CC_n, ESe_d, ESe_n, ESx_d, ESx_n = eachcol(S[:, 3:8])

    # Φ_B8 [1e6 cm⁻² s⁻¹], c₀, c₁, c₂, a₀, a₁: best fit, stat ⊕ syst (symmetrised), correlations
    observed = [5.25, 0.317, 0.0039, -0.0010, 0.046, -0.016]
    stat = [0.16, 0.016, 0.0066, 0.0029, 0.031, 0.025]
    syst = [0.12, 0.009, 0.0045, 0.0015, 0.0135, 0.0105]
    σ = sqrt.(stat .^ 2 .+ syst .^ 2)
    corr = [ 1.000 -0.723  0.302 -0.168  0.028 -0.012
            -0.723  1.000 -0.299 -0.366 -0.376  0.129
             0.302 -0.299  1.000 -0.206  0.219 -0.677
            -0.168 -0.366 -0.206  1.000  0.008 -0.035
             0.028 -0.376  0.219  0.008  1.000 -0.297
            -0.012  0.129 -0.677 -0.035 -0.297  1.000]
    cov = Symmetric(Diagonal(σ) * corr * Diagonal(σ))

    keep = (CC_d .+ ESe_d .+ CC_n .+ ESe_n) .> 0
    (
        observed = observed,
        cov = cov,
        E = E[keep],
        # counts = A·P_ee + B, for day and night (ES_μτ sees 1 − P_ee)
        A_day = (CC_d .+ ESe_d .- ESx_d)[keep], B_day = ESx_d[keep],
        A_night = (CC_n .+ ESe_n .- ESx_n)[keep], B_night = ESx_n[keep],
        site = solar_common.Site(physics, 46.475; depth_km=2.092),  # SNOLAB
    )
end

"""
    day_night_prob(params, physics, assets) -> (P_day, P_night)

Model νe survival probability for ⁸B neutrinos at the SNO energies, during the day and averaged
over the night.
"""
function day_night_prob(params, physics, assets)
    P = solar_common.survival(physics, assets.site, :b8, assets.E, params)
    w = assets.site.w_night ./ sum(assets.site.w_night)
    P.day[:, 1], vec(sum(P.night[:, :, 1] .* w', dims=2))
end

"""
    fit_polynomials(P_day, P_night, assets; iterations=6) -> Vector

Fit SNO's survival probability parameterisation (c₀, c₁, c₂, a₀, a₁) to the model's day and
night probabilities, weighting each E_ν bin by its sensitivity in the detected spectrum
(Gauss–Newton least squares on the predicted event counts with Poisson weights).
"""
function fit_polynomials(P_day, P_night, assets; iterations=6)
    x = assets.E .- 10
    N_d = assets.A_day .* P_day .+ assets.B_day
    N_n = assets.A_night .* P_night .+ assets.B_night
    w_d = assets.A_day .^ 2 ./ max.(N_d, 1e-3)
    w_n = assets.A_night .^ 2 ./ max.(N_n, 1e-3)

    # start: weighted linear fits of P_day and of the asymmetry
    X = hcat(one.(x), x, x .^ 2)
    c = (X' * (w_d .* X)) \ (X' * (w_d .* P_day))
    A_model = 2 .* (P_night .- P_day) ./ (P_night .+ P_day)
    Xa = hcat(one.(x), x)
    a = (Xa' * (w_n .* Xa)) \ (Xa' * (w_n .* A_model))
    θ = vcat(c, a)

    for _ in 1:iterations
        Pd = X * θ[1:3]
        A = Xa * θ[4:5]
        g = (2 .+ A) ./ (2 .- A)
        dg = 4 ./ (2 .- A) .^ 2
        Pn = Pd .* g
        r = vcat(P_day .- Pd, P_night .- Pn)
        J = vcat(hcat(X, zero(Xa)), hcat(g .* X, (Pd .* dg) .* Xa))
        W = vcat(w_d, w_n)
        θ = θ .+ (J' * (W .* J)) \ (J' * (W .* r))
    end
    θ
end

function get_expected(params, physics, assets)
    P_day, P_night = day_night_prob(params, physics, assets)
    vcat(physics.solar_flux.flux(:b8, params) / 1e6, fit_polynomials(P_day, P_night, assets))
end

function get_forward_model(physics, assets)
    function forward_model(params)
        MvNormal(get_expected(params, physics, assets), assets.cov)
    end
end

function get_plot(physics, assets)
    function plot(params, data=assets.observed)
        P_day, P_night = day_night_prob(params, physics, assets)
        θ = fit_polynomials(P_day, P_night, assets)
        val(v) = Float64.(solar_common.ForwardDiffValue.(v))
        x = assets.E .- 10
        poly(c) = c[1] .+ c[2] .* x .+ c[3] .* x .^ 2
        f = Figure()
        ax = Axis(f[1, 1], ylabel="P_ee (day)", title="SNO three-phase combined")
        lines!(ax, assets.E, poly(data[2:4]), color=:black, label="SNO fit")
        lines!(ax, assets.E, val(P_day), color=:red, label="Model")
        lines!(ax, assets.E, poly(val(θ[1:3])), color=:red, linestyle=:dash, label="Model, polynomial")
        axislegend(ax, position=:rt)
        ax2 = Axis(f[2, 1], xlabel="E_ν (MeV)", ylabel="A_ee")
        lines!(ax2, assets.E, data[5] .+ data[6] .* x, color=:black)
        lines!(ax2, assets.E, val(2 .* (P_night .- P_day) ./ (P_night .+ P_day)), color=:red)
        lines!(ax2, assets.E, val(θ[4] .+ θ[5] .* x), color=:red, linestyle=:dash)
        xlims!(ax, 4, 14); xlims!(ax2, 4, 14)
        f
    end
end

end
