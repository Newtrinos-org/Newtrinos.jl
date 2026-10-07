using LinearAlgebra
using StaticArrays
using StructArrays
using ArraysOfArrays
using ForwardDiff
using Distributions
using DensityInterface

# Approximate solar electron density, n_e / N_A = 245 exp(-10.54 r/R☉) mol/cm³ (Bahcall)
ne_sun(x) = 245.0 * exp(-10.54 * x)
const R_SUN_KM = 6.957e5

# Production region concentrated around x = r/R☉ ≈ 0.05 (⁸B-like); neutron density ≈ ne/2
function toy_production(; x = range(0.01, 0.15, length=29), x0 = 0.05, σ = 0.02)
    ne = ne_sun.(x)
    (; ne = collect(ne), nn = collect(0.5 .* ne), w = collect(exp.(-0.5 .* ((x .- x0) ./ σ) .^ 2)))
end

# Reference: propagate a νe produced at x0 radially out through n_shells constant-density
# shells and project onto vacuum mass eigenstates: P(νe → νᵢ) = |(U† S eₑ)ᵢ|²
function shell_mass_fractions(osc, params, e, x0; n_shells=40000)
    U, h = Newtrinos.osc._solar_matrices(osc.cfg, params)
    H_eff = U * Diagonal(h) * U'
    edges = range(x0, 1.0, length=n_shells + 1)
    mids = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    layers = [Newtrinos.osc.Layer(0.0, ne_sun(x), 0.5 * ne_sun(x)) for x in mids]
    mm = [Newtrinos.osc.compute_matter_matrices(H_eff, e, l, false, Newtrinos.osc.SI()) for l in layers]
    path = [Newtrinos.osc.Path(step(edges) * R_SUN_KM, i) for i in eachindex(layers)]
    S = Newtrinos.osc.path_amplitude(mm, path, e)
    abs2.(U' * S[:, 1])
end

@testset "Solar oscillations" begin
    osc_si = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction=Newtrinos.osc.SI()))
    osc_vac = Newtrinos.osc.configure()
    p = osc_si.params
    U = Newtrinos.osc.get_PMNS(p)
    s12², c13² = sin(p.θ₁₂)^2, cos(p.θ₁₃)^2
    prod = toy_production()

    @testset "Mass fractions" begin
        E = [0.1, 1.0, 5.0, 10.0, 15.0] .* 1e-3
        F = Newtrinos.osc.solar_mass_fractions(Newtrinos.osc._solar_matrices(osc_si.cfg, p)..., E, prod, Newtrinos.osc.SI())
        @test size(F) == (5, 3)
        @test all(sum(F, dims=2) .≈ 1)
        # νe content in ν3: two-flavour θ₁₃ matter angle with V = 2√2 G_F nₑ E, Δm²_ee = c₁₂²Δm²₃₁ + s₁₂²Δm²₃₂
        x0 = 0.05
        prod0 = (; ne = [ne_sun(x0)], nn = [0.5 * ne_sun(x0)], w = [1.0])
        F0 = Newtrinos.osc.solar_mass_fractions(Newtrinos.osc._solar_matrices(osc_si.cfg, p)..., E, prod0, Newtrinos.osc.SI())
        V = 2 * Newtrinos.osc.A * ne_sun(x0) .* E .* 1e9
        Δm²ee = p.Δm²₃₁ - s12² * p.Δm²₂₁
        c2θ13m = @. (cos(2p.θ₁₃) - V / Δm²ee) / sqrt((cos(2p.θ₁₃) - V / Δm²ee)^2 + sin(2p.θ₁₃)^2)
        @test F0[:, 3] ≈ 0.5 .* (1 .- c2θ13m) rtol=0.01
        # low energy: vacuum-like, high energy: mostly ν2
        @test F[1, 1] ≈ c13² * (1 - s12²) rtol=0.02
        @test F[5, 2] > 0.9
        # vacuum interaction: |U_ei|²
        Fv = Newtrinos.osc.solar_mass_fractions(U, [0.0, p.Δm²₂₁, p.Δm²₃₁], E, prod, Newtrinos.osc.Vacuum())
        @test all(Fv .≈ transpose(abs2.(U[1, :])))
    end

    @testset "Survival probability limits" begin
        E = [0.1e-3, 50e-3]
        Pee = osc_si.solar_prob(E, prod, p)[:, 1]
        sin²2θ12 = sin(2p.θ₁₂)^2
        s13⁴ = sin(p.θ₁₃)^4
        @test Pee[1] ≈ c13²^2 * (1 - 0.5 * sin²2θ12) + s13⁴ rtol=0.01
        # standard 2ν + θ₁₃ approximation: P_ee = c₁₃⁴ P₂(c₁₃² nₑ) + s₁₃⁴ with
        # P₂ = ½ + ½ cos2θ₁₂ cos2θ₁₂ᵐ, for a single production point
        x0 = 0.05
        prod0 = (; ne = [ne_sun(x0)], nn = [0.5 * ne_sun(x0)], w = [1.0])
        for E_MeV in (1.0, 5.0, 10.0)
            β = 2 * Newtrinos.osc.A * ne_sun(x0) * E_MeV * 1e6 * c13² / p.Δm²₂₁
            c2θ = cos(2p.θ₁₂)
            c2θm = (c2θ - β) / sqrt((c2θ - β)^2 + sin²2θ12)
            P_approx = c13²^2 * (0.5 + 0.5 * c2θ * c2θm) + s13⁴
            @test osc_si.solar_prob([E_MeV * 1e-3], prod0, p)[1, 1] ≈ P_approx atol=2e-3
        end
        # vacuum: averaged oscillations at all energies
        Pv = osc_vac.solar_prob(E, prod, p)[:, 1]
        @test all(Pv .≈ sum(abs2.(U[1, :]) .^ 2))
        # unitarity
        @test all(sum(osc_si.solar_prob(E, prod, p), dims=2) .≈ 1)
    end

    @testset "Adiabatic vs shell-by-shell propagation" begin
        U, h = Newtrinos.osc._solar_matrices(osc_si.cfg, p)
        x0 = 0.05
        prod0 = (; ne = [ne_sun(x0)], nn = [0.5 * ne_sun(x0)], w = [1.0])
        for E_MeV in (0.3, 2.0, 6.0, 12.0)
            e = E_MeV * 1e-3
            F = Newtrinos.osc.solar_mass_fractions(U, h, [e], prod0, Newtrinos.osc.SI())
            @test vec(F) ≈ shell_mass_fractions(osc_si, p, e, x0) atol=2e-3
        end
        # inverted ordering exercises the eigenvalue labelling by continuity
        osc_io = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(
            flavour=Newtrinos.osc.ThreeFlavour(ordering=:IO), interaction=Newtrinos.osc.SI()))
        U, h = Newtrinos.osc._solar_matrices(osc_io.cfg, osc_io.params)
        F = Newtrinos.osc.solar_mass_fractions(U, h, [8e-3], prod0, Newtrinos.osc.SI())
        @test vec(F) ≈ shell_mass_fractions(osc_io, osc_io.params, 8e-3, x0) atol=2e-3
    end

    @testset "Earth regeneration" begin
        earth = Newtrinos.earth_layers.configure()
        layers = earth.compute_layers()
        cz = collect(range(-1, -0.05, length=20))
        paths = earth.compute_paths(vcat(cz, 0.5), layers; r_detector=6370.0)
        E = [5.0, 10.0] .* 1e-3
        P_night = osc_si.solar_prob(E, prod, paths, layers, p)
        P_day = osc_si.solar_prob(E, prod, p)
        @test size(P_night) == (2, 21, 3)
        @test all(sum(P_night, dims=3) .≈ 1)
        # from above only ~1 km of rock is crossed
        @test P_night[:, end, :] ≈ P_day atol=1e-3
        # regeneration increases P_ee at night, by O(1%) at ⁸B energies, more at higher energy
        ΔP = vec(sum(P_night[:, 1:end-1, 1], dims=2)) ./ length(cz) .- P_day[:, 1]
        @test all(0.002 .< ΔP .< 0.05)
        @test ΔP[2] > ΔP[1]
        # vacuum Earth: no regeneration
        layers0 = StructArray{Newtrinos.osc.Layer}((radius=layers.radius, p_density=zero(layers.p_density), n_density=zero(layers.n_density)))
        P0 = osc_si.solar_prob(E, prod, paths, layers0, p)
        @test all(P0[:, i, :] ≈ P_day for i in axes(P0, 2))

        # single constant-density slab vs direct matrix exponential of the Hamiltonian
        U, h = Newtrinos.osc._solar_matrices(osc_si.cfg, p)
        slab = StructArray{Newtrinos.osc.Layer}((radius=[6371.0], p_density=[2.5], n_density=[2.6]))
        L = 3000.0
        R = Newtrinos.osc.earth_mass_to_flavour(U, h, [8e-3], VectorOfVectors{Newtrinos.osc.Path}([[Newtrinos.osc.Path(L, 1)]]), slab, Newtrinos.osc.SI())
        e = 8e-3
        V = Newtrinos.osc.A * e * 1e9 .* Diagonal([2 * 2.5 - 2.6, -2.6, -2.6])
        S = exp(-1im * Newtrinos.osc.F_units * L / e * Matrix(U * Diagonal(h) * U' + V))
        @test R[1, 1, :, :] ≈ transpose(abs2.(S * U)) rtol=1e-8
    end

    @testset "path_amplitude refactor" begin
        # osc_reduce(::Basic) still equals the matter osc_prob result
        osc_m = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction=Newtrinos.osc.SI()))
        earth = Newtrinos.earth_layers.configure()
        layers = earth.compute_layers()
        paths = earth.compute_paths([-0.8], layers)
        P = osc_m.osc_prob([5.0], paths, layers, p)
        @test all(sum(P, dims=4) .≈ 1)
    end

    @testset "ForwardDiff" begin
        earth = Newtrinos.earth_layers.configure()
        layers = earth.compute_layers()
        paths = earth.compute_paths([-0.7], layers; r_detector=6370.0)
        E = [3.0, 9.0] .* 1e-3
        f(x) = sum(osc_si.solar_prob(E, prod, paths, layers, merge(p, (θ₁₂=x[1], Δm²₂₁=x[2])))[:, :, 1])
        x0 = [p.θ₁₂, p.Δm²₂₁]
        g = ForwardDiff.gradient(f, x0)
        δ = [1e-6, 1e-11]
        g_fd = [(f(x0 .+ δ .* (1:2 .== i)) - f(x0 .- δ .* (1:2 .== i))) / 2δ[i] for i in 1:2]
        @test g ≈ g_fd rtol=1e-4
    end
end

@testset "Solar flux" begin
    sf = Newtrinos.solar_flux.configure()
    @test keys(sf.params) == keys(sf.priors)
    @test sf.nominal.b8 ≈ 5.127e6
    @test size(sf.correlation) == (8, 8) && sf.correlation ≈ sf.correlation' && all(diag(sf.correlation) .== 1)
    # continuous spectra are normalised to the total flux
    E = range(0.0, 20.0, length=40001)
    for c in (:pp, :hep, :b8, :n13, :o15, :f17)
        @test sum(sf.spectrum(c, E, sf.params)) * step(E) ≈ sf.nominal[c] rtol=1e-4
    end
    # flux normalisation and ⁸B shape parameters
    ib8 = Newtrinos.solar_flux.component_index(:b8)
    p = merge(sf.params, (solar_norms = [i == ib8 ? 1.1 : 1.0 for i in eachindex(sf.params.solar_norms)], solar_b8_shape = 1.0))
    @test sf.flux(:b8, p) ≈ 1.1 * sf.nominal.b8
    @test sum(sf.spectrum(:b8, E, p)) * step(E) ≈ 1.1 * sf.nominal.b8 rtol=1e-3
    @test sf.spectrum(:b8, [14.0], p)[1] > 1.1 * sf.spectrum(:b8, [14.0], sf.params)[1]
    # production regions: normalised, ⁸B more central than pp, central nₑ ≈ 100 N_A/cm³
    for c in Newtrinos.solar_flux.COMPONENTS
        @test sum(sf.production[c].w) ≈ 1
    end
    mean_r(c) = sum(sf.production[c].r .* sf.production[c].w)
    @test mean_r(:b8) < mean_r(:be7) < mean_r(:pp)
    @test 90 < sf.production.b8.ne[1] < 101
    # other compositions
    sf_gs = Newtrinos.solar_flux.configure(Newtrinos.solar_flux.SolarFluxConfig(model=Newtrinos.solar_flux.B23(composition=:GS98)))
    @test sf_gs.nominal.b8 != sf.nominal.b8
    # default: one vector parameter with the correlated solar-model prior
    @test sf.priors.solar_norms isa MvNormal
    @test sf.priors.solar_norms == Newtrinos.solar_flux.ssm_prior(sf)
    ibe7 = Newtrinos.solar_flux.component_index(:be7)
    Σ = cov(sf.priors.solar_norms)
    @test sqrt(Σ[ib8, ib8]) ≈ sf.fractional_error.b8
    @test Σ[ib8, ibe7] / sqrt(Σ[ib8, ib8] * Σ[ibe7, ibe7]) ≈ sf.correlation[ib8, ibe7]
    # independent scalar normalisations
    sf_ind = Newtrinos.solar_flux.configure(Newtrinos.solar_flux.SolarFluxConfig(systematics=Newtrinos.solar_flux.SSMPriors()))
    @test keys(sf_ind.params) == keys(sf_ind.priors)
    @test sf_ind.flux(:b8, merge(sf_ind.params, (solar_norm_b8 = 1.1,))) ≈ sf.flux(:b8, p)
    @test sf_ind.spectrum(:b8, E, merge(sf_ind.params, (solar_norm_b8 = 1.1, solar_b8_shape = 1.0))) ≈ sf.spectrum(:b8, E, p)
    # yearly exposure: half night, ~flux-weighted; Kamioka never has the Sun at the zenith
    ex = Newtrinos.solar_flux.nadir_exposure(36.43; cz_edges=range(-1, 1, length=201))
    @test sum(ex.w) ≈ 1
    @test sum(ex.w[ex.cz .< 0]) ≈ 0.5 atol=0.01
    # Kamioka (36.4° N): minimum solar zenith angle 36.4° - 23.4° = 13°
    @test all(ex.w[ex.cz .> 0.98] .== 0)
    @test sum(ex.w[0.96 .< ex.cz .< 0.98]) > 0
end

@testset "Solar cross sections" begin
    xs = Newtrinos.solar_xsec.configure()
    X = Newtrinos.solar_xsec
    @test keys(xs.params) == keys(xs.priors)
    E = 10.0
    T = range(0, X.t_max(E), length=20001)
    σe = sum(xs.dσdT_es.(:e, E, T)) * step(T)
    σx = sum(xs.dσdT_es.(:x, E, T)) * step(T)
    # tree level: 9.22e-44 cm² at 10 MeV; radiative corrections lower it by ~2%
    @test σe ≈ 9.22e-44 * 0.98 rtol=0.01
    @test 0.15 < σx / σe < 0.19
    @test xs.dσdT_es(:e, E, X.t_max(E) + 0.01) == 0
    p = xs.params
    @test xs.capture(:cl37, 0.5, p) == 0
    @test xs.capture(:ga71, 0.2, p) == 0
    @test xs.capture(:cl37, 10.0, p) ≈ 3.0e-42
    # ±3σ tables (Bahcall 1997), interpolated between 0.8 and 0.9 MeV
    @test xs.capture(:ga71, 0.8613, merge(p, (ga_xsec_sigma = 3.0,))) ≈ (82.61 + 0.613 * (103.1 - 82.61)) * 1e-46 rtol=1e-6
    @test xs.capture(:ga71, 0.8613, merge(p, (ga_xsec_sigma = -3.0,))) ≈ (62.45 + 0.613 * (75.05 - 62.45)) * 1e-46 rtol=1e-6
    @test_throws ArgumentError xs.capture(:h2o, 10.0, p)
    # spectrum-averaged chlorine cross sections (Bahcall, Neutrino Astrophysics, Table 8.2) [1e-46 cm²]
    sf = Newtrinos.solar_flux.configure()
    Eall = range(0.0, 19.0, length=40000)
    avg_σ(c, target) = (s = sf.spectrum(c, Eall, sf.params); sum(s .* xs.capture.(target, Eall, Ref(p))) / sum(s) / 1e-46)
    @test avg_σ(:b8, :cl37) ≈ 1.14e4 rtol=0.02
    @test avg_σ(:n13, :cl37) ≈ 1.7 rtol=0.05
    @test avg_σ(:o15, :cl37) ≈ 6.8 rtol=0.05
    @test 0.8956 * xs.capture(:cl37, 0.8613, p) ≈ 2.38e-46 rtol=1e-4  # per ⁷Be neutrino (both lines)
    # gallium (Bahcall 1997, Table II): pp 11.72, ⁸B 2.40e4
    @test avg_σ(:pp, :ga71) ≈ 11.72 rtol=0.02
    @test avg_σ(:b8, :ga71) ≈ 2.40e4 rtol=0.03
end

@testset "Radiochemical experiments" begin
    physics = Newtrinos.solar_common.default_physics()
    cl = Newtrinos.chlorine.configure(physics)
    ga = Newtrinos.gallex_gno.configure(physics)
    sg = Newtrinos.sage.configure(physics)
    p = Newtrinos.get_params((; cl, ga, sg))
    @test keys(Newtrinos.get_priors((; cl, ga, sg))) == keys(p)
    # LMA predictions: Cl ≈ 2.7–2.9 SNU, Ga ≈ 63–68 SNU
    r_cl = Newtrinos.chlorine.get_expected(p, cl.physics, cl.assets)
    r_ga = Newtrinos.gallex_gno.get_expected(p, ga.physics, ga.assets)
    @test 2.6 < sum(r_cl) < 3.0
    @test 62 < sum(r_ga) < 68
    @test r_cl.pp == 0
    @test r_ga.pp > r_ga.be7 > r_ga.b8
    # without oscillations: Cl ≈ 7.4 SNU, Ga ≈ 126 SNU for B23 MB22-met
    phys0 = merge(physics, (osc = Newtrinos.osc.configure(),))
    p0 = merge(p, (θ₁₂ = 0.0, θ₁₃ = 0.0))
    cl0 = Newtrinos.chlorine.configure(phys0)
    ga0 = Newtrinos.gallex_gno.configure(phys0)
    @test sum(Newtrinos.chlorine.get_expected(p0, cl0.physics, cl0.assets)) ≈ 7.4 rtol=0.03
    @test sum(Newtrinos.gallex_gno.get_expected(p0, ga0.physics, ga0.assets)) ≈ 126 rtol=0.03
    # joint likelihood is differentiable in the solar parameters and nuisances
    llh = Newtrinos.generate_likelihood((; cl, ga, sg))
    f(x) = logdensityof(llh, merge(p, (θ₁₂ = x[1], Δm²₂₁ = x[2], ga_xsec_sigma = x[3])))
    x0 = [p.θ₁₂, p.Δm²₂₁, 0.3]
    g = ForwardDiff.gradient(f, x0)
    δ = [1e-6, 1e-11, 1e-6]
    g_fd = [(f(x0 .+ δ .* (1:3 .== i)) - f(x0 .- δ .* (1:3 .== i))) / 2δ[i] for i in 1:3]
    @test g ≈ g_fd rtol=1e-4
end

@testset "SNO" begin
    s = Newtrinos.sno.configure()
    p = Newtrinos.get_params((; s))
    # the polynomial fit recovers exact polynomial inputs
    x = s.assets.E .- 10
    θ_true = [0.32, 0.004, -0.001, 0.05, -0.01]
    Pd = θ_true[1] .+ θ_true[2] .* x .+ θ_true[3] .* x .^ 2
    A = θ_true[4] .+ θ_true[5] .* x
    Pn = Pd .* (2 .+ A) ./ (2 .- A)
    @test Newtrinos.sno.fit_polynomials(Pd, Pn, s.assets) ≈ θ_true rtol=1e-8
    # at the SNO-only two-flavour best fit (tan²θ₁₂ = 0.427, Δm²₂₁ = 5.6e-5 eV²) the projected
    # coefficients are close to the measured ones
    p2 = merge(p, (θ₁₂ = atan(sqrt(0.427)), Δm²₂₁ = 5.6e-5, θ₁₃ = 0.0))
    e = Newtrinos.sno.get_expected(p2, s.physics, s.assets)
    @test e[2] ≈ 0.317 atol=0.005
    @test e[5] ≈ 0.046 atol=0.02
    @test e[1] ≈ s.physics.solar_flux.nominal.b8 / 1e6
    # larger Δm²₂₁ gives a smaller day/night asymmetry
    e3 = Newtrinos.sno.get_expected(merge(p2, (Δm²₂₁ = 8e-5,)), s.physics, s.assets)
    @test e3[5] < e[5]
    llh = Newtrinos.generate_likelihood((; s))
    ib8 = Newtrinos.solar_flux.component_index(:b8)
    f(y) = logdensityof(llh, merge(p, (θ₁₂ = y[1], Δm²₂₁ = y[2], solar_norms = [i == ib8 ? y[3] : one(y[3]) for i in 1:8])))
    y0 = [p.θ₁₂, p.Δm²₂₁, 1.02]
    g = ForwardDiff.gradient(f, y0)
    δ = [1e-6, 1e-11, 1e-6]
    g_fd = [(f(y0 .+ δ .* (1:3 .== i)) - f(y0 .- δ .* (1:3 .== i))) / 2δ[i] for i in 1:3]
    @test g ≈ g_fd rtol=1e-4
end

@testset "Super-K solar" begin
    physics = Newtrinos.solar_common.default_physics()
    A_DN(e, s) = (d = sum(e[s.zenith .== "day"]); n = sum(e[s.zenith .== "night"]); (d - n) / ((d + n) / 2))
    for (name, npoints) in ((:sk1_solar, 44), (:sk2_solar, 33), (:sk3_solar, 42), (:sk4_solar, 46))
        m = getproperty(Newtrinos, name)
        sk = m.configure(physics)
        p = Newtrinos.get_params((; sk))
        a = sk.assets
        @test length(a.observed) == npoints
        @test all(sum(a.W, dims=2) .≈ 1)
        mc = (a.mc_b8 .+ a.mc_hep)[a.bin_of]
        # without oscillations and pulls, our rates reproduce SK's MC up to the flux ratio
        phys0 = merge(sk.physics, (osc = Newtrinos.osc.configure(),))
        p0 = merge(p, (θ₁₂ = 0.0, θ₁₃ = 0.0))
        e0 = m.get_expected(p0, phys0, a)
        ph = m.PHASE
        Φ = sk.physics.solar_flux.nominal
        @test e0 ≈ a.mc_b8[a.bin_of] .* Φ.b8 / ph.phi_b8_mc .+ a.mc_hep[a.bin_of] .* Φ.hep / ph.phi_hep_mc rtol=1e-6
        # with oscillations: data/MC ≈ 0.35–0.5; night above day (Earth regeneration)
        e = m.get_expected(p, sk.physics, a)
        @test all(0.3 .< e ./ mc .< 0.55)
        if name in (:sk2_solar, :sk3_solar, :sk4_solar)
            @test -0.04 < A_DN(e, a.samples) < -0.01
        end
    end
    # SK-IV day/night asymmetry vs Δm²₂₁; SK expectation (arXiv:2312.12907): -3.84% and -1.72%, ours within 20%
    sk = Newtrinos.sk4_solar.configure(physics)
    p = Newtrinos.get_params((; sk))
    s = sk.assets.samples
    above = s.E_lo .>= 4.49
    A = [begin e2 = Newtrinos.sk4_solar.get_expected(merge(p, (Δm²₂₁ = dm,)), sk.physics, sk.assets);
               A_DN(e2[above], (; zenith = s.zenith[above])) end for dm in (4.8e-5, 7.5e-5)]
    @test A[1] ≈ -0.0384 rtol=0.2
    @test A[2] ≈ -0.0172 rtol=0.2
    # energy-scale pull shifts the spectrum: more events at high energy for a positive pull
    e = Newtrinos.sk4_solar.get_expected(p, sk.physics, sk.assets)
    eS = Newtrinos.sk4_solar.get_expected(merge(p, (sk4_solar_escale = 1.0,)), sk.physics, sk.assets)
    @test eS[end] > e[end]
    @test sum(eS) ≈ sum(e) rtol=0.02
    llh = Newtrinos.generate_likelihood((; sk))
    f(y) = logdensityof(llh, merge(p, (θ₁₂ = y[1], Δm²₂₁ = y[2], sk4_solar_escale = y[3], solar_b8_shape = y[4])))
    y0 = [p.θ₁₂, p.Δm²₂₁, 0.2, -0.3]
    g = ForwardDiff.gradient(f, y0)
    δ = [1e-6, 1e-11, 1e-6, 1e-6]
    g_fd = [(f(y0 .+ δ .* (1:4 .== i)) - f(y0 .- δ .* (1:4 .== i))) / 2δ[i] for i in 1:4]
    @test g ≈ g_fd rtol=1e-4
    # day and night samples merged per energy bin (for use with the amplitude-fit day/night term)
    skc = Newtrinos.sk4_solar.configure(physics; daynight = :combined)
    @test length(skc.assets.observed) == length(sk.assets.observed) ÷ 2
    @test all(skc.assets.samples.zenith .== "all")
    ec = Newtrinos.sk4_solar.get_expected(p, skc.physics, skc.assets)
    @test all(minimum.(zip(e[1:2:end], e[2:2:end])) .<= ec .<= maximum.(zip(e[1:2:end], e[2:2:end])))
    # SK day/night amplitude fit (arXiv:2312.12907): extracted curves and the expected asymmetry
    for (ds, fit61, exp61, exp75) in ((:sk4, -2.62, -2.38, -1.69), (:sk1to4, -2.86, -2.42, -1.72))
        dn = Newtrinos.sk_solar_dn.configure(physics; dataset = ds)
        a = dn.assets
        @test a.fit_spline(6.1e-5) ≈ fit61 atol=0.05
        pd = merge(Newtrinos.get_params((; dn)), (θ₁₃ = asin(sqrt(0.0218)), θ₁₂ = asin(sqrt(0.31))))
        @test Newtrinos.sk_solar_dn.expected_adn(merge(pd, (Δm²₂₁ = 6.1e-5,)), dn.physics, a) ≈ exp61 rtol=0.05
        @test Newtrinos.sk_solar_dn.expected_adn(merge(pd, (Δm²₂₁ = 7.5e-5,)), dn.physics, a) ≈ exp75 rtol=0.05
        lh = Newtrinos.generate_likelihood((; dn))
        h(x) = logdensityof(lh, merge(pd, (Δm²₂₁ = x,)))
        @test h(6.1e-5) > h(7.5e-5)          # the measured asymmetry prefers the lower Δm²₂₁
        @test ForwardDiff.derivative(h, 7.5e-5) ≈ (h(7.5e-5 * (1 + 1e-6)) - h(7.5e-5 * (1 - 1e-6))) / 1.5e-10 rtol=1e-4
    end
end

@testset "Borexino" begin
    physics = Newtrinos.solar_common.default_physics()
    b1 = Newtrinos.borexino_ph1.configure(physics)
    b2 = Newtrinos.borexino_ph2.configure(physics)
    b3 = Newtrinos.borexino_ph3.configure(physics)
    p = Newtrinos.get_params((; b1, b2, b3))
    ps = merge(p, (θ₁₂ = asin(sqrt(0.306)), θ₁₃ = asin(sqrt(0.02166)), Δm²₂₁ = 7.5e-5))
    # expected rates close to the measured ones (cpd/100 t)
    r1 = Newtrinos.borexino_ph1.get_expected(ps, b1.physics, b1.assets)
    r2 = Newtrinos.borexino_ph2.get_expected(ps, b2.physics, b2.assets)
    @test r1[1] ≈ 46.0 rtol=0.05
    @test r1[2] ≈ 0.217 rtol=0.1
    @test r2 ≈ [134.0, 2.43, 48.3] rtol=0.15
    # Phase III spectral fit: CNO rate near the SSM, total counts close to the data, both background-prior options
    e3 = Newtrinos.borexino_ph3.get_expected(ps, b3.physics, b3.assets)
    @test 4 < e3.solar.cno < 7
    @test e3.solar.pep ≈ 2.74 rtol=0.05
    @test sum(e3.total) ≈ sum(b3.assets.observed.spectrum) rtol=0.05
    @test size(e3.components) == (817, 12)
    b3_free = Newtrinos.borexino_ph3.configure(physics; background_priors=:free)
    @test b3_free.priors.borexino_ph3_c11 isa Uniform
    @test b3.priors.borexino_ph3_c11 isa Truncated
    # one-sided ²¹⁰Bi constraint: half-Gaussian penalty above 10.8 ± 1.0 on top of the spectral likelihood
    llh3 = Newtrinos.generate_likelihood((; b3))
    lspec(x) = sum(logpdf.(Poisson.(Newtrinos.borexino_ph3.get_expected(merge(ps, (borexino_ph3_bi210 = x,)), b3.physics, b3.assets).total),
                           b3.assets.observed.spectrum))
    ltot(x) = logdensityof(llh3, merge(ps, (borexino_ph3_bi210 = x,)))
    @test ltot(9.0) - ltot(10.8) ≈ lspec(9.0) - lspec(10.8) rtol=1e-8
    @test ltot(11.8) - ltot(10.8) ≈ lspec(11.8) - lspec(10.8) - 0.5 rtol=1e-8
    # oscillated / unoscillated rates; SK (arXiv:2312.12907) quotes 0.689, 0.614, 0.638 in restricted recoil windows
    phys0 = merge(physics, (osc = Newtrinos.osc.configure(),))
    b2_0 = Newtrinos.borexino_ph2.configure(phys0)
    r0 = Newtrinos.borexino_ph2.get_expected(merge(p, (θ₁₂ = 0.0, θ₁₃ = 0.0)), b2_0.physics, b2_0.assets)
    @test r2 ./ r0 ≈ [0.689, 0.614, 0.638] rtol=0.03
    # covariance of the emulated Borexino fit is positive definite
    @test isposdef(Matrix(b2.assets.cov))
    # asymmetric CNO likelihood: wider above the measurement than below
    lp(pred) = logpdf(Normal(pred, sqrt(Newtrinos.solar_common.asymmetric_variance(pred, 6.7, 2.0, 0.8))), 6.7)
    @test lp(8.7) > lp(4.7)
end
