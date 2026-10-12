using Distributions
using Test
using Newtrinos
using ForwardDiff

@testset "Cross Sections" begin

    @testset "SimpleScaling scale function" begin
        xs = Newtrinos.xsec.configure()

        # NC interaction returns nc_norm
        @test xs.scale(:numu, :NC, xs.params) == xs.params.nc_norm
        @test xs.scale(:nue, :NC, xs.params) == xs.params.nc_norm

        # nutau CC returns nutau_cc_norm
        @test xs.scale(:nutau, :CC, xs.params) == xs.params.nutau_cc_norm

        # Other CC returns 1.0 (type-stable with params)
        @test xs.scale(:numu, :CC, xs.params) ≈ 1.0

        # With modified params
        mod_params = (nc_norm=1.5, nutau_cc_norm=0.8)
        @test xs.scale(:nue, :NC, mod_params) == 1.5
        @test xs.scale(:nutau, :CC, mod_params) == 0.8
    end

    @testset "Default params within prior support" begin
        for (name, cfg) in [("SimpleScaling", Newtrinos.xsec.SimpleScaling()),
                            ("Differential_H2O", Newtrinos.xsec.Differential_H2O()),
                            ("H2O_PCA", Newtrinos.xsec.H2O_PCA())]
            xs = Newtrinos.xsec.configure(cfg)
            for key in keys(xs.params)
                @test Distributions.insupport(xs.priors[key], xs.params[key])
            end
        end
    end

    @testset "H2O_PCA configuration" begin
        xs = Newtrinos.xsec.configure(Newtrinos.xsec.H2O_PCA())
        @test xs isa Newtrinos.xsec.Xsec
        @test xs.cfg isa Newtrinos.xsec.H2O_PCA
        # unlike SimpleScaling/Differential_H2O, H2O_PCA implements the full extension API
        @test xs.scale isa Function
        @test xs.dσdE isa Function
        @test xs.scale_event isa Function
        @test xs.event_weights isa Function
        @test xs.grid_weights isa Function
    end

    @testset "H2O_PCA scale NC/CC at nominal params" begin
        xs = Newtrinos.xsec.configure(Newtrinos.xsec.H2O_PCA())
        E = [0.5, 1.0, 5.0, 10.0]

        # NC reweight ≈ 1 at nominal params (norm=1, shape=0, nubar_ratio=1)
        for flav in (:nue, :numu, :nutau), anti in (false, true)
            result = xs.scale(E, flav, :NC, anti, xs.params)
            @test all(result .≈ 1.0)
            @test length(result) == length(E)
        end

        # CC reweight ≈ 1 at nominal params: per-channel fractions sum to 1, all norms=1
        for flav in (:nue, :numu, :nutau), anti in (false, true)
            result = xs.scale(E, flav, :CC, anti, xs.params)
            @test all(isapprox.(result, 1.0; atol = 1e-6))
            @test all(result .>= 0)
            @test length(result) == length(E)
        end
    end

    @testset "H2O_PCA norm scaling" begin
        xs = Newtrinos.xsec.configure(Newtrinos.xsec.H2O_PCA())
        E = [1.0, 5.0]

        # xsec_nc_norm scales NC proportionally
        mod_params = merge(xs.params, (xsec_nc_norm = 1.3,))
        result_nc_mod = xs.scale(E, :numu, :NC, false, mod_params)
        result_nc_nom = xs.scale(E, :numu, :NC, false, xs.params)
        @test all(isapprox.(result_nc_mod ./ result_nc_nom, 1.3; atol = 1e-6))

        # doubling one CC channel norm increases the CC scale (never decreases it)
        E_wide = [2.0, 5.0, 10.0, 20.0]
        mod_params2 = merge(xs.params, (xsec_ccdis_norm = 2.0,))
        result_cc_mod = xs.scale(E_wide, :numu, :CC, false, mod_params2)
        result_cc_nom = xs.scale(E_wide, :numu, :CC, false, xs.params)
        @test all(result_cc_mod .>= result_cc_nom .- 1e-10)
        @test any(result_cc_mod .> result_cc_nom .+ 1e-10)

        # xsec_nutau_cc_norm scales only nutau CC (numu/nutau share the same underlying
        # channel curves via get_flavor_key, so their raw CC scale is otherwise identical)
        mod_params3 = merge(xs.params, (xsec_nutau_cc_norm = 0.5,))
        result_nutau = xs.scale(E, :nutau, :CC, false, mod_params3)
        result_numu = xs.scale(E, :numu, :CC, false, mod_params3)
        @test all(isapprox.(result_nutau, result_numu .* 0.5; atol = 1e-6))

        # non-negativity, result length matches input
        result = xs.scale(E_wide, :numu, :CC, false, xs.params)
        @test length(result) == length(E_wide)
        @test all(result .>= 0)
    end

    @testset "H2O_PCA ν̄/ν and νe/νμ ratio parameters" begin
        xs = Newtrinos.xsec.configure(Newtrinos.xsec.H2O_PCA())
        E = [1.0, 5.0]

        # non-unit nubar_ratio breaks ν/ν̄ symmetry
        mod_params = merge(xs.params, (xsec_cc1p1h_nubar_ratio = 1.5, xsec_nc_nubar_ratio = 1.5))
        r_nu_cc = xs.scale(E, :numu, :CC, false, mod_params)
        r_anti_cc = xs.scale(E, :numu, :CC, true, mod_params)
        @test !(r_nu_cc ≈ r_anti_cc)

        r_nu_nc = xs.scale(E, :numu, :NC, false, mod_params)
        r_anti_nc = xs.scale(E, :numu, :NC, true, mod_params)
        @test !(r_nu_nc ≈ r_anti_nc)

        # non-unit nue_numu_ratio breaks νe/νμ CC1p1h symmetry (channel norms set equal
        # so only the CC1p1h contribution, which carries the ratio, differs)
        mod_params2 = merge(xs.params, (xsec_cc1p1h_nue_numu_ratio = 1.5,))
        r_nue = xs.scale(E, :nue, :CC, false, mod_params2)
        r_numu = xs.scale(E, :numu, :CC, false, mod_params2)
        @test !(r_nue ≈ r_numu)
    end

    @testset "H2O_PCA dσdE" begin
        xs = Newtrinos.xsec.configure(Newtrinos.xsec.H2O_PCA())
        E = [0.5, 1.0, 5.0, 10.0]
        for flav in (:nue, :numu, :nutau), interaction in (:NC, :CC), anti in (false, true)
            result = xs.dσdE(E, flav, interaction, anti, xs.params)
            @test length(result) == length(E)
            @test all(isfinite, result)
            @test all(result .>= 0)
        end
    end

    @testset "H2O_PCA scale_event matches event_weights fast path" begin
        xs = Newtrinos.xsec.configure(Newtrinos.xsec.H2O_PCA())
        E = [0.5, 1.0, 2.0, 5.0, 10.0]
        codes = [0, 1, 2, 3, 0]  # GENIE codes: 0=QE→CC1p1h, 1=RES→CC1pi, 2=DIS, else→CCother

        result_direct = xs.scale_event(E, codes, :numu, :CC, false, xs.params)
        result_fast = xs.event_weights(E, codes, :numu, :CC, false)(xs.params)
        @test result_direct ≈ result_fast atol = 1e-8
        @test length(result_direct) == length(E)
        @test all(result_direct .>= 0)

        result_direct_nc = xs.scale_event(E, codes, :numu, :NC, false, xs.params)
        result_fast_nc = xs.event_weights(E, codes, :numu, :NC, false)(xs.params)
        @test result_direct_nc ≈ result_fast_nc atol = 1e-8
    end

    @testset "H2O_PCA grid_weights" begin
        xs = Newtrinos.xsec.configure(Newtrinos.xsec.H2O_PCA())
        E_grid = [0.5, 1.0, 2.0, 5.0, 10.0]

        result_cc = xs.grid_weights(E_grid, :numu, :CC, false)(xs.params)
        @test length(result_cc) == length(E_grid)
        @test all(isfinite, result_cc)
        @test all(result_cc .>= 0)

        result_nc = xs.grid_weights(E_grid, :numu, :NC, false)(xs.params)
        @test length(result_nc) == length(E_grid)
        @test all(result_nc .>= 0)
    end

    @testset "weights interface: Nominal and SimpleScaling" begin
        E = [1.0, 10.0]
        xs = Newtrinos.xsec.configure(Newtrinos.xsec.Nominal())
        @test isempty(xs.params)
        @test xs.weights(E; flavour=:numu, anti=false, current=:CC, target=:H2O, generator=:NEUT5_4_0)(xs.params) == 1.0
        xs = Newtrinos.xsec.configure()
        kw = (anti=false, target=:H2O, generator=:GENIE2_12)
        p = (nc_norm=1.3, nutau_cc_norm=0.7)
        @test xs.weights(E; flavour=:numu, current=:NC, kw...)(p) == 1.3
        @test xs.weights(E; flavour=:nutau, current=:CC, kw...)(p) == 0.7
        @test xs.weights(E; flavour=:numu, current=:CC, kw...)(p) == 1.0
        w = Newtrinos.xsec.event_weights(xs, [1.0, 2.0, 3.0], [:CC, :NC, :CC]; flavour=:nutau, kw...)(p)
        @test w == [0.7, 1.3, 0.7]
    end

    @testset "H2O_PCA weights are relative to its nominal" begin
        xs = Newtrinos.xsec.configure(Newtrinos.xsec.H2O_PCA())
        E = [0.5, 2.0, 20.0]
        w = xs.weights(E; flavour=:numu, anti=false, current=:CC, target=:H2O, generator=:NEUT5_4_0)
        @test w(xs.params) ≈ ones(3)
        @test w(merge(xs.params, (xsec_ccdis_norm = 1.1,)))[3] > 1.05
    end

    @testset "generator cross sections" begin
        X = Newtrinos.xsec
        tab = X.load_table()
        at(v, e) = X.at_energies(tab.E, v, [e])[1]
        # νμ CC σ/E per nucleon at high energy (PDG: ≈ 0.68e-38 cm²/GeV), ν̄μ about half
        for gen in (:G18_10a_02_11a, :G18_02a_00_000, :GENIE2_12)
            @test 0.64 < at(sum(X.generator_sigma(gen, :H2O, :numu, false, :CC)), 50.0) < 0.72
            @test 0.30 < at(sum(X.generator_sigma(gen, :Fe, :numu, true, :CC)), 50.0) < 0.36
        end
        # ντ CC threshold (≈ 3.5 GeV on free nucleons, slightly lower on nuclei through Fermi motion)
        στ(gen, e) = at(sum(X.generator_sigma(gen, :H2O, :nutau, false, :CC)), e)
        σμ(gen, e) = at(sum(X.generator_sigma(gen, :H2O, :numu, false, :CC)), e)
        for gen in (:G18_10a_02_11a, :NEUT5_4_0)
            @test στ(gen, 2.5) < 1e-4
            @test στ(gen, 3.0) < 0.01 * σμ(gen, 3.0)
            @test 0.3 < στ(gen, 20.0) / σμ(gen, 20.0) < 0.5
        end
        # GENIE 2.12 default has no MEC
        @test all(iszero, X.generator_sigma(:GENIE2_12, :O, :numu, false, :CC).MEC)
        @test_throws ArgumentError X.generator_sigma(:NEUT5_4_0, :Fe, :numu, false, :CC)
        @test_throws ArgumentError X.generator_sigma(:nonexistent, :H2O, :numu, false, :CC)
        # Llewellyn Smith: σ grows with M_A, ν > ν̄
        @test X.llewellyn_smith(1.0, 1.1) > X.llewellyn_smith(1.0, 1.0)
        @test X.llewellyn_smith(1.0, 1.0) > X.llewellyn_smith(1.0, 1.0; anti=true)
    end

    @testset "CorrelatedXsec" begin
        X = Newtrinos.xsec
        xs = X.configure(X.CorrelatedXsec())
        p = xs.params
        for key in keys(p)
            @test Distributions.insupport(xs.priors[key], p[key])
        end
        E = [0.3, 1.0, 3.0, 10.0, 100.0]
        kw = (flavour=:numu, anti=false, target=:H2O)
        # unit weights for the reference generator, inclusive and per process
        @test xs.weights(E; current=:CC, generator=:G18_10a_02_11a, kw...)(p) ≈ ones(5)
        @test xs.weights(E; current=:NC, generator=:G18_10a_02_11a, kw...)(p) ≈ ones(5)
        @test xs.weights(E; current=:CC, generator=:G18_10a_02_11a, process=:RES, kw...)(p) ≈ ones(5)
        # knobs act on their process only
        q = merge(p, (xsec_dis_norm = 1.1,))
        @test xs.weights(E; current=:CC, generator=:G18_10a_02_11a, process=:DIS, kw...)(q) ≈ fill(1.1, 5)
        @test xs.weights(E; current=:CC, generator=:G18_10a_02_11a, process=:QE, kw...)(q) ≈ ones(5)
        @test xs.weights(E; current=:CC, generator=:G18_10a_02_11a, kw...)(merge(p, (xsec_ma_qe = 1.0,)))[3] > 1.0
        # ν̄ ratio does not act on neutrinos
        @test xs.weights(E; current=:CC, generator=:G18_10a_02_11a, kw...)(merge(p, (xsec_nubar_ratio = 1.1,))) ≈ ones(5)
        # one shared parameter moves different targets together
        q = merge(p, (xsec_pc1 = 1.0,))
        wO = xs.weights(E; current=:CC, generator=:G18_10a_02_11a, kw...)(q)
        wC = xs.weights(E; current=:CC, generator=:G18_10a_02_11a, flavour=:numu, anti=false, target=:CH2)(q)
        @test sign(wO[1] - 1) == sign(wC[1] - 1)
        # per-event weights equal the per-process ones; MEC goes with QE for a generator without MEC
        we = xs.weights(E; current=:CC, generator=:GENIE2_12, process=[:QE, :RES, :DIS, :DIS, :DIS], kw...)(p)
        @test we[2] ≈ xs.weights(E[2:2]; current=:CC, generator=:GENIE2_12, process=:RES, kw...)(p)[1]
        @test we[1] ≈ xs.weights(E[1:1]; current=:CC, generator=:GENIE2_12, process=:QE, kw...)(p)[1]
        # ForwardDiff through the weights
        f(x) = sum(xs.weights(E; current=:CC, generator=:NEUT5_4_0, kw...)(merge(p, (xsec_pc1 = x[1], xsec_ma_qe = x[2]))))
        g = ForwardDiff.gradient(f, [0.1, 0.2])
        @test all(isfinite, g) && any(!iszero, g)
        @test_throws ArgumentError xs.scale(:numu, :CC, p)
    end

end
