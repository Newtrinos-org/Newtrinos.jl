using Distributions
using Test
using Newtrinos

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

end
