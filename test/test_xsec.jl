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
                            ("Differential_H2O", Newtrinos.xsec.Differential_H2O())]
            xs = Newtrinos.xsec.configure(cfg)
            for key in keys(xs.params)
                @test Distributions.insupport(xs.priors[key], xs.params[key])
            end
        end
    end

end
