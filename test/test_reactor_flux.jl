using Test
using Accessors
using Newtrinos

@testset "Reactor flux" begin 
    @testset "Configuration" begin
        reactor = Newtrinos.reactor_flux.configure()
        @test reactor isa Newtrinos.reactor_flux.ReactorFlux
        # TODO: how to comapre that each key of params is in priors?
        params = Newtrinos.reactor_flux.get_params(reactor.cfg.flux_model)
        priors = Newtrinos.reactor_flux.get_priors(reactor.cfg.flux_model)
    end

    @testset "Flux values" begin
        E = collect(LinRange(1, 12, 100))
        reactor = Newtrinos.reactor_flux.configure()
        flux = reactor.flux
        params = reactor.params
        @testset "vectorisation" begin
            nom1, snf1, corr1 = flux(3, params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, params.snf_scale[1], 1)
            nom2, snf2, corr2 = flux(2, params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, params.snf_scale[1], 1)
            nom12, snf12, corr12 = flux([3, 2],  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, params.snf_scale[1], 1)
            @test isapprox(nom12, [nom1, nom2])
            @test isapprox(snf12, [snf1, snf2])
            @test isapprox(corr12, [corr1, corr2])
        end
        @testset "snf" begin
            @reset params.neq_scale_R1 = zeros(4)
            nom1, snf1, corr1 = flux(2.5,  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, 1, 1)
            nom2, snf2, corr2 = flux(2.5,  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, 0, 1)
            @test isapprox(nom1, nom2)
            @test isapprox(corr1, corr2)
            @test isapprox(snf2, 0.0)
            @test 0 < snf1 / nom2 <= 0.04
        end

        @testset "neq" begin
            @reset params.neq_scale_R1 = zeros(4)
            nom1, snf1, corr1 = flux(2.5,  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, 0, 1)
            @reset params.neq_scale_R1 = ones(4)
            nom2, snf2, corr2 = flux(2.5,  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, 0, 1)
            println(nom1 / nom2)

        end
        
        @testset "thermal power scale" begin
            @reset params.reactor_thermal_power_scale[1] = 1.0
            @reset params.snf_scale = zeros(6)
            nom1, snf1, corr1 = flux(E, params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, params.snf_scale[1], 1)
            @reset params.reactor_thermal_power_scale[1] = 2.0
            nom2, snf2, corr2 = flux(E, params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, params.snf_scale[1], 1)
            @test isapprox(nom1 .* 2, nom2)
        end
    end

end