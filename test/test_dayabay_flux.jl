using Test
using Accessors
using Newtrinos

@testset "Daya Bay reactor flux" begin 
    @testset "Configuration" begin
        reactor = Newtrinos.dayabay_flux.configure()
        @test reactor isa Newtrinos.dayabay_flux.ReactorFlux
        # TODO: how to comapre that each key of params is in priors?
        params = Newtrinos.dayabay_flux.get_params(reactor.cfg.flux_model)
        priors = Newtrinos.dayabay_flux.get_priors(reactor.cfg.flux_model)
    end

    @testset "Flux values" begin
        E = collect(LinRange(1, 12, 100))
        reactor = Newtrinos.dayabay_flux.configure()
        flux = reactor.flux
        params = reactor.params
        @testset "vectorisation" begin
            nom1, neq1, snf1, corr1 = flux(3, params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, params.snf_scale[1], 1)
            nom2, neq2, snf2, corr2 = flux(2, params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, params.snf_scale[1], 1)
            nom12, neq12, snf12, corr12 = flux([3, 2],  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, params.snf_scale[1], 1)
            @test isapprox(nom12, [nom1, nom2])
            @test isapprox(snf12, [snf1, snf2])
            @test isapprox(corr12, [corr1, corr2])
        end
        @testset "snf" begin
            @reset params.neq_scale_R1 = zeros(4)
            nom1, neq1, snf1, corr1 = flux(2.5,  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, 1, 1)
            nom2, neq2, snf2, corr2 = flux(2.5,  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, 0, 1)
            @test isapprox(nom1, nom2)
            @test isapprox(corr1, corr2)
            @test isapprox(snf2, 0.0)
            # nom is per GW, snf for a reactor at nominal power (2.895 GW)
            @test 0 < snf1 / (2.895 * nom2) <= 0.04
        end

        @testset "neq" begin
            @reset params.neq_scale_R1 = zeros(4)
            nom1, neq1, snf1, corr1 = flux(2.5,  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, 0, 1)
            @reset params.neq_scale_R1 = ones(4)
            nom2, neq2, snf2, corr2 = flux(2.5,  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, 0, 1)
            @test isapprox(nom1, nom2)
            @test neq1 == 0
            # the non-equilibrium correction adds a few per cent at low energy
            @test 0 < neq2 / nom2 < 0.1
        end

        @testset "free spectrum correction" begin
            _, _, _, corr0 = flux(E, params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, zeros(19), params.neq_scale_R1, 0, 1)
            @test all(corr0 .≈ 1)
            _, _, _, corr1 = flux(E, params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, fill(0.1, 19), params.neq_scale_R1, 0, 1)
            @test all(corr1 .≈ exp(0.1))
        end
        
        @testset "thermal power scale" begin
            @reset params.reactor_thermal_power_scale[1] = 1.0
            @reset params.snf_scale = zeros(6)
            nom1, neq1, snf1, corr1 = flux(E, params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, params.snf_scale[1], 1)
            @reset params.reactor_thermal_power_scale[1] = 2.0
            nom2, neq2, snf2, corr2 = flux(E, params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls, params.neq_scale_R1, params.snf_scale[1], 1)
            @test isapprox(nom1 .* 2, nom2)
        end
    end

end