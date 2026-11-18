using Test
using Accessors
using Newtrinos

@testset "Daya Bay reactor flux" begin
    fission_fraction_scales = 
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
            nom1, neq1, snf1, corr1 = flux(3, 1.0, params.energy_per_fission, ones(4), params.spectrum_pulls, ones(4), 1.0, 1)
            nom2, neq2, snf2, corr2 = flux(2, 1.0, params.energy_per_fission, ones(4), params.spectrum_pulls, ones(4), 1.0, 1)
            nom12, neq12, snf12, corr12 = flux([3, 2],  1.0, params.energy_per_fission, ones(4), params.spectrum_pulls, ones(4), 1.0, 1)
            @test isapprox(nom12, [nom1, nom2])
            @test isapprox(snf12, [snf1, snf2])
            @test isapprox(corr12, [corr1, corr2])
        end
        @testset "snf" begin
            nom1, neq1, snf1, corr1 = flux(2.5,  1.0, params.energy_per_fission, ones(4), params.spectrum_pulls, zeros(4), 1, 1)
            nom2, neq2, snf2, corr2 = flux(2.5,  1.0, params.energy_per_fission, ones(4), params.spectrum_pulls, zeros(4), 0, 1)
            @test isapprox(nom1, nom2)
            @test isapprox(corr1, corr2)
            @test isapprox(snf2, 0.0)
            # nom is per GW, snf for a reactor at nominal power (2.895 GW)
            @test 0 < snf1 / (2.895 * nom2) <= 0.04
        end

        @testset "neq" begin
            nom1, neq1, snf1, corr1 = flux(2.5,  1.0, params.energy_per_fission, ones(4), params.spectrum_pulls, zeros(4), 0, 1)
            nom2, neq2, snf2, corr2 = flux(2.5,  1.0, params.energy_per_fission, ones(4), params.spectrum_pulls, ones(4), 0, 1)
            @test isapprox(nom1, nom2)
            @test neq1 == 0
            # the non-equilibrium correction adds a few per cent at low energy
            @test 0 < neq2 / nom2 < 0.1
        end

        @testset "free spectrum correction" begin
            _, _, _, corr0 = flux(E, 1.0, params.energy_per_fission, ones(4), zeros(19), ones(4), 0, 1)
            @test all(corr0 .≈ 1)
            _, _, _, corr1 = flux(E, 1.0, params.energy_per_fission, ones(4), fill(0.1, 19), ones(4), 0, 1)
            @test all(corr1 .≈ exp(0.1))
        end
        
        @testset "thermal power scale" begin
            nom1, neq1, snf1, corr1 = flux(E, 1.0, params.energy_per_fission, ones(4), params.spectrum_pulls, ones(4), 0, 1)
            nom2, neq2, snf2, corr2 = flux(E, 2.0, params.energy_per_fission, ones(4), params.spectrum_pulls, ones(4), 0, 1)
            @test isapprox(nom1 .* 2, nom2)
        end
        
        @testset "linear interpolation" begin
            config = Newtrinos.dayabay_flux.ReactorFluxConfig(corr_mode=:lin)
            reactor = Newtrinos.dayabay_flux.configure(config)
            flux = reactor.flux
            @test reactor isa Newtrinos.dayabay_flux.ReactorFlux
            params = Newtrinos.dayabay_flux.get_params(reactor.cfg.flux_model)
            priors = Newtrinos.dayabay_flux.get_priors(reactor.cfg.flux_model)
            _, _, _, corr0 = flux(E, 1.0, params.energy_per_fission, ones(4), zeros(19), ones(4), 0, 1)
            @test all(corr0 .≈ 1)
            _, _, _, corr1 = flux(E, 1.0, params.energy_per_fission, ones(4), fill(0.1, 19), ones(4), 0, 1)
            @test all(corr1 .≈ 1.1)
        end
    end
end