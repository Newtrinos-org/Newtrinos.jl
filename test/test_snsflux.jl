using Test 
using LinearAlgebra
using Distributions
using CSV
using DataFrames
using Newtrinos

@testset "SNS Flux" begin

    @testset "Params in support of priors" begin
        # use_data = true
        params_d = Newtrinos.sns_flux.get_params(true)
        priors_d = Newtrinos.sns_flux.get_priors(true)
        @test keys(params_d) == keys(priors_d)
        @test params_d.flux_norm ≈ mean(priors_d.flux_norm) atol=0.05

        # use_data = false
        params_s = Newtrinos.sns_flux.get_params(false)
        priors_s = Newtrinos.sns_flux.get_priors(false)
        @test keys(params_s) == keys(priors_s)
        @test params_s.sns_nu_per_POT ≈ mean(priors_s.sns_nu_per_POT) atol=0.005
    end

    @testset "Data loading (use_data=true)" begin
        sf = Newtrinos.sns_flux.configure(exposure=1.0, distance=19.3)

        # check output objects
        @test sf isa Newtrinos.sns_flux.SNSFlux
        @test haskey(sf.params, :flux_norm)
        @test !haskey(sf.params, :sns_nu_per_POT)
        @test haskey(sf.assets, :flux_e_bar)

        # Energy grid: sorted, non-negative, bounded by ecut = mmu/2
        @test issorted(sf.assets.E)
        @test all(sf.assets.E .>= 0)
        @test all(sf.assets.E .<= Newtrinos.sns_flux.mmu / 2)

        # Time edges: sorted, start at 0
        @test issorted(sf.assets.T)
        @test sf.assets.T[1] ≈ 0.0

        # Flux matrices: non-negative, correct shape (nE × nT)
        nE = length(sf.assets.E)
        nT = length(sf.assets.T) - 1  # T holds bin edges
        for fname in (:flux_mu, :flux_e, :flux_mu_bar, :flux_e_bar)
            F = getfield(sf.assets, fname)
            @test all(F .>= 0)
            @test size(F) == (nE, nT)
        end
    end

    # ── Testset: Analytic flux helpers vs hand-computed values ────────
    @testset "Analytic flux helpers" begin
        # Constants from sns_flux.jl
        mmu  = Newtrinos.sns_flux.mmu   # 105.6 MeV
        mpi  = Newtrinos.sns_flux.mpi   # 139.57 MeV
        ep   = Newtrinos.sns_flux.ep    # (mpi²-mmu²)/(2mpi)
        mmu3 = mmu^3                    # 1177583.616

        eta = 1.0
        bin_width = 1.0
        Emax = mmu / 2  # 52.8 MeV

        # ── flux_nu_e: Michel spectrum 192*(E²/mmu³)*(0.5 - E/mmu) ──
        @testset "flux_nu_e" begin
            # E = 0: E² = 0, flux must vanish
            _, f0 = Newtrinos.sns_flux.flux_nu_e([0.0], eta, Emax, bin_width)
            @test f0[1] == 0.0

            # E = 10 MeV: 192*(100/1177583.616)*(0.5 - 10/105.6) = 0.006608
            _, f10 = Newtrinos.sns_flux.flux_nu_e([10.0], eta, Emax, bin_width)
            @test f10[1] ≈ 192 * (100 / mmu3) * (0.5 - 10 / mmu) atol=1e-8

            # E = mmu/2 = 52.8: factor (0.5 - E/mmu) = 0, flux vanishes
            _, f52 = Newtrinos.sns_flux.flux_nu_e([Emax], eta, Emax, bin_width)
            @test f52[1] ≈ 0.0 atol=1e-15
        end

        # ── flux_nu_mu_bar: Michel spectrum 64*(E²/mmu³)*(0.75 - E/mmu) ──
        @testset "flux_nu_mu_bar" begin
            # E = 0: flux must vanish
            _, f0 = Newtrinos.sns_flux.flux_nu_mu_bar([0.0], eta, Emax, bin_width)
            @test f0[1] == 0.0

            # E = 10 MeV: 64*(100/1177583.616)*(0.75 - 10/105.6) = 0.003561
            _, f10 = Newtrinos.sns_flux.flux_nu_mu_bar([10.0], eta, Emax, bin_width)
            @test f10[1] ≈ 64 * (100 / mmu3) * (0.75 - 10 / mmu) atol=1e-8

            # E = 20 MeV: 64*(400/1177583.616)*(0.75 - 20/105.6) = 0.01219
            _, f20 = Newtrinos.sns_flux.flux_nu_mu_bar([20.0], eta, Emax, bin_width)
            @test f20[1] ≈ 64 * (400 / mmu3) * (0.75 - 20 / mmu) atol=1e-8
        end

        # ── flux_nu_mu: normalized delta function at ep ≈ 29.836 MeV ──
        @testset "flux_nu_mu" begin
            E = collect(0.5:0.5:52.5)  # uniform grid, step = 0.5
            _, flux = Newtrinos.sns_flux.flux_nu_mu(E, ep, eta, 0.5)

            # Total integral should equal eta (normalized delta)
            @test sum(flux) ≈ eta rtol=1e-6

            # Flux is nonzero only near ep
            i_peak = argmin(abs.(E .- ep))
            @test flux[i_peak] > 0

            # Flux is zero far from ep
            @test flux[1] == 0.0         # E = 0.5
            @test flux[end] == 0.0       # E = 52.5
            @test flux[10] == 0.0        # E = 5.0
        end
    end

    @testset "Flux closure (use_data=false)" begin
        sf = Newtrinos.sns_flux.configure(exposure=1.0, distance=19.3, use_data=false)
        result = sf.flux(sf.params)

        # check output objects
        @test sf isa Newtrinos.sns_flux.SNSFlux
        @test !haskey(sf.params, :flux_norm)
        @test haskey(sf.params, :sns_nu_per_POT)
        @test !haskey(sf.assets, :flux_e_bar)

        # Total = sum of components
        @test result.total_flux ≈ result.flux_mu .+ result.flux_e .+ result.flux_mu_bar

        # SNSFlux flux Components = assets flux * sns_nu_per_POT
        ν = sf.params.sns_nu_per_POT
        @test result.flux_mu ≈ sf.assets.flux_mu .* ν
        @test result.flux_e ≈ sf.assets.flux_e .* ν
        @test result.flux_mu_bar ≈ sf.assets.flux_mu_bar .* ν
    end

    @testset "Flux closure (use_data=true)" begin
        sf = Newtrinos.sns_flux.configure(exposure=1.0, distance=19.3, use_data=true)
        result = sf.flux(sf.params)

        # Output structure includes time grid and flux_e_bar
        @test haskey(result, :T)
        @test haskey(result, :flux_e_bar)

        # Total = (sum of all four components) * flux_norm
        raw_total = result.flux_mu .+ result.flux_e .+ result.flux_mu_bar .+ result.flux_e_bar
        @test result.total_flux ≈ raw_total .* sf.params.flux_norm

        # Component fluxes are NOT scaled by flux_norm (only total is)
        @test result.flux_mu ≈ sf.assets.flux_mu
        @test result.flux_e ≈ sf.assets.flux_e
        @test result.flux_mu_bar ≈ sf.assets.flux_mu_bar
        @test result.flux_e_bar ≈ sf.assets.flux_e_bar

        # flux_norm scaling: 2x norm → 2x total, components unchanged
        result_2x = sf.flux((flux_norm = 2.0, flux_onset = 0.0))
        @test result_2x.total_flux ≈ 2.0 .* result.total_flux
        @test result_2x.flux_mu ≈ result.flux_mu
        @test result_2x.flux_e ≈ result.flux_e

        # flux_onset shifts the time histogram (a nonzero shift changes the binning)
        result_shifted = sf.flux((flux_norm = 1.0, flux_onset = 200.0))
        @test !(result_shifted.flux_mu ≈ result.flux_mu)
        @test all(result_shifted.flux_mu .>= 0)
        @test size(result_shifted.flux_mu) == size(result.flux_mu)
    end
end
