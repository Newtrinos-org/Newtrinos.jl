using Distributions
using DataStructures
#using OrderedCollections
using Newtrinos
using Test

@testset "Atmospheric Flux" begin

    @testset "Parameters and priors" begin
        af = Newtrinos.atm_flux.configure()
        params = af.params
        priors = af.priors
        for k in keys(params)
            @test Distributions.insupport(priors[k], params[k])
        end
    end

    @testset "scale_flux" begin
        sf = Newtrinos.atm_flux.scale_flux

        # A=10, B=5, scale=1.1 -> r=2, total=15, mod_B=15/(1+2.2)=4.6875, mod_A=2.2*4.6875=10.3125
        mod_A, mod_B = sf(10.0, 5.0, 1.1)
        @test mod_A ≈ 10.3125 atol=1e-10
        @test mod_B ≈ 4.6875 atol=1e-10
        @test mod_A + mod_B ≈ 15.0 atol=1e-10

        # Identity: scale=1.0 leaves values unchanged
        mod_A, mod_B = sf(100.0, 100.0, 1.0)
        @test mod_A ≈ 100.0 atol=1e-10
        @test mod_B ≈ 100.0 atol=1e-10

        # Vectorized operation
        mod_A, mod_B = sf([10.0, 100.0], [5.0, 100.0], [1.1, 1.0])
        @test mod_A ≈ [10.3125, 100.0] atol=1e-10
        @test mod_B ≈ [4.6875, 100.0] atol=1e-10
    end

    @testset "uphorizontal" begin
        uh = Newtrinos.atm_flux.uphorizontal

        #explicit calculations for specific cases
        @test uh(0.0, 1.05) ≈ 1.05 atol=1e-10 # coszen=0 (horizontal) ->  rel_error        
        @test uh(1.0, 1.05) ≈ 1.0/1.05 atol=1e-10 # coszen=1 (vertical) -> 1/rel_error        
        @test uh(0.5, 1.0) ≈ 1.0 atol=1e-15 # Identity when rel_error=1.0
        @test uh(0.5, 1.05) ≈ 1.0 / sqrt((1.05^2 - (1/1.05)^2) * 0.25 + (1/1.05)^2) atol=1e-15 # intermediate values
        
        # plus/minus Symmetry in coszen (uh ~ coszen^2)
        @test uh(0.5, 1.1) ≈ uh(-0.5, 1.1) atol=1e-15
    end

    @testset "updown" begin
        ud = Newtrinos.atm_flux.updown

        #explicit calculations for specific cases
        @test ud(0.0, 1.1) ≈ 1.0 atol=1e-10 # coszen=0 (horizontal): tanh(0)=0, scale=1.0
        @test ud(10.0, 1.1) ≈ 1.1 atol=1e-6 # Saturated upgoing (large positive coszen): scale → udr
        @test ud(-10.0, 1.1) ≈ 1.0/1.1 atol=1e-6 # Saturated downgoing (large negative coszen): scale → 1/udr

        # Identity when udr=1.0
        @test ud(0.5, 1.0) ≈ 1.0 atol=1e-15

        # Anti-symmetry: updown(cz, r) * updown(-cz, r) ≈ 1
        @test ud(0.8, 1.05) * ud(-0.8, 1.05) ≈ 1.0 atol=1e-10
    end

    @testset "Nominal flux" begin
        #test datareading 
        datadir = joinpath(dirname(pathof(Newtrinos)), "physics")
        flux = Newtrinos.atm_flux.get_hkkm_flux(joinpath(datadir, "spl-nu-20-01-000.d"))

        #@test flux isa OrderedDict 
        @test length(flux) == 4
        @test collect(keys(flux)) == [:numu, :numubar, :nue, :nuebar]

        # Grid point: E=0.1 GeV (log10E=-1), cosZ=0.95 (is in interval [0.9, 1.0]) -> File values: numu=12001, numubar=12116, nue=6056.2, nuebar=5278.0
        @test flux[:numu](-1.0, 0.95) ≈ 12001.0 atol=1.0
        @test flux[:numubar](-1.0, 0.95) ≈ 12116.0 atol=1.0
        @test flux[:nue](-1.0, 0.95) ≈ 6056.2 atol=1.0
        @test flux[:nuebar](-1.0, 0.95) ≈ 5278.0 atol=1.0

        # Grid point: E=1.0 GeV (log10E=0), cosZ=0.85 (is in interval [0.8, 0.9]) -> File values: numu=150.06, numubar=140.22, nue=68.79, nuebar=53.40
        @test flux[:numu](0.0, 0.85) ≈ 150.06 atol=0.01
        @test flux[:numubar](0.0, 0.85) ≈ 140.22 atol=0.01
        @test flux[:nue](0.0, 0.85) ≈ 68.79 atol=0.01
        @test flux[:nuebar](0.0, 0.85) ≈ 53.40 atol=0.01

        # Grid point: E=0.1 GeV (log10E=-1), cosZ=-0.45 (is in interval [-0.5, -0.4]) -> File values: numu=10246, numubar=10406, nue=4983.7, nuebar=4837.0
        @test flux[:numu](-1.0, -0.45) ≈ 10246.0 atol=1.0
        @test flux[:numubar](-1.0, -0.45) ≈ 10406.0 atol=1.0
        @test flux[:nue](-1.0, -0.45) ≈ 4983.7 atol=0.1
        @test flux[:nuebar](-1.0, -0.45) ≈ 4837.0 atol=0.1
   
        #test closure of interpolation (values at grid points should match file values)
        af = Newtrinos.atm_flux.configure()
        # Meshgrid: [(E=0.1,cz=-0.95), (E=1.0,cz=-0.95), (E=0.1,cz=0.95), (E=1.0,cz=0.95)]
        energy = [0.1, 1.0]
        coszen = [-0.95, 0.95]
        flux_table = af.nominal_flux(energy, coszen)

        @test length(flux_table) == 4
        @test hasproperty(flux_table, :true_energy)
        @test hasproperty(flux_table, :log10_true_energy)
        @test hasproperty(flux_table, :true_coszen)
        @test flux_table.log10_true_energy ≈ log10.(flux_table.true_energy)
        @test all(flux_table.numu .> 0)
        @test all(flux_table.numubar .> 0)
        @test all(flux_table.nue .> 0)
        @test all(flux_table.nuebar .> 0)

        # check numu from file for (E=0.1, cz=-0.95)
        @test flux_table.numu[1] ≈ 10523.0 atol=1.0
        # check nuebar from file for (E=1.0, cz=-0.95)
        @test flux_table.nuebar[2] ≈ 52.770 atol=0.1
        # check numubar from file for (E=0.1, cz=0.95)
        @test flux_table.numubar[3] ≈ 12116.0 atol=1.0 
        # check nue from file for (E=1.0, cz=0.95)
        @test flux_table.nue[4] ≈ 63.863 atol=0.1
    end

    @testset "Systematic flux at nominal params" begin
        #Test with nominal params -> should return nominal flux unchanged:
        af = Newtrinos.atm_flux.configure()
        energy = [0.1, 1.0, 10.0, 100.0]
        coszen = [-0.95, -0.5, 0.0, 0.5, 0.95]
        nominal = af.nominal_flux(energy, coszen)
        sys_result = af.sys_flux(nominal, af.params)

        @test sys_result.nue ≈ nominal.nue atol=1e-6
        @test sys_result.numu ≈ nominal.numu atol=1e-6
        @test sys_result.nuebar ≈ nominal.nuebar atol=1e-6
        @test sys_result.numubar ≈ nominal.numubar atol=1e-6
    
        #Test with non-zero params -> should modify flux according to systematic effects
        # Spectral index shift modifies flux
        params_shifted = merge(af.params, (atm_flux_delta_spectral_index = 0.1,))
        sys_shifted = af.sys_flux(nominal, params_shifted)
        @test !(sys_shifted.numu ≈ nominal.numu)

        # nue/nuebar ratio scaling (split into sub-GeV/1-10 GeV/>10 GeV energy ranges)
        params_nue = merge(af.params, (atm_flux_nuenuebar_sigma_lo = 1.0, atm_flux_nuenuebar_sigma_mid = 1.0, atm_flux_nuenuebar_sigma_hi = 1.0))
        sys_nue = af.sys_flux(nominal, params_nue)
        @test sys_nue.nue .+ sys_nue.nuebar ≈ nominal.nue .+ nominal.nuebar atol=1e-6
        @test !(sys_nue.nue ≈ nominal.nue)

        # numu/numubar ratio scaling (split into sub-GeV/1-10 GeV/>10 GeV energy ranges)
        params_numu = merge(af.params, (atm_flux_numunumubar_sigma_lo = 1.0, atm_flux_numunumubar_sigma_mid = 1.0, atm_flux_numunumubar_sigma_hi = 1.0))
        sys_numu = af.sys_flux(nominal, params_numu)
        @test sys_numu.numu .+ sys_numu.numubar ≈ nominal.numu .+ nominal.numubar atol=1e-6
        @test !(sys_numu.numu ≈ nominal.numu)
    end

end
