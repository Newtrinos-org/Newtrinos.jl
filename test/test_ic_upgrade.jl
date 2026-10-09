using Newtrinos
using Distributions
using Test

@testset "IceCube Upgrade energy-scale option" begin
    e0 = Newtrinos.ic_upgrade.configure()
    @test e0.priors.ic_upgrade_energy_scale == 1.0                     # fixed by default
    e2 = Newtrinos.ic_upgrade.configure(; energy_scale_uncertainty = 0.02)
    @test e2.priors.ic_upgrade_energy_scale isa Truncated
    @test std(e2.priors.ic_upgrade_energy_scale.untruncated) == 0.02
    @test_throws ArgumentError Newtrinos.ic_upgrade.configure(; energy_scale_uncertainty = -0.1)
    @test haskey(Newtrinos.get_params((ic_upgrade = e0,)), :ic_upgrade_energy_scale)
end
