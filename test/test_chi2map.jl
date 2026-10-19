using DensityInterface
using ForwardDiff

@testset "chi2map" begin
    # quadratic χ² in (sin²θ₁₂, Δm²₂₁): the cubic spline is exact up to round-off
    ax = (range(0.2, 0.4, length = 21), range(5e-5, 9e-5, length = 41))
    chi2(s, d) = ((s - 0.31) / 0.013)^2 + ((d - 7.4e-5) / 0.2e-5)^2
    tab = [chi2(s, d) for s in ax[1], d in ax[2]]
    m = Newtrinos.chi2map.configure((p -> sin(p.θ₁₂)^2, p -> p.Δm²₂₁), ax, tab; title = "toy")
    @test isempty(m.params)

    p = (θ₁₂ = asin(sqrt(0.30)), Δm²₂₁ = 7.0e-5)
    @test Newtrinos.chi2map.value(m, p) ≈ chi2(0.30, 7.0e-5) rtol = 1e-6
    llh = Newtrinos.generate_likelihood((; m))
    @test logdensityof(llh, p) ≈ -chi2(0.30, 7.0e-5) / 2 rtol = 1e-6

    g = ForwardDiff.gradient(x -> logdensityof(llh, (θ₁₂ = x[1], Δm²₂₁ = x[2])), [p.θ₁₂, p.Δm²₂₁])
    g_exact = [-(0.30 - 0.31) / 0.013^2 * sin(2p.θ₁₂), -(7.0e-5 - 7.4e-5) / (0.2e-5)^2]
    @test g ≈ g_exact rtol = 1e-4

    @test_throws ArgumentError Newtrinos.chi2map.configure((p -> p.θ₁₂,), ax, tab)
end
