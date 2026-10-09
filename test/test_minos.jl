using Newtrinos
using DensityInterface
using Test

@testset "MINOS two-detector likelihood" begin
    osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(flavour = Newtrinos.osc.Sterile()))
    physics = (osc = osc, xsec = Newtrinos.xsec.configure())
    @test_throws ArgumentError Newtrinos.minos.configure(physics; detectors = :ND)

    m = Newtrinos.minos.configure(physics; detectors = :FD_ND)
    L = Newtrinos.generate_likelihood((minos = m,))
    A = m.assets
    @test length(A.observed.CC) == length(A.ch_data["FDCC"].observed) + length(A.ch_data["NDCC"].observed)

    # best fit of the data release (defaults of dataRelease_chi2Calc_compile.C): χ² = 99.308 including the
    # Δm²₃₂ penalty of 0.0196
    pbf = (θ₁₂ = 0.5540758073, θ₁₃ = 0.149116, θ₂₃ = 0.928598228704929918, δCP = 0.0, Δm²₂₁ = 7.54e-5,
           Δm²₃₁ = 2.43005123913581740e-03 + 7.54e-5, Δm²₄₁ = 2.32492426050590582e-03, θ₁₄ = 0.0,
           θ₂₄ = 1.05321302928372360e-02, θ₃₄ = 8.35186824552614469e-03, nc_norm = 1.0, nutau_cc_norm = 1.0)
    norm = sum(A.logdetΣ₀) + (length(A.observed.CC) + length(A.observed.NC)) * log(2π)
    χ² = -2 * logdensityof(L, pbf) - norm
    @test χ² + (2.43005123913581740e-03 - 0.0025)^2 / 0.0005^2 ≈ 99.3087 atol = 1e-3

    # a large-Δm²₄₁ point that the far-detector-only likelihood prefers is strongly disfavoured by the near detector
    p2 = merge(pbf, (Δm²₄₁ = 43.15, θ₂₄ = asin(sqrt(0.0699))))
    @test -2 * (logdensityof(L, p2) - logdensityof(L, pbf)) > 30
end
