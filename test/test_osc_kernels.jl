using ForwardDiff
using LinearAlgebra
using StaticArrays

# KernelBackend on the KernelAbstractions CPU backend must reproduce the SerialCPU results.
# (CI has no GPU; the kernels are the same code on every backend.)
@testset "KernelBackend matter oscillations" begin
    O = Newtrinos.osc
    el = Newtrinos.earth_layers.configure()
    layers = el.compute_layers()
    cz = collect(range(-1, 0.3, 23))
    paths = el.compute_paths(cz, layers)
    E = 10 .^ collect(range(-0.5, 2, 31))

    function pair(; kw...)
        serial = O.configure(O.OscillationConfig(; interaction=O.SI(), eigen_method=Newtrinos.BargerEigen(), kw...))
        kernel = O.configure(O.OscillationConfig(; interaction=O.SI(), eigen_method=Newtrinos.BargerEigen(), backend=O.KernelBackend(), kw...))
        serial, kernel
    end

    for (label, prop) in (("Basic", O.Basic()), ("Spray gaussian", O.Spray(σ_E=0.05, σ_h=10.0)),
                          ("Spray uniform", O.Spray(averaging=:uniform, σ_E=0.05, σ_h=10.0)))
        @testset "$label" begin
            serial, kernel = pair(propagation=prop)
            p = serial.params
            for anti in (false, true)
                Ps = serial.osc_prob(E, paths, layers, p; anti)
                Pk = kernel.osc_prob(E, paths, layers, p; anti)
                @test size(Pk) == size(Ps)
                @test eltype(Pk) == eltype(Ps)
                @test maximum(abs.(Pk .- Ps)) < 1e-12
            end
        end
    end

    @testset "DefaultEigen on CPU kernel backend" begin
        serial = O.configure(O.OscillationConfig(interaction=O.SI()))
        kernel = O.configure(O.OscillationConfig(interaction=O.SI(), backend=O.KernelBackend()))
        p = serial.params
        @test maximum(abs.(kernel.osc_prob(E, paths, layers, p) .- serial.osc_prob(E, paths, layers, p))) < 1e-12
    end

    @testset "ForwardDiff through kernels" begin
        for prop in (O.Basic(), O.Spray(σ_E=0.05, σ_h=10.0))
            serial, kernel = pair(propagation=prop)
            p0 = serial.params
            f(osc) = x -> sum(abs2, osc.osc_prob(E, paths, layers, merge(p0, (θ₂₃=x[1], Δm²₃₁=x[2], δCP=x[3]))))
            x0 = [p0.θ₂₃, p0.Δm²₃₁, p0.δCP]
            gs = ForwardDiff.gradient(f(serial), x0)
            gk = ForwardDiff.gradient(f(kernel), x0)
            @test gk ≈ gs rtol=1e-10
        end
    end

    @testset "Vacuum method unaffected" begin
        serial, kernel = pair()
        L = collect(range(1.0, 1e4, 7))
        @test kernel.osc_prob(E, L, serial.params) == serial.osc_prob(E, L, serial.params)
    end

    @testset "unsupported configurations are rejected at configure time" begin
        kb = O.KernelBackend()
        @test_throws ArgumentError O.configure(O.OscillationConfig(interaction=O.Vacuum(), backend=kb))
        @test_throws ArgumentError O.configure(O.OscillationConfig(interaction=O.SI(), propagation=O.Damping(σₑ=0.05), backend=kb))
        @test_throws ArgumentError O.configure(O.OscillationConfig(interaction=O.SI(), flavour=O.Sterile(), backend=kb))
        @test_throws ArgumentError O.configure(O.OscillationConfig(interaction=O.SI(), states=O.Cut(cutoff=1.0), backend=kb))
        @test O.configure(O.OscillationConfig()).cfg.backend isa O.SerialCPU
    end
end
