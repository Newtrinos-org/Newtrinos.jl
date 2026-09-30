# Benchmark OscillationConfig(backend=...) for matter oscillations: SerialCPU vs
# KernelBackend on CPU threads vs KernelBackend on CUDA, for DeepCore (Basic) and Super-K (Spray).
#
#   julia --project -t 14 benchmark/bench_gpu_osc.jl [deepcore super_k]
using Newtrinos, CUDA, BenchmarkTools, DensityInterface, Accessors, Printf
import ForwardDiff
import KernelAbstractions as KA
const O = Newtrinos.osc

osc_with(cfg, b) = O.configure(@set cfg.backend = b)
with_backend(physics, b) = @set physics.osc = osc_with(physics.osc.cfg, b)

function osc_args(name, exp)
    as, ph = exp.assets, exp.physics
    if name == "super_k"
        E = exp10.((as.loge_grid[1:end-1] .+ as.loge_grid[2:end]) ./ 2)
        (E, ph.earth_layers.compute_paths(as.cz_midpoints, as.nominal_layers), as.nominal_layers)
    else
        (as.binning.e_fine, as.paths, as.layers)
    end
end

exps = isempty(ARGS) ? ["deepcore", "super_k"] : ARGS
@printf("threads: %d, GPU: %s\n", Threads.nthreads(), CUDA.name(CUDA.device()))
for name in exps
    mod = getproperty(Newtrinos, Symbol(name))
    base = mod.default_physics()
    variants = ["SerialCPU" => O.SerialCPU(), "KernelBackend CPU" => O.KernelBackend(KA.CPU()),
                "KernelBackend CUDA" => O.KernelBackend(CUDABackend(); workgroupsize=128)]
    @printf("\n==== %s (%s propagation)\n", name, nameof(typeof(base.osc.cfg.propagation)))
    ref = nothing
    for (label, b) in variants
        exp = mod.configure(with_backend(base, b))
        p = Newtrinos.get_params((; Symbol(name) => exp))
        E, paths, layers = osc_args(name, exp)
        llh = Newtrinos.generate_likelihood((; Symbol(name) => exp))
        f(x) = logdensityof(llh, x)
        P = exp.physics.osc.osc_prob(E, paths, layers, p)
        t_osc = @belapsed $(exp.physics.osc.osc_prob)($E, $paths, $layers, $p)
        v = f(p); t_llh = @belapsed $f($p) samples=20 seconds=10
        g = collect(values(ForwardDiff.gradient(f, p))); t_g = @elapsed ForwardDiff.gradient(f, p)
        if ref === nothing
            ref = (; P, v, g, t_osc, t_llh, t_g)
            @printf("  %-20s osc_prob %8.3f ms   llh %8.2f ms   grad %8.1f ms\n", label, t_osc*1e3, t_llh*1e3, t_g*1e3)
        else
            @printf("  %-20s osc_prob %8.3f ms (%5.1fx)   llh %8.2f ms (%4.1fx)   grad %8.1f ms (%4.1fx)   max|ΔP| %.1e  |Δllh| %.1e  rel Δgrad %.1e\n",
                    label, t_osc*1e3, ref.t_osc/t_osc, t_llh*1e3, ref.t_llh/t_llh, t_g*1e3, ref.t_g/t_g,
                    maximum(abs.(P .- ref.P)), abs(v - ref.v), maximum(abs.(g .- ref.g)) / maximum(abs.(ref.g)))
        end
        flush(stdout)
    end
    exp = mod.configure(base); p = Newtrinos.get_params((; Symbol(name) => exp)); E, paths, layers = osc_args(name, exp)
    print("  CUDA workgroupsize sweep (osc_prob):")
    for wg in (32, 64, 128, 256, 512)
        osc = osc_with(base.osc.cfg, O.KernelBackend(CUDABackend(); workgroupsize=wg))
        osc.osc_prob(E, paths, layers, p)
        @printf("  %d: %.3f ms", wg, 1e3 * @belapsed($(osc.osc_prob)($E, $paths, $layers, $p)))
    end
    println(); flush(stdout)
end
