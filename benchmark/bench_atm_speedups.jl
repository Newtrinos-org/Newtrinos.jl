# Benchmark: (1) HKKM flux-table caching fix, (2) LAPACK eigen vs analytic BargerEigen
# for the atmospheric experiments.
#   julia --project benchmark/bench_atm_speedups.jl [orca deepcore super_k]
using BenchmarkTools
using DensityInterface
using Accessors
using DelimitedFiles
using Newtrinos
import ForwardDiff

const AF = Newtrinos.atm_flux

# --- pre-fix nominal_flux: re-reads and re-splines the table on every call ---
function old_nominal_flux(cfg::AF.HKKM)
    function nominal_flux(energy, coszen)
        e = vec(energy .* ones(length(coszen))')
        cz = vec(ones(length(energy)) .* coszen')
        filename = joinpath(AF.datadir, cfg.fname)
        flux_chunks = Matrix{Float32}[]
        for i in 19:-1:0
            idx = i*103 + 3: (i+1)*103
            push!(flux_chunks, Float32.(readdlm(filename)[idx, 2:5]))
        end
        l10 = LinRange(-1, 4, 101); czv = LinRange(-0.95, 0.95, 20)
        hk = permutedims(stack(flux_chunks), [1, 3, 2])
        spl = [AF.cubic_spline_interpolation((l10, czv), hk[:, :, k], extrapolation_bc=AF.Line()) for k in 1:4]
        AF.Table(true_energy=e, log10_true_energy=log10.(e), true_coszen=cz,
                 numu=spl[1].(log10.(e), cz), numubar=spl[2].(log10.(e), cz),
                 nue=spl[3].(log10.(e), cz), nuebar=spl[4].(log10.(e), cz))
    end
end
with_old_flux(af) = AF.AtmFlux(cfg=af.cfg, params=af.params, priors=af.priors,
                               nominal_flux=old_nominal_flux(af.cfg.nominal_model), sys_flux=af.sys_flux)

function with_eigen(osc, method)
    c = osc.cfg
    Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(flavour=c.flavour, interaction=c.interaction,
        propagation=c.propagation, states=c.states, eigen_method=method))
end

ms(t) = round(t * 1e3, digits=2)

exps = isempty(ARGS) ? ["orca", "deepcore", "super_k"] : ARGS

# ---------- (1) flux micro-benchmark ----------
println("="^70, "\nFLUX: nominal_flux on a 200x40 grid\n", "="^70)
E = 10 .^ range(0, 2, 200); cz = range(-1, 1, 40)
af = AF.configure()
f_new = af.nominal_flux; f_old = old_nominal_flux(af.cfg.nominal_model)
@assert all(f_new(E, cz).numu .== f_old(E, cz).numu)
t_old = @belapsed $f_old($E, $cz); t_new = @belapsed $f_new($E, $cz)
println("  old: $(ms(t_old)) ms   new: $(ms(t_new)) ms   speedup: $(round(t_old/t_new, digits=1))x"); flush(stdout)

# ---------- (2) per-experiment likelihood + gradient ----------
function bench(name, phys, mod)
    exp = (; Symbol(name) => mod.configure(phys))
    p = Newtrinos.get_params(exp)
    llh = Newtrinos.generate_likelihood(exp)
    f(x) = logdensityof(llh, x)
    v = f(p); g = ForwardDiff.gradient(f, p)
    t = @belapsed $f($p) samples=20 seconds=20
    tg = @belapsed ForwardDiff.gradient($f, $p) samples=5 seconds=60
    (; v, g, t, tg, n=length(p))
end

for name in exps
    mod = getproperty(Newtrinos, Symbol(name))
    base = mod.default_physics()
    variants = [
        "old flux + LAPACK" => (@set base.atm_flux = with_old_flux(base.atm_flux)),
        "new flux + LAPACK" => base,
        "new flux + Barger" => (@set base.osc = with_eigen(base.osc, Newtrinos.BargerEigen())),
    ]
    println("\n", "="^70, "\n$name\n", "="^70)
    res = Dict{String,Any}()
    for (label, phys) in variants
        r = bench(name, phys, mod); res[label] = r
        println(rpad("  $label", 24), " llh $(lpad(ms(r.t), 9)) ms   grad $(lpad(ms(r.tg), 9)) ms   ($(r.n) params)   llh=$(r.v)"); flush(stdout)
    end
    ref = res["new flux + LAPACK"]; bg = res["new flux + Barger"]
    gref = collect(values(ref.g)); gb = collect(values(bg.g))
    println("  |Δllh| Barger vs LAPACK: $(abs(bg.v - ref.v))")
    println("  max |Δgrad|/max|grad|:   $(maximum(abs.(gb .- gref)) / maximum(abs.(gref)))"); flush(stdout)
end
