# Best fit of each experiment alone, in its default setup (`configure()`, default priors), stored as
# `<experiment module>.toml` in this directory. Only the experiment's own parameters are read back, as start values by
# `Newtrinos.get_params(...; start = :best_fit)`; the oscillation best fit is recorded for information only.
# Each experiment is fitted from 8 starts (both mass orderings, both θ₂₃ octants, δCP = π/2 and 3π/2); the best
# one is kept.
#
#   julia -t N --project=<Newtrinos.jl> src/experiments/best_fit/generate.jl <experiment> [<experiment> ...]
using Newtrinos, Distributions, TOML, Dates, Printf, LinearAlgebra, Base.Threads
using BAT: distprod
BLAS.set_num_threads(1)
const N = Newtrinos
isempty(ARGS) && error("usage: generate.jl <experiment> [<experiment> ...]")
logmsg(s) = (println(Dates.format(now(), "HH:MM:SS "), s); flush(stdout))

setups = Dict(map(ARGS) do name
    x = getproperty(N, Symbol(name)).configure()
    exps = NamedTuple{(Symbol(name),)}((x,))
    name => (; x, p = N.get_params(exps), pr = N.get_priors(exps), llh = N.generate_likelihood(exps))
end)
const COMMIT = try strip(read(`git -C $(pkgdir(N)) rev-parse --short HEAD`, String)) catch; "unknown" end
"Write `<name>.toml` from the finished fits of `name`"
function write_best_fit(name, res)
    rs = filter(r -> r[1] == name && isfinite(r[3][2]), res)
    isempty(rs) && (logmsg("$name: no successful fit"); return)
    _, o, (llh, lp, bf) = rs[argmax([r[3][2] for r in rs])]
    own = setups[name].x.params
    isempty(own) && (logmsg("$name: no experiment parameters"); return)
    osc = Dict(string(k) => bf[k] for k in (:θ₁₂, :θ₁₃, :θ₂₃, :δCP, :Δm²₂₁, :Δm²₃₁) if haskey(bf, k))
    d = Dict("meta" => Dict("experiment" => name, "newtrinos_commit" => COMMIT, "date" => string(Dates.today()),
                            "log_likelihood" => llh, "log_posterior" => lp, "ordering" => string(o), "oscillation" => osc),
             "params" => Dict(string(k) => (bf[k] isa AbstractArray ? collect(Float64, vec(bf[k])) : Float64(bf[k])) for k in keys(own)))
    logmsg("$name best fit: " * repr(d["params"]))          # also in the log, in case writing the file fails
    open(joinpath(@__DIR__, "$name.toml"), "w") do io
        println(io, "# written by generate.jl: best fit of `$name` alone in its default setup; [params] are read by")
        println(io, "# Newtrinos.get_params(...; start = :best_fit) as start values, [meta] is for information only")
        TOML.print(io, d; sorted = true)
    end
    logmsg("$name: written ($(length(own)) parameters, $o, log posterior $(round(lp, digits = 3)))")
end

starts = [(o, s23, d) for o in (:NO, :IO) for s23 in (0.45, 0.57) for d in (π / 2, 3π / 2)]
jobs = [(name, st) for name in ARGS for st in starts]
logmsg("$(length(jobs)) fits on $(nthreads()) threads")
res = Vector{Any}(undef, length(jobs)); done = Dict(name => 0 for name in ARGS); lk = ReentrantLock()
@threads :greedy for i in eachindex(jobs)
    name, (o, s23, d) = jobs[i]
    s = setups[name]; t = time()
    pr, p = s.pr, s.p
    if haskey(pr, :Δm²₃₁) && pr[:Δm²₃₁] isa Distribution
        pr = merge(pr, (Δm²₃₁ = o == :NO ? Uniform(2e-3, 3e-3) : Uniform(-3e-3, -2e-3),))
        p = merge(p, (Δm²₃₁ = o == :NO ? 2.5e-3 : -2.45e-3,))
    end
    haskey(pr, :θ₂₃) && pr[:θ₂₃] isa Distribution && (p = merge(p, (θ₂₃ = asin(sqrt(s23)),)))
    haskey(pr, :δCP) && pr[:δCP] isa Distribution && (p = merge(p, (δCP = d,)))
    r = N.find_mle(s.llh, distprod(; pr...), p; adsel = N.ADTypes.AutoForwardDiff())
    res[i] = (name, o, r)
    logmsg(@sprintf("%s %s θ₂₃ %.2f δCP %.2f: log posterior %.3f (%.1f min)", name, o, s23, d, r[2], (time() - t) / 60))
    lock(lk) do
        (done[name] += 1) == length(starts) && write_best_fit(name, [x for x in res[[j for j in eachindex(jobs) if isassigned(res, j)]] if x[1] == name])
    end
end

