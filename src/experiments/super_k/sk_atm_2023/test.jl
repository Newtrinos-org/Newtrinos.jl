using Newtrinos
using DensityInterface
using JLD2, FileIO
using DataStructures
using CSV, DataFrames
using CairoMakie
using Printf

# Validation script for the Super-K analysis, comparing against the official
# Super-Kamiokande I-V atmospheric oscillation results (Wendell et al., DOI:
# 10.5281/zenodo.8401262).
#
# Unlike deepcore/orca's test.jl, this does NOT recompute the profile scan from
# scratch every run: a full (θ₂₃, Δm²₃₁) profiled scan over ~80 free parameters took
# ~9.9 hours to produce (see the loaded result's `meta["exec_time"]`), which is not
# practical to run routinely. Instead this script loads the precomputed best-fit and
# 2D-scan results shipped alongside it (superk_final_bestfit.jld2,
# superk_final_theta23_dm231.jld2), and only falls back to recomputing them (via
# Newtrinos.find_mle/Newtrinos.profile — the same calls used to produce the shipped
# files) if they're missing.
#
# Every stored grid point's full optimized parameter set is re-evaluated against the
# live likelihood and checked to reproduce the stored value — this is cheap (~0.5s per
# point) and catches any future code change that silently breaks reproducibility of the
# cached reference, without needing to rerun the expensive optimization itself.

experiments = (super_k = Newtrinos.super_k.configure(),)
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

bestfit_path = joinpath(@__DIR__, "superk_final_bestfit.jld2")
scan_path = joinpath(@__DIR__, "superk_final_theta23_dm231.jld2")

# --- Load (or, as a last resort, recompute) the best fit ---
if isfile(bestfit_path)
    bf = JLD2.load(bestfit_path)
else
    @warn "superk_final_bestfit.jld2 not found — recomputing via Newtrinos.find_mle " *
          "(this needs a working Optimization.LBFGS and can take a while)"
    llh, logp, params = Newtrinos.find_mle(likelihood, priors, p)
    bf = Dict("llh" => llh, "logp" => logp, "params" => params, "converged" => isfinite(llh))
    FileIO.save(bestfit_path, bf)
end

# --- Load (or, as a last resort, recompute — ~10 hours) the 2D profile scan ---
if isfile(scan_path)
    result = JLD2.load(scan_path)["result"]
else
    @warn "superk_final_theta23_dm231.jld2 not found — recomputing via Newtrinos.profile. " *
          "This took ~9.9 hours to produce the shipped reference file; expect similar here."
    vars_to_scan = OrderedDict(:θ₂₃ => 22, :Δm²₃₁ => 19)
    result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir = "test")
    FileIO.save(scan_path, Dict("result" => result))
end

# --- Correctness check: every stored grid point's parameters must still reproduce its
#     stored likelihood under the current code. Cheap (~0.5s/point) since it's plain
#     likelihood evaluation, no optimization. ---
const NON_PARAM_KEYS = (:llh, :log_posterior, :converged)

function check_reproduces(likelihood, params::NamedTuple, stored_llh; rtol = 1e-6)
    recomputed = logdensityof(likelihood, params)
    isapprox(recomputed, stored_llh; rtol = rtol), recomputed
end

println("Checking bestfit reproduces...")
ok, recomputed = check_reproduces(likelihood, bf["params"], bf["llh"])
if !ok
    error("Bestfit likelihood mismatch: stored=$(bf["llh"]) recomputed=$recomputed")
end
@printf("  stored=%.6f recomputed=%.6f  OK\n", bf["llh"], recomputed)

println("Checking all $(length(result.values.llh)) grid points reproduce...")
t0 = time()
n_checked = 0
max_reldiff = 0.0
for idx in CartesianIndices(result.values.llh)
    stored = result.values.llh[idx]
    pp = NamedTuple(k => result.values[k][idx] for k in keys(result.values) if !(k in NON_PARAM_KEYS))
    local ok, recomputed = check_reproduces(likelihood, pp, stored)
    if !ok
        error("Grid point $(idx) mismatch: stored=$stored recomputed=$recomputed")
    end
    global max_reldiff = max(max_reldiff, abs(recomputed - stored) / abs(stored))
    global n_checked += 1
end
@printf("  %d/%d grid points OK (max rel. diff %.2e) in %.1fs\n",
        n_checked, length(result.values.llh), max_reldiff, time() - t0)

# --- Contour comparison against the official published 90% C.L. result ---
# θ₁₃ is a free (optimized) parameter in our fit, so the apples-to-apples official
# comparison is the "q13-free" scan (θ₁₃ also profiled), not "q13-constrained".
official = CSV.read(joinpath(@__DIR__, "chi2", "sk_2023_q13-free_no.txt"), DataFrame;
    delim = ' ', ignorerepeated = true, comment = "#",
    header = [:dm2, :s2th23, :dcp, :s2th13, :chi2])

# profile (minimize) over δCP and sin²θ13 at each (Δm², sin²θ23) grid point
official_profiled = combine(groupby(official, [:dm2, :s2th23]), :chi2 => minimum => :chi2)
official_dm2 = sort(unique(official_profiled.dm2))
official_s2th23 = sort(unique(official_profiled.s2th23))
official_chi2 = Matrix{Float64}(undef, length(official_dm2), length(official_s2th23))
for row in eachrow(official_profiled)
    i = searchsortedfirst(official_dm2, row.dm2)
    j = searchsortedfirst(official_s2th23, row.s2th23)
    official_chi2[i, j] = row.chi2
end
official_dchi2 = official_chi2 .- minimum(official_chi2)

if !isdir(joinpath(@__DIR__, "test_output"))
    mkdir(joinpath(@__DIR__, "test_output"))
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "sin²θ₂₃", ylabel = "Δm²₃₁ (eV²)",
          title = "Super-K NO 90% C.L. contour")
contour!(ax, official_s2th23, official_dm2, permutedims(official_dchi2),
         levels = [4.61], color = :red)
converted = Newtrinos.NewtrinosResult(
    axes = (sin2theta23 = sin.(result.axes.θ₂₃) .^ 2, Δm²₃₁ = result.axes.Δm²₃₁),
    values = result.values)
Newtrinos.plot!(ax, converted, levels = [0.9], color = :blue, label = "Newtrinos")
lines!(ax, [NaN], [NaN], color = :red, label = "SK 2023 official")  # legend entry for the contour!
axislegend(ax)
save(joinpath(@__DIR__, "test_output", "contours.png"), fig)

fig2 = experiments.super_k.plot(bf["params"])
save(joinpath(@__DIR__, "test_output", "datamc.png"), fig2)

open(joinpath(@__DIR__, "README.md"), "w") do io
    write(io, "# Super-Kamiokande I-V Atmospheric Neutrino Analysis\n ## Resources\n")
    write(io, """
data source: Super-K I-V atmospheric neutrino data release, DOI: 10.5281/zenodo.8401262
Associated publication: "Atmospheric neutrino oscillation analysis with neutron tagging
and an expanded fiducial volume in Super-Kamiokande I-V"

""")
    write(io, "\n## Test output plots\n")
    write(io, "![Comparison](test_output/contours.png)\n")
    write(io, "![DataMC](test_output/datamc.png)\n")
    write(io, "## Meta Information\n")
    if haskey(result.meta, "exec_time")
        write(io, "- **exec_time**: $(result.meta["exec_time"]) s\n")
    end
    if haskey(result.meta, "date")
        write(io, "- **date**: $(result.meta["date"])\n")
    end
    if haskey(result.meta, "commit_hash")
        write(io, "- **commit_hash**: $(result.meta["commit_hash"])\n")
    end
end

println("Done.")
