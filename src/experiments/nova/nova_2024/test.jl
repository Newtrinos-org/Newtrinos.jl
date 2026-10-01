using Distributions
using DensityInterface
using BAT
using DataStructures
using Newtrinos
using FileIO
using Accessors
using CairoMakie
using HDF5

# Validation of the NOvA 2024 implementation against the official credible regions shipped with the
# data release (contours_RCDB1D_cond: Daya Bay 1D constraint on sin²2θ₁₃, normal ordering).
# Note: the official regions are Bayesian (marginalised), ours are profile-likelihood regions.

calibration = Symbol(get(ENV, "NOVA_CALIBRATION", "ref2024"))
experiments = (nova = Newtrinos.nova.configure(; calibration),)

likelihood = Newtrinos.generate_likelihood(experiments);

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

# solar parameters fixed as in the NOvA analysis; θ₁₃ from the Daya Bay sin²2θ₁₃ = 0.0851 ± 0.0024
ref = Newtrinos.nova.ref_params(calibration)
@reset p.θ₁₂ = ref.θ₁₂
@reset p.Δm²₂₁ = ref.Δm²₂₁
@reset p.θ₁₃ = ref.θ₁₃
@reset p.δCP = ref.δCP
@reset priors.θ₁₂ = p.θ₁₂
@reset priors.Δm²₂₁ = p.Δm²₂₁
@reset priors.θ₁₃ = Normal(ref.θ₁₃, 0.0024 / (4 * sqrt(0.0851 * (1 - 0.0851))))
@reset priors.δCP = Uniform(0, 2π)

# sin²θ₂₃ ∈ [0.40, 0.64], Δm²₃₂ ∈ [2.25, 2.65]e-3 eV², δCP ∈ [0, 2π]
θ₂₃_range = (asin(sqrt(0.40)), asin(sqrt(0.64)))
@reset priors.θ₂₃ = Uniform(θ₂₃_range...)
@reset priors.Δm²₃₁ = Uniform(2.25e-3 + p.Δm²₂₁, 2.65e-3 + p.Δm²₂₁)

if !isdir("test_output")
    mkdir("test_output")
end

# official conditional credible regions (Bayesian, marginalised) from the release, per mass ordering
function official(pair, mo)
    out = []
    h5open(Newtrinos.nova.datafile) do f
        g = f["contours/RCDB1D_cond"]
        for (lvl, ls) in (("068", :dash), ("090", :solid)), k in keys(g)
            occursin("$(pair)_$(mo)_cred_int_$(lvl)_", k) && push!(out, (read(g[k]["x"]), read(g[k]["y"]), ls, lvl == "090" && endswith(k, "_0")))
        end
    end
    out
end
function overlay!(ax, pair, mo)
    for (x, y, ls, lab) in official(pair, mo)
        lines!(ax, x, y, color = :red, linestyle = ls, label = lab ? "official (Bayesian)" : nothing)
    end
end

# conditional fits per mass ordering: Δm²₃₂ range, start point (NOvA frequentist best fits, Suppl. Tab. S4)
# and NOvA's frequentist 1σ δCP intervals (Feldman-Cousins corrected, Tab. S4) in units of π
orderings = (
    NO = (Δm²₃₂ = (2.25e-3, 2.65e-3), start = (Δm²₃₂ = 2.441e-3, δCP = 0.87π), dcp_1σ = ((0.0, 0.16), (0.58, 1.17), (1.97, 2.0)), color = :blue),
    IO = (Δm²₃₂ = (-2.65e-3, -2.25e-3), start = (Δm²₃₂ = -2.481e-3, δCP = 1.53π), dcp_1σ = ((1.26, 1.76),), color = :darkorange),
)
setup(o) = (@set(priors.Δm²₃₁ = Uniform((o.Δm²₃₂ .+ p.Δm²₂₁)...)),
            merge(p, (Δm²₃₁ = o.start.Δm²₃₂ + p.Δm²₂₁, δCP = o.start.δCP)))

# the θ₂₃ octants are separate local minima: profile each octant separately, keep the better one per point
function profile_octants(pr, p0, vars_to_scan, cache_dir)
    oct = map((lower = (θ₂₃_range[1], π / 4), upper = (π / 4, θ₂₃_range[2]))) do r
        Newtrinos.profile(likelihood, @set(pr.θ₂₃ = Uniform(r...)), vars_to_scan, @set(p0.θ₂₃ = sum(r) / 2), cache_dir = cache_dir)
    end
    better = oct.lower.values.log_posterior .>= oct.upper.values.log_posterior
    Newtrinos.NewtrinosResult(axes = oct.lower.axes, values = map((l, u) -> ifelse.(better, l, u), oct.lower.values, oct.upper.values), meta = oct.lower.meta)
end

results = map(keys(orderings), values(orderings)) do mo, o
    pr, p0 = setup(o)
    cache(name) = "test_cache_$(calibration)_$(mo)_$(name)"
    th23dm = Newtrinos.profile(likelihood, pr, OrderedDict(:θ₂₃ => 21, :Δm²₃₁ => 21), p0, cache_dir = cache("th23dm31"))
    dcpth23 = Newtrinos.profile(likelihood, pr, OrderedDict(:δCP => 25, :θ₂₃ => 21), p0, cache_dir = cache("dcpth23"))
    dcp = profile_octants(pr, p0, OrderedDict(:δCP => 41), cache("dcp"))
    FileIO.save("test_output/test_$(mo).jld2", Dict("th23dm31" => th23dm, "dcpth23" => dcpth23, "dcp" => dcp))
    mo => (; th23dm, dcpth23, dcp)
end |> NamedTuple
result = results.NO.th23dm

global_max = maximum(r -> maximum(r.dcp.values.log_posterior), results)
for mo in keys(results)
    Δ = 2 * (global_max - maximum(results[mo].dcp.values.log_posterior))
    println("$(mo): best -2ΔlogL relative to the global best fit = $(round(Δ, digits = 3))")
end

# --- 1) νμ disappearance: sin²θ₂₃ – Δm²₃₂ (conditional on each ordering) ---
fig = Figure(size = (1100, 450))
for (i, mo) in enumerate(keys(orderings))
    r = results[mo].th23dm
    converted = Newtrinos.NewtrinosResult(axes = (sin2theta23 = sin.(r.axes.θ₂₃) .^ 2, Δm²₃₂ = (r.axes.Δm²₃₁ .- p.Δm²₂₁) .* 1e3), values = r.values)
    ax = Axis(fig[1, i], title = "NOvA 2024 $(mo) 68%, 90% C.L. contours", xlabel = "sin²θ₂₃", ylabel = "Δm²₃₂ (10⁻³ eV²)")
    overlay!(ax, "ssth23dm32", mo)
    plot!(ax, converted, levels = [0.68, 0.9], label = "ours", color = orderings[mo].color)
    axislegend(ax, position = mo == :NO ? :rt : :rb)
end
save("test_output/contours.png", fig)

bestfit = Newtrinos.bestfit(results.NO.th23dm)
fig = experiments.nova.plot(bestfit)
save("test_output/datamc.png", fig)

# --- 2) νe appearance: δCP – sin²θ₂₃ per ordering, and δCP for both orderings ---
fig = Figure(size = (1100, 900))
for (i, mo) in enumerate(keys(orderings))
    r = results[mo].dcpth23
    converted = Newtrinos.NewtrinosResult(axes = (δCP = r.axes.δCP ./ π, sin2theta23 = sin.(r.axes.θ₂₃) .^ 2), values = r.values)
    ax = Axis(fig[1, i], title = "NOvA 2024 $(mo) 68%, 90% C.L. contours", xlabel = "δCP / π", ylabel = "sin²θ₂₃")
    overlay!(ax, "dcpssth23", mo)
    plot!(ax, converted, levels = [0.68, 0.9], label = "ours", color = orderings[mo].color)
    xlims!(ax, 0, 2); ylims!(ax, 0.40, 0.64)
    axislegend(ax, position = :lb)
end

ax = Axis(fig[2, 1:2], title = "NOvA 2024, profiled over all other parameters (relative to the global best fit)",
          xlabel = "δCP / π", ylabel = "-2 Δ log L")
for mo in keys(orderings)
    o = orderings[mo]
    for (lo, hi) in o.dcp_1σ
        vspan!(ax, lo, hi, color = (o.color, 0.12))
    end
    lines!(ax, [NaN], [NaN], color = (o.color, 0.3), linewidth = 8, label = "official $(mo) 1σ (frequentist, FC)")
    r = results[mo].dcp
    plot!(ax, Newtrinos.NewtrinosResult(axes = (δCP = r.axes.δCP ./ π,), values = r.values), max_llh = global_max,
          levels = [0.68, 0.9], label = "ours, $(mo)", color = o.color)
end
xlims!(ax, 0, 2); ylims!(ax, 0, 8)
axislegend(ax, position = :lt)
save("test_output/dcp.png", fig)

open("README.md", "w") do io
    write(io, "# NOvA 2024 (26.61E20 ν + 12.5E20 ν̄ POT)\n ## Resources\n")
    write(io, """
data source: NOvA 2024 official data release, https://doi.org/10.5281/zenodo.17822358 (FERMILAB-DATA-2025-05),
analysis: NOvA Collaboration, Phys. Rev. Lett. 136, 011802 (2026), arXiv:2509.04361.
The ROOT files were converted once to `NOvA_2024_data_release.h5` with `convert_nova_2024_to_hdf5.py`.

""")
    write(io, "\n## Test output plots\n")
    write(io, "![Comparison](test_output/contours.png)\n")
    write(io, "![dCP](test_output/dcp.png)\n")
    write(io, "![DataMC](test_output/datamc.png)\n")
    write(io, "## Meta Information\n")
    for (key, value) in result.meta
        write(io, "- **$(key)**: $(value)\n")
    end
end
