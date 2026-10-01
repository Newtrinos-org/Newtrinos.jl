using Distributions
using DensityInterface
using BAT
using DataStructures
using Newtrinos
using FileIO
using Accessors
using CairoMakie
using HDF5

# Validation of the T2K implementation against the official frequentist results of arXiv:2506.05889
# (reactor constraint on θ₁₃): Δχ²(δCP) per mass ordering, and the sin²θ₂₃–Δm² regions (digitised).
# The official Δm² contours are smeared for simulated-data studies and the δCP intervals in the paper
# are Feldman-Cousins corrected; ours are plain profile-likelihood (Wilks) results.

experiments = (t2k = Newtrinos.t2k.configure(),)
likelihood = Newtrinos.generate_likelihood(experiments);

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

# solar parameters fixed; θ₁₃ from the reactor constraint sin²θ₁₃ = 0.0220 ± 0.0007 (PDG 2021)
ref = Newtrinos.t2k.ref_params()
@reset p.θ₁₂ = ref.θ₁₂
@reset p.Δm²₂₁ = ref.Δm²₂₁
@reset p.θ₁₃ = ref.θ₁₃
@reset priors.θ₁₂ = p.θ₁₂
@reset priors.Δm²₂₁ = p.Δm²₂₁
@reset priors.θ₁₃ = Normal(ref.θ₁₃, 0.0007 / (2 * sin(ref.θ₁₃) * cos(ref.θ₁₃)))
@reset priors.δCP = Uniform(0, 2π)

θ₂₃_range = (asin(sqrt(0.35)), asin(sqrt(0.68)))
@reset priors.θ₂₃ = Uniform(θ₂₃_range...)

if !isdir("test_output")
    mkdir("test_output")
end

# conditional fits per mass ordering; the y axis of the official figure is Δm²₃₂ (NO) and |Δm²₃₁| (IO)
orderings = (
    NO = (Δm²₃₁ = (2.2e-3, 2.8e-3) .+ p.Δm²₂₁, start = (Δm²₃₁ = 2.506e-3 + p.Δm²₂₁, δCP = mod(-2.18, 2π)),
          dm2 = Δm²₃₁ -> Δm²₃₁ - p.Δm²₂₁, color = :steelblue),
    IO = (Δm²₃₁ = (-2.8e-3, -2.2e-3), start = (Δm²₃₁ = -2.47e-3, δCP = mod(-1.37, 2π)),
          dm2 = Δm²₃₁ -> -Δm²₃₁, color = :darkorange),
)
setup(o) = (@set(priors.Δm²₃₁ = Uniform(o.Δm²₃₁...)), merge(p, (Δm²₃₁ = o.start.Δm²₃₁, δCP = o.start.δCP, θ₂₃ = asin(sqrt(0.56)))))

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
    th23dm = Newtrinos.profile(likelihood, pr, OrderedDict(:θ₂₃ => 25, :Δm²₃₁ => 25), p0, cache_dir = "test_cache_$(mo)_th23dm31")
    dcp = profile_octants(pr, p0, OrderedDict(:δCP => 41), "test_cache_$(mo)_dcp")
    FileIO.save("test_output/test_$(mo).jld2", Dict("th23dm31" => th23dm, "dcp" => dcp))
    mo => (; th23dm, dcp)
end |> NamedTuple
result = results.NO.th23dm

global_max = maximum(r -> maximum(r.dcp.values.log_posterior), results)
for mo in keys(results)
    Δ = 2 * (global_max - maximum(results[mo].dcp.values.log_posterior))
    println("$(mo): best -2ΔlogL relative to the global best fit = $(round(Δ, digits = 3))")
end

official = h5open(Newtrinos.t2k.datafile) do f
    (th23dm2 = Dict(mo => Dict(l => (read(f["official/th23dm2/$mo/$l/ssth23"]), read(f["official/th23dm2/$mo/$l/dm2"])) for l in ("068", "090", "997")) for mo in ("NO", "IO")),
     dcp = Dict(mo => (read(f["official/dcp/$mo/dcp"]), read(f["official/dcp/$mo/dchi2"])) for mo in ("NO", "IO")))
end

# --- 1) sin²θ₂₃ – Δm² (conditional on each ordering) ---
fig = Figure(size = (1100, 450))
for (i, mo) in enumerate(keys(orderings))
    r = results[mo].th23dm; o = orderings[mo]
    converted = Newtrinos.NewtrinosResult(axes = (sin2theta23 = sin.(r.axes.θ₂₃) .^ 2, Δm² = o.dm2.(r.axes.Δm²₃₁) .* 1e3), values = r.values)
    ax = Axis(fig[1, i], title = "T2K $(mo) 68%, 90%, 99.7% C.L.", xlabel = "sin²θ₂₃", ylabel = mo == :NO ? "Δm²₃₂ (10⁻³ eV²)" : "|Δm²₃₁| (10⁻³ eV²)")
    for (l, ms) in (("068", 3), ("090", 2), ("997", 2))
        x, y = official.th23dm2[string(mo)][l]
        scatter!(ax, x, y .* 1e3, color = :red, markersize = ms, label = l == "090" ? "official (frequentist, digitised)" : nothing)
    end
    plot!(ax, converted, levels = [0.68, 0.9, 0.997], label = "ours", color = o.color)
    xlims!(ax, 0.35, 0.68); ylims!(ax, 2.2, 2.8)
    axislegend(ax, position = :lb)
end
save("test_output/contours.png", fig)

bestfit = Newtrinos.bestfit(results.NO.th23dm)
fig = experiments.t2k.plot(bestfit)
save("test_output/datamc.png", fig)

# --- 2) δCP for both orderings, relative to the global best fit ---
wrap(x) = mod(x + π, 2π) - π
fig = Figure(size = (800, 450))
ax = Axis(fig[1, 1], title = "T2K, profiled over all other parameters (relative to the global best fit)", xlabel = "δCP", ylabel = "-2 Δ log L")
for mo in keys(orderings)
    o = orderings[mo]
    x, y = official.dcp[string(mo)]
    lines!(ax, x, y, color = o.color, linestyle = :dash, label = "official $(mo)")
    r = results[mo].dcp
    d = wrap.(r.axes.δCP); i = sortperm(d)
    lines!(ax, d[i], (2 .* (global_max .- r.values.log_posterior))[i], color = o.color, linewidth = 2, label = "ours, $(mo)")
end
xlims!(ax, -π, π); ylims!(ax, 0, 30)
axislegend(ax, position = :lt)
save("test_output/dcp.png", fig)

open("README.md", "w") do io
    write(io, "# T2K (19.7E20 ν + 16.3E20 ν̄ POT)\n ## Resources\n")
    write(io, """
analysis: T2K Collaboration, arXiv:2506.05889 (2025); data: arXiv:2303.03222 (same exposure);
far-detector predictions: T2K data release, https://doi.org/10.5281/zenodo.15701867 (CC-BY 4.0).
All inputs were extracted from the vector PDFs (official contours of Fig. 4: digitised from a raster
image) into `T2K_2025.h5` with the scripts in `extraction/`.

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
