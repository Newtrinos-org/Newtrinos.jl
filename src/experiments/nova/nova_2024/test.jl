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

# official NO credible regions (Bayesian, marginalised) from the release, as (x, y, linestyle) per graph
function official(pair)
    out = []
    h5open(Newtrinos.nova.datafile) do f
        g = f["contours/RCDB1D_cond"]
        for (lvl, ls) in (("068", :dash), ("090", :solid)), k in keys(g)
            occursin("$(pair)_NO_cred_int_$(lvl)_", k) && push!(out, (read(g[k]["x"]), read(g[k]["y"]), ls, lvl == "090" && endswith(k, "_0")))
        end
    end
    out
end
function overlay!(ax, pair)
    for (x, y, ls, lab) in official(pair)
        lines!(ax, x, y, color = :red, linestyle = ls, label = lab ? "official (Bayesian)" : nothing)
    end
end

# --- 1) νμ disappearance: sin²θ₂₃ – Δm²₃₂ ---
vars_to_scan = OrderedDict(:θ₂₃ => 21, :Δm²₃₁ => 21)
result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir = "test_cache_$(calibration)_th23dm31")
FileIO.save("test_output/test.jld2", Dict("result" => result))

converted = Newtrinos.NewtrinosResult(axes = (sin2theta23 = sin.(result.axes.θ₂₃) .^ 2, Δm²₃₂ = (result.axes.Δm²₃₁ .- p.Δm²₂₁) .* 1e3), values = result.values);
fig = Figure()
ax = Axis(fig[1, 1], title = "NOvA 2024 NO 68%, 90% C.L. contours", xlabel = "sin²θ₂₃", ylabel = "Δm²₃₂ (10⁻³ eV²)")
overlay!(ax, "ssth23dm32")
plot!(ax, converted, levels = [0.68, 0.9], label = "ours")
axislegend(ax)
save("test_output/contours.png", fig)

bestfit = Newtrinos.bestfit(result)
fig = experiments.nova.plot(bestfit)
save("test_output/datamc.png", fig)

# --- 2) νe appearance: δCP – sin²θ₂₃ and δCP alone ---
vars_to_scan = OrderedDict(:δCP => 25, :θ₂₃ => 21)
result_dcp2d = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir = "test_cache_$(calibration)_dcpth23")
FileIO.save("test_output/test_dcp_th23.jld2", Dict("result" => result_dcp2d))

# 1D δCP profile: the θ₂₃ octants are separate local minima, so profile each octant separately
# and keep the better one per δCP point
vars_to_scan = OrderedDict(:δCP => 41)
octants = map((lower = (θ₂₃_range[1], π / 4), upper = (π / 4, θ₂₃_range[2]))) do r
    pr = @set priors.θ₂₃ = Uniform(r...)
    p0 = @set p.θ₂₃ = sum(r) / 2
    Newtrinos.profile(likelihood, pr, vars_to_scan, p0, cache_dir = "test_cache_$(calibration)_dcp")
end
better = octants.lower.values.log_posterior .>= octants.upper.values.log_posterior
result_dcp = Newtrinos.NewtrinosResult(axes = octants.lower.axes,
    values = map((l, u) -> ifelse.(better, l, u), octants.lower.values, octants.upper.values), meta = octants.lower.meta)
FileIO.save("test_output/test_dcp.jld2", Dict("result" => result_dcp))

converted = Newtrinos.NewtrinosResult(axes = (δCP = result_dcp2d.axes.δCP ./ π, sin2theta23 = sin.(result_dcp2d.axes.θ₂₃) .^ 2), values = result_dcp2d.values);
fig = Figure(size = (1100, 450))
ax = Axis(fig[1, 1], title = "NOvA 2024 NO 68%, 90% C.L. contours", xlabel = "δCP / π", ylabel = "sin²θ₂₃")
overlay!(ax, "dcpssth23")
plot!(ax, converted, levels = [0.68, 0.9], label = "ours")
axislegend(ax, position = :lb)

ax = Axis(fig[1, 2], title = "NOvA 2024 NO, profiled over all other parameters", xlabel = "δCP / π", ylabel = "-2 Δ log L")
# NOvA frequentist 1σ intervals (Feldman-Cousins corrected), 1D Daya Bay constraint, NO (Suppl. Tab. S4)
for (lo, hi) in ((0.0, 0.16), (0.58, 1.17), (1.97, 2.0))
    vspan!(ax, lo, hi, color = (:red, 0.15))
end
lines!(ax, [NaN], [NaN], color = (:red, 0.3), linewidth = 8, label = "official 1σ (frequentist, FC)")
plot!(ax, Newtrinos.NewtrinosResult(axes = (δCP = result_dcp.axes.δCP ./ π,), values = result_dcp.values), levels = [0.68, 0.9], label = "ours")
ylims!(ax, 0, 6)
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
