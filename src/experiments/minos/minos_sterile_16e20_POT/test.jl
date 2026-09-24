using Distributions
using DensityInterface
using BAT
using DataStructures
using Newtrinos
using FileIO
using Accessors
using CairoMakie
using DataFrames
using CSV

experiments = (minos = Newtrinos.minos.configure(),)

vars_to_scan = OrderedDict()
vars_to_scan[:θ₂₃] = 21
vars_to_scan[:Δm²₃₁] = 21

likelihood = Newtrinos.generate_likelihood(experiments);

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

@reset priors.Δm²₂₁ = p.Δm²₂₁
@reset priors.θ₁₂ = p.θ₁₂
@reset priors.δCP = p.δCP
@reset priors.nc_norm = p.nc_norm
@reset priors.nutau_cc_norm = p.nutau_cc_norm
@reset priors.θ₁₃ = Truncated(Normal(0.156, 0.008), 0.13, 0.18)

result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="test")

if !isdir("test_output")
    mkdir("test_output")
end

FileIO.save("test_output/test.jld2", Dict("result" => result))

converted = Newtrinos.NewtrinosResult(axes = (sin2theta23 = sin.(result.axes.θ₂₃).^2, Δm²₃₂ = result.axes.Δm²₃₁ .- p.Δm²₂₁), values=result.values);

official = CSV.read("wpd_datasets.csv", DataFrame, header=2)

fig = Figure()
ax = Axis(fig[1,1])
ax.title = "MINOS/MINOS+ NO 68%, 90% C.L. contours"
plot!(ax, converted, levels=[0.68, 0.9], label="ours")

lines!(ax, vcat(official.X, official.X[[1]]), vcat(official.Y, official.Y[[1]]), color=:red, label="official (slightly different analysis)")
lines!(ax, Float32.(vcat(official.X_1[1:end-1], official.X_1[[1]])), Float32.(vcat(official.Y_1[1:end-1], official.Y_1[[1]])), color=:red)

ax.xlabel = "sin²θ₂₃"
ax.ylabel = "Δm²₃₂ (eV²)"
axislegend(ax)
save("test_output/contours.png", fig)

bestfit = Newtrinos.bestfit(result)

fig = experiments.minos.plot(bestfit)
save("test_output/datamc.png", fig)

open("README.md", "w") do io
    write(io, "# MINOS\n ## Resources\n")
    write(io, "Data from supplemental files in https://arxiv.org/abs/1710.06488\n\n")
    write(io, "\n## Test output plots\n")
    write(io, "![Comparison](test_output/contours.png)\n")
    write(io, "![DataMC](test_output/datamc.png)\n")
    write(io, "## Meta Information\n")
    for (key, value) in result.meta
        write(io, "- **$(key)**: $(value)\n")
    end
end
