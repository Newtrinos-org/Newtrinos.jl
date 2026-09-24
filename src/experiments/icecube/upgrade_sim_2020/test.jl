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

experiments = (ic_upgrade = Newtrinos.ic_upgrade.configure(),)

vars_to_scan = OrderedDict()
vars_to_scan[:θ₂₃] = 11
vars_to_scan[:Δm²₃₁] = 11

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

# DC 3y PRL bestfit
@reset p.Δm²₃₁ = 0.00231 + p.Δm²₂₁
@reset p.θ₂₃ = asin(sqrt(0.51))

@reset priors.Δm²₂₁ = p.Δm²₂₁
@reset priors.θ₁₂ = p.θ₁₂
@reset priors.δCP = p.δCP
@reset priors.nutau_cc_norm = p.nutau_cc_norm
@reset priors.θ₁₃ = Truncated(Normal(0.156, 0.008), 0.13, 0.18)
@reset priors.Δm²₃₁ = Uniform(0.0023, 0.00255)
@reset priors.θ₂₃ = Uniform(0.685, 0.9)

asimov = Newtrinos.generate_asimov_data(experiments, p)
likelihood = Newtrinos.generate_likelihood(experiments, asimov);

result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="test")

if !isdir("test_output")
    mkdir("test_output")
end

FileIO.save("test_output/test.jld2", Dict("result" => result))

converted = Newtrinos.NewtrinosResult(axes = (sin2theta23 = sin.(result.axes.θ₂₃).^2, Δm²₃₂ = result.axes.Δm²₃₁ .- p.Δm²₂₁), values=result.values);

official = CSV.read("wpd_datasets.csv", DataFrame, header=2)

fig = Figure()
ax = Axis(fig[1,1])

lines!(ax, vcat(official.X, official.X[[1]]), vcat(official.Y, official.Y[[1]]), color=:red, label="official")

ax.title = "IceCube Upgrade 3y NO 90% C.L. contours"
plot!(ax, converted, levels=[0.9,], label="ours")

ax.xlabel = "sin²θ₂₃"
ax.ylabel = "Δm²₃₂ (eV²)"
axislegend(ax)
save("test_output/contours.png", fig)

fig = experiments.ic_upgrade.plot(p, asimov.ic_upgrade)
save("test_output/spectrum.png", fig)

open("README.md", "w") do io
    write(io, "# IceCube Upgrade 3y MC Sample\n ## Resources\n")
    write(io, "data source: https://icecube.wisc.edu/data-releases/2020/04/icecube-upgrade-neutrino-monte-carlo-simulation/")
    write(io, "\n## Test output plots\n")
    write(io, "![Comparison](test_output/contours.png)\n")
    write(io, "![DataMC](test_output/spectrum.png)\n")
    write(io, "## Meta Information\n")
    for (key, value) in result.meta
        write(io, "- **$(key)**: $(value)\n")
    end
end
