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

experiments = (juno = Newtrinos.juno.configure(),)

p = Newtrinos.get_params(experiments)

@reset p.Δm²₂₁ = 0.0000753
@reset p.Δm²₃₁ =  0.0025283
@reset p.θ₁₂ = asin(sqrt(0.307))
@reset p.δCP = 0
@reset p.θ₁₃ = asin(sqrt( 0.0218))

asimov = Newtrinos.generate_asimov_data(experiments, p)
likelihood = Newtrinos.generate_likelihood(experiments, asimov);

priors = Newtrinos.get_priors(experiments)
@reset priors.θ₂₃ = p.θ₂₃

@reset priors.Δm²₃₁ = Uniform(0.00245, 0.0026)
@reset priors.Δm²₂₁ = Uniform(0.000072, 0.000078)
@reset priors.θ₁₂ = Uniform(asin(sqrt(0.28)), asin(sqrt(0.335)))
@reset priors.θ₁₃ = Uniform(asin(sqrt(0.016)), asin(sqrt(0.027)))

if !isdir("test_output")
    mkdir("test_output")
end

vars_to_scan = OrderedDict()
vars_to_scan[:Δm²₃₁] = 31
result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="test_dm31")
FileIO.save("test_output/test_dm31.jld2", Dict("result" => result))
fig = Figure()
ax = Axis(fig[1,1])
ax.title = "JUNO NO 6 years"
plot!(ax, result, levels=[0.68, 0.9,], label="ours")
lines!(result.axes.Δm²₃₁, ((result.axes.Δm²₃₁ .- p.Δm²₃₁) ./ (0.0000047)) .^2, color=:red, label="official")
ylims!(ax, 0, 10)
ax.xlabel = "Δm²₃₁ (eV²)"
axislegend(ax)
save("test_output/dm31.png", fig)

vars_to_scan = OrderedDict()
vars_to_scan[:Δm²₂₁] = 31
result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="test_dm21")
FileIO.save("test_output/test_dm21.jld2", Dict("result" => result))
fig = Figure()
ax = Axis(fig[1,1])
ax.title = "JUNO NO 6 years"
plot!(ax, result, levels=[0.68, 0.9,], label="ours")
lines!(result.axes.Δm²₂₁, ((result.axes.Δm²₂₁ .- p.Δm²₂₁) ./ (0.00000024)) .^2, color=:red, label="official")
ylims!(ax, 0, 10)
ax.xlabel = "Δm²₂₁ (eV²)"
axislegend(ax)
save("test_output/dm21.png", fig)

vars_to_scan = OrderedDict()
vars_to_scan[:θ₁₂] = 31
result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="test_theta12")
FileIO.save("test_output/test_theta12.jld2", Dict("result" => result))
fig = Figure()
ax = Axis(fig[1,1])
ax.title = "JUNO NO 6 years"
converted = Newtrinos.NewtrinosResult(axes = (sin2theta12 = sin.(result.axes.θ₁₂).^2,), values=result.values);
plot!(ax, converted, levels=[0.68, 0.9,], label="ours")
lines!(converted.axes.sin2theta12, ((converted.axes.sin2theta12 .- sin(p.θ₁₂)^2) ./ (0.0016)) .^2, color=:red, label="official")
ylims!(ax, 0, 10)
ax.xlabel = "sin²θ₁₂"
axislegend(ax)
save("test_output/theta12.png", fig)

vars_to_scan = OrderedDict()
vars_to_scan[:θ₁₃] = 31
result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="test_theta13")
FileIO.save("test_output/test_theta13.jld2", Dict("result" => result))
fig = Figure()
ax = Axis(fig[1,1])
ax.title = "JUNO NO 6 years"
converted = Newtrinos.NewtrinosResult(axes = (sin2theta13 = sin.(result.axes.θ₁₃).^2,), values=result.values);
plot!(ax, converted, levels=[0.68, 0.9,], label="ours")
lines!(converted.axes.sin2theta13, ((converted.axes.sin2theta13 .- sin(p.θ₁₃)^2) ./ (0.0026)) .^2, color=:red, label="official")
ylims!(ax, 0, 10)
ax.xlabel = "sin²θ₁₃"
axislegend(ax)
save("test_output/theta13.png", fig)

fig = experiments.juno.plot(p, data_to_plot=asimov.juno)
save("test_output/spectrum.png", fig)

open("README.md", "w") do io
    write(io, "# JUNO\n ## Resources\n")
    write(io, "Data source: -\n")
    write(io, "\n## Test output plots\n")
    write(io, "![Comparison](test_output/dm31.png)\n")
    write(io, "![Comparison](test_output/dm21.png)\n")
    write(io, "![Comparison](test_output/theta12.png)\n")
    write(io, "![Comparison](test_output/theta13.png)\n")
    write(io, "![Comparison](test_output/spectrum.png)\n")
    write(io, "## Meta Information\n")
    for (key, value) in result.meta
        write(io, "- **$(key)**: $(value)\n")
    end
end
