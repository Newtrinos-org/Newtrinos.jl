using Distributions
using DensityInterface
using BAT
using DataStructures
using Newtrinos
using FileIO
using Accessors
using CairoMakie
using CSV, DataFrames

experiments = (dayabay = Newtrinos.dayabay_rewrite.configure(),)

vars_to_scan = OrderedDict()
vars_to_scan[:θ₁₃] = 11
vars_to_scan[:Δm²₃₁] = 11

likelihood = Newtrinos.generate_likelihood(experiments);

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

@reset priors.Δm²₃₁ = Uniform(0.0023, 0.0028)
@reset priors.θ₁₃ = Uniform(0.13, 0.16)

@reset priors.θ₁₂ = p.θ₁₂
@reset priors.θ₂₃ = p.θ₂₃
@reset priors.δCP = p.δCP
@reset priors.Δm²₂₁ = p.Δm²₂₁

result = Newtrinos.find_mle(likelihood, distprod(;priors...), p)
FileIO.save("test_output/test_rewrite_full_replaced_livetime.jld2", Dict("logllh"=> result[1], "logposterior"=>result[2], "mle"=>result[3]))

"""
result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="test")

if !isdir("test_output")
    mkdir("test_output")
end
FileIO.save("test_output/test.jld2", Dict("result" => result))

converted = Newtrinos.NewtrinosResult(axes = (sin22theta13 = sin.(2*result.axes.θ₁₃).^2, Δm²₃₂ = result.axes.Δm²₃₁ .- p.Δm²₂₁), values=result.values);
official_raw = CSV.read("DayaBay_DeltaChiSq_NO_3158days.txt", DataFrame, skipto=10, delim=" ", header=["sin22theta13", "Dm232", "Deltachi2"], ignorerepeated=true)
axes = (sin22theta13 = reshape(official_raw.sin22theta13, 100, 100)[1,:], Δm²₃₂ = reshape(official_raw.Dm232, 100, 100)[:,1])
values = Array(reshape(official_raw.Deltachi2, 100, 100))'
official = Newtrinos.NewtrinosResult(axes=axes, values=(log_posterior=-0.5 .* values,))

fig = Figure()
ax = Axis(fig[1,1])
ax.title = "DayaBay NO 1,2,3 sigma C.L. contours"
plot!(ax, converted, levels=1 .- 2*ccdf(Normal(), 1:3), label="ours")
plot!(ax, official, levels=1 .- 2*ccdf(Normal(), 1:3), label="official", color=:red)
ax.xlabel = "sin²2θ₁₃"
ax.ylabel = "Δm²₃₂ (eV²)"
axislegend(ax)
save("test_output/contours.png", fig)

bestfit = Newtrinos.bestfit(result)

"""
#fig = experiments.dayabay.plot(bestfit)
#save("test_output/datamc.png", fig)

