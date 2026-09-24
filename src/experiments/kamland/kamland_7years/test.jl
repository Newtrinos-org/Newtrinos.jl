using Distributions
using DensityInterface
using BAT
using DataStructures
using Newtrinos
using FileIO
using Accessors
using CairoMakie
using DataFrames
using DelimitedFiles

experiments = (kamland = Newtrinos.kamland.configure(),)

vars_to_scan = OrderedDict()
vars_to_scan[:θ₁₂] = 21
vars_to_scan[:Δm²₂₁] = 21

likelihood = Newtrinos.generate_likelihood(experiments);

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

@reset p.θ₁₃ = 0.
@reset priors.Δm²₃₁ = p.Δm²₃₁
@reset priors.θ₁₃ = p.θ₁₃
@reset priors.θ₂₃ = p.θ₂₃
@reset priors.δCP = p.δCP

result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="test")

if !isdir("test_output")
    mkdir("test_output")
end

FileIO.save("test_output/test.jld2", Dict("result" => result))

converted = Newtrinos.NewtrinosResult(axes = (tan2theta12 = tan.(result.axes.θ₁₂).^2, Δm²₂₁ = result.axes.Δm²₂₁), values=result.values);

df = DataFrame(readdlm("delta_chi2_4th_result.dat"), :auto)
rename!(df, [:tan2theta12, :sin2theta13, :dm2, :deltachi2]);

sh = (121, 114, 81)

ax = (θ₁₃=asin.(sqrt.(reshape(df.sin2theta13, sh)[:,1,1])),
      tan²θ₁₂=reshape(df.tan2theta12, sh)[1,:,1],
      Δm²₂₁=reshape(df.dm2, sh)[1,1,:],)

chi2 = reshape(df.deltachi2, sh);

official = NewtrinosResult(axes=ax[keys(ax)[2:3]], values = (log_posterior =  -0.5 .*chi2[1, :, :],))

fig = Figure()
ax = Axis(fig[1,1])
ax.title = "KamLAND NO 1,2,3 sigma C.L. contours"
plot!(ax, converted, levels=1 .- 2*ccdf(Normal(), 1:3), label="ours")
plot!(ax, official, levels=1 .- 2*ccdf(Normal(), 1:3), label="official", color=:red)
ax.xlabel = "tan²θ₁₂"
ax.ylabel = "Δm²₂₁ (eV²)"
axislegend(ax)
save("test_output/contours.png", fig)

bestfit = Newtrinos.bestfit(result)

fig = experiments.kamland.plot(bestfit)
save("test_output/datamc.png", fig)

open("README.md", "w") do io
    write(io, "# KamLAND\n ## Resources\n")
    write(io, "Data digitized from plots in https://arxiv.org/abs/1009.4771\n\n")
    write(io, "\n## Test output plots\n")
    write(io, "![Comparison](test_output/contours.png)\n")
    write(io, "![DataMC](test_output/datamc.png)\n")
    write(io, "## Meta Information\n")
    for (key, value) in result.meta
        write(io, "- **$(key)**: $(value)\n")
    end
end
