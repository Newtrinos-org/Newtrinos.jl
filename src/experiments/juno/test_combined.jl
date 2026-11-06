using Distributions
using DensityInterface
using BAT
using DataStructures
using Newtrinos
using LinearAlgebra
using FileIO
using Accessors
using CairoMakie
using DataFrames
using CSV

experiments = (juno=Newtrinos.juno_rewrite.configure(),)

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
likelihood = Newtrinos.generate_likelihood(experiments)

val = 0.0851
err = 0.0024
theta13 = asin(sqrt(val)) / 2
theta13_sigma = 1 / (4 * sqrt(-(val - 1) * val)) * err
spec_pulls_cov = FileIO.load("spectrum_pulls_cov_25.jld2")["cov"]
spec_pulls_mle = FileIO.load("spectrum_pulls_cov_25.jld2")["mle"]
temp = UpperTriangular(spec_pulls_cov)
temp += transpose(temp) - Diagonal(diag(temp))
@reset priors.spectrum_pulls = MvNormal(spec_pulls_mle, temp)
@reset p.spectrum_pulls = spec_pulls_mle

@reset priors.θ₂₃ = p.θ₂₃
@reset priors.δCP = p.δCP 
@reset priors.Δm²₂₁ = Uniform(7.1e-5, 7.9e-5)
@reset priors.θ₁₂ = Uniform(asin(sqrt(0.275)), asin(sqrt(0.32)))
#@reset priors.juno_geo_rate = 2.12
@reset priors.θ₁₃ = Normal(theta13, theta13_sigma)
@reset priors.Δm²₃₁ = Normal(2.509e-3, 0.027e-3)
if !isdir("combined_output")
    mkdir("combined_output")
end

vars_to_scan = OrderedDict()
vars_to_scan[:Δm²₂₁] = 10
vars_to_scan[:θ₁₂] = 10
result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="combined_scratch")

FileIO.save("test_output/test_dm21_theta12_eres_unc_pulls.jld2", Dict("result" => result))
converted = Newtrinos.NewtrinosResult(axes = (sin2theta12 = sin.(result.axes.θ₁₂).^2, delta2m21= result.axes.Δm²₂₁), values=result.values);
f = Figure()
ax = Axis(f[1, 1])
ax.title = "Juno test"
plot!(ax, converted, levels=1 .- 2*ccdf(Normal(), 1:3), label="ours")
juno_1sigma = readdlm("207_days/fig_6_contour.txt", skipstart=25);
scatter!(ax, 0.3036, 7.388e-5, label="Juno MLE")
lines!(ax, contour[:, 1], contour[:, 2] .* 1e-5)
axislegend(ax)
save("combined_output/dm21_theta12_combined_eres_unc_pulls.png", f)
