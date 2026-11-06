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

experiments = (juno = Newtrinos.juno_rewrite.configure(),)

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
likelihood = Newtrinos.generate_likelihood(experiments)

val = 0.02175
err = 0.00063
theta13_sigma = 1 / (2 * sqrt(-(val-1)*val)) * err
theta13 = asin(sqrt(0.02175))

spec_pulls_cov = FileIO.load("spectrum_pulls_cov.jld2")["cov"]
spec_pulls_mle = FileIO.load("spectrum_pulls_cov.jld2")["mle"]
temp = UpperTriangular(spec_pulls_cov)
temp += transpose(temp) - Diagonal(diag(temp))
@reset priors.spectrum_pulls = MvNormal(ones(19) .* spec_pulls_mle, temp)
@reset p.spectrum_pulls = spec_pulls_mle


@reset priors.θ₂₃ = p.θ₂₃
@reset priors.δCP = p.δCP 
@reset priors.Δm²₂₁ = Uniform(6.4e-5, 8.25e-5)
@reset priors.θ₁₂ = Uniform(asin(sqrt(0.28)), asin(sqrt(0.31)))
@reset priors.juno_geo_rate = 2.12
#@reset priors.juno_geo_rate = params.juno_geo_rate
#@reset priors.juno_lihe_rate = params.juno_lihe_rate
#@reset priors.juno_other_rate = params.juno_other_rate
#@reset priors.juno_bipo_rate = params.juno_bipo_rate
#@reset priors.juno_world_reactor_rate = params.juno_world_reactor_rate
@reset priors.θ₁₃ = Normal(theta13, theta13_sigma)
@reset priors.Δm²₃₁ = Normal(2.541e-3, 0.060e-3)
if !isdir("test_output")
    mkdir("test_output")
end

vars_to_scan = OrderedDict()
vars_to_scan[:Δm²₂₁] = 10
vars_to_scan[:θ₁₂] = 10
result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="test_dm21_theta12")
FileIO.save("test_output/test_dm21_theta12_fixed_geo.jld2", Dict("result" => result))
converted = Newtrinos.NewtrinosResult(axes = (sin2theta12 = sin.(result.axes.θ₁₂).^2, delta2m21= result.axes.Δm²₂₁), values=result.values);
f = Figure()
ax = Axis(f[1, 1])
ax.title = "Juno test"
plot!(ax, converted, levels=1 .- 2*ccdf(Normal(), 1:3), label="ours")
axislegend(ax)
save("test_output/dm21_theta12_fixed_geo.png", f)
