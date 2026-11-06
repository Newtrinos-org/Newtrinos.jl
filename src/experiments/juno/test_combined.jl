using Distributions
using DensityInterface
using BAT
using DataStructures
using Newtrinos
using LinearAlgebra
using DelimitedFiles
using FileIO
using Accessors
using CairoMakie
using DataFrames
using CSV

experiments = (juno=Newtrinos.juno_rewrite.configure(), tao=Newtrinos.tao_rewrite.configure())

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
@reset priors.Δm²₂₁ = Uniform(6e-5, 9e-5)
@reset priors.θ₁₂ = Uniform(asin(sqrt(0.25)), asin(sqrt(0.35)))
@reset priors.θ₁₃ = Normal(theta13, theta13_sigma)
@reset priors.Δm²₃₁ = Normal(2.5413e-3, 0.060e-3)

if !isdir("combined_output")
    mkdir("combined_output")
end

for i=1:4
    try
        mle = Newtrinos.find_mle(likelihood, distprod(;priors...), p)
        FileIO.save("combined_output/test_dm21_theta12_juno_tao_mle.jld2", Dict("llh"=>mle[1], "logposterior"=>mle[3], "params"=>mle[3]))
    catch e
        continue
    finally
        break
    end
end
vars_to_scan = OrderedDict()
vars_to_scan[:Δm²₂₁] = 10
vars_to_scan[:θ₁₂] = 10
@reset priors.Δm²₂₁ = Uniform(7.1e-5, 8.0e-5)
@reset priors.θ₁₂ = Uniform(asin(sqrt(0.275)), asin(sqrt(0.32)))
result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="combined_scratch")
FileIO.save("combined_output/test_dm21_theta12_juno_tao_all_params_fixed_geo.jld2", Dict("result" => result))
converted = Newtrinos.NewtrinosResult(axes = (sin2theta12 = sin.(result.axes.θ₁₂).^2, delta2m21= result.axes.Δm²₂₁), values=result.values);


f = Figure()
ax = Axis(f[1, 1])
ax.title = "Juno test"
plot!(ax, converted, levels=1 .- 2*ccdf(Normal(), 1:3), label="Newtrinos.jl 1,2,3sigma")
juno_1sigma = readdlm("207_days/fig_6_contour.txt", skipstart=25);
lines!(ax, juno_1sigma[:, 1], juno_1sigma[:, 2] .* 1e-5, label="Juno 1sigma")
scatter!(ax, 0.3036, 7.388e-5, label="Juno MLE")
scatter!(ax, sin(mle[3].θ₁₂)^2, mle[3].Δm²₂₁, label="Newtrinos.jl MLE")
axislegend(ax)
save("combined_output/dm21_theta12_juno_tao_all_params_fixed_geo.png", f)
