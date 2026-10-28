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
@reset priors.θ₁₃ = Normal(theta13, theta13_sigma)
@reset priors.Δm²₃₁ = Normal(2.541e-3, 0.060e-3)
if !isdir("test_output")
    mkdir("test_output")
end

mle = Newtrinos.find_mle(likelihood, distprod(;priors...), p)

FileIO.save("mle.jld2", Dict("mle" => mle))
