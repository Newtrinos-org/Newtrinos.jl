using Distributions
using DensityInterface
using BAT
using DataStructures
using DelimitedFiles
using Newtrinos
using LinearAlgebra
using FileIO
using Accessors
using CairoMakie
using DataFrames
using CSV

experiments = (tao=Newtrinos.tao.configure(),)

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
likelihood = Newtrinos.generate_likelihood(experiments)


if !isdir("test_output")
    mkdir("test_output")
end

mle = Newtrinos.find_mle(likelihood, distprod(;priors...), p)

FileIO.save("test_output/test.jld2", Dict("mle" => mle))
f = experiments.tao.plot(mle[3])
save("test_output/tao_spectrum.png", f)

open("README.md", "w") do io
    write(io, "# TAO \n## Resources\n")
    write(io, "Data from https://arxiv.org/pdf/2610.23331, data read off by claude\n\n")
    write(io, "\n## Test output plots\n")
    write(io, "These plots are generated using 25 pull parameters on the reactor spectrum.")
    write(io, "![Comparison](test_output/tao_spectrum.png)\n")
end