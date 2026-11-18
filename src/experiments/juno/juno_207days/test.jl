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

experiments = (juno=Newtrinos.juno.configure(), tao=Newtrinos.tao.configure())

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
likelihood = Newtrinos.generate_likelihood(experiments)



### Run parameter scan on Δm²₂₁ and sin(θ₁₂)
val = 0.0851
err = 0.0024
theta13 = asin(sqrt(val)) / 2
theta13_sigma = 1 / (4 * sqrt(-(val - 1) * val)) * err
spec_pulls_cov = FileIO.load("spectrum_pulls.jld2")["cov"]
spec_pulls_mle = FileIO.load("spectrum_pulls.jld2")["mle"]
temp = UpperTriangular(spec_pulls_cov)
temp += transpose(temp) - Diagonal(diag(temp))
@reset priors.spectrum_pulls = MvNormal(spec_pulls_mle, temp)
@reset p.spectrum_pulls = spec_pulls_mle

@reset priors.θ₂₃ = p.θ₂₃
@reset priors.δCP = p.δCP 
@reset priors.Δm²₂₁ = Uniform(7.15e-5, 7.8e-5)
@reset priors.θ₁₂ = Uniform(asin(sqrt(0.28)), asin(sqrt(0.33)))
@reset priors.θ₁₃ = Normal(theta13, theta13_sigma)
@reset priors.Δm²₃₁ = Normal(2.5413e-3, 0.060e-3)
@reset p.Δm²₃₁ = 2.5413e-3
@reset p.θ₁₃ = theta13
@reset p.Δm²₂₁ = 7.388e-5
@reset p.θ₁₂ = asin(sqrt(0.3036))

if !isdir("test_output")
    mkdir("test_output")
end

mle = Newtrinos.find_mle(likelihood, distprod(;priors...), p)
FileIO.save("test_output/test_mle.jld2", Dict("mle"=>mle))

f, ax, ax2 = experiments.juno.plot(mle[3])
save("test_output/spectrum_juno.png", f)
f = experiments.tao.plot(mle[3])
save("test_output/spectrum_tao.png", f)
println(mle[3])

#mle = FileIO.load("test_output/test_dm21_theta12_mle.jld2")["mle"]

vars_to_scan = OrderedDict()
vars_to_scan[:θ₁₂] = 10
vars_to_scan[:Δm²₂₁] = 10

result = Newtrinos.profile(likelihood, priors, vars_to_scan, mle[3], cache_dir="combined_scratch")
FileIO.save("test_output/test_dm21_theta12.jld2", Dict("result" => result))
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
save("test_output/test_dm21_theta12.png", f)

### Run parameter scan on Δm²₃₁ for inverted and normal mass ordering
results = Dict()
order = [:IO, :NO]
for (c, ordering) in enumerate(order)

    osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(
        interaction = Newtrinos.osc.SI(),
        eigen_method = Newtrinos.BargerEigen(),
        flavour=Newtrinos.osc.ThreeFlavour(ordering=ordering)
        ))
    xsec = Newtrinos.ibd_xsec.configure(Newtrinos.ibd_xsec.VogelBeacomO1())
    flux = Newtrinos.dayabay_flux.configure(Newtrinos.dayabay_flux.ReactorFluxConfig(average=true, N_pulls=25))
    physics = (; osc, xsec, flux)
    experiments = (juno=Newtrinos.juno.configure(physics), tao=Newtrinos.tao.configure(physics))

    p = Newtrinos.get_params(experiments)
    priors = Newtrinos.get_priors(experiments)
    likelihood = Newtrinos.generate_likelihood(experiments)

    val = 0.0851
    err = 0.0024
    theta13 = asin(sqrt(val)) / 2
    theta13_sigma = 1 / (4 * sqrt(-(val - 1) * val)) * err
    spec_pulls_cov = FileIO.load("spectrum_pulls_cov_25.jld2")["cov"]
    spec_pulls_mle = FileIO.load("spectrum_pulls_cov_25.jld2")["mle"]
    local temp = UpperTriangular(spec_pulls_cov)
    temp += transpose(temp) - Diagonal(diag(temp))
    @reset priors.spectrum_pulls = MvNormal(spec_pulls_mle, temp)
    @reset p.spectrum_pulls = spec_pulls_mle

    @reset priors.θ₂₃ = p.θ₂₃
    @reset priors.δCP = p.δCP 
    @reset priors.Δm²₂₁ = Uniform(7.1e-5, 7.9e-5)
    @reset priors.θ₁₂ = Uniform(asin(sqrt(0.275)), asin(sqrt(0.32)))
    @reset priors.θ₁₃ = Normal(theta13, theta13_sigma)
    @reset p.Δm²₃₁ = 2.5413e-3
    @reset p.θ₁₃ = theta13
    @reset p.Δm²₂₁ = 7.388e-5
    @reset p.θ₁₂ = asin(sqrt(0.3036))
    low = 1.8e-3
    high = 3.0e-3
    if ordering == :NO
        @reset priors.Δm²₃₁ = Uniform(low, high)
    else
        @reset priors.Δm²₃₁ = Uniform(-high, -low)
    end
    if !isdir("test_profile")
        mkdir("test_profile")
    end

    vars_to_scan = OrderedDict()
    vars_to_scan[:Δm²₃₁] = 120
    result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="combined_scratch_$(String(ordering))")
    FileIO.save("test_profile/test_dm31_juno_tao_$(String(ordering)).jld2", Dict("result" => result))
    converted = Newtrinos.NewtrinosResult(axes = (delta2m31= result.axes.Δm²₃₁,), values=result.values);
    results[ordering] = converted
end

f = Figure()
ax = Axis(f[1, 1], )


IO_off = readdlm("207_days/fig_4_chi2.txt", skipstart=25)
IO_off = convert.(Float64, IO_off[IO_off[:, 1] .== "IO", 2:3])


NO_off = readdlm("207_days/fig_4_chi2.txt", skipstart=25)
NO_off = convert.(Float64, NO_off[NO_off[:, 1] .== "NO", 2:3]);


dayabay_NO = Normal(2.5413e-3, 0.060e-3)
dayabay_IO = Normal(-2.4957e-3, 0.060e-3)
base = minimum([-maximum(results[:IO].values.log_posterior), -maximum(results[:NO].values.log_posterior)])

f = Figure()
ax_IO = Axis(f[1, 1], xlabel="Δm²₃₁ [eV²]", ylabel="-2 * ΔLLH", xminorticks=  IntervalsBetween(5), title="IO", yminorticksvisible = true, yticks=0:5:30)
ax_NO = Axis(f[1, 2], xlabel="Δm²₃₁ [eV²]", xminorticks=  IntervalsBetween(5), title="NO", yminorticksvisible = true, yticks=0:5:30,yaxisposition = :right)

axes = [ax_IO, ax_NO]
db_constraint = [dayabay_IO, dayabay_NO]
for (c, ordering) in enumerate(order)
    lines!(axes[c], results[ordering].axes.delta2m31, 2 .* (-results[ordering].values.log_posterior .- base), label="Newtrinos.jl")
end
for (a, db, res) in zip(axes, db_constraint, [results[:IO], results[:NO]])
    x = collect(LinRange(res.axes.delta2m31[1], res.axes.delta2m31[end], 1000))

    lines!(a, x, 2 * (-logdensityof.(db, x) .+ logdensityof(db, mean(db))), label="DayaBay constraint")
end

juno_off = [IO_off, NO_off]
for (a, off) in zip(axes, juno_off)
    ylims!(a, 0., 30)
    lines!(a, off[:, 1] .* 1e-3, off[:, 2], label="Juno Δχ²")
end
f[0, 1:2] = Legend(f, [ax_IO, ax_NO], orientation=:horizontal, merge=true)
f
save("test_output/test_dm31.png", f)

open("README.md", "w") do io
    write(io, "# JUNO\n ## Resources\n")
    write(io, "Data source: https://arxiv.org/abs/2609.37895, https://arxiv.org/abs/2610.23331\n")
    write(io, "Data extraction performed by claude")
    write(io, "\n## Test output plots\n")
    write(io, "![Comparison](test_output/spectrum_juno.png)\n")
    write(io, "![Comparison](test_output/spectrum_tao.png)\n")
    write(io, "![Comparison](test_output/test_dm21_theta12.png)\n")
    write(io, "![Comparison](test_output/test_dm31.png)\n")
    write(io, "## Meta Information\n")
    for (key, value) in result.meta
        write(io, "- **$(key)**: $(value)\n")
    end
end
