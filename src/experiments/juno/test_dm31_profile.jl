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


results = Dict()
for (c, ordering) in enumerate([:NO, :IO])

    osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(
        interaction = Newtrinos.osc.SI(),
        eigen_method = Newtrinos.BargerEigen(),
        flavour=Newtrinos.osc.ThreeFlavour(ordering=ordering)
        ))
    xsec = Newtrinos.ibd_xsec.configure(Newtrinos.ibd_xsec.VogelBeacomO1())
    flux = Newtrinos.dayabay_flux.configure(Newtrinos.dayabay_flux.ReactorFluxConfig(average=true, N_pulls=25))
    physics = (; osc, xsec, flux)
    experiments = (juno=Newtrinos.juno_rewrite.configure(physics), tao=Newtrinos.tao_rewrite.configure(physics))

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
    #@reset priors.Δm²₃₁ = Normal(2.509e-3, 0.027e-3)
    low = 1.8e-3
    high = 3.0e-3
    if ordering == :NO
        @reset priors.Δm²₃₁ = Uniform(low, high)
    else
        @reset priors.Δm²₃₁ = Uniform(-high, -low)
    end
    if !isdir("combined_output")
        mkdir("combined_output")
    end

    vars_to_scan = OrderedDict()
    vars_to_scan[:Δm²₃₁] = 2
    result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="combined_scratch_$(String(ordering))")
    FileIO.save("combined_output/test_dm31_juno_tao_$(String(ordering)).jld2", Dict("result" => result))
    converted = Newtrinos.NewtrinosResult(axes = (delta2m31= result.axes.Δm²₃₁), values=result.values);
    results[ordering] = converted
end
f = Figure()
ax = Axis(f[1, 1], )
"""

ax = Axis(f[1, 1])
ax.title = "Juno test"
plot!(ax, converted, levels=1 .- 2*ccdf(Normal(), 1:3), label="ours; 1,2,3sigma")
juno_1sigma = readdlm("207_days/fig_6_contour.txt", skipstart=25);
scatter!(ax, 0.3036, 7.388e-5, label="Juno MLE")
lines!(ax, juno_1sigma[:, 1], juno_1sigma[:, 2] .* 1e-5)
axislegend(ax)
save("combined_output/dm21_theta12_juno_tao.png", f)



dayabay_NO = Normal(2.5413e-3, 0.060e-3)
dayabay_IO = Normal(−2.4957, 0.060e-3)


f = Figure()
ax = Axis(f[1, 1], xlabel="Δ²₃₁", ylabel="-2 * ΔLLH", xminorticks=  IntervalsBetween(5), title="IO", yminorticksvisible = true, yticks=0:5:30)
ax2 = Axis(f[1, 2], xlabel="Δ²₃₁", xminorticks=  IntervalsBetween(5), title="NO", yminorticksvisible = true, yticks=0:5:30,yaxisposition = :right)
lines!(ax, IO.axes.Δm²₃₁, 2 .* (-IO.values.log_posterior .- base))
lines!(ax2, NO.axes.Δm²₃₁, 2 .* (-NO.values.log_posterior .- base))

for (a, db, res) in zip([ax, ax2], [dayabay_IO, dayabay_NO], [IO, NO])
    x = collect(LinRange(res.axes.Δm²₃₁[1], res.axes.Δm²₃₁[end], 1000))
 
    lines!(a, x, 1 / 2 * (-logdensityof.(db, x) .+ logdensityof(db, mean(db)))) # why is the factor 1/2 necessary?!
end


for a in [ax, ax2]
    ylims!(a, 0., 30)
end

f
save("combined_output/delta_m31.png", f)
"""
