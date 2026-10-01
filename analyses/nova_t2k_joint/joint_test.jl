using Distributions
using DensityInterface
using BAT
using DataStructures
using Newtrinos
using FileIO
using Accessors
using CairoMakie
using HDF5
using Printf

# Joint NOvA + T2K fit with the Newtrinos forward models as they are, compared with the official joint
# analysis (NOvA & T2K Collaborations, Nature 646, 818 (2025), arXiv:2510.19888).
# Caveats: the official analysis used the NOvA 2020 data set (13.6e20 ν + 12.5e20 ν̄ POT) and the
# T2K 2023 analysis, here NOvA 2024 (26.6e20 ν) and the T2K 2025 model are used; the official results
# are Bayesian (marginalised, HPD), ours are profile likelihood with Wilks' theorem.
#
#   julia -t 8 --project=../.. joint_test.jl

experiments = (nova = Newtrinos.nova.configure(), t2k = Newtrinos.t2k.configure())
single = (nova = (nova = experiments.nova,), t2k = (t2k = experiments.t2k,))

# common settings as in the official analysis: sin²θ₁₂ = 0.307, Δm²₂₁ = 7.53e-5 eV² fixed;
# reactor constraint sin²θ₁₃ = 0.0218 ± 0.0007
function settings(exps)
    p = Newtrinos.get_params(exps); priors = Newtrinos.get_priors(exps)
    θ₁₃ = asin(sqrt(0.0218))
    p = merge(p, (θ₁₂ = asin(sqrt(0.307)), Δm²₂₁ = 7.53e-5, θ₁₃ = θ₁₃))
    priors = merge(priors, (θ₁₂ = p.θ₁₂, Δm²₂₁ = p.Δm²₂₁, θ₁₃ = Normal(θ₁₃, 0.0007 / (2 * sin(θ₁₃) * cos(θ₁₃))),
                            δCP = Uniform(0, 2π), θ₂₃ = Uniform(asin(sqrt(0.35)), asin(sqrt(0.68)))))
    p, priors
end

const θ₂₃_range = (asin(sqrt(0.35)), asin(sqrt(0.68)))
orderings = (NO = (Δm²₃₂ = (2.25e-3, 2.65e-3), start = (Δm²₃₂ = 2.43e-3, δCP = mod(-0.87π, 2π)), color = :steelblue),
             IO = (Δm²₃₂ = (-2.65e-3, -2.25e-3), start = (Δm²₃₂ = -2.48e-3, δCP = mod(-0.47π, 2π)), color = :darkorange))
setup(o, p, priors) = (@set(priors.Δm²₃₁ = Uniform((o.Δm²₃₂ .+ p.Δm²₂₁)...)),
                       merge(p, (Δm²₃₁ = o.start.Δm²₃₂ + p.Δm²₂₁, δCP = o.start.δCP, θ₂₃ = asin(sqrt(0.56)))))

function profile_octants(likelihood, pr, p0, vars_to_scan, cache_dir)
    oct = map((lower = (θ₂₃_range[1], π / 4), upper = (π / 4, θ₂₃_range[2]))) do r
        Newtrinos.profile(likelihood, @set(pr.θ₂₃ = Uniform(r...)), vars_to_scan, @set(p0.θ₂₃ = sum(r) / 2), cache_dir = cache_dir)
    end
    better = oct.lower.values.log_posterior .>= oct.upper.values.log_posterior
    Newtrinos.NewtrinosResult(axes = oct.lower.axes, values = map((l, u) -> ifelse.(better, l, u), oct.lower.values, oct.upper.values), meta = oct.lower.meta)
end

isdir("test_output") || mkdir("test_output")

# --- fits ---
runs = map((joint = experiments, nova = single.nova, t2k = single.t2k)) do exps
    likelihood = Newtrinos.generate_likelihood(exps)
    p, priors = settings(exps)
    tag = join(keys(exps), "_")
    map(keys(orderings), values(orderings)) do mo, o
        pr, p0 = setup(o, p, priors)
        dcp = profile_octants(likelihood, pr, p0, OrderedDict(:δCP => 41), "test_cache_$(tag)_$(mo)_dcp")
        if length(exps) == 2
            dcpth23 = Newtrinos.profile(likelihood, pr, OrderedDict(:δCP => 25, :θ₂₃ => 21), p0, cache_dir = "test_cache_$(tag)_$(mo)_dcpth23")
            th23dm = Newtrinos.profile(likelihood, pr, OrderedDict(:θ₂₃ => 21, :Δm²₃₁ => 21), p0, cache_dir = "test_cache_$(tag)_$(mo)_th23dm31")
            mo => (; dcp, dcpth23, th23dm)
        else
            mo => (; dcp)
        end
    end |> NamedTuple
end
FileIO.save("test_output/joint.jld2", Dict(string(k) => v for (k, v) in pairs(runs)))

wrap(x) = mod(x + π, 2π) - π
gmax(r) = maximum(x -> maximum(x.dcp.values.log_posterior), r)

# --- summary vs. official (Extended Data Table III, Bayesian HPD, reactor constraint) ---
official = (NO = (dm2 = 2.43, ss23 = 0.561, dcp = -0.87), IO = (dm2 = 2.48, ss23 = 0.563, dcp = -0.47))
summary = IOBuffer()
for mo in keys(orderings)
    r = runs.joint[mo].dcp; i = argmax(r.values.log_posterior)
    Δ = 2 * (gmax(runs.joint) - r.values.log_posterior[i])
    lp = r.values.log_posterior; inside = 2 .* (maximum(lp) .- lp) .< 1
    d = sort(wrap.(r.axes.δCP[inside]) ./ π)
    @printf(summary, "%s: best fit |Δm²₃₂| = %.3f (official %.2f), sin²θ₂₃ = %.3f (%.3f), δCP = %.2fπ (%.2fπ); -2ΔlogL vs global = %.2f; 1σ δCP grid points in [%.2fπ, %.2fπ]\n",
            mo, abs(r.values.Δm²₃₁[i] - 7.53e-5) * 1e3, official[mo].dm2, sin(r.values.θ₂₃[i])^2, official[mo].ss23,
            wrap(r.axes.δCP[i]) / π, official[mo].dcp, Δ, first(d), last(d))
end
print(String(take!(copy(summary))))
println("official: mass-ordering Bayes factor 1.3 in favour of IO (with reactor constraint)")

# --- δCP–sin²θ₂₃ per ordering vs. official Fig. 3 ---
# official regions from arXiv:2510.19888 Fig. 3 (vector), extracted with extract_joint_fig3.py
regions = h5open(joinpath(@__DIR__, "joint_fig3_regions.h5")) do f
    Dict(mo => Dict(l => [(read(f["$mo/$l/$i/dcp"]), read(f["$mo/$l/$i/ssth23"])) for i in keys(f["$mo/$l"])] for l in keys(f[mo])) for mo in keys(f))
end
fig = Figure(size = (1200, 950))
for (i, mo) in enumerate(keys(orderings))
    o = orderings[mo]; r = runs.joint[mo].dcpth23
    ax = Axis(fig[1, i], title = "NOvA+T2K $(mo): 1σ, 2σ, 3σ", xlabel = "δCP", ylabel = "sin²θ₂₃")
    for (lv, a) in (("3sigma", 0.15), ("2sigma", 0.3), ("1sigma", 0.5))
        for poly in get(regions[string(mo)], lv, [])
            poly!(ax, Point2f.(poly...), color = (:gray, a), strokewidth = 0)
        end
    end
    poly!(ax, Point2f[(10, 10), (10.1, 10), (10, 10.1)], color = (:gray, 0.4), label = "official (Bayesian)")   # legend entry
    # wrap δCP to [-π, π) for display
    d = wrap.(r.axes.δCP); k = sortperm(d)
    vals = map(v -> v[k, :], r.values)
    plot!(ax, Newtrinos.NewtrinosResult(axes = (δCP = d[k], sin2theta23 = sin.(r.axes.θ₂₃) .^ 2), values = vals),
          levels = [0.6827, 0.9545, 0.9973], color = o.color, label = "ours (profile)")
    xlims!(ax, -π, π); ylims!(ax, 0.38, 0.65)
    axislegend(ax, position = :lb)
end

ax = Axis(fig[2, 1:2], title = "δCP, profiled over all other parameters (relative to the global best fit of each fit)", xlabel = "δCP / π", ylabel = "-2 Δ log L")
for (name, r, ls) in ((:joint, runs.joint, :solid), (:nova, runs.nova, :dot), (:t2k, runs.t2k, :dash))
    g = gmax(r)
    for mo in keys(orderings)
        x = r[mo].dcp; d = wrap.(x.axes.δCP); k = sortperm(d)
        lines!(ax, d[k] ./ π, 2 .* (g .- x.values.log_posterior[k]), color = orderings[mo].color, linestyle = ls,
               linewidth = name == :joint ? 3 : 1.5, label = "$(name == :joint ? "NOvA+T2K" : name == :nova ? "NOvA" : "T2K") $(mo)")
    end
end
# official joint 1σ HPD intervals (Extended Data Table III)
for (mo, lo, hi) in ((:NO, -1.08, -0.52), (:IO, -0.62, -0.30))
    vspan!(ax, lo, hi, color = (orderings[mo].color, 0.15))
end
xlims!(ax, -1, 1); ylims!(ax, 0, 25)
axislegend(ax, position = :lt, nbanks = 3)
save("test_output/joint_dcp.png", fig)

# --- sin²θ₂₃–|Δm²₃₂| per ordering ---
fig = Figure(size = (1100, 450))
for (i, mo) in enumerate(keys(orderings))
    o = orderings[mo]; r = runs.joint[mo].th23dm
    ax = Axis(fig[1, i], title = "NOvA+T2K $(mo): 1σ, 2σ, 3σ (profile)", xlabel = "sin²θ₂₃", ylabel = "|Δm²₃₂| (10⁻³ eV²)")
    plot!(ax, Newtrinos.NewtrinosResult(axes = (sin2theta23 = sin.(r.axes.θ₂₃) .^ 2, dm2 = abs.(r.axes.Δm²₃₁ .- 7.53e-5) .* 1e3), values = r.values),
          levels = [0.6827, 0.9545, 0.9973], color = o.color, label = "ours")
    scatter!(ax, [official[mo].ss23], [official[mo].dm2], color = :black, marker = :cross, markersize = 14, label = "official HPD")
    axislegend(ax, position = :lb)
end
save("test_output/joint_th23dm2.png", fig)

open("test_output/summary.txt", "w") do io
    write(io, String(take!(summary)))
end
