# Validation of the Borexino Phase-III spectral module: best fit and CNO-rate profile compared with
# the published profile (PRD 108, 102005, Fig. 2b, solid line, including systematics).
#   cd src/experiments/borexino/borexino_phase3 && julia --project=../../../.. test.jl
using Newtrinos, Distributions, DensityInterface, Optim, ADTypes, ForwardDiff, CairoMakie, DelimitedFiles, Printf
const B = Newtrinos.borexino_ph3
mkpath(joinpath(@__DIR__, "test_output"))

physics = B.default_physics()
pub = readdlm(joinpath(B.DATADIR, "Phase3Final_BiULLarge_Golden_Energy_Radial_May18_HybridMethod.txt"))

function run(background_priors)
    bx = B.configure(physics; background_priors)
    exps = (borexino_ph3 = bx,)
    llh = Newtrinos.generate_likelihood(exps)
    p0 = Newtrinos.get_params(exps)
    priors = Newtrinos.get_priors(exps)
    # free: ⁷Be and common CNO normalisation (flat), pep (SSM prior), backgrounds and response (module priors);
    # oscillations, pp, ⁸B and hep fixed at their defaults
    nuis = vcat([k for k in keys(bx.params)], [:solar_norm_pep])
    names = vcat([:solar_norm_be7, :cno], nuis)
    function params(x)
        cno = x[2]
        merge(p0, (solar_norm_n13 = cno, solar_norm_o15 = cno, solar_norm_f17 = cno),
              NamedTuple{Tuple(names[[1; 3:end]])}(Tuple(x[[1; 3:end]])))
    end
    logprior(x) = sum(logpdf(priors[k], x[i]) for (i, k) in enumerate(names) if k in nuis)
    nlp(x) = -(logdensityof(llh, params(x)) + logprior(x))
    x0 = vcat(1.0, 1.0, [p0[k] for k in nuis])
    t = @elapsed nlp(x0); tg = @elapsed ForwardDiff.gradient(nlp, x0)
    t = @elapsed nlp(x0); tg = @elapsed ForwardDiff.gradient(nlp, x0)
    @printf "[%s] -log L at defaults: %.1f  (%.3f s, gradient %.3f s)\n" background_priors nlp(x0) t tg
    lower = vcat(0.3, 0.0, [minimum(priors[k]) for k in nuis]); upper = vcat(2.0, 4.0, [maximum(priors[k]) for k in nuis])
    fit(f, x, lower=lower, upper=upper) = (r = optimize(f, lower, upper, clamp.(x, lower .+ 1e-6, upper .- 1e-6), Fminbox(LBFGS()),
                              Optim.Options(iterations=500, outer_iterations=10, g_tol=1e-5); autodiff=ADTypes.AutoForwardDiff());
                 (Optim.minimizer(r), Optim.minimum(r)))
    xb, mb = fit(nlp, x0)
    rates = B.solar_rates(params(xb), bx.physics, bx.assets)
    cno_per_norm = rates.cno / xb[2]
    @printf "[%s] best fit: ⁷Be=%.2f pep=%.3f CNO=%.2f cpd/100t | %s\n" background_priors rates.be7 rates.pep rates.cno join(
        [@sprintf("%s=%.3g", replace(string(k), "borexino_ph3_" => ""), v) for (k, v) in zip(nuis, xb[3:end])], " ")
    grid = [2.0, 3.0, 4.0, 5.0, 5.9, 7.5, 8.6, 10.0, 12.0, 14.0]
    prof = map(grid) do c
        f(y) = nlp(vcat(y[1], c / cno_per_norm, y[2:end]))
        _, m = fit(f, xb[[1; 3:end]], lower[[1; 3:end]], upper[[1; 3:end]])
        2 * (m - mb)
    end
    f = Figure(); ax = Axis(f[1, 1], xlabel="CNO rate (cpd/100 t)", ylabel="−2Δln L", title="Borexino Phase-III CNO ($background_priors)")
    lines!(ax, pub[:, 1], pub[:, 3], color=:black, label="Borexino (with syst.)")
    o = sortperm(vcat(grid, rates.cno))
    scatterlines!(ax, vcat(grid, rates.cno)[o], vcat(prof, 0.0)[o], color=:red, label="Newtrinos")
    xlims!(ax, 0, 15); ylims!(ax, 0, 20); axislegend(ax, position=:ct)
    save(joinpath(@__DIR__, "test_output", "cno_profile_$(background_priors).png"), f)
    save(joinpath(@__DIR__, "test_output", "spectrum_$(background_priors).png"), bx.plot(params(xb)))
    println("[$background_priors] CNO profile: ", join([@sprintf("%.1f:%.2f(pub %.2f)", c, d, pub[argmin(abs.(pub[:, 1] .- c)), 3]) for (c, d) in zip(grid, prof)], "  "))
end

run(:constrained)
run(:free)
