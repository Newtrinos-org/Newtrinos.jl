# Calibration of the effective true-energy response used in t2k.jl (RESPONSE): for each μ-like sample
# the response of mode A (σA, biasA) and mode B (σB, shiftB) is fitted such that the released
# oscillated mode breakdown divided by the bin-averaged ⟨P_μμ⟩ at the reference point reproduces the
# released unoscillated CC spectrum.
#
#   julia --project=../../../.. calibrate_response.jl
using Newtrinos, Optim, Printf
const T = Newtrinos.t2k

physics = T.default_physics()
base = T.get_assets(physics)
Pref = T.channel_probabilities(physics, base.Et, base.paths, base.layers, T.ref_params())

function unosc_from_osc(k, x)
    σA, biasA, σB, shiftB = x
    s = base.samples[k]
    fE = T.spectrum_proxy(base.Et, s.edges, max.(s.unosc .- s.nc, 0.0))
    P = T.SAMPLES[k].anti ? Pref.nubar.mm : Pref.nu.mm
    pA = T.response_weights(base.Et, s.edges, fE, σA; bias = biasA) * P
    pB = T.response_weights(base.Et, s.edges, fE, σB; shift = shiftB) * P
    s.A ./ max.(pA, 0.01) .+ s.B ./ max.(pB, 0.01)
end

function chi2(k, x)
    (x[1] < 0.02 || x[3] < 0.02 || x[4] < -0.1 || abs(x[2]) > 0.2) && return 1e9
    s = base.samples[k]
    u_cc = max.(s.unosc .- s.nc, 0.0)
    sum((unosc_from_osc(k, x) .- u_cc) .^ 2 ./ max.(u_cc, 1.0))
end

for k in (:fhc1rmu, :rhc1rmu, :fhcnumucc1pi)
    x0 = collect(values(T.RESPONSE[k]))
    r = optimize(x -> chi2(k, x), x0, NelderMead(), Optim.Options(iterations = 2000))
    x = Optim.minimizer(r); s = base.samples[k]
    u = unosc_from_osc(k, x); u_cc = max.(s.unosc .- s.nc, 0.0)
    @printf("%-13s σA = %.3f  biasA = %+.3f  σB = %.3f  shiftB = %.3f GeV   χ²-like %.2f (start %.2f)   Σunosc CC: released %.1f, from osc %.1f\n",
            k, x..., Optim.minimum(r), chi2(k, x0), sum(u_cc), sum(u))
end
