# Calibration of the effective νμ energy response used in nova.jl (CALIBRATIONS): per quartile and beam
# mode, fit a fractional resolution σ (and reco-energy bias b) such that
#   NoOscillations_Signal × <P_μμ(θ_ref)>  ≈  Oscillated_Signal
# for the released predictions, at a given reference oscillation point θ_ref.
#
#   julia --project src/experiments/nova/nova_2024/calibrate_numu_resolution.jl
using Newtrinos, HDF5, StaticArrays, Optim, Printf, StructArrays, ArraysOfArrays, LinearAlgebra
using SpecialFunctions: erf
const O = Newtrinos.osc

f = h5open(Newtrinos.nova.datafile)
hist(g) = (edges = read(g["edges"]), values = read(g["values"]))

osc = O.configure(O.OscillationConfig(interaction = O.SI(), eigen_method = Newtrinos.BargerEigen()))
layers = StructArray{Newtrinos.Layer}((radius = [6371.0], p_density = [2.84 * 0.5], n_density = [2.84 * 0.5]))
paths = VectorOfVectors{Newtrinos.Path}([[Newtrinos.Path(810.0, 1)]])

Et = exp10.(range(log10(0.1), log10(20.0), 800))          # true-energy grid
ref(; ss23 = 0.55, dm32 = 2.441e-3) = (θ₁₂ = asin(sqrt(0.851)) / 2, θ₁₃ = asin(sqrt(0.0851)) / 2, θ₂₃ = asin(sqrt(ss23)),
                                      δCP = 0.87π, Δm²₂₁ = 7.53e-5, Δm²₃₁ = dm32 + 7.53e-5)
Pmm(p, anti) = osc.osc_prob(Et, paths, layers, p; anti)[:, 1, 2, 2]

Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
# smearing weights W[b, i] ∝ f(E_i) · P(E_reco ∈ bin b | E_i), true-spectrum proxy f = unoscillated reco density
function weights(edges, noosc, σ)
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    dens = noosc ./ diff(edges)
    fE = [Ei < edges[1] || Ei >= edges[end] ? 0.0 : dens[searchsortedlast(edges, Ei)] for Ei in Et] .+ 1e-6 * maximum(dens)
    dE = vcat(diff(Et), diff(Et)[end])
    W = [fE[i] * dE[i] * (Φ((edges[b+1] - Et[i]) / (σ * Et[i])) - Φ((edges[b] - Et[i]) / (σ * Et[i]))) for b in 1:length(noosc), i in eachindex(Et)]
    W ./ max.(sum(W, dims = 2), 1e-300)
end

samples = [(mode, q) for mode in ("fhc", "rhc") for q in 1:4]
data = Dict(s => (noosc = hist(f["predictions/prediction_components_numu_$(s[1])_Quartile$(s[2])/NoOscillations_Signal"]),
                  osc   = hist(f["predictions/prediction_components_numu_$(s[1])_Quartile$(s[2])/Oscillated_Signal"])) for s in samples)

function sample_chi2(s, σ, P)
    d = data[s]; W = weights(d.noosc.edges, d.noosc.values, σ)
    pred = d.noosc.values .* (W * P)
    sum((pred .- d.osc.values) .^ 2 ./ max.(d.osc.values, 0.5))   # Pearson-like, events
end


# response with fractional resolution σ and fractional bias b: E_reco ~ N((1+b) E_true, σ E_true)
function weights_b(edges, noosc, σ, b)
    dens = noosc ./ diff(edges)
    fE = [Ei < edges[1] || Ei >= edges[end] ? 0.0 : dens[searchsortedlast(edges, Ei)] for Ei in Et] .+ 1e-6 * maximum(dens)
    dE = vcat(diff(Et), diff(Et)[end])
    W = [fE[i] * dE[i] * (Φ((edges[k+1] - (1 + b) * Et[i]) / (σ * Et[i])) - Φ((edges[k] - (1 + b) * Et[i]) / (σ * Et[i]))) for k in 1:length(noosc), i in eachindex(Et)]
    W ./ max.(sum(W, dims = 2), 1e-300)
end
function chi2_b(s, x, P)
    d = data[s]; pred = d.noosc.values .* (weights_b(d.noosc.edges, d.noosc.values, x[1], x[2]) * P)
    sum((pred .- d.osc.values) .^ 2 ./ max.(d.osc.values, 0.5))
end
for (label, pt) in (("2024 reference (0.55, 2.441e-3, 0.87π)", ref(; ss23 = 0.55, dm32 = 2.441e-3)),
                    ("refitted effective (0.5681, 2.4073e-3)", ref(; ss23 = 0.5681, dm32 = 2.4073e-3)))
    P = Dict(false => Pmm(pt, false), true => Pmm(pt, true)); tot = 0.0
    println("== σ + bias per quartile at ", label)
    for s in samples
        r = optimize(x -> chi2_b(s, x, P[s[1] == "rhc"]), [0.08, 0.0], NelderMead()); x = Optim.minimizer(r); tot += Optim.minimum(r)
        d = data[s]; pred = d.noosc.values .* (weights_b(d.noosc.edges, d.noosc.values, x...) * P[s[1] == "rhc"])
        @printf("  %s Q%d: σ = %.3f  bias = %+.4f  χ²-like = %.3f  (Σosc %.2f, Σpred %.2f)\n", s[1], s[2], x..., Optim.minimum(r), sum(d.osc.values), sum(pred))
    end
    @printf("  total χ²-like %.3f\n", tot)
end
