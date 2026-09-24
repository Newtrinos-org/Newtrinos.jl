using Newtrinos
using BenchmarkTools

# Configure oscillation physics with standard interactions (matter effects)
osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour = Newtrinos.osc.ThreeFlavour(ordering=:NO),
    interaction = Newtrinos.osc.SI(),
    propagation = Newtrinos.osc.Basic(),
)
osc = Newtrinos.osc.configure(osc_cfg)

# Configure Earth layers (PREM model)
el = Newtrinos.earth_layers.configure()
layers = el.compute_layers()

# Energy grid: 100 points log-spaced from 1 to 100 GeV
log10e_bins = LinRange(0, 2, 101)
E = 10 .^ (0.5 .* (log10e_bins[1:end-1] .+ log10e_bins[2:end]))

# Coszen grid: 100 points from -1 to 0 (upgoing neutrinos through Earth)
cz_bins = LinRange(-1, 0, 101)
cz = 0.5 .* (cz_bins[1:end-1] .+ cz_bins[2:end])

# Compute paths through Earth for each coszen
paths = el.compute_paths(cz, layers)

println("Grid: $(length(E)) energies × $(length(cz)) coszen angles")
println("Energy range: $(round(E[1], digits=2)) - $(round(E[end], digits=1)) GeV")
println("Coszen range: $(round(cz[1], digits=3)) - $(round(cz[end], digits=3))")
println()

# Warmup
osc.osc_prob(E, paths, layers, osc.params)
osc.osc_prob(E, paths, layers, osc.params, anti=true)

# Benchmark neutrino oscillation probabilities
println("=== Neutrino oscillation probabilities (matter, 100×100) ===")
b_nu = @benchmark $(osc.osc_prob)($E, $paths, $layers, $(osc.params))
display(b_nu)
println()

# Benchmark antineutrino
println("=== Antineutrino oscillation probabilities (matter, 100×100) ===")
b_anti = @benchmark $(osc.osc_prob)($E, $paths, $layers, $(osc.params), anti=true)
display(b_anti)
println()

# Benchmark vacuum oscillation for comparison
osc_vac_cfg = Newtrinos.osc.OscillationConfig(
    flavour = Newtrinos.osc.ThreeFlavour(ordering=:NO),
    interaction = Newtrinos.osc.Vacuum(),
    propagation = Newtrinos.osc.Basic(),
)
osc_vac = Newtrinos.osc.configure(osc_vac_cfg)

# For vacuum, we need baselines instead of paths
L = [sum(segment.length for segment in path) for path in paths]

# Warmup
osc_vac.osc_prob(E, L, osc_vac.params)

println("=== Vacuum oscillation probabilities (100×100) ===")
b_vac = @benchmark $(osc_vac.osc_prob)($E, $L, $(osc_vac.params))
display(b_vac)
println()

# Summary
println("=== Summary ===")
println("  Vacuum:       $(round(median(b_vac).time / 1e6, digits=2)) ms (median)")
println("  Matter (ν):   $(round(median(b_nu).time / 1e6, digits=2)) ms (median)")
println("  Matter (ν̄):   $(round(median(b_anti).time / 1e6, digits=2)) ms (median)")
println("  Ratio matter/vacuum: $(round(median(b_nu).time / median(b_vac).time, digits=1))×")
