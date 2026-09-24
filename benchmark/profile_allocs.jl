using Newtrinos
using BenchmarkTools
using LinearAlgebra

# Setup (same as bench_osc.jl)
osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour = Newtrinos.osc.ThreeFlavour(ordering=:NO),
    interaction = Newtrinos.osc.SI(),
    propagation = Newtrinos.osc.Basic(),
)
osc = Newtrinos.osc.configure(osc_cfg)

el = Newtrinos.earth_layers.configure()
layers = el.compute_layers()

log10e_bins = LinRange(0, 2, 101)
E = 10 .^ (0.5 .* (log10e_bins[1:end-1] .+ log10e_bins[2:end]))

cz_bins = LinRange(-1, 0, 101)
cz = 0.5 .* (cz_bins[1:end-1] .+ cz_bins[2:end])

paths = el.compute_paths(cz, layers)

# Warmup
osc.osc_prob(E, paths, layers, osc.params)

println("=" ^60)
println("ALLOCATION PROFILE: Matter oscillation (100×100)")
println("=" ^60)

# Step 1: get_matrices
println("\n--- Step 1: get_matrices ---")
matrices_fn = Newtrinos.osc.get_matrices(osc_cfg.flavour)
@btime $matrices_fn($(osc.params))

# Step 2: compute_matter_matrices for a single energy and all layers
U, h_raw = matrices_fn(osc.params)
h = h_raw .- minimum(h_raw)
H_eff = U * Diagonal(h) * adjoint(U)

println("\n--- Step 2: compute_matter_matrices (single layer) ---")
@btime Newtrinos.osc.compute_matter_matrices($H_eff, $(E[50]), $(layers[2]), false, $(Newtrinos.osc.SI()))

println("\n--- Step 2b: compute_matter_matrices (all layers, broadcast) ---")
@btime Newtrinos.osc.compute_matter_matrices.(Ref($H_eff), $(E[50]), $layers, false, Ref($(Newtrinos.osc.SI())))

# Step 3: osc_kernel
matter_matrices = Newtrinos.osc.compute_matter_matrices.(Ref(H_eff), E[50], layers, false, Ref(Newtrinos.osc.SI()))
test_path = paths[50]  # a single path

println("\n--- Step 3: osc_kernel (single section) ---")
section = test_path[1]
@btime Newtrinos.osc.osc_kernel($(matter_matrices[section.layer_idx])..., $(E[50]), $(section.length))

# Step 4: osc_reduce for a single path
println("\n--- Step 4: osc_reduce (single path, Basic) ---")
@btime Newtrinos.osc.osc_reduce($matter_matrices, $test_path, $(E[50]), $(Newtrinos.osc.Basic()))

# Step 5: matter_osc_per_e for single energy
println("\n--- Step 5: matter_osc_per_e (single energy, all paths) ---")
@btime Newtrinos.osc.matter_osc_per_e($H_eff, $(E[50]), $layers, $paths, false, $(Newtrinos.osc.Basic()), $(Newtrinos.osc.SI()))

# Step 6: Full propagate
println("\n--- Step 6: propagate (all energies, all paths) ---")
Uc = U
h_shifted = h
@btime Newtrinos.osc.propagate($Uc, $h_shifted, $E, $paths, $layers, $(Newtrinos.osc.Basic()), $(Newtrinos.osc.SI()), false)

# Step 7: Full osc_prob
println("\n--- Step 7: Full osc_prob ---")
@btime $(osc.osc_prob)($E, $paths, $layers, $(osc.params))

# Breakdown per energy
println("\n" * "=" ^60)
println("PER-ENERGY COST BREAKDOWN")
println("=" ^60)
n_e = length(E)
n_cz = length(cz)

t_matrices = @belapsed $matrices_fn($(osc.params))
t_per_e = @belapsed Newtrinos.osc.matter_osc_per_e($H_eff, $(E[50]), $layers, $paths, false, $(Newtrinos.osc.Basic()), $(Newtrinos.osc.SI()))
t_total = @belapsed $(osc.osc_prob)($E, $paths, $layers, $(osc.params))

println("  get_matrices:        $(round(t_matrices*1e6, digits=1)) μs (once)")
println("  matter_osc_per_e:    $(round(t_per_e*1e6, digits=1)) μs (per energy)")
println("  Expected n_e × per_e: $(round(n_e * t_per_e * 1e3, digits=2)) ms")
println("  Actual total:        $(round(t_total*1e3, digits=2)) ms")
println("  Overhead:            $(round((t_total - n_e * t_per_e - t_matrices)*1e3, digits=2)) ms")

# Allocation breakdown
println("\n" * "=" ^60)
println("ALLOCATION COUNTS")
println("=" ^60)
a_matrices = @ballocated $matrices_fn($(osc.params))
a_per_e = @ballocated Newtrinos.osc.matter_osc_per_e($H_eff, $(E[50]), $layers, $paths, false, $(Newtrinos.osc.Basic()), $(Newtrinos.osc.SI()))
a_total = @ballocated $(osc.osc_prob)($E, $paths, $layers, $(osc.params))
a_cmm = @ballocated Newtrinos.osc.compute_matter_matrices($H_eff, $(E[50]), $(layers[2]), false, $(Newtrinos.osc.SI()))
a_kernel = @ballocated Newtrinos.osc.osc_kernel($(matter_matrices[section.layer_idx])..., $(E[50]), $(section.length))
a_reduce = @ballocated Newtrinos.osc.osc_reduce($matter_matrices, $test_path, $(E[50]), $(Newtrinos.osc.Basic()))

n_layers = length(layers)

println("  osc_kernel (1 call):              $(a_kernel) bytes")
println("  compute_matter_matrices (1 call): $(a_cmm) bytes")
println("  osc_reduce (1 path):              $(a_reduce) bytes")
println("  matter_osc_per_e (1 energy):      $(a_per_e) bytes")
println("  get_matrices:                     $(a_matrices) bytes")
println("  Full osc_prob:                    $(a_total) bytes ($(round(a_total/1024/1024, digits=2)) MiB)")
println("  Per-energy share:                 $(round(a_per_e * n_e / 1024 / 1024, digits=2)) MiB")
