# Validation: the Newtrinos no-oscillation prompt spectrum (Daya Bay flux model, Huber–Mueller, Vogel–Beacom O(1/M)
# IBD with the official two-fold Gauss-Legendre folding, resolution, selection efficiency) vs KamLAND's published
# expectation for the 2002–2009 data (PRD 83, 052002, Fig. 1: data/noosc_2011.csv, with that data set's efficiency
# data/efficiency_2011.csv); shapes compared, rate as a ratio.
#   julia --project=<Newtrinos.jl> validate_noosc.jl
using Newtrinos, Printf, CSV, DataFrames, Accessors, FileIO
const K = Newtrinos.kamland
const D = joinpath(@__DIR__, "data")
noosc = CSV.read(joinpath(D, "noosc_2011.csv"), DataFrame; comment = "#")[1:length(K.EDGES)-1, :]
eff = CSV.read(joinpath(D, "efficiency_2011.csv"), DataFrame; comment = "#")
eff2011(e) = K.lin(clamp(e, eff.E_MeV[1], eff.E_MeV[end]), eff.E_MeV, eff.efficiency)
flux = Newtrinos.dayabay_flux.configure(Newtrinos.dayabay_flux.ReactorFluxConfig(average = true))
# load DayaBays' best fitting spectrum pulls
spec_pulls_mle = FileIO.load(joinpath(@__DIR__, "../../juno/juno_207days/spectrum_pulls_cov.jld2"))["mle"]
@reset flux.params.spectrum_pulls = spec_pulls_mle
xsec = Newtrinos.ibd_xsec.configure(Newtrinos.ibd_xsec.VogelBeacomO1())
E_fine = collect(0.7:0.025:8.45)
E_fine_c = (E_fine[1:end-1] + E_fine[2:end]) / 2
mesh = Newtrinos.Helpers.ibd_integral_mesh(xsec, E_fine)
main, neq, snf, correction = flux.flux_on_mesh(mesh.E_nu_mesh)(1.0, flux.params.energy_per_fission, ones(4), flux.params.spectrum_pulls, ones(4), 1.0, 1)
spec = vec(sum(reshape((main .* correction .+ neq + snf) .* mesh.integral_weights, mesh.NQ, mesh.NFc), dims = 1))
spec = Newtrinos.Helpers.smear_erf(spec, K.RESOLUTION * sqrt.(E_fine_c), E_fine)
o = Newtrinos.Helpers.rebin(spec .* eff2011.(E_fine_c), Newtrinos.Helpers.rebin_index(E_fine, K.EDGES), length(K.EDGES) - 1)
kl = noosc.events
r = (o ./ sum(o)) ./ (kl ./ sum(kl))
@printf "shape χ² (Poisson errors of %.0f events) = %.2f for %d bins\n" sum(kl) sum(abs2, (o ./ sum(o) .- kl ./ sum(kl)) ./ sqrt.(kl ./ sum(kl)^2)) length(kl)
println("ratio ours/KamLAND: ", join([@sprintf("%.3f", x) for x in r], " "))
