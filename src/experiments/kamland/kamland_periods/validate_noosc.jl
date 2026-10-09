# Validation: the Newtrinos no-oscillation prompt spectrum (Huber–Mueller × IBD, resolution, selection efficiency) vs
# KamLAND's published expectation for the 2002–2009 data (PRD 83, 052002, Fig. 1: data/noosc_2011.csv, with that data
# set's efficiency data/efficiency_2011.csv and fission fractions); shapes compared, rate as a ratio.
#   julia --project=<Newtrinos.jl> validate_noosc.jl
using Newtrinos, Printf, CSV, DataFrames
const K = Newtrinos.kamland
const D = joinpath(@__DIR__, "data")
rf = Newtrinos.reactor_flux.configure()
noosc = CSV.read(joinpath(D, "noosc_2011.csv"), DataFrame; comment = "#")[1:length(K.EDGES)-1, :]
eff = CSV.read(joinpath(D, "efficiency_2011.csv"), DataFrame; comment = "#")
eff2011(e) = K.lin(clamp(e, eff.E_MeV[1], eff.E_MeV[end]), eff.E_MeV, eff.efficiency)
fr2011 = (U235 = 0.571, U238 = 0.078, Pu239 = 0.295, Pu241 = 0.056)
Et = collect(1.0:0.01:9.0)
o = K.response(Et, eff2011) * [rf.spectrum(e + K.ΔE_NU, fr2011) * rf.ibd_xsec(e + K.ΔE_NU) for e in Et]
kl = noosc.events
r = (o ./ sum(o)) ./ (kl ./ sum(kl))
@printf "shape χ² (Poisson errors of %.0f events) = %.2f for %d bins\n" sum(kl) sum(abs2, (o ./ sum(o) .- kl ./ sum(kl)) ./ sqrt.(kl ./ sum(kl)^2)) length(kl)
println("ratio ours/KamLAND: ", join([@sprintf("%.3f", x) for x in r], " "))
