# IceCube Upgrade 3y MC Sample
 ## Resources
data source: https://icecube.wisc.edu/data-releases/2020/04/icecube-upgrade-neutrino-monte-carlo-simulation/
## Test output plots
![Comparison](test_output/contours.png)
![DataMC](test_output/spectrum.png)
## Meta Information
- **repo_clean**: false
- **exec_time**: 1733.0327620506287
- **username**: peller
- **repo**: /mnt/c/Users/peller/work/Newtrinos
- **cache_dir**: test
- **hostname**: flippy
- **params**: (atm_flux_delta_spectral_index = 0.0, atm_flux_nuenumu_sigma = 0.0, atm_flux_nunubar_sigma = 0.0, atm_flux_uphorizontal_sigma = 0.0, ic_upgrade_energy_scale = 1.0, ic_upgrade_lifetime = 3.0, nc_norm = 1.0, nutau_cc_norm = 1.0, Δm²₂₁ = 7.53e-5, Δm²₃₁ = 0.0023853, δCP = 1.0, θ₁₂ = 0.5872523687443223, θ₁₃ = 0.1454258194533693, θ₂₃ = 0.7953988301841435)
- **date**: 2025-10-23 09:51:45
- **task**: profile
- **vars_to_scan**: OrderedDict{Any, Any}(:θ₂₃ => 11, :Δm²₃₁ => 11)
- **commit_hash**: 0bdd199ef69cd33e341cb6a7744e6a8f28bffcc0
- **priors**: (atm_flux_delta_spectral_index = Truncated(Normal{Float64}(μ=0.0, σ=0.1); lower=-0.3, upper=0.3), atm_flux_nuenumu_sigma = Truncated(Normal{Float64}(μ=0.0, σ=1.0); lower=-3.0, upper=3.0), atm_flux_nunubar_sigma = Truncated(Normal{Float64}(μ=0.0, σ=1.0); lower=-3.0, upper=3.0), atm_flux_uphorizontal_sigma = Truncated(Normal{Float64}(μ=0.0, σ=1.0); lower=-3.0, upper=3.0), ic_upgrade_energy_scale = Truncated(Normal{Float64}(μ=1.0, σ=0.02); lower=0.5, upper=1.5), ic_upgrade_lifetime = Uniform{Float64}(a=2.0, b=4.0), nc_norm = Truncated(Normal{Float64}(μ=1.0, σ=0.2); lower=0.4, upper=1.6), nutau_cc_norm = 1.0, Δm²₂₁ = 7.53e-5, Δm²₃₁ = Uniform{Float64}(a=0.0023, b=0.00255), δCP = 1.0, θ₁₂ = 0.5872523687443223, θ₁₃ = Truncated(Normal{Float64}(μ=0.156, σ=0.008); lower=0.13, upper=0.18), θ₂₃ = Uniform{Float64}(a=0.685, b=0.9))
