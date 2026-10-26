# Solar and Reactor Physics API

## Solar Flux

The standard solar model is accessed via `Newtrinos.solar_flux` (B23, default MB22-met composition): fluxes and
spectra of the pp, pep, hep, ⁷Be, ⁸B, ¹³N, ¹⁵O and ¹⁷F neutrinos, their production regions in the Sun (input to
`osc.solar_prob`), and the flux normalisations with the correlated B23 prior (`CorrelatedSSMPriors`, default, as one
vector parameter `solar_norms`) or as independent scalars (`SSMPriors`). The ⁸B spectrum is Winter et al. (2006) by
default (`b8_spectrum = :bahcall` for Bahcall et al. 1996).

```@docs
Newtrinos.solar_flux.SolarFluxConfig
Newtrinos.solar_flux.SolarFlux
Newtrinos.solar_flux.B23
Newtrinos.solar_flux.B16
Newtrinos.solar_flux.CorrelatedSSMPriors
Newtrinos.solar_flux.SSMPriors
Newtrinos.solar_flux.configure
Newtrinos.solar_flux.nadir_exposure
Newtrinos.solar_flux.read_spectra
Newtrinos.solar_flux.read_fluxes
Newtrinos.solar_flux.read_production
Newtrinos.solar_flux.COMPONENTS
Newtrinos.solar_flux.ssm_prior
Newtrinos.solar_flux.flux_norm
```

## Solar Cross-Sections

ν–e elastic scattering (with radiative corrections) and the ³⁷Cl and ⁷¹Ga capture cross sections, accessed via
`Newtrinos.solar_xsec`.

```@docs
Newtrinos.solar_xsec.SolarXsec
Newtrinos.solar_xsec.configure
```

## Reactor Flux

Reactor ν̄e fluxes and IBD detection shared by reactor experiments (`Newtrinos.reactor_flux`): Huber–Mueller isotope
spectra, energy per fission, IBD cross section and kinematics, Bugey-4 rate anchoring, and a reactor database
(`src/physics/reactors/`: IAEA PRIS units of Japan and Korea with thermal power, load factors and site coordinates).

```@docs
Newtrinos.reactor_flux.ReactorFlux
Newtrinos.reactor_flux.bugey4_scale
Newtrinos.reactor_flux.neutrino_energy
Newtrinos.reactor_flux.positron_energy
Newtrinos.reactor_flux.load_units
Newtrinos.reactor_flux.daily_power
```

## IBD Cross-Section and Daya Bay Flux

`Newtrinos.ibd_xsec` provides inverse-beta-decay cross sections (Strumia–Vissani table, Vogel–Beacom first order);
`Newtrinos.dayabay_flux` the reactor flux model of the full Daya Bay analysis (Huber–Mueller spectra, non-equilibrium
and spent-fuel corrections, free spectrum correction), used by `Newtrinos.dayabay_rewrite`.

```@docs
Newtrinos.ibd_xsec.StrumiaVissani
Newtrinos.ibd_xsec.VogelBeacomO1
Newtrinos.dayabay_flux.get_flux
Newtrinos.dayabay_flux.get_flux_on_mesh
```

## Published Δχ² maps

`Newtrinos.chi2map` turns a published Δχ² table into an experiment (or a penalty term) that can be added to a joint
likelihood, for data sets whose full likelihood is not available.

```@docs
Newtrinos.chi2map.Chi2Map
Newtrinos.chi2map.Chi2Penalty
```
