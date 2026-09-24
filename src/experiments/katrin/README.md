source: KATRIN, “First direct neutrino-mass measurement with sub-eV sensitivity,” 05, 2021.
arxiv.org/abs/2105.08533

Comparison between upper limit on effective electron neutrino mass measured by KATRIN and expected mass from different models.

## Mass functions

### `get_neutrinomass(cfg::Newtrinos.osc.ThreeFlavour)`

Returns a function that computes the effective mass-squared prediction for a
three-flavour model.

The implementation fixes the lightest mass parameter to `m₀ = 0.1` before
calling `Newtrinos.osc.get_abs_masses`.

### `get_neutrinomass(cfg::Newtrinos.osc.NND)`

Returns a function for an NND model. It obtains the PMNS matrix and the mass
eigenvalues and mixing matrices from `Newtrinos.osc.get_matrices(cfg)`, removes
modes with mass-squared above `1` eV, renormalizes the retained mixing-matrix
columns, and sums the retained contributions:

### `get_neutrinomass_all_modes(cfg=NND)`

Returns a related NND calculation retaining modes with mass-squared up to
`(18.6 × 10^3)^2`. Unlike `get_neutrinomass`, it does not apply the
`1 - 1/N_i^2` denominator.

### `mixing_angles(params, cfg=NND)`

Returns six arrays:
```julia
mass_e, mass_m, mass_t, angles_e, angles_m, angles_t
```

The three `mass_*` arrays contain the flavour-separated eigenvalues. The
`angles_*` arrays contain the absolute PMNS electron-row component multiplied
by the corresponding first-row mixing-matrix component.
