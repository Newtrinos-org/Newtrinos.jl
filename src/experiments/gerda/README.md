
source: GERDA Collaboration, “Final results of GERDA on the search for neutrinoless double-beta decay,” Physical
Review Letters 125 no. 25, (12, 2020) 252502.
http://dx.doi.org/10.1103/PhysRevLett.125.252502
 
Comparison of the effective Majorana-neutrino mass upper limit for GERDA experiment
and predicted from different models via neutrinoless double-beta-decay half-life limit.


## Mass functions

### `get_neutrinomass(cfg::Newtrinos.osc.ThreeFlavour)`

Returns a function for a three-flavour effective-mass quantity:

```math
m_{\beta\beta} = \sum_{i=1}^{3} \left|U_{ei}^2 m_i\right|.
```

The implementation sets `m₀ = 0.1` by merging it into `params` before calling
`Newtrinos.osc.get_abs_masses`.

### `get_neutrinomass(cfg::Newtrinos.osc.NNM)`

Returns an NNM effective-mass function. It obtains mass eigenvalues and mixing
matrices from `Newtrinos.osc.get_matrices(cfg)`, separates the eigenvalues into
three flavour towers, and sums contributions of the form:

```math
\left|X_{ij} U_{ei}\right|^2 \sqrt{m_{ij}}\, f(m_{ij}).
```

The mass-dependent factor `f` is a correction for heavy modes:

| Mass range | Factor |
| --- | --- |
| `m < 1e8` | `1` |
| `1e8 ≤ m ≤ 1e12` | `0.7 × factor / (factor + m)` |
| `m > 1e12` | `factor / (factor + m)` |

where `factor = m_e m_p (194 / 5.27)`, with `m_e = 0.511e6` and
`m_p = 938.27e6`.

### `get_neutrinomassSTD(cfg=NNM(...))`

Returns a related NNM effective-mass calculation without the mass-dependent
factor and without cutoff for comparison. It sums:


### `mixing_angles(params, cfg=NNM)`

Returns six arrays:

```julia
mass_e, mass_m, mass_t, angles_e, angles_m, angles_t
```

The `mass_*` arrays are the three flavour-separated eigenvalue sequences. The
`angles_*` arrays are the absolute values of the first row of each corresponding
mixing matrix.

## Half-life calculation

`get_halftime(cfg=Newtrinos.osc.NNM())` returns a function that converts the
effective mass into a half-life:

```math
T_{1/2}^{-1} =
\frac{G_g\, g_A^4\, M^2\, m_{\beta\beta}^2}{m_e^2},
\qquad
T_{1/2} = \frac{1}{T_{1/2}^{-1}}.
```

The constants used by the implementation are `Gg = 3.37e-15`, `g_a = 1.27`,
`M = 5.27`, and `m_e = 0.511e6`.
