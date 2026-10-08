module sk_solar_dn

using Distributions
using DelimitedFiles
using CairoMakie
using Interpolations
import ..Newtrinos
import ..Newtrinos.solar_common
import ..Newtrinos.sk_solar_common
import ..Newtrinos.sk4_solar

"""
Super-Kamiokande solar day/night asymmetry from the amplitude fit of the solar zenith-angle
variation, K. Abe et al., Phys. Rev. D 109, 092001 (2024), arXiv:2312.12907.

SK fits the amplitude of the expected zenith-angle variation of the ⁸B rate, which reduces the
systematic uncertainties of a direct day/night comparison. The result is quoted as a day/night
asymmetry ``A_{DN} = (D − N) / ((D + N)/2)`` that depends on the Δm²₂₁ assumed for the shape of
the variation (Figs. "SK-IV day/night amplitude fit dependence on Δm²₂₁" and the SK-I–IV
combined version, extracted from the vector graphics):

- `dataset = :sk1to4` (default): SK-I–IV combined, ``A_{DN} = −2.86 ± 0.85 (stat) ± 0.32 (syst) %``
  at Δm²₂₁ = 6.1×10⁻⁵ eV²,
- `dataset = :sk4`: SK-IV, ``A_{DN} = −2.62 ± 1.07 (stat) ± 0.30 (syst) %``.

The extracted curves are smoothed with a running mean over ±0.5×10⁻⁵ eV² (removing the
digitisation noise of the figure) and interpolated with cubic splines, so that the likelihood
is smooth in Δm²₂₁. The likelihood is Gaussian, ``A_{DN}^{fit}(Δm²₂₁) − A_{DN}^{exp}(θ) ∼ N(0, σ(Δm²₂₁))``, with the
statistical band of the figure and the quoted systematic uncertainty added in quadrature. The
expected asymmetry is computed with Newtrinos (⁸B + hep elastic scattering with the SK-IV
response, Earth regeneration at Kamioka) from the day and night rates with recoil kinetic energy
above `E_min` (default 4.49 MeV, the SK-I energy range in which SK expresses the asymmetry), so
the term applies to any oscillation model. With the default solar physics (continental crust
instead of PREM's ocean layer) this reproduces SK's expected asymmetry curves within 1–2 %.

Use together with the SK spectra configured with `daynight = :combined` (e.g.
`Newtrinos.sk4_solar.configure(physics; daynight = :combined)`) to avoid counting the day/night
information twice.
"""
const DATASETS = (
    sk1to4 = (file = "sk1to4_adn_vs_dm2.csv", syst = 0.32, title = "SK-I–IV day/night amplitude fit"),
    sk4 = (file = "sk4_adn_vs_dm2.csv", syst = 0.30, title = "SK-IV day/night amplitude fit"),
)

@kwdef struct SKSolarDN <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

default_physics() = solar_common.default_physics()

function configure(physics=default_physics(); dataset::Symbol=:sk1to4, E_min=4.49)
    haskey(DATASETS, dataset) || throw(ArgumentError("dataset must be one of $(keys(DATASETS))"))
    physics = (; physics.osc, physics.solar_flux, physics.solar_xsec, physics.earth_layers)
    assets = get_assets(physics, dataset, E_min)
    SKSolarDN(
        physics = physics,
        params = (;),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets),
    )
end

function get_assets(physics, dataset, E_min)
    ds = DATASETS[dataset]
    @info "Loading $(ds.title)"
    d = readdlm(joinpath(@__DIR__, ds.file), ',', Float64; skipstart=4)
    dm2 = d[:, 1] .* 1e-5
    @assert all(diff(d[:, 1]) .≈ 0.1) "expected a uniform 0.1×10⁻⁵ eV² grid"
    grid = range(dm2[1], dm2[end], length=length(dm2))
    smooth(y; w=5) = [sum(y[max(1, i - w):min(end, i + w)]) / length(max(1, i - w):min(length(y), i + w)) for i in eachindex(y)]
    spline(y) = cubic_spline_interpolation(grid, smooth(y), extrapolation_bc=Flat())
    # SK-IV day/night samples and response; only the samples above E_min enter the asymmetry
    sk = sk_solar_common.get_assets(sk4_solar.PHASE, physics)
    s = sk.samples
    (
        observed = [0.0],
        dm2 = dm2, adn_fit = d[:, 2], sigma_stat = (d[:, 4] .- d[:, 3]) ./ 2, syst = ds.syst, adn_sk_expected = d[:, 5],
        fit_spline = spline(d[:, 2]), sigma_spline = spline((d[:, 4] .- d[:, 3]) ./ 2),
        title = ds.title, E_min = E_min, sk = sk,
        day = (s.zenith .== "day") .& (s.E_lo .>= E_min - 1e-6),
        night = (s.zenith .== "night") .& (s.E_lo .>= E_min - 1e-6),
    )
end

# linear interpolation in Δm²₂₁, constant outside the tabulated range (AD-safe in Δm²₂₁)
function interp(x, xs, ys)
    x <= xs[1] && return ys[1] + zero(x)
    x >= xs[end] && return ys[end] + zero(x)
    i = searchsortedlast(xs, x)
    t = (x - xs[i]) / (xs[i+1] - xs[i])
    ys[i] + t * (ys[i+1] - ys[i])
end

"""
    expected_adn(params, physics, assets) -> A_DN [%]

Expected day/night asymmetry of the SK-IV ⁸B + hep rate above `E_min`.
"""
function expected_adn(params, physics, assets)
    p = merge(params, (sk4_solar_escale = 0.0, sk4_solar_ereso = 0.0, sk4_solar_norm = 0.0))
    e = sk_solar_common.get_expected(sk4_solar.PHASE, p, physics, assets.sk)
    D, N = sum(e[assets.day]), sum(e[assets.night])
    100 * (D - N) / ((D + N) / 2)
end

function get_forward_model(physics, assets)
    function forward_model(params)
        dm2 = params.Δm²₂₁
        fit = assets.fit_spline(dm2)
        σ = sqrt(assets.sigma_spline(dm2)^2 + assets.syst^2)
        MvNormal([fit - expected_adn(params, physics, assets)], [σ^2;;])
    end
end

function get_plot(physics, assets)
    function plot(params, data=assets.observed)
        x = assets.dm2 .* 1e5
        f = Figure()
        ax = Axis(f[1, 1], title=assets.title, xlabel="Δm²₂₁ (10⁻⁵ eV²)", ylabel="Day/night asymmetry (%)")
        band!(ax, x, assets.adn_fit .- assets.sigma_stat, assets.adn_fit .+ assets.sigma_stat, color=(:gray, 0.4), label="SK amplitude fit (stat)")
        lines!(ax, x, assets.adn_fit, color=:black)
        lines!(ax, x, assets.adn_sk_expected, color=:red, label="SK expected")
        dms = range(2e-5, 22e-5, length=41)
        lines!(ax, dms .* 1e5, [solar_common.ForwardDiffValue(expected_adn(merge(params, (Δm²₂₁ = d,)), physics, assets)) for d in dms],
               color=:dodgerblue, linestyle=:dash, label="Newtrinos expected")
        hlines!(ax, [0.0], color=:gray)
        ylims!(ax, -5, 1)
        axislegend(ax, position=:rb, labelsize=10)
        f
    end
end

end
