module minos

using LinearAlgebra
using Distributions
using HDF5
using BAT
using DataStructures
using CairoMakie
using Logging
import ..Newtrinos


@kwdef struct Minos <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function default_physics()
    osc = Newtrinos.osc.configure()
    xsec = Newtrinos.xsec.configure()
    (; osc, xsec)
end

"""
    configure(physics=default_physics(); detectors=:FD) -> Minos

Configure the MINOS/MINOS+ two-detector νμ CC and NC analysis (data release of arXiv:1710.06488).

# Keywords
- `detectors::Symbol = :FD`: likelihood construction.
  - `:FD`: far-detector spectra conditioned on the near-detector data (Gaussian conditioning
    with the release covariance). Appropriate for three-flavour fits, where the near-detector
    prediction does not depend on the oscillation parameters.
  - `:FD_ND`: joint far+near spectrum with the release covariance ``V = μμ^T ∘ V_{rel} + \\mathrm{diag}(μ)``,
    as in the χ² of the data release (`dataRelease_chi2Calc_compile.C`). Required for
    sterile-neutrino fits, where the near detector oscillates. The covariance depends on the
    prediction in the quadratic form, while the normalisation (log-determinant) is fixed at the
    nominal prediction (see [`FixedNormGaussian`](@ref)); otherwise the log-determinant
    rewards parameter points with lower predicted rates.
"""
function configure(physics=default_physics(); detectors::Symbol = :FD)
    detectors in (:FD, :FD_ND) || throw(ArgumentError("detectors must be :FD or :FD_ND, got :$(detectors)"))
    physics = (;physics.osc, physics.xsec)
    assets = get_assets(physics; detectors)
    return Minos(
        physics = physics,
        params = (;),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

"""
    FixedNormGaussian(μ, Σ, logdetΣ₀) <: ContinuousMultivariateDistribution

Gaussian likelihood term whose covariance `Σ` may depend on the prediction `μ` in the quadratic
form, while the normalisation uses a fixed log-determinant `logdetΣ₀`:

``\\log L = -\\tfrac12 \\left[(x-μ)^T Σ^{-1} (x-μ) + \\log\\det Σ_0 + n \\log 2π\\right]``

so that ``-2Δ\\log L`` is the usual covariance-matrix χ². With a prediction-dependent `Σ` the
exact Gaussian normalisation ``\\log\\det Σ(μ)`` would favour lower predictions independently of
the data. `mean`, `cov`, `var` and `rand` refer to the Gaussian ``N(μ, Σ)``.
"""
struct FixedNormGaussian{V<:AbstractVector, M<:AbstractMatrix} <: Distributions.ContinuousMultivariateDistribution
    μ::V
    Σ::M
    logdetΣ₀::Float64
end
Base.length(d::FixedNormGaussian) = length(d.μ)
Base.eltype(d::FixedNormGaussian) = eltype(d.μ)
Distributions.mean(d::FixedNormGaussian) = d.μ
Distributions.cov(d::FixedNormGaussian) = d.Σ
Distributions.var(d::FixedNormGaussian) = diag(d.Σ)
function Distributions._logpdf(d::FixedNormGaussian, x::AbstractArray)
    r = x .- d.μ
    -(dot(r, d.Σ \ r) + d.logdetΣ₀ + length(r) * log(2π)) / 2
end
Distributions._rand!(rng::Distributions.AbstractRNG, d::FixedNormGaussian, x::AbstractVector) =
    copyto!(x, rand(rng, MvNormal(d.μ, Symmetric(d.Σ))))

function get_assets(physics; datadir = @__DIR__, detectors::Symbol = :FD)
    @info "Loading minos data"

    h5file = h5open(joinpath(datadir, "dataRelease.h5"), "r")

    channels = ["FDCC", "FDNC", "NDCC", "NDNC"]
    experiments = ["minos", "minosPlus"]

    L=735.0

    ch_data = Dict([
        channel => (
            observed = sum([read(h5file["data$(channel)_$(ex)_hist"]) for ex in experiments]),
            bin_edges = read(h5file["data$(channel)_minosPlus_bins"]),
            smearings = (
                E=(x -> L ./((x[1:end-1] .+ x[2:end]) ./ 2))(read(h5file["hRecoToTrue$(channel)SelectedNuMu_minos_bins1"])),
                energy_edges=read(h5file["hRecoToTrue$(channel)SelectedNuMu_minos_bins2"]),
                NuMu=sum([read(h5file["hRecoToTrue$(channel)SelectedNuMu_$(ex)_hist"]) for ex in experiments]),
                TrueNC=sum([read(h5file["hRecoToTrue$(channel)SelectedTrueNC_$(ex)_hist"]) for ex in experiments]),
                BeamNue=sum([read(h5file["hRecoToTrue$(channel)SelectedBeamNue_$(ex)_hist"]) for ex in experiments]),
                AppNue=sum([read(h5file["hRecoToTrue$(channel)SelectedAppNue_$(ex)_hist"]) for ex in experiments]),
                AppNuTau=sum([read(h5file["hRecoToTrue$(channel)SelectedAppNuTau_$(ex)_hist"]) for ex in experiments])
            ),
            L=L,
        )
        for channel in channels
    ])

    TotalCCCovar = (x->reshape(x, fill(Int(sqrt(length(x))), 2)...))(read(h5file["TotalCCCovar"]))
    TotalNCCovar = (x->reshape(x, fill(Int(sqrt(length(x))), 2)...))(read(h5file["TotalNCCovar"]))
    close(h5file)

    joint(ch) = vcat(ch_data["FD"*ch].observed, ch_data["ND"*ch].observed)
    assets = (
        ch_data = ch_data,
        detectors = detectors,
        TotalCCCovar = TotalCCCovar,
        TotalNCCovar = TotalNCCovar,
        observed = detectors == :FD ? (CC = ch_data["FDCC"].observed, NC = ch_data["FDNC"].observed) :
                                      (CC = joint("CC"), NC = joint("NC")),
    )
    if detectors == :FD_ND
        # fixed normalisation: log det of the joint covariance at the nominal prediction
        p0 = merge(physics.osc.params, physics.xsec.params)
        logdet0 = Dict(ch => logdet(Symmetric(joint_covariance(joint_expected(p0, physics, ch, assets), ch, assets)))
                       for ch in ("CC", "NC"))
        assets = merge(assets, (logdetΣ₀ = (CC = logdet0["CC"], NC = logdet0["NC"]),))
    end
    assets
end

# expected joint [FD; ND] spectrum and its covariance (release convention: FD bins first)
joint_expected(params, physics, channel, assets) =
    vcat(get_expected_per_channel(params, physics, assets.ch_data["FD"*channel]),
         get_expected_per_channel(params, physics, assets.ch_data["ND"*channel])) * physics.xsec.scale(:any, Symbol(channel), params)
function joint_covariance(μ, channel, assets)
    V = (μ * μ') .* (channel == "CC" ? assets.TotalCCCovar : assets.TotalNCCovar) + diagm(μ)
    (V + V') / 2
end

function get_expected_per_channel(params, physics, assets)
    # Minos baseline:
    s = assets.smearings
    p = physics.osc.osc_prob(s.E, [assets.L], params)
    NuMu = s.NuMu * p[:,[1],2,2]
    TrueNC = s.TrueNC * dropdims(sum(p[:,[1],2,1:3], dims=3), dims=3)
    BeamNue = s.BeamNue * p[:,[1],1,1]
    AppNue = s.AppNue * p[:,[1],2,1]
    AppNuTau = s.AppNuTau * p[:,[1],2,3]
    dropdims(NuMu + TrueNC + BeamNue + AppNue + AppNuTau, dims=2)
end

function forward_model_per_channel(params, physics, channel, assets)
   
    observed_far = assets.ch_data["FD"*channel].observed
    expected_far = get_expected_per_channel(params, physics, assets.ch_data["FD"*channel]) * physics.xsec.scale(:any, Symbol(channel), params)
    observed_near = assets.ch_data["ND"*channel].observed
    expected_near = get_expected_per_channel(params, physics, assets.ch_data["ND"*channel]) * physics.xsec.scale(:any, Symbol(channel), params)
    
    cov = channel == "CC" ? assets.TotalCCCovar : assets.TotalNCCovar

    tot = vcat(expected_far, expected_near)
    cov = (tot * tot') .* cov + diagm(tot)
        
    cov11 = cov[1:length(observed_far), 1:length(observed_far)]
    cov12 = cov[1:length(observed_far), end-length(observed_near)+1:end]
    cov21 = cov[end-length(observed_near)+1:end, 1:length(observed_far)]
    cov22 = cov[end-length(observed_near)+1:end, end-length(observed_near)+1:end]

    cov22inv = inv(cov22)

    x = cov12 * (cov22inv * (observed_near - expected_near))
    expected = expected_far + x

    cov = cov11 - cov12 * cov22inv * cov21

    Distributions.MvNormal(expected, (cov+cov')/2)
end

function forward_model_two_detector(params, physics, channel, assets)
    μ = joint_expected(params, physics, channel, assets)
    FixedNormGaussian(μ, joint_covariance(μ, channel, assets), assets.logdetΣ₀[Symbol(channel)])
end

function get_forward_model(physics, assets)
    fm = assets.detectors == :FD ? forward_model_per_channel : forward_model_two_detector
    function forward_model(params)
        distprod(
            CC = fm(params, physics, "CC", assets),
            NC = fm(params, physics, "NC", assets),
        )
    end
end

function get_plot(physics, assets)

    function plot(params, d=assets.observed)
        f = Figure()
    
        m = mean(get_forward_model(physics, assets)(params))
        v = var(get_forward_model(physics, assets)(params))
        
        for (i, ch) in enumerate([:CC, :NC])
            nfd = length(assets.ch_data["FD"*String(ch)].observed)     # with detectors = :FD_ND, show the FD part
            m, v, d = merge(m, (; ch => m[ch][1:nfd])), merge(v, (; ch => v[ch][1:nfd])), merge(d, (; ch => d[ch][1:nfd]))
        
            ax = Axis(f[1,i])
            energy_bins = assets.ch_data["FD"*String(ch)].bin_edges
            energy = 0.5 .* (energy_bins[1:end-1] .+ energy_bins[2:end])
            
            plot!(ax, energy, d[ch] ./ diff(energy_bins), color=:black, label="Observed")
            stephist!(ax, energy, weights=m[ch] ./ diff(energy_bins), bins=energy_bins, label="Expected")
            barplot!(ax, energy, (m[ch] .+ sqrt.(v[ch])) ./ diff(energy_bins), width=diff(energy_bins), gap=0, fillto= (m[ch] .- sqrt.(v[ch])) ./ diff(energy_bins), alpha=0.5, label="Standard Deviation")
            
            ax.ylabel="Counts / GeV"
            ax.title="MINOS/MINOS+ Far Detector "*String(ch)
            axislegend(ax, framevisible = false)
            
            
            ax2 = Axis(f[2,i])
            plot!(ax2, energy, d[ch] ./ m[ch], color=:black, label="Observed")
            hlines!(ax2, 1, label="Expected")
            barplot!(ax2, energy, 1 .+ sqrt.(v[ch]) ./ m[ch], width=diff(energy_bins), gap=0, fillto= 1 .- sqrt.(v[ch])./m[ch], alpha=0.5, label="Standard Deviation")
            ylims!(ax2, 0.7, 1.3)
            
            ax.xticksvisible = false
            ax.xticklabelsvisible = false
            
            rowsize!(f.layout, 1, Relative(3/4))
            rowgap!(f.layout, 1, 0)
            
            ax2.xlabel="Reconstructed Energy (GeV)"
            ax2.ylabel="Counts/Expected"
            
            xlims!(ax, minimum(energy_bins), maximum(energy_bins))
            xlims!(ax2, minimum(energy_bins), maximum(energy_bins))
        
        end
        
        ylims!(f.content[1], 0, 800)
        ylims!(f.content[4], 0, 600)
        
        f
    end
end

end
