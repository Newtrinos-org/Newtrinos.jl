# Correlated cross-section systematics for global fits (included by xsec.jl).
#
# Every experiment's prediction is reweighted from the cross-section model of its own simulation
# ("generator": a GENIE tune, or NEUT 5.4.0 on water) to one common reference model whose uncertainties
# are described by one shared set of parameters, so that the same parameter moves all experiments
# consistently (targets, flavours, ν/ν̄, processes).

using HDF5

# ------------------------------------------------------------------------------------------------
# cross-section tables
# ------------------------------------------------------------------------------------------------
const SPLINE_FILE = joinpath(@__DIR__, "xsec_data", "genie_splines.h5")

"Interaction processes of the CC cross section (NC is treated as one inclusive process)"
const PROCESSES = (:QE, :MEC, :RES, :DIS, :COH)

const NUCLEUS_A = Dict("H1" => 1, "C12" => 12, "O16" => 16, "Ar40" => 40, "Fe56" => 56)

"""
Detector targets as nucleus multiplicities. Cross sections of a target are per nucleon:
`Σᵢ nᵢ σᵢ / Σᵢ nᵢ Aᵢ`. Ice and sea water are `:H2O`; NOvA's liquid scintillator is close to `:CH2`.
"""
const TARGETS = Dict(
    :H2O => ("H1" => 2, "O16" => 1),
    :CH => ("C12" => 1, "H1" => 1),
    :CH2 => ("C12" => 1, "H1" => 2),
    :C => ("C12" => 1,),
    :O => ("O16" => 1,),
    :Ar => ("Ar40" => 1,),
    :Fe => ("Fe56" => 1,),
    :H => ("H1" => 1,),
)

"""
    XsecTable

Per-process cross sections σ/E (10⁻³⁸ cm²/GeV per nucleus) on a logarithmic energy grid `E` (GeV),
`σ[(tune, nucleus, flavour, current, process)]`, with `flavour` one of `nue, nuebar, numu, numubar,
nutau, nutaubar`, `current` `"CC"`/`"NC"` and `process` one of `PROCESSES`.
"""
struct XsecTable
    E::Vector{Float64}
    σ::Dict{NTuple{5, String}, Vector{Float64}}
    tunes::Vector{String}
end

const _table_cache = Ref{Union{Nothing, XsecTable}}(nothing)
const _table_lock = ReentrantLock()

function load_table()
    lock(_table_lock) do
        isnothing(_table_cache[]) || return _table_cache[]
        σ = Dict{NTuple{5, String}, Vector{Float64}}()
        local E, tunes
        h5open(SPLINE_FILE) do h
            E = read(h["E"])
            tunes = filter(!=("E"), keys(h))
            for tune in tunes, nuc in keys(h[tune]), fl in keys(h[tune][nuc]), cur in keys(h[tune][nuc][fl]), p in keys(h[tune][nuc][fl][cur])
                # stored per nucleus σ; keep σ/E per nucleus
                σ[(tune, nuc, fl, cur, p)] = Float64.(read(h[tune][nuc][fl][cur][p])) ./ E
            end
        end
        _table_cache[] = XsecTable(E, σ, tunes)
    end
end

flavour_key(flavour::Symbol, anti::Bool) = string(flavour, anti ? "bar" : "")

"""
    generator_sigma(generator, target, flavour, anti, current) -> NamedTuple

σ/E per nucleon (10⁻³⁸ cm²/GeV) on the table grid `load_table().E`, per process (`QE, MEC, RES, DIS, COH`;
for NC only the total, in `NC`), for a cross-section `generator`:

- a GENIE tune from the table (e.g. `:G18_10a_02_11a`, `:G18_02a_00_000`, `:AR23_20i_00_000`,
  `:G2_12_10_Default_MEC`); `:GENIE2_12` is GENIE 2.12 default without MEC;
- `:NEUT5_4_0`: NEUT 5.4.0 on water (digitised, T. Wester, Super-K PhD thesis, Fig. 4.7; valid up to
  26 GeV and continued with the G18_10a_02_11a shape above, ντ from νμ times the G18_10a_02_11a ντ/νμ
  ratio). Only for `target = :H2O`.
"""
function generator_sigma(generator::Symbol, target::Symbol, flavour::Symbol, anti::Bool, current::Symbol)
    tab = load_table()
    generator === :NEUT5_4_0 && return neut_sigma(target, flavour, anti, current)
    tune, drop = generator === :GENIE2_12 ? ("G2_12_10_Default_MEC", (:MEC,)) : (string(generator), ())
    tune in tab.tunes || throw(ArgumentError("unknown cross-section generator $generator; available: $(tab.tunes), :GENIE2_12, :NEUT5_4_0"))
    haskey(TARGETS, target) || throw(ArgumentError("unknown target $target; available: $(collect(keys(TARGETS)))"))
    fl = flavour_key(flavour, anti)
    comp = TARGETS[target]
    nucleons = sum(n * NUCLEUS_A[nuc] for (nuc, n) in comp)
    per_nucleon(p) = sum(n .* get(tab.σ, (tune, nuc, fl, string(current), string(p)), zeros(length(tab.E))) for (nuc, n) in comp) ./ nucleons
    if current === :NC
        return (NC = sum(per_nucleon(p) for p in PROCESSES),)
    end
    NamedTuple{PROCESSES}(Tuple(p in drop ? zeros(length(tab.E)) : per_nucleon(p) for p in PROCESSES))
end

# NEUT 5.4.0 (water) channels → processes. CC multi-π ("CCother", W < 2 GeV) is non-resonant
# inelastic and goes with DIS, as GENIE's DIS includes the low-W region.
const NEUT_CHANNELS = (QE = ("CC1p1h",), MEC = ("CC2p2h",), RES = ("CC1pi",), DIS = ("CCother", "CCDIS"), COH = ())

function neut_sigma(target, flavour, anti, current)
    target === :H2O || throw(ArgumentError("NEUT 5.4.0 cross sections are only available for water (:H2O)"))
    tab = load_table()
    data = load(joinpath(@__DIR__, "xsec_genie_data.jld2"))
    En, w = data["E_grid"], data["wester_xsec"]
    E_valid = 26.0
    genie(fl, cur) = generator_sigma(:G18_10a_02_11a, :H2O, fl, anti, cur)
    # NEUT curve on the table grid, continued above E_valid with the GENIE shape (matched at E_valid)
    function onto_table(neut_vals, genie_vals)
        itp = extrapolate(interpolate((En,), neut_vals, Gridded(Linear())), Flat())
        v = itp.(tab.E)
        i = findlast(tab.E .<= E_valid)
        gi, ni = genie_vals[i], v[i]
        gi > 0 && ni > 0 && (v[i+1:end] .= genie_vals[i+1:end] .* (ni / gi))
        max.(v, 0.0)
    end
    base = flavour === :nutau ? :numu : flavour
    fk = flavour_key(base, anti)
    σ = if current === :NC
        (NC = onto_table(w[fk]["NC"], genie(base, :NC).NC),)
    else
        gc = genie(base, :CC)
        NamedTuple{PROCESSES}(Tuple(isempty(NEUT_CHANNELS[p]) ? zeros(length(tab.E)) :
                                    onto_table(sum(w[fk][ch] for ch in NEUT_CHANNELS[p]), getfield(gc, p)) for p in PROCESSES))
    end
    if flavour === :nutau && current === :CC
        # ντ CC: τ threshold and suppression from the GENIE ντ/νμ ratio per process
        gτ, gμ = genie(:nutau, :CC), genie(:numu, :CC)
        σ = map((s, a, b) -> s .* ifelse.(b .> 0, a ./ max.(b, 1e-300), 0.0), σ, gτ, gμ)
    end
    σ
end

# log-linear interpolation of a tabulated curve at energies E
function at_energies(tabE, v, E)
    itp = extrapolate(interpolate((log.(tabE),), v, Gridded(Linear())), Flat())
    itp.(log.(E))
end

# ------------------------------------------------------------------------------------------------
# axial-mass response of quasi-elastic scattering (Llewellyn Smith, dipole form factors)
# ------------------------------------------------------------------------------------------------
"""
    llewellyn_smith(E, MA; m_l, anti) -> σ (arbitrary units)

Free-nucleon CCQE cross section (Llewellyn Smith) with dipole vector (M_V = 0.84 GeV) and axial
(`MA`) form factors and the PCAC pseudoscalar form factor, integrated over the kinematically allowed Q².
Only used for ratios σ(M_A)/σ(M_A,0).
"""
function llewellyn_smith(E, MA; m_l = 0.10566, anti = false, n_q = 400)
    M = 0.93892; mπ = 0.13957; MV = 0.84; ξ = 3.706; gA = -1.2723
    s = M^2 + 2M * E
    s <= (M + m_l)^2 && return 0.0
    √s = sqrt(s)
    pν = (s - M^2) / (2√s)
    El = (s + m_l^2 - M^2) / (2√s)
    pl = sqrt(max(El^2 - m_l^2, 0.0))
    Q2min = max(-m_l^2 + 2pν * (El - pl), 0.0)
    Q2max = -m_l^2 + 2pν * (El + pl)
    Q2max <= Q2min && return 0.0
    function dσ(Q2)
        τ = Q2 / (4M^2)
        GE = 1 / (1 + Q2 / MV^2)^2
        GM = (1 + ξ) * GE
        F1 = (GE + τ * GM) / (1 + τ)
        ξF2 = (GM - GE) / (1 + τ)
        FA = gA / (1 + Q2 / MA^2)^2
        FP = 2M^2 * FA / (mπ^2 + Q2)
        A = (m_l^2 + Q2) / M^2 * ((1 + τ) * FA^2 - (1 - τ) * F1^2 + τ * (1 - τ) * ξF2^2 + 4τ * F1 * ξF2 -
             m_l^2 / (4M^2) * ((F1 + ξF2)^2 + (FA + 2FP)^2 - 4(1 + τ) * FP^2))
        B = Q2 / M^2 * FA * (F1 + ξF2)
        C = (FA^2 + F1^2 + τ * ξF2^2) / 4
        su = 4M * E - Q2 - m_l^2
        max(A + (anti ? 1 : -1) * B * su / M^2 + C * su^2 / M^4, 0.0) / E^2
    end
    # Simpson on a uniform Q² grid
    h = (Q2max - Q2min) / n_q
    acc = dσ(Q2min) + dσ(Q2max)
    for i in 1:n_q-1
        acc += (isodd(i) ? 4 : 2) * dσ(Q2min + i * h)
    end
    acc * h / 3
end

const LEPTON_MASS = Dict(:nue => 0.000511, :numu => 0.10566, :nutau => 1.77686)

"""
Quadratic response of σ_QE to a relative axial-mass shift `x` (in units of `σ_rel`):
`σ(M_A (1 + σ_rel x)) / σ(M_A) ≈ 1 + a(E) x + b(E) x²`, fitted at x = ±1, ±2.
"""
function ma_qe_response(E, flavour, anti; MA0 = 1.0, σ_rel = 0.1)
    m = LEPTON_MASS[flavour]
    a = similar(E, Float64); b = similar(E, Float64)
    xs = (-2.0, -1.0, 1.0, 2.0)
    for (i, e) in enumerate(E)
        σ0 = llewellyn_smith(e, MA0; m_l = m, anti)
        if σ0 <= 0
            a[i] = 0.0; b[i] = 0.0
            continue
        end
        r = [llewellyn_smith(e, MA0 * (1 + σ_rel * x); m_l = m, anti) / σ0 - 1 for x in xs]
        # least squares for r = a x + b x²
        X = hcat(collect(xs), collect(xs) .^ 2)
        a[i], b[i] = X \ r
    end
    a, b
end

# ------------------------------------------------------------------------------------------------
# the model
# ------------------------------------------------------------------------------------------------
"""
    CorrelatedXsec <: XsecModel

Cross-section model for global fits: every experiment is reweighted from its own simulation's
cross sections (`generator`, declared by the experiment) to one common `reference` GENIE tune, whose
uncertainties are one shared parameter set, so that all experiments are correlated through it.

The uncertainty model has two parts:

1. **Model spread** (`xsec_pc1 … xsec_pcK`, standard normal): principal components of the spread of
   all `tunes` (σ/E per process, around their mean, relative to the reference total), computed jointly
   over targets (H, C, O, Ar, Fe), flavours (νe, νμ, ντ and antineutrinos), processes (CC QE, MEC, RES,
   DIS, COH; NC) and energy (`pca_range`). A component therefore moves e.g. QE on oxygen and on carbon
   together, as the generators do. Mode k is `U_k S_k / √(N-1)`, i.e. one unit is the 1σ spread.
2. **Physics knobs**, independent of the tune spread (which underestimates uncertainties the tunes share):
   - `xsec_ma_qe`: axial mass of QE in units of 10 % (Llewellyn Smith response, per flavour, ν/ν̄);
   - `xsec_res_norm`, `xsec_dis_norm`, `xsec_coh_norm`: process normalisations (10 %, 3 %, 30 %);
   - `xsec_nc_norm` (20 %), `xsec_nutau_cc_norm` (20 %);
   - `xsec_nubar_ratio`: ν̄ CC QE + MEC + RES + COH relative to ν (5 %);
   - `xsec_nue_numu_ratio`: νe/ν̄e CC QE + MEC + RES relative to νμ (3 %).

# Fields
- `reference::Symbol = :G18_10a_02_11a`: the reference model (unit weights at nominal parameters).
- `tunes`: the GENIE tunes spanning the model spread (default: all in the table).
- `n_modes::Int = 4`: number of principal components.
- `pca_range = (0.1, 100.0)`: energy range (GeV) of the PCA; outside, the relative mode is held constant.
"""
@kwdef struct CorrelatedXsec <: XsecModel
    reference::Symbol = :G18_10a_02_11a
    tunes::Union{Nothing, Vector{Symbol}} = nothing
    n_modes::Int = 4
    pca_range::Tuple{Float64, Float64} = (0.1, 100.0)
end

const NORM_PRIORS = (
    xsec_res_norm = Truncated(Normal(1, 0.10), 0.5, 1.5),
    xsec_dis_norm = Truncated(Normal(1, 0.03), 0.85, 1.15),
    xsec_coh_norm = Truncated(Normal(1, 0.30), 0.0, 2.5),
    xsec_nc_norm = Truncated(Normal(1, 0.20), 0.2, 1.8),
    xsec_nutau_cc_norm = Truncated(Normal(1, 0.20), 0.2, 1.8),
    xsec_nubar_ratio = Truncated(Normal(1, 0.05), 0.75, 1.25),
    xsec_nue_numu_ratio = Truncated(Normal(1, 0.03), 0.85, 1.15),
)

pc_names(cfg::CorrelatedXsec) = Tuple(Symbol("xsec_pc$k") for k in 1:cfg.n_modes)

function get_params(cfg::CorrelatedXsec)
    pcs = NamedTuple{pc_names(cfg)}(ntuple(_ -> 0.0, cfg.n_modes))
    merge(pcs, (xsec_ma_qe = 0.0,), map(_ -> 1.0, NORM_PRIORS))
end

function get_priors(cfg::CorrelatedXsec)
    pcs = NamedTuple{pc_names(cfg)}(ntuple(_ -> Normal(0, 1), cfg.n_modes))
    merge(pcs, (xsec_ma_qe = Normal(0, 1),), NORM_PRIORS)
end

const PCA_NUCLEI = ("H1", "C12", "O16", "Ar40", "Fe56")
const PCA_FLAVOURS = ("nue", "nuebar", "numu", "numubar", "nutau", "nutaubar")

"""
    xsec_modes(cfg) -> (modes, explained)

Principal components of the tune spread. `modes[k][(nucleus, flavour, current, process)]` is the
absolute change of σ/E per nucleus (table grid) for one unit of `xsec_pck`; `explained` the fraction of
the spread variance per component.
"""
function xsec_modes(cfg::CorrelatedXsec)
    tab = load_table()
    tunes = isnothing(cfg.tunes) ? tab.tunes : string.(cfg.tunes)
    ref = string(cfg.reference)
    ref in tab.tunes || throw(ArgumentError("reference $(cfg.reference) not in the table: $(tab.tunes)"))
    E = tab.E
    inrange = (E .>= cfg.pca_range[1]) .& (E .<= cfg.pca_range[2])
    ie = findall(inrange)
    blocks = Tuple{NTuple{4, String}, Vector{Float64}}[]   # (key, reference total for normalisation)
    for nuc in PCA_NUCLEI, fl in PCA_FLAVOURS
        tot_cc = sum(get(tab.σ, (ref, nuc, fl, "CC", string(p)), zeros(length(E))) for p in PROCESSES)
        tot_nc = sum(get(tab.σ, (ref, nuc, fl, "NC", string(p)), zeros(length(E))) for p in PROCESSES)
        for p in PROCESSES
            push!(blocks, ((nuc, fl, "CC", string(p)), tot_cc))
        end
        push!(blocks, ((nuc, fl, "NC", "NC"), tot_nc))
    end
    curve(t, key) = key[4] == "NC" ? sum(get(tab.σ, (t, key[1], key[2], "NC", string(p)), zeros(length(E))) for p in PROCESSES) :
                                     get(tab.σ, (t, key...), zeros(length(E)))
    # rows: (block, energy); columns: tunes. Entries relative to the reference total, where that is > 0
    rows = Vector{Float64}[]
    for t in tunes
        col = Float64[]
        for (key, tot) in blocks
            c = curve(t, key)[ie]
            append!(col, ifelse.(tot[ie] .> 0, c ./ max.(tot[ie], 1e-300), 0.0))
        end
        push!(rows, col)
    end
    D = reduce(hcat, rows)
    D .-= sum(D, dims = 2) ./ size(D, 2)
    U, S, _ = svd(D)
    nt = length(tunes)
    K = min(cfg.n_modes, length(S))
    explained = S .^ 2 ./ sum(S .^ 2)
    modes = map(1:K) do k
        v = U[:, k] .* S[k] ./ sqrt(nt - 1)
        out = Dict{NTuple{4, String}, Vector{Float64}}()
        nE = length(ie)
        for (b, (key, tot)) in enumerate(blocks)
            rel = v[(b-1)*nE+1:b*nE]
            # hold the relative mode constant outside the PCA range
            full = [rel[clamp(searchsortedfirst(ie, i), 1, nE)] for i in eachindex(E)]
            full[E .< cfg.pca_range[1]] .= rel[1]
            out[key] = full .* tot
        end
        out
    end
    modes, explained[1:K]
end

"Reference cross sections and modes for a detector target, per nucleon, on the table grid"
function target_curves(cfg::CorrelatedXsec, modes, target, flavour, anti, current)
    tab = load_table()
    comp = TARGETS[target]
    nucleons = sum(n * NUCLEUS_A[nuc] for (nuc, n) in comp)
    fl = flavour_key(flavour, anti)
    ref = generator_sigma(cfg.reference, target, flavour, anti, current)
    combine(k, key4) = sum(n .* get(modes[k], (nuc, fl, key4...), zeros(length(tab.E))) for (nuc, n) in comp) ./ nucleons
    Δ = if current === :NC
        [(NC = combine(k, ("NC", "NC")),) for k in eachindex(modes)]
    else
        [NamedTuple{PROCESSES}(Tuple(combine(k, ("CC", string(p))) for p in PROCESSES)) for k in eachindex(modes)]
    end
    ref, Δ
end

function get_scale(cfg::CorrelatedXsec)
    function scale(args...)
        throw(ArgumentError("CorrelatedXsec has no scalar `scale`; use `weights` (see Newtrinos.xsec.weights)"))
    end
end

"""
    get_weights(cfg) -> weights(E; flavour, anti, current, target, generator, process = :all)

Precomputed reweighting closure: `weights(...)` returns `params -> w`, with `w` a vector over `E`
(true neutrino energies, GeV) or a scalar where the model has no energy dependence.

- `flavour ∈ (:nue, :numu, :nutau)`, `anti::Bool`, `current ∈ (:CC, :NC)`;
- `target` (see `TARGETS`) and `generator` (see [`generator_sigma`](@ref)) describe the experiment's
  simulation and are only used by models that reweight between cross-section models;
- `process`: `:all` (inclusive: total σ ratio at each energy), one process or a tuple of processes
  (`:QE, :MEC, :RES, :DIS, :COH`; the ratio of their sums), or a vector with one process per entry of `E`
  (per-event reweighting with the event's interaction process).

If the generator has no MEC, the reference MEC cross section is assigned to the QE events.
"""
function get_weights(cfg::CorrelatedXsec)
    modes, explained = xsec_modes(cfg)
    @info "CorrelatedXsec configured" reference=cfg.reference n_modes=cfg.n_modes explained=round.(explained, digits=3)
    tab = load_table()
    pcs = pc_names(cfg)
    cache = Dict{Any, Any}()

    function weights(E; flavour::Symbol, anti::Bool, current::Symbol, target::Symbol, generator::Symbol, process = :all)
        Ev = collect(Float64, vec(E))
        process = process isa AbstractArray ? vec(process) : process
        key = (target, flavour, anti, current)
        ref, Δ = get!(() -> target_curves(cfg, modes, target, flavour, anti, current), cache, key)
        gen = generator_sigma(generator, target, flavour, anti, current)
        at(v) = at_energies(tab.E, v, Ev)
        if current === :NC
            r = at(ref.NC); g = at(gen.NC); d = [at(Δk.NC) for Δk in Δ]
            inv_g = ifelse.(g .> 0, 1 ./ max.(g, 1e-300), 0.0)
            return function (params)
                σ = r .+ sum(getproperty(params, pcs[k]) .* d[k] for k in eachindex(d))
                max.(σ, zero(eltype(σ))) .* inv_g .* params.xsec_nc_norm
            end
        end
        # CC: per-process reference, modes and knobs
        a_qe, b_qe = ma_qe_response(Ev, flavour, anti)
        R = map(at, ref); G = map(at, gen); D = [map(at, Δk) for Δk in Δ]
        no_mec = all(iszero, gen.MEC)
        sel(p) = process === :all ? true : process isa Symbol ? p === process :
                 process isa Tuple ? p in process : nothing
        lowE = (:QE, :MEC, :RES, :COH)
        knob(p, params) = (p === :RES ? params.xsec_res_norm : p === :DIS ? params.xsec_dis_norm :
                           p === :COH ? params.xsec_coh_norm : one(params.xsec_res_norm)) *
                          (anti && p in lowE ? params.xsec_nubar_ratio : one(params.xsec_res_norm)) *
                          (flavour === :nue && p in (:QE, :MEC, :RES) ? params.xsec_nue_numu_ratio : one(params.xsec_res_norm))
        σp(p, params) = begin
            s = getfield(R, p) .+ sum(getproperty(params, pcs[k]) .* getfield(D[k], p) for k in eachindex(D))
            s = max.(s, zero(eltype(s))) .* knob(p, params)
            p === :QE ? s .* (1 .+ a_qe .* params.xsec_ma_qe .+ b_qe .* params.xsec_ma_qe^2) : s
        end
        τ(params) = flavour === :nutau ? params.xsec_nutau_cc_norm : one(params.xsec_nutau_cc_norm)
        if process isa AbstractArray
            length(process) == length(Ev) || throw(DimensionMismatch("one process per energy expected"))
            masks = NamedTuple{PROCESSES}(Tuple(Float64.(process .=== p) for p in PROCESSES))
            inv_g = NamedTuple{PROCESSES}(Tuple(begin
                g = getfield(G, p)
                ifelse.(g .> 0, 1 ./ max.(g, 1e-300), 0.0)
            end for p in PROCESSES))
            return function (params)
                w = sum(masks[p] .* σp(p, params) .* inv_g[p] for p in PROCESSES if p !== :MEC || !no_mec)
                if no_mec
                    w = w .+ masks.QE .* σp(:MEC, params) .* inv_g.QE
                end
                w .* τ(params)
            end
        end
        ps = Tuple(p for p in PROCESSES if sel(p))
        isempty(ps) && throw(ArgumentError("no process selected by $process"))
        num_ps = no_mec && :QE in ps && !(:MEC in ps) ? (ps..., :MEC) : ps
        g = sum(getfield(G, p) for p in ps)
        inv_g = ifelse.(g .> 0, 1 ./ max.(g, 1e-300), 0.0)
        return params -> sum(σp(p, params) for p in num_ps) .* inv_g .* τ(params)
    end
end

"""
    get_sigma(cfg) -> sigma(E; flavour, anti, current, target, process = :all)

Absolute σ/E per nucleon (10⁻³⁸ cm²/GeV) of the reference model with its uncertainties, as a closure
`params -> σ`, for experiments that build predictions from flux × cross section.
"""
function get_sigma(cfg::CorrelatedXsec, w = get_weights(cfg))
    function sigma(E; flavour::Symbol, anti::Bool, current::Symbol, target::Symbol, process = :all)
        Ev = collect(Float64, vec(E))
        ref = generator_sigma(cfg.reference, target, flavour, anti, current)
        tabE = load_table().E
        base = current === :NC ? at_energies(tabE, ref.NC, Ev) :
               sum(at_energies(tabE, getfield(ref, p), Ev) for p in PROCESSES if process === :all || p === process || (process isa Tuple && p in process))
        f = w(Ev; flavour, anti, current, target, generator = cfg.reference, process)
        params -> base .* f(params)
    end
end
