# KernelAbstractions implementation of matter oscillations, selected with
# `OscillationConfig(backend=KernelBackend(...))`.
#
# Two kernels reuse the CPU building blocks, so the physics is defined in one place:
#   1. over (energy, layer): `compute_matter_matrices` -> matter eigensystem per layer
#   2. over (energy, path):  `osc_reduce` (Basic) or `_spray_reduce` + `spray_average` (Spray)
#      along the path's layer sections, written directly in the `osc_prob` output layout
#      P[E, path, in, out].
# Threads with consecutive energies share a path, so they run the same number of loop iterations.

"""
    check_backend(cfg::OscillationConfig)

Throw an informative error if `cfg.backend` cannot run the requested configuration.
Called when the oscillation module is configured.
"""
check_backend(cfg::OscillationConfig) = check_backend(cfg, cfg.backend)
check_backend(cfg::OscillationConfig, ::SerialCPU) = nothing
function check_backend(cfg::OscillationConfig, kb::KernelBackend)
    unsupported(what) = throw(ArgumentError("KernelBackend does not support $what (supported: 3-flavour models, SI interactions, Basic/Spray propagation, All states). Use backend=SerialCPU() for this configuration."))
    cfg.interaction isa SI || unsupported("interaction $(nameof(typeof(cfg.interaction)))")
    cfg.propagation isa Union{Basic, Spray} || unsupported("propagation $(nameof(typeof(cfg.propagation)))")
    cfg.states isa All || unsupported("state selector $(nameof(typeof(cfg.states)))")
    U, _ = get_matrices(cfg.flavour, cfg.eigen_method)(get_params(cfg))
    U isa SMatrix{3,3} || unsupported("flavour model $(nameof(typeof(cfg.flavour))) ($(size(U, 1)) states)")
    if !(kb.backend isa KA.CPU) && !(cfg.eigen_method isa Newtrinos.BargerEigen)
        throw(ArgumentError("KernelBackend on $(nameof(typeof(kb.backend))) requires eigen_method=Newtrinos.BargerEigen() (LAPACK-based $(nameof(typeof(cfg.eigen_method))) cannot run on the device)."))
    end
    nothing
end

_to_device(backend, x::AbstractArray) = (y = KA.allocate(backend, eltype(x), size(x)); copyto!(y, x); y)
_to_device(::KA.CPU, x::AbstractArray) = collect(x)
_to_host(::KA.CPU, x) = x
_to_host(backend, x) = Array(x)

@kernel function _matter_matrices_kernel!(mm, H_eff, @Const(E), @Const(radius), @Const(pd), @Const(nd), anti, interaction, eigen_method)
    ie, il = @index(Global, NTuple)
    @inbounds mm[ie, il] = compute_matter_matrices(H_eff, E[ie], Layer(radius[il], pd[il], nd[il]), anti, interaction, eigen_method)
end

@kernel function _basic_paths_kernel!(P, mm, @Const(E), @Const(ptr), sections, propagation)
    ie, ip = @index(Global, NTuple)
    @inbounds begin
        path = view(sections, ptr[ip]:(ptr[ip + 1] - 1))
        p = osc_reduce(view(mm, ie, :), path, E[ie], propagation)
        n = size(p, 1)
        for b in 1:n, a in 1:n
            P[ie, ip, b, a] = p[a, b]
        end
    end
end

@kernel function _spray_paths_kernel!(P, mm, spray_data, @Const(E), @Const(ptr), sections, dldh, σ_E, σ_h, averaging, eigen_method)
    ie, ip = @index(Global, NTuple)
    @inbounds begin
        rng = ptr[ip]:(ptr[ip + 1] - 1)
        e = E[ie]
        S, K_E, K_Theta = _spray_reduce(view(mm, ie, :), spray_data, view(sections, rng), e, view(dldh, rng))
        p = spray_average(S, K_E, K_Theta, σ_E * e, σ_h, averaging, eigen_method)
        n = size(p, 1)
        for b in 1:n, a in 1:n
            P[ie, ip, b, a] = p[a, b]
        end
    end
end

# Matter eigensystems for every (energy, layer) on the device
# (`E` is the host vector, used only to determine the element type; `E_d` is its device copy)
function _device_matter_matrices(kb::KernelBackend, H_eff, E, E_d, layers, anti, interaction, eigen_method)
    be = kb.backend
    radius, pd, nd = (_to_device(be, getproperty(layers, f)) for f in (:radius, :p_density, :n_density))
    MM = typeof(compute_matter_matrices(H_eff, first(E), first(layers), anti, interaction, eigen_method))
    mm = KA.allocate(be, MM, (length(E_d), length(layers)))
    _matter_matrices_kernel!(be, kb.workgroupsize)(mm, H_eff, E_d, radius, pd, nd, anti, interaction, eigen_method; ndrange=size(mm))
    mm
end

_output_type(H_eff, E, layers) = promote_type(real(eltype(H_eff)), eltype(E), eltype(layers.p_density), eltype(layers.n_density))

function _device_propagate(kb::KernelBackend, H_eff, E, paths, layers, propagation::Basic, interaction, anti, eigen_method)
    be = kb.backend
    E_d = _to_device(be, E)
    mm = _device_matter_matrices(kb, H_eff, E, E_d, layers, anti, interaction, eigen_method)
    ptr = _to_device(be, Int32.(paths.elem_ptr))
    sections = _to_device(be, flatview(paths))
    n = size(H_eff, 1)
    P = KA.allocate(be, _output_type(H_eff, E, layers), (length(E), length(paths), n, n))
    _basic_paths_kernel!(be, kb.workgroupsize)(P, mm, E_d, ptr, sections, propagation; ndrange=(length(E), length(paths)))
    KA.synchronize(be)
    _to_host(be, P)
end

function _device_propagate(kb::KernelBackend, H_eff, E, paths, layers, propagation::Spray, interaction, anti, eigen_method)
    be = kb.backend
    E_d = _to_device(be, E)
    mm = _device_matter_matrices(kb, H_eff, E, E_d, layers, anti, interaction, eigen_method)
    spray_data = _to_device(be, [compute_dVdE(layer, anti, interaction, _val_nflav(H_eff)) for layer in layers])
    ptr = _to_device(be, Int32.(paths.elem_ptr))
    sections = _to_device(be, flatview(paths))
    dldh = _to_device(be, reduce(vcat, _spray_dldh(paths, layers)))
    n = size(H_eff, 1)
    P = KA.allocate(be, _output_type(H_eff, E, layers), (length(E), length(paths), n, n))
    averaging = propagation.averaging === :gaussian ? Val(:gaussian) : Val(:uniform)
    _spray_paths_kernel!(be, kb.workgroupsize)(P, mm, spray_data, E_d, ptr, sections, dldh, propagation.σ_E, propagation.σ_h, averaging, eigen_method;
                                               ndrange=(length(E), length(paths)))
    KA.synchronize(be)
    _to_host(be, P)
end

# Matter method of `osc_prob` for KernelBackend configs (see `check_backend` for what is supported).
# With `All` states there is no incoherent remainder, so no `_add_rest_and_permute` is needed:
# the kernels write P[E, path, in, out] directly.
function _osc_prob(cfg::OscillationConfig{<:FlavourModel, <:SI, <:Union{Basic, Spray}, All, <:EigenMethod, <:KernelBackend},
                   E::AbstractVector{<:Real}, paths::VectorOfVectors{Path}, layers::StructVector{Layer}, params::NamedTuple, anti)
    U, h_raw = get_matrices(cfg.flavour, cfg.eigen_method)(params)
    h = h_raw .- minimum(h_raw)
    Uc = anti ? conj.(U) : U
    H_eff = Uc * Diagonal(h) * adjoint(Uc)
    _device_propagate(cfg.backend, H_eff, E, paths, layers, cfg.propagation, cfg.interaction, anti, cfg.eigen_method)
end
