using LinearAlgebra
using StaticArrays
using ..Newtrinos

include("eigen_hermitian_3x3.jl")

"""
    BargerEigen <: Newtrinos.osc.EigenMethod

Analytic 3×3 Hermitian eigendecomposition using closed-form expressions.
Uses the method of Deledalle et al. 2017, which computes eigenvalues via
the cubic formula and eigenvectors via closed-form expressions.

Only valid for 3×3 Hermitian matrices (i.e. three-flavour oscillations).
Falls back to Julia's `eigen` for non-3×3 matrices.
"""
struct BargerEigen <: Newtrinos.osc.EigenMethod end

function Newtrinos.osc.decompose(H::Hermitian{T, <:SMatrix{3,3,T}}, ::BargerEigen) where T
    fast_eigen(H)
end

# Fallback for non-3×3 (e.g. sterile, ADD models)
function Newtrinos.osc.decompose(H::Hermitian, ::BargerEigen)
    eigen(H)
end
