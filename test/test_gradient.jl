using Test
using ForwardDiff
using Newtrinos

function f(x::Vector)
    return x[1]^2 + x[1] * x[2] + x[3]
end


function g(x::NamedTuple)
    return x.a^2 + x.a * x.b + x.c
end

function h(x::NamedTuple)
    return x.a[1]^2 + x.a[1] * x.a[2] + x.c
end


@testset "autodiff" begin
    p = (a=3, b=2, c=1,)
    l = (a=[3, 2], c=1,)
    v = [3, 2, 1]
    @testset "gradient" begin
        grad_vec = ForwardDiff.gradient(f, v)
        grad_named_tup = values(collect(Newtrinos.gradient(g, p)))
        grad_nested = Newtrinos.gradient(h, l)
        @test grad_vec == grad_named_tup
        @test grad_named_tup == vcat(grad_nested.a, [grad_nested.c])
    end

    @testset "hessian" begin
        hess_vec = ForwardDiff.hessian(f, v)
        hess_named_tup = Newtrinos.hessian(g, p)
        hess_nested = Newtrinos.hessian(h, l)
        @test hess_vec == reduce(hcat, values(hess_named_tup))
        @test hess_vec == reduce(vcat, [values(hess_nested)[1], reshape(values(hess_nested)[2], (1, 3))])
    end
end