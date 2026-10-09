using ForwardDiff
import ForwardDiff: gradient

function gradient(f, p::NamedTuple)

    pnames = Tuple(keys(p))
    pvals = values(p)
    sizes = ntuple(i -> pvals[i] isa AbstractVector ? length(pvals[i]) : 1, length(pnames))

    flat = Float64[]
    for v in pvals
        append!(flat, v isa AbstractVector ? v : [v])
    end

    function wrapped_f(x)
        pos = 1
        fields = Any[]
        for s in sizes
            push!(fields, s == 1 ? x[pos] : x[pos:pos+s-1])
            pos += s
        end
        f(NamedTuple{pnames}(fields))
    end

    g = ForwardDiff.gradient(wrapped_f, flat)

    pos = 1
    gvals = Any[]
    for s in sizes
        push!(gvals, s == 1 ? g[pos] : g[pos:pos+s-1])
        pos += s
    end

    return NamedTuple{pnames}(gvals)

end


function hessian(f, p::NamedTuple)

    pnames = Tuple(keys(p))
    pvals = values(p)
    sizes = ntuple(i -> pvals[i] isa AbstractVector ? length(pvals[i]) : 1, length(pnames))

    flat = Float64[]
    for v in pvals
        append!(flat, v isa AbstractVector ? v : [v])
    end

    function wrapped_f(x)
        pos = 1
        fields = Any[]
        for s in sizes
            push!(fields, s == 1 ? x[pos] : x[pos:pos+s-1])
            pos += s
        end
        f(NamedTuple{pnames}(fields))
    end

    g = ForwardDiff.hessian(wrapped_f, flat)

    pos = 1
    gvals = Any[]
    for s in sizes
        push!(gvals, s == 1 ? g[pos, :] : g[pos:pos+s-1, :])
        pos += s
    end

    return NamedTuple{pnames}(gvals)

end