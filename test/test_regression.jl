using Test
using Newtrinos
using DensityInterface
using Printf
# All reference likelihood values at default parameters.
# If a value changes beyond rtol=1e-6, it indicates a regression.
# To update: run the experiment and replace the reference value.
const ALL_REFERENCE_VALUES = (
    dayabay      = -168.90003273508322,
    kamland      = -144.63869819545005,
    minos        = -268.4046154382636,
    deepcore     = -942.5706065958955,
    super_k      = -3432.5312315130445,
    orca         = -1165.3469481902216,  # Frejus flux without mountain, as prescribed by the data release
    t2k          = -353.00066927575386,
    nova         = -329.9281060402088,
    coherent_csi = -276.22310206660876,
    coherent_lAr = -1755.0426661113386,
    katrin       = 0.4549394122046937,
    gerda        = -85.50961704258683,
    chlorine     = -0.3161211491885203,
    gallex_gno   = -6.3130576725955,
    sage         = -2.3240030626211583,
    sno          = 18.25318881919382,
    sk1_solar    = -127.27197625330082,
    sk2_solar    = -50.54386516276435,
    sk3_solar    = -83.55199911835112,
    sk4_solar    = -45.788946357271385,
    borexino_ph1 = 0.6366322757818439,
    borexino_ph2 = -5.065586938970489,
    borexino_ph3 = -2290.7264526090985,
)

# Parse --experiments from ARGS if present, otherwise run all
function select_experiments()
    idx = findfirst(==("--experiments"), ARGS)
    if idx === nothing
        return ALL_REFERENCE_VALUES
    end
    names = Symbol[]
    for i in (idx+1):length(ARGS)
        startswith(ARGS[i], "-") && break
        push!(names, Symbol(lowercase(ARGS[i])))
    end
    isempty(names) && error("--experiments requires at least one experiment name")
    for n in names
        haskey(ALL_REFERENCE_VALUES, n) || error("Unknown experiment: $n. Available: $(join(keys(ALL_REFERENCE_VALUES), ", "))")
    end
    return (; (n => ALL_REFERENCE_VALUES[n] for n in names)...)
end

selected = select_experiments()

@testset "Likelihood Regression" begin
    # Collect results for summary table
    results = NamedTuple{(:name, :ref, :actual, :rdiff, :status), Tuple{Symbol, Float64, Float64, Float64, Symbol}}[]

    for (name, ref_llh) in pairs(selected)
        @testset "$name" begin
            mod = getproperty(Newtrinos, name)
            exp = mod.configure()
            experiments = (; name => exp)
            params = Newtrinos.get_params(experiments)
            likelihood = Newtrinos.generate_likelihood(experiments)
            llh = logdensityof(likelihood, params)

            diff = llh - ref_llh
            rdiff = abs(diff) / abs(ref_llh)

            if rdiff > 1e-6
                if llh > ref_llh
                    push!(results, (name=name, ref=ref_llh, actual=llh, rdiff=rdiff, status=:improved))
                    @warn "$name likelihood improved: $ref_llh → $llh (rdiff=$rdiff). Update reference value."
                    @test_broken llh ≈ ref_llh rtol=1e-6
                else
                    push!(results, (name=name, ref=ref_llh, actual=llh, rdiff=rdiff, status=:regressed))
                    @test llh ≈ ref_llh rtol=1e-6
                end
            else
                push!(results, (name=name, ref=ref_llh, actual=llh, rdiff=rdiff, status=:ok))
                @test llh ≈ ref_llh rtol=1e-6
            end
        end
    end

    # Print summary table
    green  = "\e[32m"
    orange = "\e[33m"
    red    = "\e[31m"
    bold   = "\e[1m"
    reset  = "\e[0m"

    println()
    println("$(bold)╔══════════════════════════════════════════════════════════════════════════════════════╗$(reset)")
    println("$(bold)║                         Likelihood Regression Summary                              ║$(reset)")
    println("$(bold)╠══════════════╦══════════════════╦══════════════════╦══════════════╦═════════════════╣$(reset)")
    println("$(bold)║ Experiment   ║ Reference        ║ Actual           ║ Rel. Diff    ║ Status          ║$(reset)")
    println("$(bold)╠══════════════╬══════════════════╬══════════════════╬══════════════╬═════════════════╣$(reset)")

    for r in results
        color = r.status == :ok ? green : r.status == :improved ? orange : red
        status_str = r.status == :ok ? "$(green)OK$(reset)" :
                     r.status == :improved ? "$(orange)IMPROVED$(reset)" :
                     "$(red)REGRESSED$(reset)"

        @printf("║ %-12s ║ %16.6f ║ %16.6f ║ %12.2e ║ %-24s║\n",
                r.name, r.ref, r.actual, r.rdiff, status_str)
    end

    println("$(bold)╚══════════════╩══════════════════╩══════════════════╩══════════════╩═════════════════╝$(reset)")
    println()
end
