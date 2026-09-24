using Distributions
using Distributed
using DensityInterface
using BAT
using DataStructures
using MeasureBase
using ADTypes
using Newtrinos
using FileIO
using Accessors
using ArgParse

include("cli_common.jl")

function parse_command_line()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--experiments"
        help = "List of experiments to run"
        nargs = '+'
        required = true

        "--name"
        help = "Name for outputs"
        arg_type = String
        required = true

        "--task"
        help = "Task to perform: Choice of NestedSampling, ImportanceSampling, Profile, Scan"
        arg_type = String
        required = true

        "--plot"
        help = "Enable plotting"
        action = :store_true

        "--workers"
        help = "Number of distributed workers (default: 1, no distributed)"
        arg_type = Int
        default = 1

        "--threads"
        help = "Number of threads per worker (only used when --workers > 1)"
        arg_type = Int
        default = 1
    end

    return parse_args(s)
end

# dynamically construct named tuple from experiment names
function configure_experiments(experiment_list)
    pairs = (Symbol(exp) => getproperty(getproperty(Newtrinos, Symbol(exp)), :configure)() for exp in experiment_list)
    #pairs = (Symbol(exp) => getproperty(getproperty(Newtrinos, Symbol(exp)), :configure)(ff_model=:klein_nystrand) for exp in experiment_list)
    #pairs = (Symbol(exp) => getproperty(getproperty(Newtrinos, Symbol(exp)), :configure)(ff_model=:sym_fermi) for exp in experiment_list)
    return (; pairs...)
end

args = parse_command_line()

name = args["name"]
n_workers = args["workers"]
n_threads = args["threads"]
use_distributed = n_workers > 1

# Set up distributed workers if requested
map_func = nothing
if use_distributed
    addprocs(n_workers; exeflags="--threads=$n_threads")

    @everywhere args = $args

    @everywhere begin
        using Distributions
        using DensityInterface
        using BAT
        using MeasureBase
        using ADTypes
        using Newtrinos

        include(joinpath(@__DIR__, "cli_common.jl"))

        ##### PHYSICS CONFIG #####
        # To use defaults:
        experiments = configure_experiments(args["experiments"])

        # To override physics (e.g. for IO, sterile models, custom flux, etc.), uncomment and modify:
        # osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(
        #     flavour=Newtrinos.osc.ThreeFlavour(ordering=:IO),
        #     interaction=Newtrinos.osc.SI(),
        # ))
        # atm_flux = Newtrinos.atm_flux.configure()
        # earth_layers = Newtrinos.earth_layers.configure()
        # xsec = Newtrinos.xsec.configure()
        # physics = (; osc, atm_flux, earth_layers, xsec)
        # experiments = configure_experiments(args["experiments"], physics)

        likelihood = Newtrinos.generate_likelihood(experiments)
    end

    map_func = pmap
else
    adsel = AutoForwardDiff()
    context = set_batcontext(ad = adsel)
end

##### PHYSICS CONFIG #####
# To use defaults:
if !use_distributed
    experiments = configure_experiments(args["experiments"])
end

# To override physics (e.g. for IO, sterile models, custom flux, etc.), uncomment and modify:
# osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(
#     flavour=Newtrinos.osc.ThreeFlavour(ordering=:IO),
#     interaction=Newtrinos.osc.SI(),
# ))
# atm_flux = Newtrinos.atm_flux.configure()
# earth_layers = Newtrinos.earth_layers.configure()
# xsec = Newtrinos.xsec.configure()
# physics = (; osc, atm_flux, earth_layers, xsec)
# experiments = configure_experiments(args["experiments"], physics)

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

# Variables to condition on (=fix)
conditional_vars = Dict(:θ₁₂=>p.θ₁₂, :δCP=>-1.89, :Δm²₂₁=>p.Δm²₂₁)


# For profile / scan task only: choose scan grid
vars_to_scan = OrderedDict()
vars_to_scan[:θ₂₃] = 11
vars_to_scan[:Δm²₃₁] = 11

###### END CONFIG ######

if !use_distributed
    likelihood = Newtrinos.generate_likelihood(experiments);
end

priors = Newtrinos.condition(priors, conditional_vars, p)

@reset priors.Δm²₃₁ = Uniform(0.0018, 0.0028)
@reset priors.θ₂₃ = Uniform(pi/4-0.2, pi/4+0.2)

if lowercase(args["task"]) == "nestedsampling"
    import UltraNest
    prior = distprod(;priors...)
    posterior = PosteriorMeasure(likelihood, prior)
    samples = bat_sample(posterior, ReactiveNestedSampling()).result
    FileIO.save(name * ".jld2", Dict("samples" => samples))

elseif lowercase(args["task"]) == "importancesampling"
    prior = distprod(;priors...)
    posterior = PosteriorMeasure(likelihood, prior)

    #seed_points = load("darkdim_seeds.jld2")["df"]
    #seed_points = seed_points[seed_points.ca3 .< 0, :]
    init_samples =  make_init_samples(posterior, 10, 1_000_000)
    nsamples = 100_000
    #init_samples =  make_init_samples(posterior, 10, 100_000)
    #FileIO.save(name * "_init_samples.jld2", Dict(String(a)=>init_samples[a] for a in keys(init_samples)))
    whack_samples = whack_many_moles(posterior, init_samples, target_samplesize=nsamples, cache_dir=nothing, maxiter=40)
    resampled = bat_sample(whack_samples.samples_user, RandResampling(nsamples=nsamples)).result
    FileIO.save(name * ".jld2", Dict("samples" => resampled))
else
    if lowercase(args["task"]) == "profile"
        result = Newtrinos.profile(likelihood, priors, vars_to_scan, p; cache_dir=name, map_func=map_func)
    elseif lowercase(args["task"]) == "scan"
        result = Newtrinos.scan(likelihood, priors, vars_to_scan, p)
    end

    save_result(result, name)

    if args["plot"]
        using CairoMakie
        title = args["task"] * ": " * join(args["experiments"], " + ")
        plot_result(result, name, vars_to_scan; title=title)
    end
end
