using LinearAlgebra
using Distributions
using DensityInterface
using Base
using ForwardDiff
using BAT
using IterTools
using DataStructures
using MeasureBase
using ADTypes
using Newtrinos
using FileIO
using Accessors
using ArgParse

function parse_command_line()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--experiments"
        help = "List of experiments to run"
        nargs = '+'
        required = true

        "--ordering"
        help = "NMO: either NO or IO"
        arg_type = String
        default = "NO"      

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

        # "--lambda"
        # arg_type = Float64
        # default = 1.

        #"--ca"
        #arg_type = Float64
        #default = 1.

        #"--scan"
        #arg_type = String
        #required = true

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

adsel = AutoForwardDiff()
context = set_batcontext(ad = adsel)

name = args["name"]

osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour=Newtrinos.osc.ThreeFlavour(ordering=Symbol(args["ordering"])),
    propagation=Newtrinos.osc.Basic(),
    states=Newtrinos.osc.All(),
    interaction=Newtrinos.osc.SI()
    )
osc = Newtrinos.osc.configure(osc_cfg)

atm_flux = Newtrinos.atm_flux.configure(
        Newtrinos.atm_flux.AtmFluxConfig(nominal_model=Newtrinos.atm_flux.HKKM("kam-ally-20-01-mtn-solmin.d")
        )
    )
earth_layers = Newtrinos.earth_layers.configure()
xsec=Newtrinos.xsec.configure(Newtrinos.xsec.Differential_H2O())

#physics = (; osc, atm_flux, earth_layers, xsec);

# Choose experiments to include
#experiments = (
#    deepcore = Newtrinos.deepcore.configure(physics),
#    dayabay = Newtrinos.dayabay.configure(physics),
#    kamland = Newtrinos.kamland.configure(physics),
#    minos = Newtrinos.minos.configure(physics),
#    orca = Newtrinos.orca.configure(physics),
#);

experiments = configure_experiments(args["experiments"])
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

# Variables to condition on (=fix)
#conditional_vars = [:θ₁₂, :δCP, :Δm²₂₁]
#conditional_vars = Dict(:θ₁₂=>p.θ₁₂, :δCP=>-1.89, :Δm²₂₁=>p.Δm²₂₁)
#conditional_vars = [:Darkdim_radius, :δCP, :λ₁, :λ₂, :λ₃]

#conditional_vars = Dict(:δCP=>0., :ca1=>args["ca"], :ca2=>args["ca"], :ca3=>args["ca"], :nutau_cc_norm=>1., :nc_norm=>1.)
#conditional_vars = Dict(:δCP=>0., :λ₁=>args["lambda"], :λ₂=>args["lambda"], :λ₃=>args["lambda"], :nutau_cc_norm=>1., :nc_norm=>1.)
#conditional_vars = Dict(:δCP=>0., :nutau_cc_norm=>1., :nc_norm=>1.)
conditional_vars = [:cevns_xsec_a, :cevns_xsec_b, :cevns_xsec_c, :cevns_xsec_d]
#conditional_vars = []


# For profile / scan task only: choose scan grid
vars_to_scan = OrderedDict()
vars_to_scan[:θ₂₃] = 11
vars_to_scan[:Δm²₃₁] = 11


###### END CONFIG ######

likelihood = Newtrinos.generate_likelihood(experiments);


priors = Newtrinos.condition(priors, conditional_vars, p)

#@reset priors.Δm²₃₁ = Uniform(0.0018, 0.0028)
#@reset priors.θ₂₃ = Uniform(pi/4-0.2, pi/4+0.2)
    
if lowercase(args["task"]) == "nestedsampling"
    #import NestedSamplers
    import UltraNest
    prior = distprod(;priors...)
    posterior = PosteriorMeasure(likelihood, prior)
    samples = bat_sample(posterior, ReactiveNestedSampling()).result
    #samples = bat_sample(posterior, EllipsoidalNestedSampling()).result
    #samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^4, nchains = 4)).result
    FileIO.save(name * ".jld2", Dict("samples" => samples))
    
elseif lowercase(args["task"]) == "importancesampling"
    prior = distprod(;priors...)
    posterior = PosteriorMeasure(likelihood, prior)
    #init_samples =  make_prior_samples(posterior, 1_000)

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
        result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir=name)

    elseif lowercase(args["task"]) == "scan"
        result = Newtrinos.scan(likelihood, priors, vars_to_scan, p)
    end 
    FileIO.save(name * ".jld2", Dict("result" => result))

    if args["plot"]
        using CairoMakie
        fig = Figure()
        ax = Axis(fig[1,1])
        plot!(ax, result)
        ax.xlabel = String(collect(keys(vars_to_scan))[1])

        if length(vars_to_scan) == 1
            ax.ylabel = "-2ΔLLH"
        else
            ax.ylabel = String(collect(keys(vars_to_scan))[2])
        end
        ax.title = args["task"] * ": " * join(args["experiments"], " + ")
        save(name * ".png", fig)
    end
end
