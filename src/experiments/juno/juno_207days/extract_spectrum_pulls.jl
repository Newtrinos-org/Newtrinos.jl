using FileIO
using DensityInterface
using BAT
using Distributions
using MeasureBase
using Newtrinos

db = Newtrinos.dayabay.configure()
p = Newtrinos.get_params((;db,))
priors = Newtrinos.get_priors((;db,))
likelihood = Newtrinos.generate_likelihood((;db,))

llh(x) = logdensityof(PosteriorMeasure(likelihood, distprod(;priors...)), x)

wrapped_llh, kept, skipped = Newtrinos.Helpers.wrap_func(llh, mle[3], [:spectrum_pulls])

hessian = Newtrinos.hessian(wrapped_llh, kept)
covariance = -inv(hessian)

FileIO.save("spectrum_pulls_cov.jld2", Dict("mle"=>kept.spectrum_pulls, "cov"=>covariance))
