# Newtrinos.jl

**Newtrinos.jl** is a Julia package for the **global analysis of neutrino data**. It is fully open source and free to use under the MIT license.

The package is built to support flexible and modular analysis of neutrino physics, combining experimental data with physics models and inference tools. It supports both **Frequentist** (profile likelihood) and **Bayesian** (nested sampling, importance sampling) inference methods.

The key features of the package are: 

- **More than 30 data sets**: atmospheric (IceCube DeepCore, Super-Kamiokande, KM3NeT/ORCA, IceCube Upgrade), accelerator (T2K, NOvA, MINOS), reactor (Daya Bay, KamLAND, JUNO, TAO), solar (Homestake, GALLEX/GNO, SAGE, SNO, Super-Kamiokande I–IV, Borexino Phases I–III), CEvNS (COHERENT CsI/LAr), and neutrino mass (KATRIN, GERDA) — see [Experiments](tutorials/experiments.md)
- **Modular physics**: Oscillations (3-flavour, sterile, ADD, dark dimensions; Earth and solar matter effects with SI and NSI), atmospheric, solar and reactor fluxes, and cross-sections, including a correlated cross-section model shared between experiments for global fits
- **Validated**: the implementations are compared with the publications on the [verifications site](https://newtrinos-org.github.io/newtrinos-verifications/)
- **Fully differentiable**: All code supports ForwardDiff automatic differentiation for gradient-based optimization
- **Scalable**: Threaded and distributed parallelism for profile likelihood scans

## Typical Workflow
An example of a typical workflow for a global analysis of Newtrinos.jl is visualized in the below picture. The layers communicate through two well-defined interfaces: NamedTuples of parameters and priors, and callable functions stored in structs.

![png](Workflow_Diagram.png)

## Julia programming language

The package is written in Julia. If you're not yet familiar with Julia and want to learn more about the language, here are a few resources to get started:
- the [Julia Website](https://julialang.org/) offers many links to introductory videos and written tutorials. 
- there also exists a [MATLAB-Python-Julia cheatsheet](https://cheatsheets.quantecon.org/)
- this [article](https://www.stochasticlifestyle.com/why-numba-and-cython-are-not-substitutes-for-julia/) explains how Julia adresses several fundamental challenges inherent to scientific high-performance computing

## References

Software paper:

- [Newtrinos.jl: A Julia Package for Global Analysis of Neutrino Data](https://doi.org/10.21105/joss.09644) — Journal of Open Source Software, 11(125), 9644

Newtrinos.jl has been used to produce the results presented in:

- [Testing the number of neutrino species with a global fit of neutrino data](https://arxiv.org/abs/2402.00490) — Phys.Rev.D 109 (2024) 9, 095016
- [Constraints on non-unitary neutrino mixing in light of atmospheric and reactor neutrino data](https://arxiv.org/abs/2407.20388) — JHEP 05 (2025) 130
- [A neutrino data analysis of extra-dimensional theories with massive bulk fields](https://arxiv.org/abs/2508.04274) — Phys.Rev.D 112 (2025) 5, 055009
- [Atmospheric Neutrino Oscillations: the Full Picture](https://arxiv.org/abs/2606.09714) — Phys.Rev.D 114 (2026) 5, 053005
- [Searching for the N-naturalness tower of neutrinos](https://arxiv.org/abs/2607.14243) — Phys.Rev.D 114 (2026) 7, 075022
