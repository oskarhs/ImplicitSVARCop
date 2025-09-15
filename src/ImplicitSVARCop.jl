module ImplicitSVARCop

# Package imports
using Distributions, Random, LinearAlgebra, Pigeons
import Polyester: @batch
import LoopVectorization: @turbo, vsum, vmap
import SparseArrays: sparse
import LogDensityProblems
using ProgressMeter: Progress, next!
import BSplineKit: BSplineOrder, BSplineBasis
import ForwardDiff
import DifferentiationInterface: prepare_jacobian, value_and_jacobian, AutoForwardDiff, Constant

include(joinpath("..", "..", "BandwidthSelectors.jl", "src", "BandwidthSelectors.jl"))
using .BandwidthSelectors
export fit, SSVKernel, UnivariateKDE, InterpKDEQF, InterpKDECDF

include(joinpath("helpers", "Bspline_basis_matrix.jl"))
export B_spline_basis_matrix

include(joinpath("helpers", "fit_marginals.jl"))
export fit_marginals


include(joinpath("conditionals", "conditional_beta.jl"))
#include(joinpath("conditionals_igprior", "conditional_beta.jl"))
export sample_conditional_β, logp_conditional_β, grad_logp_conditional_β

include(joinpath("conditionals", "conditional_tau.jl"))
#include(joinpath("conditionals_igprior", "conditional_tau.jl"))
export sample_conditional_τ, logp_conditional_τ_k, grad_logp_conditional_τ_k, hess_logp_conditional_τ_k, logp_conditional_τ, grad_logp_conditional_τ, sample_mh_τ_all

include(joinpath("conditionals", "conditional_xi.jl"))
#include(joinpath("conditionals_igprior", "conditional_xi.jl"))
export logp_conditional_ξ, grad_logp_conditional_ξ

include(joinpath("helpers", "helpers_g.jl"))
export compute_Σ
include(joinpath("conditionals", "conditional_g.jl"))
#include(joinpath("conditionals_igprior", "conditional_g.jl"))
export logp_conditional_γ, grad_logp_conditional_γ

include("VARModel.jl")
export VARModel, get_varsymbols

include(joinpath("conditionals", "conditional_joint.jl"))
#include(joinpath("conditionals_igprior", "conditional_joint.jl"))
export logp_joint, logp_and_grad_joint, grad_logp_joint

include(joinpath("conditionals", "conditional_joint_autodiff.jl"))
#include(joinpath("conditionals_igprior", "conditional_joint_autodiff.jl"))

#include(joinpath("conditionals", "conditional_joint_xi_gamma.jl"))
#export logp_joint_xi_gamma, grad_logp_joint_xi_gamma, grad_logp_joint_xi_gamma_nuts

include(joinpath("variational", "grad_elbo.jl"))
include(joinpath("variational", "adadelta_step.jl"))
include(joinpath("variational", "VIPosterior.jl"))
include(joinpath("variational", "fitBayesianSVARCopVI.jl"))
export VIPosterior, fitBayesianSVARCopVI, logpdf, predict_response

include(joinpath("mcmc", "grad_and_logp_elbo_conditional_xi_gamma.jl"))
include(joinpath("mcmc", "sample_conditional_xi_gamma.jl"))
include(joinpath("mcmc", "composite_gibbs_vi.jl"))
export composite_gibbs_vi

include(joinpath("mcmc2", "composite_gibbs_mh.jl"))
export composite_gibbs_mh


end # end module