module ImplicitSVARCop

# Package imports
using Distributions, Random, LinearAlgebra, SliceSampling, ARS
import LoopVectorization: @turbo, vsum, vmap
import SparseArrays: sparse
import LogDensityProblems
import AbstractMCMC
using ProgressMeter: Progress, next!
import BSplineKit: BSplineOrder, BSplineBasis
import ForwardDiff
import DifferentiationInterface: prepare_jacobian, value_and_jacobian, AutoForwardDiff, Constant
import Bijectors

# Required in order to use within Gibbs sampler
const MetaSliceSamplers = Union{
    SliceSampling.GibbsState,
    SliceSampling.HitAndRunState
}

AbstractMCMC.getparams(state::MetaSliceSamplers) = state.transition.params


include(joinpath("..", "BandwidthSelectors.jl", "src", "BandwidthSelectors.jl"))
using .BandwidthSelectors
export fit, SSVKernel, ISJ, UnivariateKDE, InterpKDEQF, InterpKDECDF, ISJ



include("VARModel.jl")
export VARModel, get_varsymbols

include(joinpath("helpers", "Bspline_basis_matrix.jl"))
export B_spline_basis_matrix

include(joinpath("helpers", "create_dummy_encoding.jl"))
export create_dummy_encoding

include(joinpath("helpers", "fit_marginals.jl"))
export fit_marginals

include(joinpath("helpers", "predict_response.jl"))
export predict_response, predict_response_plugin

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
include(joinpath("conditionals", "conditional_gamma.jl"))
#include(joinpath("conditionals_igprior", "conditional_g.jl"))
export logp_conditional_γ, grad_logp_conditional_γ

include(joinpath("conditionals", "conditional_rho.jl"))
#include(joinpath("conditionals_igprior", "conditional_g.jl"))
export logp_conditional_rho, grad_logp_conditional_rho


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
export VIPosterior, fitBayesianSVARCopVI, logpdf, predict_response, predict_response_plugin, cov

include(joinpath("variational_lkj", "grad_elbo_lkj.jl"))
include(joinpath("variational_lkj", "adadelta_step_lkj.jl"))
include(joinpath("variational_lkj", "VIPosterior_lkj.jl"))
include(joinpath("variational_lkj", "fitBayesianSVARCopVI_lkj.jl"))
export VIPosterior_lkj, fitBayesianSVARCopVI_lkj, logpdf, predict_response, predict_response_plugin, cov


include(joinpath("mcmc", "composite_gibbs_mh.jl"))
export composite_gibbs_mh

include(joinpath("mcmc", "composite_gibbs_slice.jl"))
export composite_gibbs_abstractmcmc

include(joinpath("mcmc", "composite_gibbs_slice_lkj.jl"))
export composite_gibbs_abstractmcmc_lkj


end # end module