using Turing, LinearAlgebra, Distributions, Random, ADTypes

# Custom lognormal distribution that exploits the sparse structure of the covariance (Σ ∗ P⁻¹)
struct MvNormalPrior{T, R, S} <: ContinuousMultivariateDistribution
    μ::Vector{T} # mean vector (this is always 0 in practice, can remove later for a slight performance increase)
    L::AbstractArray{R} # cholesky decomposition of residual correlation matrix Σ
    P_inv_root::Vector{S} # vector of size K*J containing the square root of the nonzero elements of P⁻¹ (recall that P is diagonal as a modelling assumption, at least conditional on hyperpar)
    J::Int # number of parameters for each response variable (#params per block)
    K::Int # dimension of response (number of blocks)
end

# New version using reshaped arrays
function Distributions.rand(rng::Random.AbstractRNG, d::MvNormalPrior)
    N = d.J*d.K
    u = rand(rng, Normal(), N)
    U = reshape(u, (d.J, d.K))
    B = similar(U)
    mul!(B, U, d.L') # this is more efficient if L is stored as a lower triangular matrix
    β = vec(B)
    return @.(d.P_inv_root * β + d.μ)
end

# logpdf (up to proportionality) implementation that exploits sparsity of the Kronecker product between Σ and I_K
function Distributions.logpdf(d::MvNormalPrior, β::AbstractArray{T, M}) where {M, T<:Real}
    #log_det_L = sum(log, abs.(diag(d.L))) # this is multiplied by 2*1/2 in the loglikelihood expression [det(Σ) = det(L)^2].
    log_det_L = logdet(d.L)
    
    # compute quadratic form
    β_tilde = @. 1.0/d.P_inv_root * β # multiply by sqrt(P)
    B_tilde = reshape(β_tilde, (d.J, d.K))
    #g = vec(B_tilde*inv(d.L)')
    g = vec(transpose(d.L) \ transpose(B_tilde))
    #quadratic_form = dot(g, g)
    quadratic_form = sum(abs2, g)

    return -d.J*d.K*log_det_L - 0.5 * quadratic_form
end

Distributions.length(d::MvNormalPrior) = length(d.μ)

Turing.bijector(d::MvNormalPrior) = identity

# Now make a distribution type for the likelihood.

struct MvNormalLikelihood{T, R, S, U} <: ContinuousMultivariateDistribution # note: elements of F can have a different type
    F::Matrix{T} # design matrix
    β::AbstractArray{R} # regression coefficient vector
    L::AbstractArray{S} # cholesky decomposition of residual correlation matrix Σ
    P_inv_root::Vector{U} # vector of the standardization matrix in the marginal Gaußian inverse copula
    J::Int # number of parameters for each response variable (#params per block)
    K::Int # dimension of response (number of blocks)
end


function Distributions.logpdf(d::MvNormalLikelihood, z::AbstractArray{T, M}) where {M, T<:Real}
    Tsubp = size(d.F, 1)

    #log_det_L = sum(log, abs.(diag(d.L))) # this is multiplied by 2*1/2 in the loglikelihood expression [det(Σ) = det(L)^2].
    log_det_L = logdet(d.L)

    #S_vec_inv = sqrt.(1.0 .+ vec(F_tilde * d.P_inv_root'))
    F_tilde = d.F .^2
    S_vec_inv = sqrt.(1.0 .+ vec(F_tilde * reshape(d.P_inv_root, (d.J, d.K))))

    # compute Xβ (after reshaping β)
    B_tilde = d.F * reshape(d.β, (d.J, d.K))
    G_tilde = reshape(S_vec_inv .* z - vec(B_tilde), (Tsubp, d.K))
    g = vec(transpose(d.L) \ transpose(G_tilde))
    quadratic_form = sum(abs2, g)

    return -Tsubp * log_det_L - 0.5 * quadratic_form
end

# Try using Turing.@addlogprob! instead
function mvnormalloglikelihood(z::AbstractArray{V}, F::AbstractArray{T, M}, β::AbstractArray{R}, L::AbstractArray{S}, P_inv_root::AbstractArray{U}, J::Int, K::Int) where {M, T<:Real, R, S, U, V}
    Tsubp = size(F, 1)

    log_det_L = logdet(L)

    # compute standardization matrix S
    F_tilde = F .^2 
    S_vec_inv = sqrt.(1.0 .+ vec(F_tilde * reshape(P_inv_root, (d.J, d.K))))

    # compute Xβ (after reshaping β)
    B_tilde = F * reshape(β, (J, K))
    G_tilde = reshape(S_vec_inv .* z - vec(B_tilde), (Tsubp, K))
    g = vec(transpose(d.L) \ transpose(G_tilde))
    quadratic_form = sum(abs2, g)

    return -Tsubp * log_det_L - 0.5 * quadratic_form
end

# This is currently not working, but it is not necessary to implement this correctly to be able to sample from the posterior.
#= function Distributions.rand(rng::Random.AbstractRNG, d::MvNormalLikelihood)
    # Create the F matrix based on simulations instead!!!
    Tsubp = size(d.F, 1)

    F_tilde = repeat(d.F, d.K, 1) .* d.P_inv_root
    S_vec = 1.0 ./sqrt.(1.0 .+ dot.(eachrow(F_tilde), eachcol(F_tilde))) # this can be differentiated by ReverseDiff

    # Add mean and return
    Z_tilde = rnorm(Tsubp)
    B = reshape(d.β, (d.J, d.K))
    return vec(Z_tilde).+ vec(d.F*B)
end =#

Distributions.length(d::MvNormalLikelihood) = size(d.F, 1)

# Let J be the number of covariates per variable
# We assume here that the marginal models have already been fitted 

@model function implicit_var(z::Vector{T}, F::Matrix{T}, J::Int, K::Int) where T
    Tsubp = size(F, 1)

    # Priors on τ, Σ:
    #τ ~ filldist(InverseGamma(3, 3), K*J)
    τ = [1.0]
    #L = reshape([1.0], 1, 1)
    #Σ ~ LKJ(K, 1.0)
    L = LinearAlgebra.cholesky(diagm([1.0]))
    #L ~ LKJCholesky(K, 1.0)

    # Prior on β:
    P_inv_root = @.(1.0/sqrt(τ))
    β ~ MvNormalPrior(zeros(typeof(τ[1]), K*J), L.L, P_inv_root, J, K)

    # Likelihood:
    z ~ MvNormalLikelihood(F, β, L.L, P_inv_root, J, K)
end

# predict y_{t+1,j} based on posterior samples from β and τ (univariate for now). NB! does not predict the joint distribution, only marginals...
function predict_obs(rng::Random.AbstractRNG, kdest, x, β, τ, J, K)
    N_mc = 200
    y_pred = Matrix{Float64}(undef, N_mc, K)
    kdest_qf = InterpKDEQF(kdest)
    for j = 1:N_mc
        i = rand(rng, DiscreteUniform(1, size(β, 1)))            # sample a random row
        P_inv_root = 1.0 ./ sqrt.(τ[i,:])
        #s = 1.0 ./ sqrt.(1.0 .+ x.^2 * P_inv_root)
        #z_pred_t = rand(Normal(s[1]*dot(x, β[i,:]), s[1]))
        s = 1.0 ./ sqrt.( 1.0 .+ vec( x' .^2 * reshape(P_inv_root, (J, K)) ) )
        μ = s .* vec(x' * reshape(β[i,:], J, K))
        z_pred_t = Vector{Float64}(undef, K)
        for k in 1:K
            z_pred_t[k] = rand(rng, Normal(μ[k], s[k]))
        end
        y_pred[j,:] = quantile.(kdest_qf, cdf.(Normal(), z_pred_t))
    end
    return mean(y_pred; dims=1) # return column means
end