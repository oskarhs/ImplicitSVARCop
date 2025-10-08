# Put functions for computing log full-conditionals and their gradients here
# (except β, as this is samples in a Gibbs step)
# Here, θ is the unconstrained parameter vector
# θ[1:K*J] are transformed β (identity)
# θ[K*J+1:2*K*J] are transformed ξ
# θ[2*K*J+1:2*K*J+K] are tramsformed τ
# (later) θ[2*K*J+K+1:2*K*J+K+num_prior_Σ]
# this is the way the untransformed parameter vector has been encoded in Klein et al. (2025)

"""
    stable_logplus1exp(x::Real)

Numerically stable variant of log(1 + exp(x)) for both positive and negative x.
"""
stable_logplus1exp(x::Real) = ifelse(x ≥ 0, x + log(1 + exp(-x)), log(1 + exp(x)))

"""
    stable_sigmoid(x::Real)

Numerically stable variant of exp(x) / (1 + exp(x)) for positive and negative x.
"""
stable_sigmoid(x::Real) = ifelse(x ≥ 0, 1/(1 + exp(-x)), exp(x)/(1 + exp(x)))

"""
    logp_conditional_τ_k(log_τ_k::Real, log_ξ_k::AbstractArray{<:Real}, J::Int, K::Int)

Evaluate the full conditional of log(τ_k).
"""
function logp_conditional_τ_k(log_τ_k::Real, log_ξ_k::AbstractArray{<:Real}, J::Int, K::Int)
    logp = -(J-1.0)*log_τ_k - stable_logplus1exp(2.0*log_τ_k) - sum( stable_logplus1exp.(2.0*log_ξ_k .- 2.0*log_τ_k) ) # some minor optimization by factoring this more cleverly can be done later (this is not so critical for now)
    return logp
end

"""
    grad_logp_conditional_τ_k(log_τ_k::Real, log_ξ_k::AbstractArray{<:Real}, J::Int, K::Int)

Evaluate the derivative of log p(log(τ_k) | ⋯) with respect to log(τ_k).
"""
function grad_logp_conditional_τ_k(log_τ_k::Real, log_ξ_k::AbstractArray{<:Real}, J::Int, K::Int)
    #grad = -(J-1.0) - 2.0 * exp(2.0*log_τ_k) / (1.0 + exp(2.0*log_τ_k)) + 2.0 * sum( exp.(2.0*log_ξ_k .- 2.0*log_τ_k) ./ (1.0 .+ exp.(2.0*log_ξ_k .- 2.0*log_τ_k)) )
    grad = -(J-1.0) - 2.0 * stable_sigmoid(2.0 * log_τ_k) + 2.0 * sum( stable_sigmoid.(2.0*log_ξ_k .- 2.0*log_τ_k) )
    return grad
end

"""
    logp_conditional_τ(log_τ::AbstractArray{<:Real}, log_ξ_k::AbstractArray{<:Real}, J::Int, K::Int)

Evaluate the full joint conditional of log(τ).
"""
function logp_conditional_τ(log_τ::AbstractArray{<:Real}, log_ξ::AbstractArray{<:Real}, J::Int, K::Int)
    logp = 0.0
    for k in 1:K
        logp += logp_conditional_τ_k(log_τ[k], view(log_ξ, (k-1)*J + 1:k*J), J, K)
    end
    return logp
end

"""
    grad_logp_conditional_τ(log_τ::AbstractArray{<:Real}, log_ξ_k::AbstractArray{<:Real}, J::Int, K::Int)

Evaluate the derivative of log p(log(τ) | ⋯) with respect to log(τ).
"""
function grad_logp_conditional_τ(log_τ::AbstractArray{<:Real}, log_ξ::AbstractArray{<:Real}, J::Int, K::Int)
    grad = Vector{eltype(log_τ)}(undef, K)
    for k = 1:K
        grad[k] = grad_logp_conditional_τ_k(log_τ[k], view(log_ξ, (k-1)*J + 1:k*J), J, K)
    end
    return grad
end

"""
    hess_logp_conditional_τ_k(log_τ_k::Real, log_ξ_k::AbstractArray{<:Real}, J::Int, K::Int)

Evaluate the second derivative of log p(log(τ_k) | ⋯) with respect to log(τ_k).
"""
function hess_logp_conditional_τ_k(log_τ_k::Real, log_ξ_k::AbstractArray{<:Real}, J::Int, K::Int)
    hess_logp = - 4.0 * exp(2.0*log_τ_k) / (1.0 + exp(2.0*log_τ_k))^2 - 4.0 * sum( exp.(2.0*log_ξ_k .- 2.0*log_τ_k) ./ (1.0 .+ exp.(2.0*log_ξ_k .- 2.0*log_τ_k)).^2 )
    return hess_logp
end

"""
    sample_mh_τ_k(rng::Random.AbstractRNG, log_τ_k::Real, log_ξ_k::AbstractArray{<:Real}, J::Int, K::Int)

Perform a Metropolis-Hastings step for log(τ_k), using a 2nd order Taylor expansion around the current value to form a Gaussian proposal.
"""
function sample_mh_τ_k(rng::Random.AbstractRNG, log_τ_k::Real, log_ξ_k::AbstractArray{<:Real}, J::Int, K::Int) # optimize implementation later, for now just make sure that this works
    # compute gradient, hessian for proposal:

    grad_prop = grad_logp_conditional_τ_k(log_τ_k, log_ξ_k, J, K)
    hess_prop = hess_logp_conditional_τ_k(log_τ_k, log_ξ_k, J, K)
    var_prop = max(-1.0 / hess_prop, 1e-10)             # compute mean and variance of proposal distribution
    mean_prop = log_τ_k - grad_prop / hess_prop
    log_τ_k_prop = rand(rng, Normal(mean_prop, sqrt(var_prop))) # generate proposal
    
    grad_rev = grad_logp_conditional_τ_k(log_τ_k_prop, log_ξ_k, J, K)
    hess_rev = hess_logp_conditional_τ_k(log_τ_k_prop, log_ξ_k, J, K)
    var_rev = max(-1.0 / hess_rev, 1e-10)
    mean_rev = log_τ_k_prop - grad_rev / hess_rev
    
    # NB! calling logpdf here is slightly slower than implementing the logpdf up to additive constants, but makes for more readable code
    if isnan(var_rev)
        return log_τ_k
    end
    log_acc = logpdf(Normal(mean_rev, sqrt(var_rev)), log_τ_k) - logpdf(Normal(mean_prop, sqrt(var_prop)), log_τ_k_prop)  # MH ratio contribution from proposal and the reverse proposal
    log_acc += logp_conditional_τ_k(log_τ_k_prop, log_ξ_k, J, K) - logp_conditional_τ_k(log_τ_k, log_ξ_k, J, K)               # MH ratio contribution from target conditional

    log_τ_k_new = ifelse(log(rand(rng)) < log_acc, log_τ_k_prop, log_τ_k)
    return log_τ_k_new
end

# At a later point in time: perhaps change this to take in vector of log_τ_k and vector of log_ξ_k
function sample_mh_τ_all(rng::Random.AbstractRNG, log_τ::AbstractVector, log_ξ::AbstractVector, J::Int, K::Int)
    for k = 1:K
        log_τ[k] = sample_mh_τ_k(rng, log_τ[k], log_ξ[(k-1)*J+1:k*J], J, K)
    end
    return log_τ
end


function sample_τ_k_ars(rng::Random.AbstractRNG, log_τ_k::Real, log_ξ_k::AbstractArray{<:Real}, J::Int, K::Int)
    maxiter = 20
    δ = 1
    for i in 1:maxiter
        log_τ_k_lower = log_τ_k - δ
        log_τ_k_upper = log_τ_k + δ
        δ = 2*δ
    end
    if δ ≥ 2^maxiter
        throw(error("Unable to find bracketing interval for the maximum of log π(log τ_k | ⋯)."))
    end

    obj = ARS.Objective(log_τ_k -> logp_conditional_τ_k(log_τ_k, log_ξ_k, J, K), log_τ_k -> grad_logp_conditional_τ_k(log_τ_k, log_ξ_k, J, K))
end