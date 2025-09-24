function compute_inv_Σ_ρ(atanh_ρ::AbstractVector{<:Real})
    ρ = tanh(atanh_ρ[1])
    inv_Σ = Symmetric([1.0 -ρ; -ρ 1.0]) / (1-ρ^2)
    return inv_Σ
end


"""
    logp_conditional_ρ(γ::Real, inv_Σ::AbstractMatrix, β::AbstractArray, P_root::AbstractArray, inv_S::AbstractArray, z::AbstractArray, F::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)

Function for computing the log-conditional density log p(γ | ⋯) with respect to γ.

# Arguments
* `ρ`: Unconstrained parameter vector that parametrizes the covariance matrix.
* `inv_Σ`: The inverse of the covariance matrix corresponding to γ, e.g. inv_Σ = inv(Σ(v)).
"""
function logp_conditional_ρ(atanh_ρ::AbstractVector{<:Real}, inv_Σ::AbstractMatrix, β::AbstractVector, P_root::AbstractArray, Mlik::AbstractArray, J::Int, K::Int, Tsubp::Int)
    η = 1.0 # Uniform prior on ρ

    C = cholesky(inv_Σ).L

    logp = 0.0

    # Contribution from p(γ)
    logp += sech(atanh_ρ[1])^(2*η)

    # Contribution from p(β | Σ(γ))
    logp += J * logdet(C)
    temp1 = reshape(P_root * β, (J, K)) * C
    logp -= 0.5*sum(abs2, temp1)

    # contribution from likelihood
    logp += Tsubp * logdet(C)
    #temp2 = (reshape(inv_Sz, (Tsubp, K)) - XB) * C
    temp2 = Mlik * C
    logp -= 0.5*sum(abs2, temp2)
    
    return logp
end

function logp_conditional_ρ_autodiff(atanh_ρ::AbstractVector{<:Real}, β::AbstractVector, P_root::AbstractArray, Mlik::AbstractArray, J::Int, K::Int, Tsubp::Int)
    inv_Σ = compute_inv_Σ_ρ(atanh_ρ)
    return logp_conditional_ρ(atanh_ρ, inv_Σ, β, P_root, Mlik, J, K, Tsubp)
end

"""
    grad_logp_conditional_ρ(ρ::Real, inv_Σ::AbstractMatrix, β::AbstractArray, P_root::AbstractArray, inv_S::AbstractArray, z::AbstractArray, F::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)

Function for computing the gradient of the log-conditional density log p(ρ | ⋯) with respect to ρ.

# Arguments
* `atanh_ρ`: Unconstrained parameter vector that parametrizes the covariance matrix.
* `inv_Σ`: The inverse of the covariance matrix corresponding to γ, e.g. inv_Σ = inv(Σ(v)).
"""
function grad_logp_conditional_ρ(atanh_ρ::AbstractVector{<:Real}, β::AbstractArray, P_root::AbstractArray, vec_MliktMlik_t::AbstractArray, J::Int, K::Int, Tsubp::Int)
    η = 1

    #Σ = Symmetric([1.0 tanh(atanh_ρ[1]); tanh(atanh_ρ[1]) 1.0])
    inv_Σ = compute_inv_Σ_ρ(atanh_ρ)
    Dldv = [0.0, sech(atanh_ρ[1])^2, sech(atanh_ρ[1])^2, 0.0]
    #Dldv = deriv_Σ_g03(γ, K, M)
    temp_0 = sum(Dldv .* vec(inv_Σ), dims=1)
    temp_02 = kron(inv_Σ, inv_Σ) * Dldv

    # Contribution from p(ρ)
    grad_log_dens_ρ = -2*η*sech(atanh_ρ[1])^(2*η)*tanh(atanh_ρ[1])

    # Contribution from p(β |ρ, ξ)
    temp_1 = -0.5*J*temp_0 # from logdeterminant of Σ(ρ)
    t1 = reshape(P_root * β, (J, K))
    temp_2 = 0.5 * (vec(t1' * t1)') * temp_02

    grad_log_dens_β = temp_1[1] + temp_2
    grad_log_dens_β + grad_log_dens_ρ

    # Contribution from likelihood
    temp_3 = -0.5*Tsubp * temp_0 # from logdeterminant of Σ(v)
    #Mlik = reshape(inv_Sz, (Tsubp, K)) - XB # this is needed for both the loglikelihood and the gradient, move it outside later
    #temp_4 = 0.5 * (vec(Mlik' * Mlik)') * temp_02
    temp_4 = 0.5 * vec_MliktMlik_t * temp_02
    grad_log_dens_y = temp_3[1] + temp_4

    grad = grad_log_dens_β + grad_log_dens_ρ + grad_log_dens_y
    return [grad]
end

struct Conditional_ρ{T<:AbstractMatrix{<:Real}, S<:AbstractMatrix{<:Real}}
    β::Vector{Float64}
    P_root::T
    Mlik::Matrix{Float64}
    vec_MliktMlik_t::S
    J::Int
    K::Int
    Tsubp::Int
end
LogDensityProblems.dimension(cond::Conditional_ρ) = 1
LogDensityProblems.capabilities(::Type{<:Conditional_ρ}) = LogDensityProblems.LogDensityOrder{1}() # We can provide the gradient

function LogDensityProblems.logdensity(cond::Conditional_ρ, atanh_ρ)
    (; β, P_root, Mlik, vec_MliktMlik_t, J, K, Tsubp) = cond
    return logp_conditional_ρ_autodiff(atanh_ρ, β, P_root, Mlik, J, K, Tsubp)
end
function LogDensityProblems.logdensity_and_gradient(cond::Conditional_ρ, atanh_ρ) # Can be optimized, there is some overlap with logdensity calculation
    (; β, P_root, Mlik, vec_MliktMlik_t, J, K, Tsubp) = cond
    logp = logp_conditional_ρ_autodiff(atanh_ρ, β, P_root, Mlik, J, K, Tsubp)
    grad = grad_logp_conditional_ρ(atanh_ρ, β, P_root, vec_MliktMlik_t, J, K, Tsubp)
    return logp, grad
end


"""
NEED TO WRAP THE TARGET CONDITIONAL IN A LOGDENSITYPROBLEM PARAMETERIZED BY THE VALUES OF THE OTHER PARAMETERS. THEN USE AbstractMCMC.LogDensityModel TO CREATE A NEW OBJECT IN THIS ITERATION.
"""
function abstractmcmc_sample_ρ(
    rng::Random.AbstractRNG,
    sampler_ρ,
    state_ρ,
    atanh_ρ::AbstractVector{<:Real},
    β::AbstractArray{<:Real},
    P_root::AbstractArray{<:Real},
    Mlik::AbstractArray{<:Real},
    vec_MliktMlik::AbstractArray{<:Real},
    J::Int,
    K::Int,
    Tsubp::Int;
    n_adapts::Int
)   
    # Create target LogDensityModel
    Cond = AbstractMCMC.LogDensityModel(Conditional_ρ(β, P_root, Mlik, vec_MliktMlik, J, K, Tsubp))
    if isnothing(state_ρ)
        transition_ρ, state_ρ = AbstractMCMC.step(rng, Cond, sampler_ρ; initial_params=atanh_ρ, n_adapts=n_adapts)
    else
        # Update state
        state_ρ = AbstractMCMC.setparams!!(Cond, state_ρ, atanh_ρ)

        # Slice sample log_ξ
        transition_ρ, state_ρ = AbstractMCMC.step(rng, Cond, sampler_ρ, state_ρ; n_adapts=n_adapts)
    end
    atanh_ρ = AbstractMCMC.getparams(state_ρ)
    return atanh_ρ, state_ρ
end