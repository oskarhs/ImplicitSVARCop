"""
    logp_conditional_γ(γ::AbstractVector{<:Real}, inv_Σ::AbstractMatrix, β::AbstractArray, P_root::AbstractArray, inv_S::AbstractArray, z::AbstractArray, F::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)

Function for computing the log-conditional density log p(γ | ⋯) with respect to γ.

# Arguments
* `γ`: Unconstrained parameter vector that parametrizes the covariance matrix.
* `inv_Σ`: The inverse of the covariance matrix corresponding to γ, e.g. inv_Σ = inv(Σ(v)).
"""
function logp_conditional_γ(γ::AbstractVector{<:Real}, C::AbstractMatrix, transformed_dist, P_rootβrs::AbstractArray, Mlik::AbstractArray, J::Int, K::Int, Tsubp::Int)

    #C = transpose(inv(bij(γ))) # inv(Σ) = inv(L * L') = inv(L') * inv(L) = C * C', C = inv(L')
    #C = cholesky(inv_Σ).L

    logp = 0.0

    # Contribution from p(γ)
    logp += logpdf(transformed_dist, γ)

    # Contribution from p(β | Σ(γ))
    logp += J * logdet(C)
    temp1 = P_rootβrs * C
    logp -= 0.5*sum(abs2, temp1)

    # contribution from likelihood
    logp += Tsubp * logdet(C)
    #temp2 = (reshape(inv_Sz, (Tsubp, K)) - XB) * C
    temp2 = Mlik * C
    logp -= 0.5*sum(abs2, temp2)
    
    return logp
end


"""
    grad_logp_conditional_γ(γ::Real, inv_Σ::AbstractMatrix, β::AbstractArray, P_root::AbstractArray, inv_S::AbstractArray, z::AbstractArray, F::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)

Function for computing the gradient of the log-conditional density log p(γ | ⋯) with respect to γ.


# Arguments
* `γ`: Unconstrained parameter vector that parametrizes the correlation matrix.
* `C`: Inverse of 
"""
function grad_logp_conditional_γ(γ::AbstractVector{<:Real}, C::AbstractMatrix, transformed_dist, to_chol, P_rootβrs::AbstractArray, vec_MliktMlik_t::AbstractArray, J::Int, K::Int, Tsubp::Int)
    inv_Σ = C * transpose(C)

    Dldv = vec(ForwardDiff.jacobian(γ -> vec(Matrix(to_chol(γ))), γ))
    #Dldv = [0.0, sech(γ[1])^2, sech(γ[1])^2, 0.0]
    temp_0 = sum(Dldv .* vec(inv_Σ), dims=1)
    temp_02 = kron(inv_Σ, inv_Σ) * Dldv

    # Contribution from p(γ)
    grad_log_dens_γ = ForwardDiff.gradient(γ -> logpdf(transformed_dist, γ), γ)
    #grad_log_dens_γ = [-2*tanh(γ[1])]


    # Contribution from p(β |γ, ξ)
    temp_1 = -0.5*J*temp_0 # from logdeterminant of Σ(γ)
    t1 = P_rootβrs
    temp_2 = 0.5 * (vec(t1' * t1)') * temp_02
    temp_2 = ifelse(temp_2 isa Vector, temp_2, [temp_2])

    grad_log_dens_β = temp_1 + temp_2

    # Contribution from likelihood
    temp_3 = -0.5*Tsubp * temp_0 # from logdeterminant of Σ(v)
    #Mlik = reshape(inv_Sz, (Tsubp, K)) - XB # this is needed for both the loglikelihood and the gradient, move it outside later
    #temp_4 = 0.5 * (vec(Mlik' * Mlik)') * temp_02
    temp_4 = 0.5 * vec_MliktMlik_t * temp_02
    grad_log_dens_y = temp_3 + ifelse(temp_4 isa Vector, temp_4, [temp_4])

    grad = grad_log_dens_β + grad_log_dens_γ + grad_log_dens_y
    return grad
end

struct Conditional_γ{F, G, T<:AbstractMatrix{<:Real}, S<:AbstractMatrix{<:Real}}
    transformed_dist::F
    to_chol::G
    P_rootβrs::T
    Mlik::Matrix{Float64}
    vec_MliktMlik_t::S
    J::Int
    K::Int
    Tsubp::Int
end
LogDensityProblems.dimension(cond::Conditional_γ) = 1
LogDensityProblems.capabilities(::Type{<:Conditional_γ}) = LogDensityProblems.LogDensityOrder{1}() # We can provide the gradient

function LogDensityProblems.logdensity(cond::Conditional_γ, γ)
    (; transformed_dist, to_chol, P_rootβrs, Mlik, vec_MliktMlik_t, J, K, Tsubp) = cond
    C = transpose(inv(to_chol(γ).L)) # inv_Σ = C * C'
    return logp_conditional_γ(γ, C, transformed_dist, P_rootβrs, Mlik, J, K, Tsubp)
end
function LogDensityProblems.logdensity_and_gradient(cond::Conditional_γ, γ) # Can be optimized, there is some overlap with logdensity calculation
    (; transformed_dist, to_chol, P_rootβrs, Mlik, vec_MliktMlik_t, J, K, Tsubp) = cond
    C = transpose(inv(to_chol(γ).L)) # inv_Σ = C * C'
    logp = logp_conditional_γ(γ, C, transformed_dist, P_rootβrs, Mlik, J, K, Tsubp)
    grad = grad_logp_conditional_γ(γ, C, transformed_dist, to_chol, P_rootβrs, vec_MliktMlik_t, J, K, Tsubp)
    return logp, grad
end


"""
NEED TO WRAP THE TARGET CONDITIONAL IN A LOGDENSITYPROBLEM PARAMETERIZED BY THE VALUES OF THE OTHER PARAMETERS. THEN USE AbstractMCMC.LogDensityModel TO CREATE A NEW OBJECT IN THIS ITERATION.
"""
function abstractmcmc_sample_γ(
    rng::Random.AbstractRNG,
    sampler_γ,
    state_γ,
    γ::AbstractVector{<:Real},
    transformed_dist,
    to_chol,
    P_rootβrs::AbstractArray{<:Real},
    Mlik::AbstractArray{<:Real},
    vec_MliktMlik::AbstractArray{<:Real},
    J::Int,
    K::Int,
    Tsubp::Int;
    n_adapts::Int
)   
    # Create target LogDensityModel
    Cond = AbstractMCMC.LogDensityModel(Conditional_γ(transformed_dist, to_chol, P_rootβrs, Mlik, vec_MliktMlik, J, K, Tsubp))
    if isnothing(state_γ)
        transition_γ, state_γ = AbstractMCMC.step(rng, Cond, sampler_γ; initial_params=γ, n_adapts=n_adapts)
    else
        # Update state
        state_γ = AbstractMCMC.setparams!!(Cond, state_γ, γ)

        # Sample γ
        transition_γ, state_γ = AbstractMCMC.step(rng, Cond, sampler_γ, state_γ; n_adapts=n_adapts)
    end 
    γ = AbstractMCMC.getparams(state_γ)
    return γ, state_γ
end