"""
    logp_conditional_γ(γ::AbstractVector, inv_Σ::AbstractMatrix, β::AbstractArray, P_root::AbstractArray, inv_S::AbstractArray, z::AbstractArray, F::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)

Function for computing the log-conditional density log p(γ | ⋯) with respect to γ.

# Arguments
* `γ`: Unconstrained parameter vector that parametrizes the covariance matrix.
* `inv_Σ`: The inverse of the covariance matrix corresponding to γ, e.g. inv_Σ = inv(Σ(v)).
"""
function logp_conditional_γ(γ::AbstractVector, inv_Σ::AbstractMatrix, β::AbstractVector, P_root::AbstractArray, Mlik::AbstractArray, J::Int, K::Int, Tsubp::Int)
    a_γ = 3.0
    b_γ = 1.0

    C = cholesky(inv_Σ).L

    logp = 0.0

    # Contribution from p(γ)
    f = let b_γ = b_γ
        x -> log(1.0 + abs(x)/b_γ)
    end
    logp -= (a_γ + 1.0) * sum(f, γ)

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

function logp_conditional_γ_autodiff(γ::AbstractVector, β::AbstractVector, P_root::AbstractArray, Mlik::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)
    Σ = compute_Σ(γ, K, M)
    inv_Σ = inv(Σ)
    return logp_conditional_γ(γ, inv_Σ, β, P_root, Mlik, J, K, Tsubp)
end

"""
    grad_logp_conditional_γ(γ::AbstractVector, inv_Σ::AbstractMatrix, β::AbstractArray, P_root::AbstractArray, inv_S::AbstractArray, z::AbstractArray, F::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)

Function for computing the gradient of the log-conditional density log p(γ | ⋯) with respect to γ.

# Arguments
* `γ`: Unconstrained parameter vector that parametrizes the covariance matrix.
* `inv_Σ`: The inverse of the covariance matrix corresponding to γ, e.g. inv_Σ = inv(Σ(v)).
"""
function grad_logp_conditional_γ(γ::AbstractVector, β::AbstractArray, P_root::AbstractArray, vec_MliktMlik_t::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)
    a_γ = 3.0
    b_γ = 1.0

    Σ = compute_Σ(γ, K, M)
    inv_Σ = inv(Σ)
    Dldv = deriv_Σ_g03(γ, K, M)
    temp_0 = sum(Dldv .* vec(inv_Σ), dims=1)
    temp_02 = kron(inv_Σ, inv_Σ) * Dldv

    # Contribution from p(γ)
    # veclG is the first K*M - M*(M - 1) ÷ 2 elements of γ
    # elli are the last M
    num_nondiag = K*M - div(M*(M+1), 2)
    grad_log_dens_γ = similar(γ)
    for i in 1:num_nondiag # non-diagonal elements
        abs_γ_i = abs(γ[i])
        grad_log_dens_γ[i] = -(a_γ + 1) * (γ[i] / ((b_γ + abs_γ_i) * abs_γ_i))
    end
    for i in num_nondiag+1:num_nondiag+M # diagonal elements
        exp_γ_i = exp(γ[i])
        grad_log_dens_γ[i] = -(a_γ + 1.0) * ( exp_γ_i / (b_γ + exp_γ_i) - 1.0 / (a_γ + 1.0) )
    end

    # Contribution from p(β |γ, ξ)
    temp_1 = -0.5*J*temp_0 # from logdeterminant of Σ(v)
    t1 = reshape(P_root * β, (J, K))
    temp_2 = 0.5 * (vec(t1' * t1)') * temp_02

    grad_log_dens_β = vec(temp_1 + temp_2)

    # Contribution from likelihood
    temp_3 = -0.5*Tsubp * temp_0 # from logdeterminant of Σ(v)
    #Mlik = reshape(inv_Sz, (Tsubp, K)) - XB # this is needed for both the loglikelihood and the gradient, move it outside later
    #temp_4 = 0.5 * (vec(Mlik' * Mlik)') * temp_02
    temp_4 = 0.5 * vec_MliktMlik_t * temp_02
    grad_log_dens_y = vec(temp_3 + temp_4)

    grad = grad_log_dens_β + grad_log_dens_γ + grad_log_dens_y
    return grad
end


function grad_logp_conditional_γ_unpacked(γ::AbstractVector, inv_Σ::AbstractMatrix, P_rootβrs::AbstractArray,
                                          Mlik::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)
    a_γ = 3.0
    b_γ = 1.0

    Dldv = deriv_Σ_g03(γ, K, M)
    temp_0 = sum(Dldv .* vec(inv_Σ), dims=1)
    temp_02 = kron(inv_Σ, inv_Σ) * Dldv

    # veclG is the first K*M - M*(M - 1) ÷ 2 elements of γ
    # elli are the last M
    num_nondiag = K*M - div(M*(M+1), 2)
    grad_log_dens_γ = similar(γ)
    for i in 1:num_nondiag # non-diagonal elements
        abs_γ_i = abs(γ[i])
        grad_log_dens_γ[i] = -(a_γ + 1) * (γ[i] / ((b_γ + abs_γ_i) * abs_γ_i))
    end
    for i in num_nondiag+1:num_nondiag+M # diagonal elements
        exp_γ_i = exp(γ[i])
        grad_log_dens_γ[i] = -(a_γ + 1.0) * ( exp_γ_i / (b_γ + exp_γ_i) - 1.0 / (a_γ + 1.0) )
    end

    # Contribution from p(β |γ, ξ)
    temp_1 = -0.5*K*temp_0 # from logdeterminant of Σ(γ)
    t1 = P_rootβrs
    temp_2 = 0.5 * (vec(t1' * t1)') * temp_02

    grad_log_dens_β = vec(temp_1 + temp_2)

    # Contribution from likelihood
    temp_3 = -0.5*Tsubp * temp_0 # from logdeterminant of Σ(γ)
    temp_4 = 0.5 * transpose(vec(transpose(Mlik) * Mlik)) * temp_02
    grad_log_dens_y = vec(temp_3 + temp_4)

    grad = grad_log_dens_β + grad_log_dens_γ + grad_log_dens_y
    return grad
end


struct Conditional_γ{T<:AbstractMatrix{<:Real}, S<:AbstractMatrix{<:Real}}
    β::Vector{Float64}
    P_root::T
    Mlik::Matrix{Float64}
    vec_MliktMlik_t::S
    J::Int
    K::Int
    Tsubp::Int
    M::Int
    M_γ::Int
end
LogDensityProblems.dimension(cond::Conditional_γ) = cond.M_γ
LogDensityProblems.capabilities(::Type{<:Conditional_γ}) = LogDensityProblems.LogDensityOrder{1}() # We can provide the gradient

function LogDensityProblems.logdensity(cond::Conditional_γ, γ)
    (; β, P_root, Mlik, vec_MliktMlik_t, J, K, M, Tsubp, M_γ) = cond
    return logp_conditional_γ_autodiff(γ, β, P_root, Mlik, M, J, K, Tsubp)
end
function LogDensityProblems.logdensity_and_gradient(cond::Conditional_γ, γ) # Can be optimized, there is some overlap with logdensity calculation
    (; β, P_root, Mlik, vec_MliktMlik_t, J, K, M, Tsubp, M_γ) = cond
    logp = logp_conditional_γ_autodiff(γ, β, P_root, Mlik, M, J, K, Tsubp)
    grad = grad_logp_conditional_γ(γ, β, P_root, vec_MliktMlik_t, M, J, K, Tsubp)
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
    β::AbstractArray{<:Real},
    P_root::AbstractArray{<:Real},
    Mlik::AbstractArray{<:Real},
    vec_MliktMlik::AbstractArray{<:Real},
    J::Int,
    K::Int,
    Tsubp::Int,
    M::Int,
    M_γ::Int;
    n_adapts::Int
)   
    # Create target LogDensityModel
    Cond = AbstractMCMC.LogDensityModel(Conditional_γ(β, P_root, Mlik, vec_MliktMlik, J, K, Tsubp, M, M_γ))
    if isnothing(state_γ)
        transition_γ, state_γ = AbstractMCMC.step(rng, Cond, sampler_γ; initial_params=γ, n_adapts=n_adapts)
    else
        # Update state
        state_γ = AbstractMCMC.setparams!!(Cond, state_γ, γ)

        # Slice sample log_ξ
        transition_γ, state_γ = AbstractMCMC.step(rng, Cond, sampler_γ, state_γ; n_adapts=n_adapts)
    end
    γ = AbstractMCMC.getparams(state_γ)
    return γ, state_γ
end