function logp_conditional_ξ(log_ξ::AbstractArray{<:Real}, log_τ::AbstractArray{<:Real}, β::AbstractArray, inv_Σ::AbstractArray, P_root::AbstractArray, inv_S::AbstractArray, z::AbstractArray, F::AbstractArray, J::Int, K::Int, Tsubp::Int, df_ξ::Real) # remove F_sq from args later
    ξ = exp.(log_ξ) # dimension J*K
    τ = exp.(log_τ) # probably better to pass these directly here
    logp = 0.0
    for k in 1:K
        for j in 1:J
            logp += log_ξ[(k-1)*J + j]-0.5*(df_ξ + 1.0) * log(1.0 + (ξ[(k-1)*J + j] / τ[k])^2 / df_ξ) # contribution from prior
        end
    end

    # contribution from logdeterminants
    logp += logdet(inv_S) + logdet(P_root)
    C = cholesky(inv_Σ).L

    # quadratic form in prior
    temp1 = reshape(P_root * β, (J, K)) * C
    logp -= 0.5*sum(abs2, temp1)

    # quadratic form in likelihood
    XB = F * reshape(β, (J, K))
    temp2 = (reshape(inv_S * z, (Tsubp, K)) - XB) * C
    logp -= 0.5*sum(abs2, temp2)
    return logp
end

# Version of the above to be tested with autodiff
function logp_conditional_ξ_autodiff(log_ξ::AbstractArray{<:Real}, log_τ::AbstractArray{<:Real}, β::AbstractArray, inv_Σ::AbstractArray, z::AbstractArray,
                                     F::AbstractArray, F_sq::AbstractMatrix, XB::AbstractMatrix, J::Int, K::Int, Tsubp::Int, df_ξ::Real) # remove F_sq from args later
    ξ = exp.(log_ξ) # dimension J*K
    τ = exp.(log_τ)
    P_root = Diagonal(1.0 ./ ξ)
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( abs2.(ξ), (J, K) ) ) ) ) # should square P_root here.
    logp = 0.0
    for k in 1:K
        for j in 1:J
            logp += log_ξ[(k-1)*J + j]-0.5*(df_ξ + 1.0) * log(1.0 + (ξ[(k-1)*J + j] / τ[k])^2 / df_ξ) # contribution from prior
        end
    end
    # contribution from logdeterminants
    logp += logdet(inv_S) + logdet(P_root)
    C = cholesky(inv_Σ).L

    # quadratic form in prior
    temp1 = reshape(P_root * β, (J, K)) * C
    logp -= 0.5*sum(abs2, temp1)

    # quadratic form in likelihood
    temp2 = (reshape(inv_S * z, (Tsubp, K)) - XB) * C
    logp -= 0.5*sum(abs2, temp2)
    return logp
    #return logp_conditional_ξ(log_ξ, log_τ, β, inv_Σ, P_root, inv_S, z, F, J, K, Tsubp, df_ξ)
end

function logp_conditional_ξ_nt(log_ξ::AbstractArray{<:Real}, log_τ::AbstractArray{<:Real}, β::AbstractArray, C::AbstractMatrix, z::AbstractArray,
                                     F_sq::AbstractMatrix, XB::AbstractMatrix, J::Int, K::Int, Tsubp::Int, df_ξ::Real) # remove F_sq from args later
    ξ = exp.(log_ξ) # dimension J*K
    τ = exp.(log_τ)
    P_root = Diagonal(1.0 ./ ξ)
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( abs2.(ξ), (J, K) ) ) ) ) # should square P_root here.
    logp = 0.0
    for k in 1:K
        for j in 1:J
            logp += log_ξ[(k-1)*J + j]-0.5*(df_ξ + 1.0) * log(1.0 + (ξ[(k-1)*J + j] / τ[k])^2 / df_ξ) # contribution from prior
        end
    end
    # contribution from logdeterminants
    logp += logdet(inv_S) + logdet(P_root)

    # quadratic form in prior
    temp1 = reshape(P_root * β, (J, K)) * C
    logp -= 0.5*sum(abs2, temp1)

    # quadratic form in likelihood
    temp2 = (reshape(inv_S * z, (Tsubp, K)) - XB) * C
    logp -= 0.5*sum(abs2, temp2)
    return logp
    #return logp_conditional_ξ(log_ξ, log_τ, β, inv_Σ, P_root, inv_S, z, F, J, K, Tsubp, df_ξ)
end


function grad_logp_conditional_ξ_unpacked(log_ξ::AbstractArray{<:Real}, ξ::AbstractArray{<:Real}, τ::AbstractArray{<:Real}, β::AbstractArray,
                                          P_root::AbstractArray, inv_Σ::AbstractArray, inv_S::AbstractArray, z::AbstractArray, MSvecinvz::AbstractArray,
                                          F::AbstractArray, F_sq::AbstractArray, J::Int, K::Int, Tsubp::Int, df_ξ::Real)

    # Reuse reshaped versions of β and Pvec
    Mz = reshape(z, (Tsubp, K))
    Mβ = reshape(β, (J, K))
    Pvec = reshape(diag(P_root).^2, (J, K))  # replaces diag(P_root).^2
    sqPvec = sqrt.(Pvec)
    sqPvecinv = 1.0 ./ sqPvec
    Pvecinv = sqPvecinv .^ 2

    dPdtheta = -2.0 .* Pvec  # derivative of Pvec w.r.t. log_ξ
    dsqPdtheta = @. 0.5 * sqPvecinv * dPdtheta

    # Part of gradient from prior on β
    sqPbeta = sqPvec .* Mβ
    dsqPdthetabeta = dsqPdtheta .* Mβ
    temp1 = -(sqPbeta * inv_Σ)
    temp1 .*= dsqPdthetabeta

    grad_log_dens_β = vec(temp1) + vec(@. 0.5 * Pvecinv * dPdtheta)

    # Prior contribution from ξ
    τ_rep = repeat(τ, inner=J)
    grad_log_dens_θ = @. 1.0 - (((df_ξ+1.0) * ξ / τ_rep ^2) / (df_ξ + ξ^2 / τ_rep ^2)) * ξ

    # Likelihood contribution from ξ
    S_vec = 1.0 ./ diag(inv_S)
    MSvec2 = reshape(S_vec .^ 2, (Tsubp, K))
    MSvec = reshape(S_vec, (Tsubp, K))

    temp3 = @. Pvecinv ^ 2 * dPdtheta
    inv_Σ_Mβ = Mβ * inv_Σ

    # Precompute reusable matrices
    fac_ret2 = MSvecinvz * inv_Σ            # Tsubp × K
    fac_ret3 = F * inv_Σ_Mβ                 # Tsubp × K

    # Elementwise operations
    scalar_factors = -MSvec2 .+ (fac_ret2 .- fac_ret3) .* (Mz .* MSvec)  # Tsubp × K

    # Matrix multiply then elementwise multiply with temp3
    tmp = transpose(F_sq) * scalar_factors           # J × K
    ret_mat = tmp .* temp3                  # J × K

    # Final gradient
    grad_log_dens_y = vec(0.5 .* ret_mat)   # J*K vector


    # Final gradient is sum of three contributions
    grad = grad_log_dens_β + grad_log_dens_θ + grad_log_dens_y
    return grad
end


function grad_logp_conditional_ξ_nt(log_ξ::AbstractArray{<:Real}, τ::AbstractArray{<:Real}, β::AbstractArray, inv_Σ::AbstractArray,
                                    z::AbstractArray, F_sq::AbstractArray, fac_ret3::AbstractMatrix, J::Int, K::Int, Tsubp::Int, df_ξ::Real)
    # Precompute relevant quantities.
    ξ = map(exp, log_ξ)
    P_root = Diagonal(map(inv, ξ))
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ .^2, (J, K) ) ) ) )
    MSvecinvz = reshape(inv_S * z, (Tsubp, K))


    # Reuse reshaped versions of β and Pvec
    Mz = reshape(z, (Tsubp, K))
    Mβ = reshape(β, (J, K))
    Pvec = reshape(diag(P_root).^2, (J, K))  # replaces diag(P_root).^2
    sqPvec = sqrt.(Pvec)
    sqPvecinv = 1.0 ./ sqPvec
    Pvecinv = sqPvecinv .^ 2

    dPdtheta = -2.0 .* Pvec  # derivative of Pvec w.r.t. log_ξ
    dsqPdtheta = @. 0.5 * sqPvecinv * dPdtheta

    # Part of gradient from p(β | ξ, Σ)
    sqPbeta = sqPvec .* Mβ
    dsqPdthetabeta = dsqPdtheta .* Mβ
    temp1 = -(sqPbeta * inv_Σ)
    temp1 .*= dsqPdthetabeta

    grad_log_dens_β = vec(temp1) + vec(@. 0.5 * Pvecinv * dPdtheta)

    # Contribution from p(ξ | τ)
    τ_rep = repeat(τ, inner=J)
    grad_log_dens_θ = @. 1.0 - (((df_ξ+1.0) * ξ / τ_rep ^2) / (df_ξ + ξ^2 / τ_rep ^2)) * ξ

    # Likelihood contribution from p(z | ξ, β, Σ)
    S_vec = 1.0 ./ diag(inv_S)
    MSvec2 = reshape(S_vec .^ 2, (Tsubp, K))
    MSvec = reshape(S_vec, (Tsubp, K))

    temp3 = @. Pvecinv ^ 2 * dPdtheta

    # Precompute reusable matrices
    fac_ret2 = MSvecinvz * inv_Σ            # Tsubp × K

    # Elementwise operations
    scalar_factors = -MSvec2 .+ (fac_ret2 .- fac_ret3) .* (Mz .* MSvec)  # Tsubp × K

    # Matrix multiply then elementwise multiply with temp3
    tmp = transpose(F_sq) * scalar_factors           # J × K
    ret_mat = tmp .* temp3                           # J × K

    # Final gradient
    grad_log_dens_y = vec(0.5 .* ret_mat)   # J*K vector

    # Final gradient is sum of three contributions
    grad = grad_log_dens_β + grad_log_dens_θ + grad_log_dens_y
    return grad
end



struct Conditional_log_ξ{A<:AbstractMatrix}
    log_τ::Vector{Float64}
    β::Vector{Float64}
    C::A
    inv_Σ::Matrix{Float64}
    z::Vector{Float64}
    F_sq::Matrix{Float64}
    XB::Matrix{Float64}
    fac_ret3::Matrix{Float64}
    J::Int
    K::Int
    Tsubp::Int
    df_ξ::Float64
end
LogDensityProblems.dimension(cond::Conditional_log_ξ) = cond.J * cond.K
LogDensityProblems.capabilities(::Type{<:Conditional_log_ξ}) = LogDensityProblems.LogDensityOrder{1}() # We can provide the gradient

function LogDensityProblems.logdensity(cond::Conditional_log_ξ, log_ξ)
    (; log_τ, β, C, _, z, F_sq, XB, _, J, K, Tsubp, df_ξ) = cond
    return logp_conditional_ξ_nt(log_ξ, log_τ, β, C, z, F_sq, XB, J, K, Tsubp, df_ξ)

end
function LogDensityProblems.logdensity_and_gradient(cond::Conditional_log_ξ, log_ξ) # Can be optimized, there is some overlap with logdensity calculation
    (; log_τ, β, C, inv_Σ, z, F_sq, XB, fac_ret3, J, K, Tsubp, df_ξ) = cond
    logp = logp_conditional_ξ_nt(log_ξ, log_τ, β, C, z, F_sq, XB, J, K, Tsubp, df_ξ)
    grad = grad_logp_conditional_ξ_nt(log_ξ, exp.(log_τ), β, inv_Σ, z, F_sq, fac_ret3, J, K, Tsubp, df_ξ)
    return logp, grad
end


"""
NEED TO WRAP THE TARGET CONDITIONAL IN A LOGDENSITYPROBLEM PARAMETERIZED BY THE VALUES OF THE OTHER PARAMETERS. THEN USE AbstractMCMC.LogDensityModel TO CREATE A NEW OBJECT IN THIS ITERATION.
"""
function abstractmcmc_sample_log_ξ(
    rng::Random.AbstractRNG,
    sampler_ξ,
    state_ξ,
    log_ξ::AbstractArray{<:Real},
    log_τ::AbstractArray{<:Real},
    β::AbstractArray{<:Real},
    C::AbstractMatrix,
    inv_Σ::AbstractArray{<:Real},
    z::AbstractArray{<:Real},
    F_sq::AbstractArray{<:Real},
    XB::AbstractArray{<:Real},
    fac_ret3::AbstractArray,
    J::Int,
    K::Int,
    Tsubp::Int,
    df_ξ::Real;
    n_adapts::Int
)   
    # Create target LogDensityModel
    Cond = AbstractMCMC.LogDensityModel(Conditional_log_ξ(log_τ, β, C, inv_Σ, z, F_sq, XB, fac_ret3, J, K, Tsubp, df_ξ))
    if isnothing(state_ξ)
        transition_ξ, state_ξ = AbstractMCMC.step(rng, Cond, sampler_ξ; initial_params=log_ξ, n_adapts=n_adapts)
    else
        # Update state
        state_ξ = AbstractMCMC.setparams!!(Cond, state_ξ, log_ξ)

        # Slice sample log_ξ
        transition_ξ, state_ξ = AbstractMCMC.step(rng, Cond, sampler_ξ, state_ξ; n_adapts=n_adapts)
    end
    log_ξ = AbstractMCMC.getparams(state_ξ)
    return log_ξ, state_ξ
end