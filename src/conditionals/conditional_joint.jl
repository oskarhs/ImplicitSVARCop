"""
    logp_joint(model::VARModel, θ::AbstractVector)

Compute the full logposterior density.

# Arguments
* `model`: VARModel object holding hyperparameters and (transformed) data
* `θ`: Unconstrained model parameters.
"""
function logp_joint(model::VARModel, θ::AbstractVector)
    # Fixed prior parameters (can make them part of model later)
    a_γ = 3.0
    b_γ = 1.0

    # Unpack fixed model parameters
    z = model.z
    F = model.F
    F_sq = model.F_sq
    K = model.K
    J = model.J
    M = model.M
    Tsubp = model.Tsubp

    # Unpack parameter vector
    β = θ[1:K*J]
    log_ξ = θ[K*J+1:2*K*J]
    log_τ = θ[2*K*J+1:2*K*J+K]
    M_γ = K*M - div(M*(M+1), 2) + M
    γ = θ[2*K*J+K+1:2*K*J+K+M_γ]

    # Transform parameters to facilitate computation
    ξ = vmap(exp, log_ξ)
    ξ2 = vmap(abs2, ξ)
    τ = vmap(exp, log_τ)
    #ξinv2 = ξinv .^ 2   # diag(P_root).^2 = (1 ./ ξ).^2

    # Compute the Matrix square root of P and the standardization matrix:
    P_root = Diagonal(vmap(inv, ξ))
    #inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ .^2, (J, K) ) ) ) ) # should square P_root here.
    
    # Try with a for loop instead: (faster, but perhaps less friendly to autodiff. Uncomment this portion later)
    inv_S_vec = Vector{eltype(ξ)}(undef, K * Tsubp)
    @turbo for k in 1:K # I love this macro
        for t in 1:Tsubp
            a = zero(eltype(ξ))
            for j in 1:J
                a += F_sq[t,j] * ξ2[(k-1)*J + j]
            end
            inv_S_vec[(k-1)*Tsubp + t] = sqrt(1.0 + a)
        end
    end
    inv_S = Diagonal(inv_S_vec)

    Σ = compute_Σ(γ, K, M)
    inv_Σ = inv(Σ)

    # Compute intermediate quantities that are reused in evaluation of the logdensity and the gradients
    C = cholesky(inv_Σ).L
    XB = F * reshape(β, (J, K))
    Mlik = reshape(inv_S * z, (Tsubp, K)) - XB
    P_rootβrs = reshape(P_root * β, (J, K))

    # Compute likelihood:
    logp = logp_unpacked_params(log_ξ, ξ, log_τ, τ, γ, C, inv_S_vec, P_root, P_rootβrs, Mlik, K, J, Tsubp, a_γ, b_γ)

    return logp
end

# Evaluate the joint logposterior density, where intermediate quantities that can be reused for gradients have been precomputed.
function logp_unpacked_params(log_ξ, ξ, log_τ, τ, γ, C, inv_S_vec, P_root, P_rootβrs, Mlik, K, J, Tsubp, a_γ, b_γ, df = 1) # NB! Σ itself is not needed, only its Cholesky decomp.
    # Start computing the logdensity
    logp = 0.0

    # Contribution from log p(τ)
    f1 = let J = J
        log_τ -> -(J-1.0)*log_τ - log(1.0 + exp(2.0*log_τ))
    end
    logp += vsum(f1, log_τ)

    # Contribution from log p(ξ | τ)
    logp += vsum(log_ξ .- 0.5 * (df+1.0) * log.(1.0 .+ (ξ ./ τ[repeat(1:K, inner=J)]).^2 / df))
    #logp += vsum(log_ξ .- log.(1.0 .+ (ξ ./ τ[repeat(1:K, inner=J)]).^2))

    # Contribution from log p(g)
    f2 = let b_γ = b_γ
        x -> log(1.0 + abs(x)/b_γ)
    end
    logp -= (a_γ + 1.0) * vsum(f2, γ)

    # Contribution from log p(β | g, ξ)
    #temp1 = reshape(P_root * β, (J, K)) * C
    temp1 = P_rootβrs * C
    logp += J*logdet(C)
    logp -= 0.5*vsum(abs2, temp1)

    # Contribution from likelihood
    temp2 = Mlik * C
    logp += vsum(log, inv_S_vec) + logdet(P_root) + Tsubp * logdet(C)
    logp -= 0.5*vsum(abs2, temp2)

    return logp
end

"""
    logp_and_grad_joint(model::VARModel, θ::AbstractVector)

Compute the full logposterior density and its gradient.

# Arguments
* `model`: VARModel object holding hyperparameters and (transformed) data
* `θ`: Unconstrained model parameters.

# Returns
* `logp`: The joint logdensity evaluated at θ.
* `grad_θ`: The gradient of the joint logensity with respect to θ.
"""
function logp_and_grad_joint(model::VARModel, θ::AbstractVector)
    # Fixed prior parameters (can make them part of model later)
    a_γ = model.a_γ
    b_γ = model.b_γ
    df_ξ = model.df_ξ

    # Unpack fixed model parameters
    z = model.z
    F = model.F
    F_sq = model.F_sq
    K = model.K
    J = model.J
    M = model.M
    Tsubp = model.Tsubp

    # Unpack parameter vector
    β = θ[1:K*J]
    log_ξ = θ[K*J+1:2*K*J]
    log_τ = θ[2*K*J+1:2*K*J+K]
    M_γ = K*M - div(M*(M+1), 2) + M
    γ = θ[2*K*J+K+1:2*K*J+K+M_γ]

    # Transform parameters to facilitate computation
    ξ = map(exp, log_ξ)
    ξ2 = map(abs2, ξ)
    τ = map(exp, log_τ)

    # Compute the Matrix square root of P and the standardization matrix:
    P_root = Diagonal(map(inv, ξ))
    #inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ .^2, (J, K) ) ) ) ) # should square P_root here.
    
    # Try with a for loop instead: (faster, but perhaps less friendly to autodiff. Uncomment this portion later)
    inv_S_vec = Vector{eltype(ξ)}(undef, K * Tsubp)
    @turbo for k in 1:K # I love this macro
        for t in 1:Tsubp
            a = zero(eltype(ξ))
            for j in 1:J
                a += F_sq[t,j] * ξ2[(k-1)*J + j]
            end
            inv_S_vec[(k-1)*Tsubp + t] = sqrt(1.0 + a)
        end
    end
    inv_S = Diagonal(inv_S_vec)

    Σ = compute_Σ(γ, K, M)
    inv_Σ = inv(Σ)

    # Compute intermediate quantities that are reused in evaluation of the logdensity and the gradients
    C = cholesky(inv_Σ).L
    XB = F * reshape(β, (J, K))
    MSvecinvz = reshape(inv_S * z, (Tsubp, K))
    Mlik = MSvecinvz - XB
    P_rootβrs = reshape(P_root * β, (J, K))

    # Compute likelihood:
    logp = logp_unpacked_params(log_ξ, ξ, log_τ, τ, γ, C, inv_S_vec, P_root, P_rootβrs, Mlik, K, J, Tsubp, a_γ, b_γ)

    # Compute gradients:
    grad_β = grad_logp_conditional_β_unpacked(β, z, inv_Σ,  P_root, F, Mlik, J, K, Tsubp) # checked!

    grad_log_ξ = grad_logp_conditional_ξ_unpacked(log_ξ, ξ, τ, β, P_root, inv_Σ, inv_S, z, MSvecinvz, F, F_sq, J, K, Tsubp, df_ξ) # checked!

    grad_log_τ = grad_logp_conditional_τ(log_τ, log_ξ, J, K) # checked!

    grad_γ = grad_logp_conditional_γ_unpacked(γ, inv_Σ, P_rootβrs, Mlik, M, J, K, Tsubp) # checked!

    # Concatenate gradients
    grad_θ = vcat(grad_β, grad_log_ξ, grad_log_τ, grad_γ)

    return logp, grad_θ
end



# Evaluate the joint logposterior density, where intermediate quantities that can be reused for gradients have been precomputed.
function logp_unpacked_params_lkj(log_ξ, ξ, log_τ, τ, atanh_ρ, C, inv_S_vec, P_root, P_rootβrs, Mlik, K, J, Tsubp, a_γ, b_γ, df = 1) # NB! Σ itself is not needed, only its Cholesky decomp.
    η = 1.0
    
    # Start computing the logdensity
    logp = 0.0

    # Contribution from log p(τ)
    f1 = let J = J
        log_τ -> -(J-1.0)*log_τ - log(1.0 + exp(2.0*log_τ))
    end
    logp += vsum(f1, log_τ)

    # Contribution from log p(ξ | τ)
    logp += vsum(log_ξ .- 0.5 * (df+1.0) * log.(1.0 .+ (ξ ./ τ[repeat(1:K, inner=J)]).^2 / df))
    #logp += vsum(log_ξ .- log.(1.0 .+ (ξ ./ τ[repeat(1:K, inner=J)]).^2))

    # Contribution from log p(ρ)
    logp += sech(atanh_ρ[1])^(2*η)

    # Contribution from log p(β | g, ξ)
    #temp1 = reshape(P_root * β, (J, K)) * C
    temp1 = P_rootβrs * C
    logp += J*logdet(C)
    logp -= 0.5*vsum(abs2, temp1)

    # Contribution from likelihood
    temp2 = Mlik * C
    logp += vsum(log, inv_S_vec) + logdet(P_root) + Tsubp * logdet(C)
    logp -= 0.5*vsum(abs2, temp2)

    return logp
end