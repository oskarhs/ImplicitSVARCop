function logp_joint_xi_gamma(log_ξ::AbstractVector, γ::AbstractVector, log_τ, β, z, F_sq, XB, J, K, M, Tsubp)
    # Fixed prior parameters (can make them part of model later)
    a_γ = 3.0
    b_γ = 1.0

    # Transform parameters to facilitate computation
    ξ = vmap(exp, log_ξ)
    τ = vmap(exp, log_τ)
    ξ2 = vmap(abs2, ξ)

    # Compute the Matrix square root of P and the standardization matrix:
    P_root = Diagonal(vmap(inv, ξ))
    
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
    MSvecinvz = reshape(inv_S * z, (Tsubp, K))
    Mlik = MSvecinvz - XB
    P_rootβrs = reshape(P_root * β, (J, K))

    # Compute likelihood:
    # Start computing the logdensity
    logp = 0.0

    # Contribution from log p(ξ | τ)
    logp += vsum(log_ξ .- log.(1.0 .+ (ξ ./ τ[repeat(1:K, inner=J)]).^2))

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



function grad_logp_joint_xi_gamma(log_ξ::AbstractVector, γ::AbstractVector, log_τ, β, z, F, F_sq, XB, J, K, M, Tsubp)
    # Fixed prior parameters (can make them part of model later)
    a_γ = 3.0
    b_γ = 1.0

    # Transform parameters to facilitate computation
    ξ = vmap(exp, log_ξ)
    τ = vmap(exp, log_τ)
    ξ2 = vmap(x -> x^2, ξ)

    # Compute the Matrix square root of P and the standardization matrix:
    P_root = Diagonal(@. 1.0 / ξ)
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ .^2, (J, K) ) ) ) ) # should square P_root here.
    
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
    MSvecinvz = reshape(inv_S * z, (Tsubp, K))
    Mlik = MSvecinvz - XB
    P_rootβrs = reshape(P_root * β, (J, K))

    # Compute gradients:
    grad_log_ξ = grad_logp_conditional_ξ_unpacked(log_ξ, ξ, τ, β, P_root, inv_Σ, inv_S, z, MSvecinvz, F, F_sq, J, K, Tsubp) # checked!

    grad_γ = grad_logp_conditional_γ_unpacked(γ, inv_Σ, P_rootβrs, Mlik, M, J, K, Tsubp) # checked!

    # Return gradients
    return grad_log_ξ, grad_γ
end


function grad_logp_joint_xi_gamma_nuts(log_ξ::AbstractVector, γ::AbstractVector, log_τ, β, z, F, F_sq, XB, J, K, M, Tsubp)
    # Fixed prior parameters (can make them part of model later)
    a_γ = 3.0
    b_γ = 1.0

    # Transform parameters to facilitate computation
    ξ = vmap(exp, log_ξ)
    τ = vmap(exp, log_τ)
    ξ2 = vmap(x -> x^2, ξ)

    # Compute the Matrix square root of P and the standardization matrix:
    P_root = Diagonal(@. 1.0 / ξ)
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ .^2, (J, K) ) ) ) ) # should square P_root here.
    
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
    MSvecinvz = reshape(inv_S * z, (Tsubp, K))
    Mlik = MSvecinvz - XB
    P_rootβrs = reshape(P_root * β, (J, K))

    # Compute likelihood:
    # Start computing the logdensity
    logp = 0.0

    # Contribution from log p(ξ | τ)
    logp += vsum(log_ξ .- log.(1.0 .+ (ξ ./ τ[repeat(1:K, inner=J)]).^2))

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

    # Compute gradients:
    grad_log_ξ = grad_logp_conditional_ξ_unpacked(log_ξ, ξ, τ, β, P_root, inv_Σ, inv_S, z, MSvecinvz, F, F_sq, J, K, Tsubp) # checked!

    grad_γ = grad_logp_conditional_γ_unpacked(γ, inv_Σ, P_rootβrs, Mlik, M, J, K, Tsubp) # checked!

    # Return gradients
    return logp, vcat(grad_log_ξ, grad_γ)
end