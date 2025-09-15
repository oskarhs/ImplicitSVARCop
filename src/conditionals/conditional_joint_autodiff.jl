"""
    logp_joint(model::VARModel, θ::AbstractVector)

Compute the full logposterior density.

# Arguments
* `model`: VARModel object holding hyperparameters and (transformed) data
* `θ`: Unconstrained model parameters.
"""
function logp_joint_autodiff(model::VARModel, θ::AbstractVector)
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
    ξ = exp.(log_ξ)
    #ξ2 = ξ .^2
    τ = exp.(log_τ)
    #ξinv2 = ξinv .^ 2   # diag(P_root).^2 = (1 ./ ξ).^2

    # Compute the Matrix square root of P and the standardization matrix:
    P_root = Diagonal(@. 1.0 / ξ)
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ .^2, (J, K) ) ) ) ) # should square P_root here.

    Σ = compute_Σ(γ, K, M)
    inv_Σ = inv(Σ)

    # Compute intermediate quantities that are reused in evaluation of the logdensity and the gradients
    C = cholesky(inv_Σ).L
    XB = F * reshape(β, (J, K))
    Mlik = reshape(inv_S * z, (Tsubp, K)) - XB
    P_rootβrs = reshape(P_root * β, (J, K))

    # Start computing the logdensity
    logp = 0.0

    # Contribution from log p(τ)
    f1 = let J = J
        log_τ -> -(J-1.0)*log_τ - log(1.0 + exp(2.0*log_τ))
    end
    logp += sum(f1, log_τ)

    # Contribution from log p(ξ | τ)
    logp += sum(log_ξ .- log.(1 .+ (ξ ./ τ[repeat(1:K, inner=J)]).^2))

    # Contribution from log p(g)
    f2 = let b_γ = b_γ
        x -> log(1.0 + abs(x)/b_γ)
    end
    logp -= (a_γ + 1.0) * sum(f2, γ)

    # Contribution from log p(β | g, ξ)
    temp1 = P_rootβrs * C
    logp += J*logdet(C)
    logp -= 0.5*sum(abs2, temp1)

    # Contribution from likelihood
    temp2 = Mlik * C
    logp += logdet(inv_S) + logdet(P_root) + Tsubp * logdet(C)
    logp -= 0.5*sum(abs2, temp2)

    return logp
end