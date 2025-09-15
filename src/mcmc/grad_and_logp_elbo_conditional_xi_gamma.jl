"""
    grad_and_logp_elbo(
    )

Function for computing the gradient of the evidence lower bound with respect to variational parameters.

# Arguments
* `model`: VARModel object.
* `θ`: Unconstrained parameter vector
* `Bfac`: B matrix in the factor covariance matrix expression BB' + Δ²
* `d`: Vector of diagonal entries of Δ in the expression BB' + Δ²
* `w1`: Random variable generated to estimate the gradient, of low dimension (factor structure)
* `w2`: Random variable generated to estimate the gradient, of dimension Dimension(model)
* `Signinvpart`: Inverse matrix used in variational lower bound, (BB' + Δ²)⁻¹
"""
function grad_and_logp_elbo_conditional_ξ_γ(
    model::VARModel, 
    η::AbstractVector{<:Real},
    β::AbstractVector,
    log_τ::AbstractVector,
    XB::AbstractMatrix,
    Bfac::AbstractMatrix{<:Real}, 
    d::AbstractVector{<:Real},
    w1::AbstractVector{<:Real},
    w2::AbstractVector{<:Real},
    Siginvpart::AbstractMatrix{<:Real}
    )
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
    log_ξ = η[1:K*J]
    M_γ = K*M - div(M*(M+1), 2) + M
    γ = η[K*J+1:K*J+M_γ]

    # Transform parameters to facilitate computation
    ξ = vmap(exp, log_ξ)
    ξ2 = vmap(abs2, ξ)
    τ = vmap(exp, log_τ)

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
    #XB = F * reshape(β, (J, K)) can be precomputed
    MSvecinvz = reshape(inv_S * z, (Tsubp, K))
    Mlik = MSvecinvz - XB
    P_rootβrs = reshape(P_root * β, (J, K))

    # Compute likelihood:
    logp = logp_unpacked_params(log_ξ, ξ, log_τ, τ, γ, C, inv_S_vec, P_root, P_rootβrs, Mlik, K, J, Tsubp, a_γ, b_γ)

    # Compute gradients:
    grad_log_ξ = grad_logp_conditional_ξ_unpacked(log_ξ, ξ, τ, β, P_root, inv_Σ, inv_S, z, MSvecinvz, F, F_sq, J, K, Tsubp) # checked!

    grad_γ = grad_logp_conditional_γ_unpacked(γ, inv_Σ, P_rootβrs, Mlik, M, J, K, Tsubp) # checked!

    # Concatenate gradients
    grad_η = vcat(grad_log_ξ, grad_γ)

    # Compute VI gradient:
    d2 = 1.0 ./ d.^2
    inv_ΔsqBfac = Bfac .* d2
    Bz_deps = Bfac * w1 + d .* w2
    DBz_deps = Bz_deps .* d2
    Half2 = inv_ΔsqBfac * (Siginvpart * (transpose(Bfac) * DBz_deps))
    SiginvBz_deps = DBz_deps - Half2

    L_μ = grad_η + SiginvBz_deps
    L_B = grad_η * transpose(w1) + SiginvBz_deps * transpose(w1)
    L_d = grad_η .* w2 + SiginvBz_deps .* w2

    return L_μ, L_B, L_d, logp
end