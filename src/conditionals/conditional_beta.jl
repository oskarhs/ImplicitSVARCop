"""
    sample_conditional_β([rng::Random.AbstractRNG,] z::AbstractArray, inv_Σ::AbstractArray,  P_root::AbstractArray, inv_S::AbstractArray, F_t::AbstractArray, FtF::AbstractArray, J::Int, K::Int, Tsubp::Int)

Generate a sample from the full conditional distribution of β.

# Fields
- `rng`: Controls the seed used by the sampler. Defaults to `Random.default_rng()`
- `z`: Vector of latent observations.
- `inv_Σ`: Inverse of covariance matrix.
- `P_root`: Diagonal matrix appearing in the prior covariance of β
- `inv_S`: Inverse of the diagonal standardization matrix `S`
- `F_t`: Transpose of design matrix `F`
- `FtF`: The result of the matrix product `F' * F`
- `J`: Number of covariates per response variable
- `K`: Number of response variables
- `Tsubp`: Number of observations minus the length of the series

# Returns
- `β`: A sample from the full conditional distribution of β
"""
function sample_conditional_β(rng::Random.AbstractRNG, inv_Σ::AbstractArray, P_root::AbstractArray, inv_Sz::AbstractArray, F_t::AbstractArray, FtF::AbstractArray, J::Int, K::Int, Tsubp::Int)
    inv_Σ_β = Symmetric(kron(inv_Σ, FtF) + P_root * kron(inv_Σ, I(J)) * P_root)
    C_β = cholesky(inv_Σ_β).L  # inv(Σ_β) = C_β * C_β'
    U_β = inv(C_β)             # Σ_β = U_β' * U_β

    μ_β = U_β * transpose(U_β) * kron(inv_Σ, F_t) * inv_Sz

    # NB! Better to compute cholesky of inv(Σ_β) first, then invert the Cholesky decomposition, as inversion is only O(n²) in this case
    # Will keep this inefficient for now, as this is more in line with the paper draft for now, and as such will be a bit easier to debug.

    u = rand(rng, Normal(), K*J)
    β = transpose(U_β) * u + μ_β
    return β
end

function sample_conditional_β(z::AbstractArray, inv_Σ::AbstractArray,  P_root::AbstractArray, inv_S::AbstractArray, F_t::AbstractArray, FtF::AbstractArray, J::Int, K::Int, Tsubp::Int)
    return sample_conditional_β(Random.default_rng(), z, inv_Σ, P_root, inv_S, F_t, FtF, J,K, Tsubp)
end

function logp_conditional_β(β::AbstractArray, z::AbstractArray, inv_Σ::AbstractArray,  P_root::AbstractArray, inv_S::AbstractArray, F::AbstractArray, J::Int, K::Int, Tsubp::Int)
    C = cholesky(inv_Σ).L
    logp = 0.0

    # Contribution from p(β | Σ, θ)
    #temp1 = vec(reshape(P_root * β, (J, K)) * C)
    temp1 = reshape(P_root * β, (J, K)) * C
    logp -= 0.5*sum(abs2, temp1)

    # Contribution from likelihood
    XB = F * reshape(β, (J, K)) # this can be reused
    #temp2 = vec( (reshape(inv_S * z, (Tsubp, K)) - XB) * C)
    temp2 = (reshape(inv_S * z, (Tsubp, K)) - XB) * C
    logp -= 0.5*sum(abs2, temp2)
    return logp
end

function grad_logp_conditional_β(β::AbstractArray, z::AbstractArray, inv_Σ::AbstractArray,  P_root::AbstractArray, inv_S::AbstractArray, F_t::AbstractArray, FtF::AbstractArray, J::Int, K::Int, Tsubp::Int)
    grad_log_dens_β = vec(-transpose(β) * P_root * kron(inv_Σ, I(J)) * P_root) # = -β' * (Σ ∗ inv(P))⁻¹

    grad_log_dens_y = vec( (F_t * reshape(inv_S * z, (Tsubp, K)) - FtF * reshape(β, (J, K)) ) * inv_Σ) # (F_t * reshape(inv_S * z, (Tsubp, K)) is fixed here, can be computed in advance
    
    grad = grad_log_dens_β + grad_log_dens_y
    return grad
end

function grad_logp_conditional_β_unpacked(β::AbstractArray, z::AbstractArray, inv_Σ::AbstractArray,  P_root::AbstractArray, F::AbstractArray, Mlik::AbstractArray,  J::Int, K::Int, Tsubp::Int)
    grad_log_dens_β = vec(-transpose(β) * P_root * kron(inv_Σ, I(J)) * P_root) # = -β' * (Σ ∗ inv(P))⁻¹

    grad_log_dens_y = vec( (transpose(F) * Mlik ) * inv_Σ)
    
    grad = grad_log_dens_β + grad_log_dens_y
    return grad
end

#= function test_derivatives_β() # it does not matter that the input args for e.g. F_tilde are not correct
    rng = Random.Xoshiro(1)
    K = 4
    J = 20
    Tsubp = 49999

    log_τ = rand(rng, Normal(), K)
    log_ξ = rand(rng, Normal(), K*J)
    β = rand(rng, Normal(), K*J)
    Σ = Symmetric(rand(rng, LKJ(K, 1)))
    inv_Σ = inv(Σ)

    P_root = Diagonal(@. 1.0 / exp(log_ξ))

    F = rand(rng, Normal(), (Tsubp, J))
    z = rand(rng, Normal(), Tsubp*K)
    F_tilde = F .^2
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_tilde * reshape( diag(inv(P_root)) .^2, (J, K) ) ) ) )
    F_t = F'
    FtF = F' * F
    
    logp_conditional_β_cl = let inv_Σ = inv_Σ, z = z, F = F, J = J, K = K, Tsubp = Tsubp
        β -> logp_conditional_β(β, z, inv_Σ, P_root, inv_S, F, J, K, Tsubp)
    end

    ForwardDiff.gradient(logp_conditional_β_cl, β) |> println
    grad_logp_conditional_β(β, z, inv_Σ, P_root, inv_S, F_t, FtF, J, K, Tsubp) |> println

    @code_warntype logp_conditional_β(β, z, inv_Σ, P_root, inv_S, F, J, K, Tsubp)
    
    @assert ForwardDiff.gradient(logp_conditional_β_cl, β) ≈ grad_logp_conditional_β(β, z, inv_Σ, P_root, inv_S, F_t, FtF, J, K, Tsubp)
    #grad_logp_conditional_ξ(log_ξ, log_τ, β, inv_Σ, inv_S, z, F, J, K, Tsubp)

    @benchmark ForwardDiff.gradient($logp_conditional_β_cl, $β)
    @benchmark grad_logp_conditional_β(β, z, inv_Σ, P_root, inv_S, F_t, FtF, J, K, Tsubp)

    @benchmark logp_conditional_β(β, z, inv_Σ, P_root, inv_S, F, J, K, Tsubp)
end =#