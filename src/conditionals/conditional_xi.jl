function logp_conditional_ξ(log_ξ::AbstractArray{<:Real}, log_τ::AbstractArray{<:Real}, β::AbstractArray, inv_Σ::AbstractArray, P_root::AbstractArray, inv_S::AbstractArray, z::AbstractArray, F::AbstractArray, J::Int, K::Int, Tsubp::Int, df_ξ::Real) # remove F_sq from args later
    ξ = exp.(log_ξ) # dimension J*K
    τ = exp.(log_τ) # probably better to pass these directly here
    logp = 0.0
    for k in 1:K
        for j in 1:J
            logp += log_ξ[(k-1)*J + j]-log(1.0 + (ξ[(k-1)*J + j] / τ[k])^2) # contribution from prior
        end
    end

    # contribution from logdeterminants
    #logp = logp + logdet(inv_S) - logdet(P_root)
    logp += logdet(inv_S) + logdet(P_root)
    C = cholesky(inv_Σ).L

    # quadratic form in prior
    #temp1 = vec(reshape(P_root * β, (J, K)) * C)
    temp1 = reshape(P_root * β, (J, K)) * C
    logp -= 0.5*sum(abs2, temp1)

    # quadratic form in likelihood
    XB = F * reshape(β, (J, K))
    #temp2 = vec( (reshape(inv_S * z, (Tsubp, K)) - XB) * C)
    temp2 = (reshape(inv_S * z, (Tsubp, K)) - XB) * C
    logp -= 0.5*sum(abs2, temp2)
    return logp
end

# Version of the above to be tested with autodiff
function logp_conditional_ξ_autodiff(log_ξ::AbstractArray{<:Real}, log_τ::AbstractArray{<:Real}, β::AbstractArray, inv_Σ::AbstractArray, z::AbstractArray, F::AbstractArray, F_sq::AbstractMatrix, J::Int, K::Int, Tsubp::Int, df_ξ::Real) # remove F_sq from args later
    ξ = exp.(log_ξ) # dimension J*K
    P_root = Diagonal(1.0 ./ ξ)
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( diag(inv(P_root)) .^2, (J, K) ) ) ) ) # should square P_root here.
    return logp_conditional_ξ(log_ξ, log_τ, β, inv_Σ, P_root, inv_S, z, F, J, K, Tsubp, df_ξ)
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


function grad_logp_conditional_ξ_nt(log_ξ::AbstractArray{<:Real}, τ::AbstractArray{<:Real}, β::AbstractArray,
                                        inv_Σ::AbstractArray, z::AbstractArray,
                                        F::AbstractArray, F_sq::AbstractArray, J::Int, K::Int, Tsubp::Int, df_ξ::Real)
    # Precompute relevant quantities.
    ξ = map(exp, log_ξ)
    P_root = Diagonal(map(inv, ξ))
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ .^2, (J, K) ) ) ) ) # should square P_root here.
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

function softabsmax(x::AbstractVector{<:Real}, α::Real)
    return ifelse.(x .!= 0.0, x .* coth.(α * x), 1.0/α)
end

function softabsmax(x::Real, α::Real)
    return ifelse(x != 0.0, x * coth(α * x), 1.0/α)
end


function sample_mh_log_ξ( # Use the eigendecomposition to compute cholesky decomp directly later
    rng::Random.AbstractRNG,
    backend,
    prep_grad_ξ,
    log_ξ::AbstractArray{<:Real},
    log_τ::AbstractArray{<:Real},
    β::AbstractArray,
    inv_Σ::AbstractArray, z::AbstractArray,
    F::AbstractArray,
    F_sq::AbstractArray,
    J::Int,
    K::Int,
    Tsubp::Int,
    df_ξ::Real
)   
    τ = map(exp, log_τ)

    lr = 1.0

    # Compute gradient and hessian of current state
    grad_prop, hess_prop = value_and_jacobian(
        ImplicitSVARCop.grad_logp_conditional_ξ_nt,
        backend,
        prep_grad_ξ,
        log_ξ,
        Constant(τ),
        Constant(β),
        Constant(inv_Σ),
        Constant(z),
        Constant(F),
        Constant(F_sq),
        Constant(J),
        Constant(K),
        Constant(Tsubp),
        Constant(df_ξ)
    )
    α = 1e6 # Use the Betancourt trick
    vals, vecs = eigen(-hess_prop)
    vals = Diagonal(softabsmax(vals, α))
    Ψ_prop = Symmetric(vecs * vals * transpose(vecs))
    mean_prop = log_ξ + lr * Ψ_prop \ grad_prop

    # Draw proposal
    log_ξ_prop = rand(rng, MvNormal(mean_prop, inv(Ψ_prop)))

    # Compute reverse proposal:
    grad_rev, hess_rev = value_and_jacobian(
        ImplicitSVARCop.grad_logp_conditional_ξ_nt,
        backend,
        prep_grad_ξ,
        log_ξ_prop,
        Constant(τ),
        Constant(β),
        Constant(inv_Σ),
        Constant(z),
        Constant(F),
        Constant(F_sq),
        Constant(J),
        Constant(K),
        Constant(Tsubp),
        Constant(df_ξ)
    )
    if any(isnan.(grad_rev)) || any(isnan.(hess_rev)) || any(isinf.(grad_rev)) || any(isinf.(hess_rev))
        println(log_ξ_prop)
    end
    if rank(hess_rev) < length(grad_rev) # If matrix is rank deficient, use Tikhonov!
        hess_rev += 1e-2 * I
    end
    vals, vecs = eigen(-hess_rev)
    #println(vals)
    vals = Diagonal(softabsmax(vals, α))
    #println(diag(vals))
    Ψ_rev = Symmetric(vecs * vals * transpose(vecs))
    if rank(Ψ_rev) < length(grad_rev) # If matrix is rank deficient, use Tikhonov!
        Ψ_rev += 1e-2 * I
    end
    mean_rev = log_ξ_prop + lr * Ψ_rev \ grad_rev
    # Compute MH ratio:

    log_mh_ratio = logpdf(MvNormal(mean_rev, inv(Ψ_rev)), log_ξ_prop) - logpdf(MvNormal(mean_prop, inv(Ψ_prop)), log_ξ)
    #println(log_mh_ratio)

    log_mh_ratio += logp_conditional_ξ_autodiff(log_ξ_prop, log_τ, β, inv_Σ, z, F, F_sq, J, K, Tsubp, df_ξ) - logp_conditional_ξ_autodiff(log_ξ, log_τ, β, inv_Σ, z, F, F_sq, J, K, Tsubp, df_ξ)

    #println(log_mh_ratio)

    # Accept or reject
    u = rand(rng, Uniform(0,1))
    if log(u) < log_mh_ratio
        log_ξ = log_ξ_prop
    end
    return log_ξ
end




function sample_mh_log_ξ_diag( # Use the eigendecomposition to compute cholesky decomp directly later
    rng::Random.AbstractRNG,
    backend,
    prep_grad_ξ,
    log_ξ::AbstractArray{<:Real},
    log_τ::AbstractArray{<:Real},
    β::AbstractArray,
    inv_Σ::AbstractArray, z::AbstractArray,
    F::AbstractArray,
    F_sq::AbstractArray,
    J::Int,
    K::Int,
    Tsubp::Int,
    df_ξ::Real
)   
    τ = map(exp, log_τ)

    lr = 1.0
    log_ξ_prop = copy(log_ξ)

    for j in 1:J
    # Compute gradient and hessian of current state
        
        grad_prop, hess_prop = value_and_jacobian(
            ImplicitSVARCop.grad_logp_conditional_ξ_nt,
            backend,
            prep_grad_ξ,
            log_ξ,
            Constant(τ),
            Constant(β),
            Constant(inv_Σ),
            Constant(z),
            Constant(F),
            Constant(F_sq),
            Constant(J),
            Constant(K),
            Constant(Tsubp),
            Constant(df_ξ)
        )
        var_prop = max(eps(), -1.0/hess_prop[j,j])
        mean_prop = log_ξ[j] + lr * grad_prop[j] * var_prop

        # Draw proposal
        log_ξ_prop[j] = rand(rng, Normal(mean_prop, sqrt(var_prop)))

        # Compute reverse proposal:
        grad_rev, hess_rev = value_and_jacobian(
            ImplicitSVARCop.grad_logp_conditional_ξ_nt,
            backend,
            prep_grad_ξ,
            log_ξ_prop,
            Constant(τ),
            Constant(β),
            Constant(inv_Σ),
            Constant(z),
            Constant(F),
            Constant(F_sq),
            Constant(J),
            Constant(K),
            Constant(Tsubp),
            Constant(df_ξ)
        )
        var_rev = max(eps(), -1.0/hess_rev[j,j])
        mean_rev = log_ξ_prop[j] + lr * grad_rev[j] * var_rev

        if !(isinf(var_rev) || isnan(var_rev) || var_rev == zero(var_rev))
            log_mh_ratio = logpdf(Normal(mean_rev, sqrt(var_rev)), log_ξ[j]) - logpdf(Normal(mean_prop, sqrt(var_prop)), log_ξ_prop[j])

            if any(isinf.(map(exp, log_ξ_prop))) || any(0.0 .== map(exp, log_ξ_prop))
                println(mean_prop)
            end

            log_mh_ratio += logp_conditional_ξ_autodiff(log_ξ_prop, log_τ, β, inv_Σ, z, F, F_sq, J, K, Tsubp, df_ξ) - logp_conditional_ξ_autodiff(log_ξ, log_τ, β, inv_Σ, z, F, F_sq, J, K, Tsubp, df_ξ)

            # Accept or reject
            u = rand(rng, Uniform(0,1))
            if log(u) < log_mh_ratio
                log_ξ[j] = log_ξ_prop[j]
            else
                log_ξ_prop[j] = log_ξ[j]
            end
        else
            #println(j, " ", mean_prop, " ", sqrt(var_prop), " ", grad_prop[j], " ", log_ξ_prop[j], " ", log_ξ[j])
            log_ξ_prop[j] = log_ξ[j]
        end
    end
    return log_ξ
end