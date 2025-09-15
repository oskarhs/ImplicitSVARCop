using ForwardDiff, LinearAlgebra, Distributions

function ddlogFCuj(u, tBbeta, S2, dS2, ddS2, z, betaj, tau)
    lambda2 = exp(u)
    term1 = -0.5 * betaj .* betaj ./ lambda2
    term2 = -(lambda2 / tau^2) ./ (1 .+ lambda2 ./ tau^2) .^ 2
    term3 = -0.5 * sum(ddS2 ./ S2 .- dS2 .* dS2 ./ (S2 .* S2))
    term4 = -0.5 * sum(z .^ 2 .* (2 .* dS2 .^ 2 ./ S2 .^ 3 .- ddS2 ./ S2 .^ 2))
    term5 = sum(tBbeta .* ((0.75 .* dS2 .^ 2 ./ S2 .^ 2.5 .- 0.5 .* ddS2 ./ S2 .^ 1.5) .* z))
    return term1 + term2 + term3 + term4 + term5
end

function dlogFCuj(u, tBbeta, S2, dS2, z, betaj, tau)
    lambda2 = exp(u)
    term1 = 0.5 * betaj .* betaj ./ lambda2
    term2 = -(lambda2 / tau^2) ./ (1 .+ lambda2 ./ tau^2)
    term3 = -0.5 * sum(dS2 ./ S2)
    term4 = -0.5 * sum(z .^ 2 .* (-dS2 ./ S2 .^ 2))
    term5 = sum(tBbeta .* ((-0.5 .* dS2 ./ S2 .^ 1.5) .* z))
    return term1 + term2 + term3 + term4 + term5
end

function logFCuj(log_ξ, j, tBbeta, z, betaj, tau, F_sq)
    u = log_ξ[j]
    ξ = map(exp, log_ξ)
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ .^2, (J, K) ) ) ) ) # should square P_root here.
    S2 = 1.0 ./ diag(inv_S) .^2
    lambda2 = exp(u)
    #term1 = -0.5 * betaj .* betaj ./ lambda2
    #term2 = -log.(1.0 .+ lambda2 / tau[1]^2)
    term3 = -0.5 * sum(log.(S2))
    term4 = tBbeta * (z ./ sqrt.(S2))
    term5 = -0.5 * sum(z .^ 2 ./ S2)
    return term1 + term2 + term3 + term4 + term5
end

function logFCu(ω, tBbeta, z, β, tau, F_sq)
    ξ2 = map(exp, ω)
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ2, (J, K) ) ) ) ) # should square P_root here.
    S2 = 1.0 ./ diag(inv_S) .^2
    term1 = -0.5 * sum(β.^2 ./ ξ2)
    term2 = -sum(log.(1.0 .+ ξ2 / tau[1]^2))
    term3 = -0.5 * sum(log.(S2))
    term4 = tBbeta * (z ./ sqrt.(S2))
    term5 = -0.5 * sum(z .^ 2 ./ S2)
    #println(term1, "\n", term2, "\n", term3, "\n", term4 + term5)
    return term1 + term2 + term3 + term4 + term5
end

# NB! Here, we use ω = 2log_ξ as parameter instead.
function logp_conditional_ξ(ω::AbstractArray{<:Real}, log_τ::AbstractArray{<:Real}, β::AbstractArray, inv_Σ::AbstractArray, z::AbstractArray, F::AbstractArray, F_sq, J::Int, K::Int, Tsubp::Int) # remove F_sq from args later
    ξ2 = exp.(ω) # dimension J*K
    τ = exp.(log_τ) # probably better to pass these directly here
    term2 = 0.0
    for k in 1:K
        for j in 1:J
            term2 += 0.5 * ω[(k-1)*J + j]-log(1.0 + (ξ2[(k-1)*J + j] / τ[k]^2)) # contribution from prior
            #logp += -log(1.0 + (ξ2[(k-1)*J + j] / τ[k]^2)) # contribution from prior
        end
    end

    P_root = Diagonal(map(inv, sqrt.(ξ2)))
    term2 = term2 + logdet(P_root)

    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ2, (J, K) ) ) ) ) # should square P_root here.

    # contribution from logdeterminants
    term3 = logdet(inv_S)
    C = cholesky(inv_Σ).L

    # quadratic form in prior
    temp1 = reshape(P_root * β, (J, K)) * C
    term1 = -0.5*sum(abs2, temp1)

    # quadratic form in likelihood
    XB = F * reshape(β, (J, K))
    temp2 = (reshape(inv_S * z, (Tsubp, K)) - XB) * C
    #-0.5*sum(abs2, temp2) .+ 0.5*transpose(XB*C) * (XB*C)
    term4 = -0.5*sum(abs2, temp2)

    #println(term1, "\n", term2, "\n", term3, "\n", term4 .+ 0.5*transpose(XB*C) * (XB*C))
    return term1 + term2 + term3 + term4
end

# SO far, the term from the prior and the logdet term from S matches. Also, term from prior on β matches

function test_derivative_shit()
    K = 1
    J = 20
    M = 1
    Tsubp = 100

    β = rand(Normal(), J)
    ω1 = rand(Normal(), J)
    ω2 = rand(Normal(), J)
    log_τ = rand(Normal(), 1)
    
    inv_Σ = [1.0;;]
    
    # Simulate data:
    z = rand(Normal(), Tsubp)
    F = rand(Normal(), (Tsubp, J))

    F_sq = F.^2

    tBbeta = transpose(F * β)

    f_my = ω -> logp_conditional_ξ(ω, log_τ, β, inv_Σ, z, F, F_sq, J, K, Tsubp)
    f_sm = ω -> logFCu(ω, tBbeta, z, β, map(exp, log_τ), F_sq)

    f_my(ω1)
    f_sm(ω1)
    nothing

    f_my(ω1) - f_my(ω2)
    f_sm(ω1) - f_sm(ω2)

    ForwardDiff.gradient(f_my, ω1)
    ForwardDiff.gradient(f_sm, ω1)


    grad_logp_conditional_ξ_nt(ω1, map(exp, log_τ), β, inv_Σ, z, F, F_sq, J, K, Tsubp, 1.0)

    ForwardDiff.gradient(f_my, log_ξ) - ForwardDiff.gradient(f_sm, log_ξ)
end



function grad_logp_conditional_ξ_nt(ω::AbstractArray{<:Real}, τ::AbstractArray{<:Real}, β::AbstractArray,
                                    inv_Σ::AbstractArray, z::AbstractArray,
                                    F::AbstractArray, F_sq::AbstractArray, J::Int, K::Int, Tsubp::Int, df_ξ::Real)
    # Precompute relevant quantities.
    ξ2 = map(exp, ω)
    P_root = Diagonal(map(inv, sqrt.(ξ2)))
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ2, (J, K) ) ) ) ) # should square P_root here.
    MSvecinvz = reshape(inv_S * z, (Tsubp, K))


    # Reuse reshaped versions of β and Pvec
    Mz = reshape(z, (Tsubp, K))
    Mβ = reshape(β, (J, K))
    Pvec = reshape(diag(P_root).^2, (J, K))  # replaces diag(P_root).^2
    sqPvec = sqrt.(Pvec)
    sqPvecinv = 1.0 ./ sqPvec
    Pvecinv = sqPvecinv .^ 2

    #dPdtheta = -2.0 .* Pvec  # derivative of Pvec w.r.t. ω
    dPdtheta = -Pvec  # derivative of Pvec w.r.t. ω
    dsqPdtheta = @. 0.5 * sqPvecinv * dPdtheta

    # Part of gradient from p(β | ξ, Σ)
    sqPbeta = sqPvec .* Mβ
    dsqPdthetabeta = dsqPdtheta .* Mβ
    temp1 = -(sqPbeta * inv_Σ)
    temp1 .*= dsqPdthetabeta

    grad_log_dens_β = vec(temp1) + vec(@. 0.5 * Pvecinv * dPdtheta)

    # Contribution from p(ξ | τ)
    τ_rep = repeat(τ, inner=J)
    #grad_log_dens_θ = zeros(J)
    grad_log_dens_θ = @. 0.5 - 0.5 * (((df_ξ+1.0) * ξ2 / τ_rep ^2) / (df_ξ + ξ2 / τ_rep ^2))

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