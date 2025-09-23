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