using ImplicitSVARCop, LinearAlgebra
using Test
import ForwardDiff, Random, Distributions

@testset "Derivatives conditional β" begin
    rng = Random.Xoshiro(1)
    K = 4
    J = 20
    Tsubp = 49999

    log_τ = rand(rng, Distributions.Normal(), K)
    log_ξ = rand(rng, Distributions.Normal(), K*J)
    β = rand(rng, Distributions.Normal(), K*J)
    Σ = Symmetric(rand(rng, Distributions.LKJ(K, 1)))
    inv_Σ = inv(Σ)

    P_root = Diagonal(@. 1.0 / exp(log_ξ))

    F = rand(rng, Distributions.Normal(), (Tsubp, J))
    z = rand(rng, Distributions.Normal(), Tsubp*K)
    F_tilde = F .^2
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_tilde * reshape( diag(inv(P_root)) .^2, (J, K) ) ) ) )
    F_t = F'
    FtF = F' * F
    
    logp_conditional_β_cl = let inv_Σ = inv_Σ, z = z, F = F, J = J, K = K, Tsubp = Tsubp
        β -> logp_conditional_β(β, z, inv_Σ, P_root, inv_S, F, J, K, Tsubp)
    end
    
    @test ForwardDiff.gradient(logp_conditional_β_cl, β) ≈ grad_logp_conditional_β(β, z, inv_Σ, P_root, inv_S, F_t, FtF, J, K, Tsubp)
end

#= @testset "Derivatives conditional τ_k" begin
    rng = Random.Xoshiro(1)
    K = 1
    J = 5
    log_τ_k = rand(rng, Distributions.Normal())
    log_ξ_k = rand(rng, Distributions.Normal(), J)

    logp_conditional_τ_k_cl = let log_ξ_k = log_ξ_k, J = J, K = K
        log_τ_k -> logp_conditional_τ_k(log_τ_k, log_ξ_k, J, K)
    end

    # First derivative (used in proposal)
    @test grad_logp_conditional_τ_k(log_τ_k, log_ξ_k, J, K) ≈ ForwardDiff.derivative(logp_conditional_τ_k_cl, log_τ_k)

    # Second derivative (used in proposal)
    @test hess_logp_conditional_τ_k(log_τ_k, log_ξ_k, J, K) ≈ ForwardDiff.derivative(x -> ForwardDiff.derivative(logp_conditional_τ_k_cl, x), log_τ_k)
end =#

@testset "Derivatives conditional ξ" begin
    rng = Random.Xoshiro(1)
    K = 4
    J = 20
    Tsubp = 49999

    log_τ = rand(rng, Distributions.Normal(), K)
    log_ξ = rand(rng, Distributions.Normal(), K*J)
    β = rand(rng, Distributions.Normal(), K*J)
    Σ = Symmetric(rand(rng, Distributions.LKJ(K, 1)))
    inv_Σ = inv(Σ)

    P_root = Diagonal(@. 1.0 / exp(log_ξ))

    F = rand(rng, Distributions.Normal(), (Tsubp, J))
    z = rand(rng, Distributions.Normal(), Tsubp*K)
    F_tilde = F .^2
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_tilde * reshape( diag(inv(P_root)) .^2, (J, K) ) ) ) )
    
    logp_conditional_ξ_cl = let log_τ = log_τ, β = β, inv_Σ = inv_Σ, z = z, F = F, J = J, K = K, Tsubp = Tsubp
        log_ξ -> ImplicitSVARCop.logp_conditional_ξ_autodiff(log_ξ, log_τ, β, inv_Σ, z, F, J, K, Tsubp)
    end

    @test grad_logp_conditional_ξ(log_ξ, log_τ, β, inv_Σ, inv_S, z, F, J, K, Tsubp) ≈ ForwardDiff.gradient(logp_conditional_ξ_cl, log_ξ) 
end

@testset "Derivatives conditional γ" begin
    rng = Random.Xoshiro(1)
    K = 4
    J = 20
    M = 4
    M_γ = K*M - div(M*(M+1), 2) + M
    Tsubp = 49999

    log_τ = rand(rng, Distributions.Normal(), K)
    log_ξ = rand(rng, Distributions.Normal(), K*J)
    β = rand(rng, Distributions.Normal(), K*J)
    γ = rand(rng, Distributions.Normal(), M_γ)
    Σ = ImplicitSVARCop.compute_Σ(γ, K, M)
    inv_Σ = inv(Σ)

    P_root = Diagonal(@. 1.0 / exp(log_ξ))

    F = rand(rng, Distributions.Normal(), (Tsubp, J))
    z = rand(rng, Distributions.Normal(), Tsubp*K)
    F_tilde = F .^2
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_tilde * reshape( diag(inv(P_root)) .^2, (J, K) ) ) ) )
    
    logp_conditional_γ_cl = let β = β, P_root = P_root, inv_S = inv_S, z = z, F = F, M = M, J = J, K = K, Tsubp = Tsubp
        γ -> ImplicitSVARCop.logp_conditional_γ_autodiff(γ, β, P_root, inv_S, z, F, M, J, K, Tsubp)
    end

    # NB! We have plenty of non-diagonal matrix inversions here. Thus, we expect more numerical error. I think the autodiff version is right here.
    @test isapprox(ForwardDiff.gradient(logp_conditional_γ_cl, γ), grad_logp_conditional_γ(γ, inv_Σ, β, P_root, inv_S, z, F, M, J, K, Tsubp); rtol=1e-4)
end

@testset "Derivatives conditional joint" begin
    rng = Random.Xoshiro(1)
    K = 4
    J = 20
    M = 3
    M_γ = K*M - div(M*(M+1), 2) + M
    Tsubp = 49999

    log_τ = rand(rng, Distributions.Normal(), K)
    log_ξ = rand(rng, Distributions.Normal(), K*J)
    β = rand(rng, Distributions.Normal(), K*J)
    γ = rand(rng, Distributions.Normal(), M_γ)

    F = rand(rng, Distributions.Normal(), (Tsubp, J))
    z = rand(rng, Distributions.Normal(), Tsubp*K)
    F_sq = F .^2

    # Set up model and θ
    model = VARModel(z, F, K, J, M, Tsubp)
    θ = vcat(β, log_ξ, log_τ, γ)

    # Create a partial function for autodiff
    logp_joint_ad = let model=model
        θ -> ImplicitSVARCop.logp_joint_autodiff(model, θ)
    end
    grad_θ_ad = ForwardDiff.gradient(logp_joint_ad, θ)

    logp, grad_θ = logp_and_grad_joint(model, θ)

    #@benchmark logp_and_grad_joint($model, $θ)

    #@benchmark logp_joint($model, $θ)

    @test isapprox(grad_θ, grad_θ_ad, rtol=1e-5)
end