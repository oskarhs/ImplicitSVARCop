using LinearAlgebra, ForwardDiff, DifferentiationInterface

include(joinpath(@__DIR__, "..", "src", "ImplicitSVARCop.jl"))
using .ImplicitSVARCop


function logp_conditional_ξ_nt(log_ξ::AbstractArray{<:Real}, log_τ::AbstractArray{<:Real}, β::AbstractArray, inv_Σ::AbstractArray, z::AbstractArray, F::AbstractArray, F_sq, J::Int, K::Int, Tsubp::Int) # remove F_sq from args later
    ξ = exp.(log_ξ) # dimension J*K
    τ = exp.(log_τ) # probably better to pass these directly here
    logp = 0.0
    for k in 1:K
        for j in 1:J
            logp += log_ξ[(k-1)*J + j]-log(1.0 + (ξ[(k-1)*J + j] / τ[k])^2) # contribution from prior
        end
    end

    P_root = Diagonal(map(inv, ξ))
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ .^2, (J, K) ) ) ) ) # should square P_root here.

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


function grad_logp_conditional_ξ_nt(log_ξ::AbstractArray{<:Real}, τ::AbstractArray{<:Real}, β::AbstractArray,
                                    inv_Σ::AbstractArray, z::AbstractArray,
                                    F::AbstractArray, F_sq::AbstractArray, J::Int, K::Int, Tsubp::Int)
    ξ = map(exp, log_ξ)
    #P_root = Diagonal(map(inv, ξ))
    P_root_vec = map(inv, ξ)
    #inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ .^2, (J, K) ) ) ) ) # should square P_root here.
    inv_S_vec = sqrt.( 1.0 .+ vec( F_sq * reshape( ξ .^2, (J, K) ) ) ) # should square P_root here.
    Mz = reshape(z, (Tsubp, K))
    MSvecinvz = reshape(inv_S_vec .* z, (Tsubp, K))

    # Reuse reshaped versions of β and Pvec
    Mβ = reshape(β, (J, K))
    #Pvec = reshape(diag(P_root).^2, (J, K))  # replaces diag(P_root).^2
    Pvec = reshape(P_root_vec.^2, (J, K))  # replaces diag(P_root).^2
    sqPvec = sqrt.(Pvec)
    sqPvecinv = 1.0 ./ sqPvec
    Pvecinv = sqPvecinv .^ 2

    dPdtheta = -2.0 .* Pvec  # derivative of Pvec w.r.t. log_ξ
    dsqPdtheta = @. 0.5 * sqPvecinv * dPdtheta

    # Part of gradient from prior on β
    sqPbeta = sqPvec .* Mβ
    dsqPdthetabeta = dsqPdtheta .* Mβ
    temp1 = -(sqPbeta * inv_Σ) .* dsqPdthetabeta
    #temp1 .*= dsqPdthetabeta
    #grad_log_dens_β = vec(temp1)

    grad_log_dens_β = vec(temp1) + vec(@. 0.5 * Pvecinv * dPdtheta)

    # Prior contribution from ξ
    τ_rep = repeat(τ, inner=J)
    grad_log_dens_θ = @. 1.0 - ((2.0 * ξ / τ_rep ^2) / (1.0 + ξ^2 / τ_rep ^2)) * ξ

    # Likelihood contribution from ξ
    S_vec = 1.0 ./ inv_S_vec
    MSvec2 = reshape(S_vec .^ 2, (Tsubp, K))
    MSvec = reshape(S_vec, (Tsubp, K))

    temp3 = @. Pvecinv ^ 2 * dPdtheta
    inv_Σ_Mβ = Mβ * inv_Σ

    #= tf1 = let MSvec2 = MSvec2, inv_Σ = inv_Σ, MSvec = MSvec, F_sq_t = transpose(F_sq),
              Mz = Mz, temp3 = temp3
        function (k)
            # Copy relevant slice of Matrices to avoid repeatedly creating new views within the performance-critial loop
            fac_ret2_arr = MSvecinvz * view(inv_Σ, k, :)
            fac_ret3_arr = F * view(inv_Σ_Mβ, :, k)

            # Compute all scalar_factors for each t:
            scalar_factors = @. @views -MSvec2[:, k] + (fac_ret2_arr - fac_ret3_arr) * (Mz[:, k] * MSvec[:, k])

            ret1 = (F_sq_t * scalar_factors) .* view(temp3, :, k)
            return 0.5*ret1
        end
    end =#

    # Compute gradients for log_ξ_1, …, log_ξ_K
    #grad_log_dens_y = mapreduce(tf1, vcat, 1:K)

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


function example_here()
    rng = Random.default_rng()
    p = 1          # order
    T = 50_000   # number of observations
    J = 50 # Number of covariates per variable
    K = 4 # Dimension of response
    M = 3
    Tsubp = T - p
    #β = [0.8, 0.4]       # autoregressive parameter, covariate

    #= z = Vector{Float64}(undef, Tsubp+1)
    x_exo = Vector{Float64}(undef, Tsubp)
    z[1] = rand(rng, Normal(0.0, 1.0/sqrt(1.0-β[1]^2-β[2]^2)))
    for t in 2:Tsubp+1
        x_exo[t-1] = rand(rng, Normal())
        z[t] = β[1]*z[t-1] + β[2]*x_exo[t-1] + rand(rng, Normal())
    end
    d_true = Exponential(4.0)
    y = quantile.(d_true, cdf.(Normal(0.0, 1.0/sqrt(1.0-β[1]^2-β[2]^2)), z))

    kdest = fit(UnivariateKDE, y, SSVKernel())
    z = quantile.(Normal(0.0, 1.0), cdf.(kdest, y))

    F = hcat(z[1:end-1], x_exo)
    z = z[2:end] =#
    z = randn(K * Tsubp)
    F = randn((Tsubp, J))

    model = VARModel(z, F, K, J, M, Tsubp)
    D = LogDensityProblems.dimension(model)
    θ_init = rand(rng, Normal(), D)
    θ = θ_init

    β = θ[1:K*J]
    log_ξ = θ[K*J+1:2*K*J]
    log_τ = θ[2*K*J+1:2*K*J+K]
    M_γ = K*M - div(M*(M+1), 2) + M
    γ = θ[2*K*J+K+1:2*K*J+K+M_γ]

    Σ = compute_Σ(γ, K, M)
    inv_Σ = inv(Σ)

    f_wrapped(log_ξ) = logp_conditional_ξ_nt(log_ξ, log_τ, β, inv_Σ, z, F, model.F_sq, J, K, Tsubp)


    grad_logp_conditional_ξ_nt(log_ξ, map(exp, log_τ), β, inv_Σ, z, F, model.F_sq, J, K, Tsubp)
    ForwardDiff.gradient(f_wrapped, log_ξ)

    @benchmark grad_logp_conditional_ξ_nt(log_ξ, map(exp, log_τ), β, inv_Σ, z, F, model.F_sq, J, K, Tsubp)

    @benchmark ForwardDiff.gradient(f_wrapped, log_ξ)

    grad_wrapped = log_ξ -> grad_logp_conditional_ξ_nt(log_ξ, map(exp, log_τ), β, inv_Σ, z, F, model.F_sq, J, K, Tsubp)
    ForwardDiff.jacobian(grad_wrapped, log_ξ)
    ForwardDiff.hessian(f_wrapped, log_ξ)

    backend = AutoForwardDiff()
    prep = prepare_jacobian(grad_logp_conditional_ξ_nt, backend, log_ξ, Constant(map(exp, log_τ)), Constant(β), Constant(inv_Σ), Constant(z), Constant(F), Constant(model.F_sq), Constant(J), Constant(K), Constant(Tsubp))

    DifferentiationInterface.value_and_jacobian(grad_logp_conditional_ξ_nt, prep, backend, log_ξ, Constant(map(exp, log_τ)), Constant(β), Constant(inv_Σ), Constant(z), Constant(F), Constant(model.F_sq), Constant(J), Constant(K), Constant(Tsubp))

    @benchmark DifferentiationInterface.jacobian(grad_logp_conditional_ξ_nt, prep, backend, log_ξ, Constant(map(exp, log_τ)), Constant(β), Constant(inv_Σ), Constant(z), Constant(F), Constant(model.F_sq), Constant(J), Constant(K), Constant(Tsubp))


    @benchmark ForwardDiff.jacobian(grad_wrapped, log_ξ)


    @benchmark ForwardDiff.hessian(f_wrapped, log_ξ)


    grad_enzyme = log_ξ -> enzyme_test(log_ξ, map(exp, log_τ), β, inv_Σ, z, F, model.F_sq, J, K, Tsubp)

    Enzyme.jacobian(Forward, grad_enzyme, log_ξ)

    f_part(log_ξ) = obj_part2(log_ξ, model, β, log_τ, inv_Σ)

    g1, H1 = symbolic_grad_and_hess(θ_init, model)
end