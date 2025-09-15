function adadelta_step_conditional_ξ_γ(
    rng::Random.AbstractRNG,
    model::VARModel,
    β::AbstractVector,
    log_τ::AbstractVector,
    XB::AbstractMatrix,
    μ::AbstractVector{<:Real}, 
    Bfac::AbstractMatrix{<:Real},
    d::AbstractVector{<:Real}, 
    N_fac::Int,
    Siginvpart::AbstractMatrix{<:Real},
    ADA::ADADELTAState
)
    J = model.J
    K = model.K
    M = model.M
    M_γ = K*M - div(M*(M+1), 2) + M
    D = K*J + M_γ

    ρ = ADA.ρ
    ϵ = ADA.ϵ

    oldEδ2_μ = ADA.Eδ2_μ
    oldEg2_μ = ADA.Eg2_μ

    oldEδ2_B = ADA.Eδ2_B
    oldEg2_B = ADA.Eg2_B

    oldEδ2_d = ADA.Eδ2_d
    oldEg2_d = ADA.Eg2_d

    w1 = Base.rand(rng, Normal(), N_fac)
    w2 = Base.rand(rng, Normal(), D)
    
    η = μ + Bfac * w1 + d .* w2

    L_μ, L_B, L_d, logp = grad_and_logp_elbo_conditional_ξ_γ(model, η, β, log_τ, XB, Bfac, d, w1, w2, Siginvpart) # get gradients and logdensity

    # Set upper triangle of L_B (including diagonal) to 0:
    tril!(L_B, -1)

    # update μ
    newEg2_μ = ρ*oldEg2_μ + (1.0 - ρ)*L_μ .^2
    change_δ_μ = @. sqrt(oldEδ2_μ + ϵ) / sqrt(newEg2_μ + ϵ)*L_μ
    μ = μ + change_δ_μ
    newEδ2_μ = ρ * oldEδ2_μ + (1.0 - ρ) * change_δ_μ .^2

    # update B
    newEg2_B = ρ * oldEg2_B + (1.0 - ρ) * L_B .^2
    change_δ_B = @. sqrt(oldEδ2_B + ϵ)/sqrt(newEg2_B + ϵ)*L_B
    Bfac = Bfac + change_δ_B # How do we ensure that Bfac remains lower triangular here?
    newEδ2_B = ρ * oldEδ2_B + (1.0 - ρ) * change_δ_B .^2

    # update d
    newEg2_d = ρ*oldEg2_d + (1.0 - ρ)*L_d .^2
    change_δ_d = @. sqrt(oldEδ2_d + ϵ)/sqrt(newEg2_d + ϵ)*L_d
    d = d + change_δ_d
    newEδ2_d = ρ * oldEδ2_d + (1.0 - ρ) * change_δ_d .^2

    d[abs.(d) .<= 1e-4] .= 1e-4

    # Update ADADELTA state:
    ADA = ADADELTAState(ρ, ϵ, newEδ2_μ, newEg2_μ, newEδ2_B, newEg2_B, newEδ2_d, newEg2_d)

    Bz_deps = Bfac * w1 + d .* w2

    d2 = 1.0 ./ d .^2
    Dinv2B = Bfac .* d2
    DBz_deps = Bz_deps .* d2
    Siginvpart = inv(I(N_fac) + transpose(Bfac) * Dinv2B)

    Blogdet = logdet(I(N_fac) + transpose(Dinv2B) * Bfac) + vsum(x -> log(abs2(x)), d)
    Half2 = Dinv2B * (Siginvpart * (transpose(Bfac) * DBz_deps))
    quadform = transpose(Bz_deps) * (DBz_deps - Half2)
    LowerB = logp + 0.5 * D * log(2.0*pi) + 0.5*Blogdet + 0.5 * quadform

    return LowerB, μ, Bfac, d, ADA, Siginvpart, L_μ, L_B, L_d
end


function fit_vi_proposal_conditional_ξ_γ(rng::Random.AbstractRNG, model::VARModel, last_conditional::VIPosterior, β::AbstractVector, log_τ::AbstractVector, XB::AbstractMatrix, n_iter::Int, N_fac::Int)
    J = model.J
    K = model.K
    M_γ = K*model.M - div(model.M*(model.M+1), 2) + model.M
    D = K*J + M_γ

    # Initialize variational parameters:
    μ = last_conditional.μ
    d = last_conditional.d
    Bfac = last_conditional.Bfac
    #tril!(Bfac, -1) # sets upper triangle (including diagonal) of Bfac to 0.

    # Initialize ADADELTA state
    Eδ2_μ = zeros(Float64, length(μ))
    Eg2_μ = zeros(Float64, length(μ))

    Eδ2_B = zeros(Float64, size(Bfac))
    Eg2_B = zeros(Float64, size(Bfac))

    Eδ2_d = zeros(Float64, length(d))
    Eg2_d = zeros(Float64, length(d))

    ρ = 0.95
    ϵ = 1e-6

    ADA = ADADELTAState(ρ, ϵ, Eδ2_μ, Eg2_μ, Eδ2_B, Eg2_B, Eδ2_d, Eg2_d)

    # Compute inverse of factor matrix
    d2 = 1.0 ./ d .^2
    Dinv2B = Bfac .* d2
    Siginvpart = inv(I(N_fac) + transpose(Bfac) * Dinv2B)

    # Store values of lower bound
    ELBOs = Vector{Float64}(undef, n_iter)
    μ_c = zeros(Float64, length(μ))
    Bfac_c = zeros(Float64, size(Bfac))
    d_c = zeros(Float64, length(d))
    n_avg = 0

    # Perform SGA iterations
    for it in 1:n_iter
        ELBOs[it], μ, Bfac, d, ADA, Siginvpart, L_μ, L_B, L_d = adadelta_step_conditional_ξ_γ(rng, model, β, log_τ, XB, μ, Bfac, d, N_fac, Siginvpart, ADA)
        # Parameter averaging strategy
        if it < 0.8 * n_iter
            μ_c += μ
            Bfac_c += Bfac
            d_c += d
            n_avg += 1
        end
    end
    μ_c = μ_c / n_avg
    Bfac_c = Bfac_c / n_avg
    d_c = d_c / n_avg
    
    return VIPosterior(μ_c, Bfac_c, d_c, J, K, M_γ), ELBOs
end

# Proposal based on a local VI approximation
function sample_conditional_ξ_γ_vi(rng::Random.AbstractRNG, log_ξ::AbstractVector, γ::AbstractVector, model::VARModel, last_conditional::VIPosterior, β::AbstractVector, log_τ::AbstractVector, XB::AbstractMatrix, n_iter::Int, N_fac::Int)
    # Fit variational distribution to the joint conditional p(log_ξ, γ | ⋯)
    new_conditional, _ = fit_vi_proposal_conditional_ξ_γ(rng, model, last_conditional, β, log_τ, XB, n_iter, N_fac)

    # Generate proposal
    η_prop = rand(rng, new_conditional)
    log_ξ_prop = η_prop[1:model.K*model.J]
    γ_prop = η_prop[model.K*model.J+1:end]

    # Evaluate MH ratio
    log_mh_ratio = logp_joint_xi_gamma(log_ξ_prop, γ_prop, log_τ, β, model.z, model.F_sq, XB, model.J, model.K, model.M, model.Tsubp) - logp_joint_xi_gamma(log_ξ, γ, log_τ, β, model.z, model.F_sq, XB, model.J, model.K, model.M, model.Tsubp)
    #println(log_mh_ratio)
    log_mh_ratio += logpdf(new_conditional, vcat(log_ξ, γ)) - logpdf(new_conditional, vcat(log_ξ_prop, γ_prop))
    if isinf(log_mh_ratio) || isnan(log_mh_ratio)
        println("Underflow.")
    end
    #println(log_ξ_prop, γ_prop)
    #println(log_mh_ratio)
    log_ξ, γ = ifelse(
        log(Base.rand(rng)) < log_mh_ratio,
        (log_ξ_prop, γ_prop),
        (log_ξ, γ)
    )
    return log_ξ, γ, new_conditional
end

# Random walk proposal using the multivariate T distribution:
function sample_conditional_ξ_γ_rw(rng::Random.AbstractRNG, log_ξ::AbstractVector, γ::AbstractVector, model::VARModel, last_conditional::VIPosterior, β::AbstractVector, log_τ::AbstractVector, XB::AbstractMatrix)
    df = 8.0

    # Retrieve covariance matrix of VI posterior from last iteration:
    V = 0.25 * covariance(last_conditional)

    # Positive definiteness master hack (confidential)
    eig = eigen(Symmetric(V))
    D = Diagonal(max.(eig.values, 1e-6*maximum(eig.values))) # Eigenvalues should be negative
    V = Symmetric(eig.vectors * D * transpose(eig.vectors))

    # Generate proposal
    d_prop = MvTDist(df, vcat(log_ξ, γ), Matrix(V))
    η_prop = rand(rng, d_prop)
    log_ξ_prop = η_prop[1:model.K*model.J]
    γ_prop = η_prop[model.K*model.J+1:end]

    # Distribution for reverse proposal
    #d_rev = MvTDist(df, η_prop, V)

    # Evaluate MH ratio
    log_mh_ratio = logp_joint_xi_gamma(log_ξ_prop, γ_prop, log_τ, β, model.z, model.F_sq, XB, model.J, model.K, model.M, model.Tsubp) - logp_joint_xi_gamma(log_ξ, γ, log_τ, β, model.z, model.F_sq, XB, model.J, model.K, model.M, model.Tsubp)
    #println(log_mh_ratio)
    #symmetry, proposal distribution cancels in MH ratio
    #log_mh_ratio += logpdf(d_rev, vcat(log_ξ, γ)) - logpdf(d_prop, vcat(log_ξ_prop, γ_prop))
    if isinf(log_mh_ratio) || isnan(log_mh_ratio)
        println("Underflow.")
    end
    #println(log_ξ_prop, γ_prop)
    #println(log_mh_ratio)
    log_ξ, γ = ifelse(
        log(Base.rand(rng)) < log_mh_ratio,
        (log_ξ_prop, γ_prop),
        (log_ξ, γ)
    )
    return log_ξ, γ, last_conditional
end