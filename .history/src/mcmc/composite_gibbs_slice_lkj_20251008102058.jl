function composite_gibbs_abstractmcmc_lkj(rng::Random.AbstractRNG, model::VARModel, sampler_ξ, sampler_γ, θ_init::AbstractVector, num_samples::Int; n_adapts=n_adapts, progress=true)
    # Unpack data
    z = model.z
    F = model.F
    F_sq = model.F_sq
    K = model.K
    J = model.J
    M = model.M
    Tsubp = model.Tsubp
    df_ξ = model.df_ξ
    F_t = transpose(F)

    FtF = F_t * F

    # Unpack parameter vector
    β = θ_init[1:K*J]
    log_ξ = θ_init[K*J+1:2*K*J]
    log_τ = θ_init[2*K*J+1:2*K*J+K]
    M_γ = K*M - div(M*(M+1), 2) + M
    γ = θ_init[2*K*J+K+1:end]

    # Set up LKJ distribution for covariance matrix
    η = 1
    transformed_dist = Bijectors.transformed(LKJCholesky(K, η))
    to_chol = Bijectors.inverse(Bijectors.bijector(LKJCholesky(K, η)))
    C = transpose(inv(to_chol(γ).L))

    # Set AbstractMCMC sampler states:
    inv_Σ = C * C'
    XB = F * reshape(β, (J, K))
    Mβ = reshape(β, (J, K))
    fac_ret3 = F * (Mβ * inv_Σ)                 # Tsubp × K. This can be precomputed, and time save should be decent 
    state_ξ = nothing
    log_ξ, state_ξ = abstractmcmc_sample_log_ξ(rng, sampler_ξ, state_ξ, log_ξ, log_τ, β, C, inv_Σ, z, F, F_sq, XB, fac_ret3, J, K, Tsubp, df_ξ; n_adapts=n_adapts)

    # Precompute some quantities
    ξ = vmap(exp, log_ξ)
    ξ2 = vmap(abs2, ξ)
    P_root = Diagonal(vmap(inv, ξ))
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ2, (J, K) ) ) ) )
    inv_Sz = inv_S * z # some computation can be reused here
    Mlik = reshape(inv_Sz, (Tsubp, K)) - XB
    vec_MliktMlik_t = transpose(vec(Mlik' * Mlik))
    P_rootβrs = reshape(P_root * β, (J, K))
    state_γ = nothing
    γ, state_γ = abstractmcmc_sample_γ(rng, sampler_γ, state_γ, γ, transformed_dist, to_chol, P_rootβrs, Mlik, vec_MliktMlik_t, J, K, Tsubp; n_adapts=n_adapts)

    # Upate inverse covariance matrix
    C = transpose(inv(to_chol(γ).L))
    inv_Σ = C * C'
    # Set up progressbar
    pm = progress ? Progress(num_samples; desc="Generating samples", barlen=31) : nothing

    
    # Generate samples
    samples = Vector{Vector{Float64}}(undef, num_samples)
    for b in 1:num_samples
        # Sample β using a Gibbs step
        β = sample_conditional_β(rng, inv_Σ, P_root, inv_Sz, F_t, FtF, J, K, Tsubp)

        # Update model quantities to reflect the new value of β
        Mβ = reshape(β, (J, K))
        XB = F * Mβ
        fac_ret3 = XB * inv_Σ

        # Sample log_τ using a Metropolis-Hastings step
        #log_τ = sample_mh_τ_all(rng, log_τ, log_ξ, J, K)
        log_τ = sample_τ_ars(rng, log_τ, log_ξ, J, K)

        # Sample covariance parameters jointly using NUTS
        log_ξ, state_ξ = abstractmcmc_sample_log_ξ(rng, sampler_ξ, state_ξ, log_ξ, log_τ, β, C, inv_Σ, z, F_sq, XB, fac_ret3, J, K, Tsubp, df_ξ; n_adapts=n_adapts)

        # Update relevant model quantities to reflect the new ξ
        ξ = vmap(exp, log_ξ)
        ξ2 = vmap(abs2, ξ)
        P_root = Diagonal(vmap(inv, ξ))
        inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_sq * reshape( ξ2, (J, K) ) ) ) )
        inv_Sz = inv_S * z # some computation can be reused here
        Mlik = reshape(inv_Sz, (Tsubp, K)) - XB
        vec_MliktMlik_t = transpose(vec(transpose(Mlik) * Mlik))
        P_rootβrs = reshape(P_root * β, (J, K))
        

        # For now, just leave γ as is (we run on 1D examples, so this parameter has no effect)
        γ, state_γ = abstractmcmc_sample_γ(rng, sampler_γ, state_γ, γ, transformed_dist, to_chol, P_rootβrs, Mlik, vec_MliktMlik_t, J, K, Tsubp; n_adapts=n_adapts)

        C = transpose(inv(to_chol(γ).L))
        inv_Σ = C * C'
        
        # Store the values of the chain
        samples[b] = vcat(β, log_ξ, log_τ, γ)

        if !isnothing(pm)
            next!(pm)
        end
    end
    return samples
end