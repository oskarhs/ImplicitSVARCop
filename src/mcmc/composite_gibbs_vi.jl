function composite_gibbs_vi(rng::Random.AbstractRNG, model::VARModel, θ_init::AbstractVector, N_fac::Int, N_iter_vi::Int, num_samples::Int; progress=true)
    # Unpack data
    z = model.z
    F = model.F
    F_sq = model.F_sq
    K = model.K
    J = model.J
    M = model.M
    Tsubp = model.Tsubp
    F_t = transpose(F)

    FtF = F_t * F

    # Unpack parameter vector
    β = θ_init[1:K*J]
    log_ξ = θ_init[K*J+1:2*K*J]
    log_τ = θ_init[2*K*J+1:2*K*J+K]
    M_γ = K*M - div(M*(M+1), 2) + M
    γ = θ_init[2*K*J+K+1:2*K*J+K+M_γ]

    # Set up VI posterior for first iteration:
    conditional = VIPosterior(K*J + M_γ, N_fac, J, K, M_γ)

    # Set up progressbar
    pm = progress ? Progress(num_samples; desc="Generating samples", barlen=31) : nothing

    
    # Generate samples
    samples = Vector{Vector{Float64}}(undef, num_samples)
    for b in 1:num_samples
        # Update model quantities:
        ξ = vmap(exp, log_ξ)
        ξ2 = vmap(abs2, ξ)
        P_root = Diagonal(vmap(inv, ξ))
        inv_S, inv_S_vec = compute_S(ξ2, F_sq, J, K, Tsubp)
        inv_Σ = inv(compute_Σ(γ, K, M)) # can invert this using Woodbury later.
        inv_Sz = inv_S * z # some computation can be reused here

        # Sample β using a Gibbs step
        β = sample_conditional_β(rng, inv_Σ, P_root, inv_Sz, F_t, FtF, J, K, Tsubp)

        # Update model quantities to reflect the new value of β
        XB = F * reshape(β, (J, K))

        # Sample log_τ using a Metropolis-Hastings step
        log_τ = sample_mh_τ_all(rng, log_τ, log_ξ, J, K)

        # Sample covariance parameters jointly using VI
        if b == 1
            log_ξ, γ, conditional = sample_conditional_ξ_γ_vi(rng, log_ξ, γ, model, conditional, β, log_τ, XB, 500, N_fac)
        elseif b%2 == 1
            log_ξ, γ, conditional = sample_conditional_ξ_γ_vi(rng, log_ξ, γ, model, conditional, β, log_τ, XB, N_iter_vi, N_fac)
        else
            log_ξ, γ, conditional = sample_conditional_ξ_γ_rw(rng, log_ξ, γ, model, conditional, β, log_τ, XB)
        end

        # Store the values of the chain
        samples[b] = vcat(β, log_ξ, log_τ, γ)

        if !isnothing(pm)
            next!(pm)
        end
    end
    return samples
end


function compute_S(ξ2::AbstractVector, F_sq::AbstractArray, J::Int, K::Int, Tsubp::Int)
    inv_S_vec = Vector{eltype(ξ2)}(undef, K * Tsubp)
    @turbo for k in 1:K # I love this macro
        for t in 1:Tsubp
            a = zero(eltype(ξ2))
            for j in 1:J
                a += F_sq[t,j] * ξ2[(k-1)*J + j]
            end
            inv_S_vec[(k-1)*Tsubp + t] = sqrt(1.0 + a)
        end
    end
    inv_S = Diagonal(inv_S_vec)
    return inv_S, inv_S_vec
end