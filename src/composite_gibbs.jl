using LinearAlgebra, LoopVectorization
function composite_gibbs(rng::Random.AbstractRNG, model::VARModel, θ_init::AbstractVector, δ::Real, num_samples::Int, B_adapt::Int)
    # Unpack data
    z = model.z
    F = model.F
    F_sq = model.F_sq
    K = model.K
    J = model.J
    M = model.M
    Tsubp = model.Tsubp
    F_t = transpose(F)
    D = LogDensityProblems.dimension(model)

    FtF = F_t * F

    # Unpack parameter vector
    β = θ_init[1:K*J]
    log_ξ = θ_init[K*J+1:2*K*J]
    log_τ = θ_init[2*K*J+1:2*K*J+K]
    M_γ = K*M - div(M*(M+1), 2) + M
    γ = θ_init[2*K*J+K+1:2*K*J+K+M_γ]

    # Initialize HMCDA hyperparameters
    XB = F * reshape(β, (J, K))
    logp_cond = let log_τ = log_τ, β = β, z = z, F_sq = F_sq, XB = XB, J = J, K = K, M = M, Tsubp = Tsubp
        η -> logp_joint_xi_gamma(η[1:K*J], η[K*J+1:K*J+M_γ], log_τ, β, z, F_sq, XB, J, K, M, Tsubp)
    end
    grad_logp_cond = let log_τ = log_τ, z = z, β = β, F = F, F_sq = F_sq, XB = XB, J = J, K = K, M = M, Tsubp = Tsubp
        η -> grad_logp_joint_xi_gamma_nuts(η[1:K*J], η[K*J+1:K*J+M_γ], log_τ, β, z, F, F_sq, XB, J, K, M, Tsubp)
    end
    ϵ = get_initial_ϵ(rng, vcat(log_ξ, γ), logp_cond, grad_logp_cond)
    ϵ0 = ϵ
    ϵ_bar = 1.0
    #μ = log(10.0*ϵ)
    H_bar = 0.0

    #metric = Diagonal(ones(Float64, K*J+M_γ))  # Start with identity covariance
    #inv_metric = Diagonal(ones(Float64, K*J+M_γ))
    inv_metric = Diagonal([0.0001422928350203481, 9.601752323857944e-5, 0.5903456869141932])
    metric = inv(inv_metric)

    running_cov = RunningCovariance(K*J+M_γ)

    println("Found initial stepsize ϵ = ", ϵ)

    # Generate samples
    samples = Vector{Vector{Float64}}(undef, num_samples)
    for b in 1:num_samples
        # Update model quantities:
        ξ = vmap(exp, log_ξ)
        ξ2 = vmap(x -> x^2, ξ)
        P_root = Diagonal(vmap(inv, ξ))
        inv_S, inv_S_vec = compute_S(ξ2, F, J, K, Tsubp)
        inv_Σ = inv(compute_Σ(γ, K, M)) # can invert this using Woodbury later.
        inv_Sz = inv_S * z

        # Sample β using a Gibbs step
        β = sample_conditional_β(rng, inv_Σ, P_root, inv_Sz, F_t, FtF, J, K, Tsubp)

        # Update model quantities to reflect the new value of β
        XB = F * reshape(β, (J, K))

        # Sample log_τ using a Metropolis-Hastings step
        log_τ = sample_mh_τ_all(rng, log_τ, log_ξ, J, K)

        # Sample covariance parameters jointly using HMCDA
        logp_cond = let log_τ = log_τ, β = β, z = z, F_sq = F_sq, XB = XB, J = J, K = K, M = M, Tsubp = Tsubp
            η -> logp_joint_xi_gamma(η[1:K*J], η[K*J+1:K*J+M_γ], log_τ, β, z, F_sq, XB, J, K, M, Tsubp)
        end
        grad_logp_cond = let log_τ = log_τ, z = z, β = β, F = F, F_sq = F_sq, XB = XB, J = J, K = K, M = M, Tsubp = Tsubp
            η -> grad_logp_joint_xi_gamma_nuts(η[1:K*J], η[K*J+1:K*J+M_γ], log_τ, β, z, F, F_sq, XB, J, K, M, Tsubp)
        end


        η, ϵ, ϵ_bar, H_bar = nuts_step2(rng, vcat(log_ξ, γ), δ, logp_cond, grad_logp_cond, ϵ, ϵ_bar, ϵ0, H_bar, b, B_adapt, metric, inv_metric, running_cov)
        log_ξ, γ = η[1:K*J], η[K*J+1:K*J+M_γ]
        if mod(b, 100) == 0
            println("b = ", b)
        end

        # Store the values of the chain
        samples[b] = vcat(β, log_ξ, log_τ, γ)
    end
    return samples
end

function composite_gibbs(model::VARModel, θ_init::AbstractVector, δ::Real, num_samples::Int, B_adapt::Int)
    return composite_gibbs(Random.default_rng(), model, θ_init, δ, num_samples, B_adapt)
end


function compute_S(ξ2::AbstractVector, F::AbstractArray, J::Int, K::Int, Tsubp::Int)
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

function predict_obs_mcmc(rng::Random.AbstractRNG, kdest::UnivariateKDE, x::AbstractVector, chain::AbstractVector, J::Int, K::Int) # chain is a vector of vectors here
    N_mc = 200
    y_pred = Matrix{Float64}(undef, N_mc, K)
    kdest_qf = InterpKDEQF(kdest)
    for j = 1:N_mc
        i = rand(rng, DiscreteUniform(1, length(chain)))            # sample a random row
        θ = chain[i]                                                # put parameter vector equal to the draw
        
        # Unpack parameter vector
        β = θ[1:K*J]
        log_ξ = θ[K*J+1:2*K*J]
        #log_τ = θ[2*K*J+1:2*K*J+K]
        #M_γ = K*M - div(M*(M+1), 2) + M
        #γ = θ[2*K*J+K+1:2*K*J+K+M_γ]

        ξ = vmap(exp, log_ξ)
        ξ2 = ξ .^2
            
        s_vec = 1.0 ./ sqrt.( 1.0 .+ vec( transpose(x) .^2 * reshape(ξ2, (J, K)) ) )
        μ = s_vec .* vec(transpose(x) * reshape(β, J, K))
        z_pred_t = Vector{Float64}(undef, K)
        for k in 1:K
            z_pred_t[k] = rand(rng, Normal(μ[k], s_vec[k]))
        end
        y_pred[j,:] = quantile.(kdest_qf, cdf.(Normal(), z_pred_t))
    end
    return mean(y_pred; dims=1) # return column means
end


function composite_gibbs2(rng::Random.AbstractRNG, model::VARModel, θ_init::AbstractVector, δ::Real, num_samples::Int, B_adapt::Int)
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

    logp_nuts = let log_τ = log_τ, β = β, z = z, F_sq = F_sq, XB = F * reshape(β, (J, K)), J = J, K = K, M = M, Tsubp = Tsubp
        η -> logp_joint_xi_gamma(η[1:K*J], η[K*J+1:K*J + M_γ], log_τ, β, z, F_sq, XB, J, K, M, Tsubp)
    end
    grad_nuts = let log_τ = log_τ, z = z, β = β, F = F, F_sq = F_sq, XB = F*reshape(β, (J, K)), J = J, K = K, M = M, Tsubp = Tsubp
        η -> (logp_joint_xi_gamma(η[1:K*J], η[K*J+1:K*J + M_γ], log_τ, β, z, F_sq, XB, J, K, M, Tsubp), grad_logp_joint_xi_gamma_nuts(η[1:K*J], η[K*J+1:K*J + M_γ], log_τ, β, z, F, F_sq, XB, J, K, M, Tsubp))
    end

    ϵ = get_initial_ϵ(log_ξ, γ, logp_nuts, grad_nuts)
    #log_ξ, γ, ϵ = nuts_step(rng, log_ξ, γ, logp_nuts, grad_nuts, ϵ, δ, 501, 500)
    δ = 0.5

    # Generate samples
    samples = Vector{Vector{Float64}}(undef, num_samples)
    for b in 1:num_samples
        # Update model quantities:
        ξ = vmap(exp, log_ξ)
        ξ2 = vmap(x -> x^2, ξ)
        P_root = Diagonal(vmap(inv, ξ))
        inv_S, inv_S_vec = compute_S(ξ2, F, J, K, Tsubp)
        inv_Σ = inv(compute_Σ(γ, K, M)) # can invert this using Woodbury later.
        inv_Sz = inv_S * z

        # Sample β using a Gibbs step
        β = sample_conditional_β(rng, inv_Σ, P_root, inv_Sz, F_t, FtF, J, K, Tsubp)

        # Update model quantities to reflect the new value of β
        #XB = F * reshape(β, (J, K))

        # Sample log_τ using a Metropolis-Hastings step
        log_τ = sample_mh_τ_all(rng, log_τ, log_ξ, J, K)

        # Sample covariance parameters jointly using HMCDA
        # need to implement: version of logp and grad for this joint conditional
        logp_nuts = let log_τ = log_τ, β = β, z = z, F_sq = F_sq, XB = F * reshape(β, (J, K)), J = J, K = K, M = M, Tsubp = Tsubp
        η -> logp_joint_xi_gamma(η[1:K*J], η[K*J+1:K*J + M_γ], log_τ, β, z, F_sq, XB, J, K, M, Tsubp)
        end
        grad_nuts = let log_τ = log_τ, z = z, β = β, F = F, F_sq = F_sq, XB = F*reshape(β, (J, K)), J = J, K = K, M = M, Tsubp = Tsubp
            η -> (logp_joint_xi_gamma(η[1:K*J], η[K*J+1:K*J + M_γ], log_τ, β, z, F_sq, XB, J, K, M, Tsubp), grad_logp_joint_xi_gamma_nuts(η[1:K*J], η[K*J+1:K*J + M_γ], log_τ, β, z, F, F_sq, XB, J, K, M, Tsubp))
        end

        log_ξ, γ = nuts_step(rng, log_ξ, γ, logp_nuts, grad_nuts, ϵ, δ, 2, 1)

        # Store the values of the chain
        samples[b] = vcat(β, log_ξ, log_τ, γ)
    end
end



function composite_gibbs3(rng::Random.AbstractRNG, model::VARModel, θ_init::AbstractVector, δ::Real, num_samples::Int, B_adapt::Int)
    # Unpack data
    z = model.z
    F = model.F
    F_sq = model.F_sq
    K = model.K
    J = model.J
    M = model.M
    Tsubp = model.Tsubp
    F_t = transpose(F)
    D = LogDensityProblems.dimension(model)

    FtF = F_t * F

    # Unpack parameter vector
    β = θ_init[1:K*J]
    log_ξ = θ_init[K*J+1:2*K*J]
    log_τ = θ_init[2*K*J+1:2*K*J+K]
    M_γ = K*M - div(M*(M+1), 2) + M
    γ = θ_init[2*K*J+K+1:2*K*J+K+M_γ]

    # Initialize HMCDA hyperparameters
    XB = F * reshape(β, (J, K))
    logp_cond = let log_τ = log_τ, β = β, z = z, F_sq = F_sq, XB = XB, J = J, K = K, M = M, Tsubp = Tsubp
        η -> logp_joint_xi_gamma(η[1:K*J], η[K*J+1:K*J+M_γ], log_τ, β, z, F_sq, XB, J, K, M, Tsubp)
    end
    grad_logp_cond = let log_τ = log_τ, z = z, β = β, F = F, F_sq = F_sq, XB = XB, J = J, K = K, M = M, Tsubp = Tsubp
        η -> grad_logp_joint_xi_gamma_nuts(η[1:K*J], η[K*J+1:K*J+M_γ], log_τ, β, z, F, F_sq, XB, J, K, M, Tsubp)
    end

    metric = DiagEuclideanMetric(K*J + M_γ)
    hamiltonian = Hamiltonian(metric, logp_cond, grad_logp_cond)

    # Define a leapfrog solver, with the initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, vcat(log_ξ, γ))
    integrator = Leapfrog(initial_ϵ)

    # For NUTS, uncomment the following lines:
    #kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(max_depth=10)))
    #adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    # For HMCDA, uncomment the following lines:
    δ, λ = 0.8, 1.0
    kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedIntegrationTime(λ)))
    adaptor = StepSizeAdaptor(δ, initial_ϵ)

    hamiltonian, t = AdvancedHMC.sample_init(rng, hamiltonian, vcat(log_ξ, γ))

    println("Found initial stepsize ϵ = ", initial_ϵ)

    # Generate samples
    samples = Vector{Vector{Float64}}(undef, num_samples)
    for b in 1:num_samples
        # Update model quantities:
        ξ = vmap(exp, log_ξ)
        ξ2 = vmap(x -> x^2, ξ)
        P_root = Diagonal(vmap(inv, ξ))
        inv_S, inv_S_vec = compute_S(ξ2, F, J, K, Tsubp)
        inv_Σ = inv(compute_Σ(γ, K, M)) # can invert this using Woodbury later.
        inv_Sz = inv_S * z

        # Sample β using a Gibbs step
        β = sample_conditional_β(rng, inv_Σ, P_root, inv_Sz, F_t, FtF, J, K, Tsubp)

        # Update model quantities to reflect the new value of β
        XB = F * reshape(β, (J, K))

        # Sample log_τ using a Metropolis-Hastings step
        log_τ = sample_mh_τ_all(rng, log_τ, log_ξ, J, K)

        # Sample covariance parameters jointly using HMCDA
        logp_cond = let log_τ = log_τ, β = β, z = z, F_sq = F_sq, XB = XB, J = J, K = K, M = M, Tsubp = Tsubp
            η -> logp_joint_xi_gamma(η[1:K*J], η[K*J+1:K*J+M_γ], log_τ, β, z, F_sq, XB, J, K, M, Tsubp)
        end
        grad_logp_cond = let log_τ = log_τ, z = z, β = β, F = F, F_sq = F_sq, XB = XB, J = J, K = K, M = M, Tsubp = Tsubp
            η -> grad_logp_joint_xi_gamma_nuts(η[1:K*J], η[K*J+1:K*J+M_γ], log_τ, β, z, F, F_sq, XB, J, K, M, Tsubp)
        end

        # Define new Hamiltonian for target conditional densities
        hamiltonian = Hamiltonian(hamiltonian.metric, logp_cond, grad_logp_cond) # NB! Important to use the Hamiltonian (adapted) metric here
        t = AdvancedHMC.transition(rng, hamiltonian, kernel, t.z)
        # Adapt Hamiltonian
        tstat = AdvancedHMC.stat(t)
        hamiltonian, kernel, isadapted = AdvancedHMC.adapt!(hamiltonian, kernel, adaptor, b, B_adapt, t.z.θ, tstat.acceptance_rate)
        tstat = merge(tstat, (is_adapt=isadapted,))
        # Extract sampled parameters
        η = t.z.θ

        #kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(max_depth=10)))
        #adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
        log_ξ, γ = η[1:K*J], η[K*J+1:K*J+M_γ]
        #log_ξ, γ, ϵ, ϵ_bar, H_bar = hmcda_step(rng, log_ξ, γ, logp_hmcda, grad_hmcda, δ, b, B_adapt, ϵ, ϵ_bar, H_bar, μ)
        if mod(b, 10) == 0
            println("b = ", b)
        end

        # Store the values of the chain
        samples[b] = vcat(β, log_ξ, log_τ, γ)
    end
    return samples
end