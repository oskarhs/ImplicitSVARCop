using Random, Distributions, SliceSampling, AdvancedHMC
include(joinpath(@__DIR__, "..", "src", "ImplicitSVARCop.jl"))
using .ImplicitSVARCop
include(joinpath(@__DIR__, "..", "src", "composite_gibbs.jl"))
include(joinpath(@__DIR__, "..", "src", "nuts_step.jl"))
include(joinpath(@__DIR__, "..", "src", "nuts_step2.jl"))



using AdvancedHMC, LogDensityProblems, MCMCChains, StatsPlots

function estimate_best_true_var1_cov(rng, d_true, x, β)
    N_mc = 500 # number of Monte Carlo samples used to compute the estimate.
    # sample a random row from the chain:
    y_pred = Vector{Float64}(undef, N_mc)
    for j = 1:N_mc
        z = dot(x, β) + rand(rng, Normal())                      # first predict on latent scale
        y_pred[j] = quantile(d_true, cdf(Normal(0.0, 1.0/sqrt(1.0-β[1]^2-β[2]^2)), z))
    end
    return mean(y_pred)
end

# Now do a AR(1) process with 1 exogenous covariate
function test_AR1_cov()
    rng = Random.default_rng()
    p = 1          # order
    T = 100   # number of observations
    J = 2 # Number of covariates per variable
    K = 1 # Dimension of response
    M = 1
    Tsubp = T - p
    β = [0.8, 0.4]       # autoregressive parameter, covariate

    z = Vector{Float64}(undef, Tsubp+1)
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
    z = z[2:end]
    
    model = VARModel(z, F, K, J, M, Tsubp)
    D = LogDensityProblems.dimension(model)

    # Number of samples and adaptation steps
    # Set the number of samples to draw and warmup iterations
    n_samples, n_adapts = 5_000, 2_000


    # Define a Hamiltonian system
    metric = DenseEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, θ -> logp_joint(model, θ), θ -> logp_and_grad_joint(model, θ))

    initial_θ = rand(rng, Normal(), D)    

    # Define a leapfrog solver, with the initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)

    # Define NUTS sampler
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(max_depth=8)))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.5, integrator))

    # Run the sampler
    samples, stats = sample(
        hamiltonian, kernel, initial_θ, n_samples, adaptor, n_adapts; progress=true, (pm_next!) = AdvancedHMC.simple_pm_next!
    )

    chain = MCMCChains.Chains(samples[n_adapts+1:end], get_varsymbols(model))
    describe(chain)

    newdata = VARModel(z, F, K, J, M, Tsubp)
    y_pred = predict_response(rng, samples[n_adapts+1:end], [kdest], newdata)

    sqrt(mean((y[2:end] - y_pred).^2))

    # Here: do simple ML estimation on log-scale for comparison
    log_y = log.(y)
    X = hcat(log_y[1:end-1], x_exo)
    model = lm(X, log_y[2:end])
    σ2 = 1.0/(Tsubp-1) * sum(residuals(model).^2)

    log_pred = exp.(predict(model) .+ 0.5*σ2)
    sqrt(mean((y[2:end] - log_pred).^2))

    plot(1:Tsubp, y_pred .- y[2:end])
    plot!(1:Tsubp, log_pred .- y[2:end])

    MCMCChains.Chains(samples[n_adapts+1:end], get_varsymbols(model)) |> describe
    autocor(chain)
    autocorplot(chain)

    plot(MCMCChains.Chains(samples[n_adapts+1:end], get_varsymbols(model)))

    chain_no_burnin = MCMCChains.Chains(samples[n_adapts+1:end], get_varsymbols(model))
    describe(chain_no_burnin)

    accept_probs = [s.acceptance_rate for s in stats]

    # Compute average acceptance rate
    mean_acceptance_rate = mean(accept_probs)
    
    # Generate some more data and check predictions
    z_new = Vector{Float64}(undef, Tsubp+1)
    x_exo_new = Vector{Float64}(undef, Tsubp)
    z_new[1] = rand(rng, Normal(0.0, 1.0/sqrt(1.0-β[1]^2-β[2]^2)))
    for t in 2:Tsubp+1
        x_exo_new[t-1] = rand(rng, Normal())
        z_new[t] = β[1]*z_new[t-1] + β[2]*x_exo_new[t-1] + rand(rng, Normal())
    end
    y_new = quantile.(d_true, cdf.(Normal(0.0, 1.0/sqrt(1.0-β[1]^2-β[2]^2)), z_new))
    z_new_est = quantile.(Normal(0.0, 1.0), cdf.(kdest, y_new))
    F_new = hcat(z_new_est[1:end-1], x_exo_new)
    z_new_est = z_new_est[2:end]

    # get predictions
    y_pred = Vector{Float64}(undef, Tsubp)
    y_best_true = Vector{Float64}(undef, Tsubp)
    for t in 1:Tsubp
        y_pred[t] = predict_obs_mcmc(rng, kdest, F_new[t,:], samples[n_adapts+1:end], J, K)[1,1]
        y_best_true[t] = estimate_best_true_var1_cov(rng, d_true, [z_new[t], x_exo_new[t]], β)
    end
    println(sqrt(mean((y_new[2:end] - y_pred).^2)))   # RMSE of model
    println(sqrt(mean((y_new[2:end] .- mean(y)).^2))) # RMSE of null model
    println(sqrt(mean((y_new[2:end] .- y_best_true).^2))) # RMSE of predictions based on true model

    # Again, our approach proves to not be so unreasonable.
end


# Now do a AR(1) process with 1 exogenous covariate
function test_AR1_cov()
    rng = Random.default_rng()
    p = 1          # order
    T = 20   # number of observations
    J = 2 # Number of covariates per variable
    K = 1 # Dimension of response
    M = 1
    Tsubp = T - p
    β = [0.8, 0.4]       # autoregressive parameter, covariate

    z = Vector{Float64}(undef, Tsubp+1)
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
    z = z[2:end]

    model = VARModel(z, F, K, J, M, Tsubp)
    D = LogDensityProblems.dimension(model)
    #θ_init = rand(rng, Normal(), D)
    θ_init = zeros(Float64, D)
    #n_samples = 3_000
    n_samples = 5_000
    n_adapts = 1_000

    #sampler = NUTS(0.8)
    sampler = RandPermGibbs(SliceDoublingOut(2.0; max_proposals=10^4))
    
    samples = composite_gibbs_abstractmcmc(rng, model, sampler, θ_init, n_samples; n_adapts = n_adapts, progress = true)
    #samples = composite_gibbs_mh(rng, model, θ_init, n_samples)
    #samples = composite_gibbs_vi(rng, model, θ_init, N_fac, N_iter_vi, n_samples)

    log_τ_chain = vec(MCMCChains.Chains(samples, get_varsymbols(model))[Symbol("log_τ[1]")])
    num_accepted = sum(log_τ_chain[2:end] .!= log_τ_chain[1:end-1])
    acceptance_ratio = num_accepted / (length(log_τ_chain) - 1)

    log_ξ_chain = vec(MCMCChains.Chains(samples, get_varsymbols(model))[Symbol("log_ξ[1]")])
    num_accepted = sum(log_ξ_chain[2:end] .!= log_ξ_chain[1:end-1])
    acceptance_ratio = num_accepted / (length(log_ξ_chain) - 1)

    chain = MCMCChains.Chains(samples[1001:end], get_varsymbols(model))
    describe(chain)


    plot(MCMCChains.Chains(samples[1001:end], get_varsymbols(model)))
    autocorplot(MCMCChains.Chains(samples[1001:end], get_varsymbols(model)))

    describe(MCMCChains.Chains(samples[1001:3000], get_varsymbols(model)))


    chain_no_burnin = MCMCChains.Chains(samples[B_adapt+1:end], get_varsymbols(model))
    describe(chain_no_burnin)

    accept_probs = [s.acceptance_rate for s in stats]

    # Compute average acceptance rate
    mean_acceptance_rate = mean(accept_probs)
    
    # Generate some more data and check predictions
    z_new = Vector{Float64}(undef, Tsubp+1)
    x_exo_new = Vector{Float64}(undef, Tsubp)
    z_new[1] = rand(rng, Normal(0.0, 1.0/sqrt(1.0-β[1]^2-β[2]^2)))
    for t in 2:Tsubp+1
        x_exo_new[t-1] = rand(rng, Normal())
        z_new[t] = β[1]*z_new[t-1] + β[2]*x_exo_new[t-1] + rand(rng, Normal())
    end
    y_new = quantile.(d_true, cdf.(Normal(0.0, 1.0/sqrt(1.0-β[1]^2-β[2]^2)), z_new))
    z = quantile.(Normal(0.0, 1.0), cdf.(kdest, y_new))
    F = hcat(z[1:end-1], x_exo_new)
    z = z[2:end]

    # get predictions
    y_pred = Vector{Float64}(undef, Tsubp)
    y_best_true = Vector{Float64}(undef, Tsubp)
    for t in 1:Tsubp
        #z_pred_t = mean(F[t] * β_chain)                      # first predict on latent scale
        #y_pred[t] = quantile(marg, cdf.(Normal(), z_pred_t)) # revert to observed scale
        y_pred[t] = predict_obs_mcmc(rng, kdest, F[t,:], samples, J, K)[1,1]
        #y_best_true[t] = quantile(Exponential(4.0), cdf.(Normal(0.0, 1.0/sqrt(1.0-a^2)), z_new[t])) # best prediction based on true modely_best_true[t]
        y_best_true[t] = estimate_best_true_var1_cov(rng, d_true, [z_new[t], x_exo_new[t]], β)
    end
    println(sqrt(mean((y_new[2:end] - y_pred).^2)))   # RMSE of model
    println(sqrt(mean((y_new[2:end] .- mean(y)).^2))) # RMSE of null model
    println(sqrt(mean((y_new[2:end] .- y_best_true).^2))) # RMSE of predictions based on true model

    # Again, our approach proves to not be so unreasonable.
end

#test_AR1_cov()


function test_AR1_cov()
    rng = Random.default_rng()
    p = 1          # order
    T = 50_000   # number of observations
    J = 2 # Number of covariates per variable
    K = 1 # Dimension of response
    M = 1
    Tsubp = T - p
    β = [0.8, 0.4]       # autoregressive parameter, covariate

    z = Vector{Float64}(undef, Tsubp+1)
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
    z = z[2:end]

    model = VARModel(z, F, K, J, M, Tsubp)
    
    # Fit VI posterior
    posterior, ELBOs = fitBayesianSVARCopVI(rng, model, 2000, 4)

    
    # Generate some more data and check predictions
    z_new = Vector{Float64}(undef, Tsubp+1)
    x_exo_new = Vector{Float64}(undef, Tsubp)
    z_new[1] = rand(rng, Normal(0.0, 1.0/sqrt(1.0-β[1]^2-β[2]^2)))
    for t in 2:Tsubp+1
        x_exo_new[t-1] = rand(rng, Normal())
        z_new[t] = β[1]*z_new[t-1] + β[2]*x_exo_new[t-1] + rand(rng, Normal())
    end
    y_new = quantile.(d_true, cdf.(Normal(0.0, 1.0/sqrt(1.0-β[1]^2-β[2]^2)), z_new))
    z_new_est = quantile.(Normal(0.0, 1.0), cdf.(kdest, y_new))
    F_new = hcat(z_new_est[1:end-1], x_exo_new)
    z_new_est = z_new_est[2:end]

    newdata = VARModel(z_new_est, F_new, K, J, M, Tsubp)

    y_pred = predict_response(rng, posterior, [kdest], newdata)
    

    # get predictions
    y_best_true = Vector{Float64}(undef, Tsubp)
    for t in 1:Tsubp
        y_best_true[t] = estimate_best_true_var1_cov(rng, d_true, [z_new[t], x_exo_new[t]], β)
    end
    println(sqrt(mean((y_new[2:end] - y_pred).^2)))   # RMSE of model
    println(sqrt(mean((y_new[2:end] .- mean(y)).^2))) # RMSE of null model
    println(sqrt(mean((y_new[2:end] .- y_best_true).^2))) # RMSE of predictions based on true model

    # Again, our approach proves to not be so unreasonable.
end