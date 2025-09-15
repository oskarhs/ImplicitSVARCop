using Random, Distributions, BSplineKit, StatsPlots, AdvancedHMC, LogDensityProblems, MCMCChains, LinearAlgebra, ForwardDiff
include(joinpath(@__DIR__, "..", "src", "ImplicitSVARCop.jl"))
using .ImplicitSVARCop

"""
The following example is taken from Klein & Smith (2019), Implicit Copulas from Bayesian Regularized Regression Smoothers.

We implement the Horseshoe Copula prior as detailed in that paper, using a total of 25 spline basis functions on [0,1], with uniform spacings excluding the edges.
The simulations undertaken here follow case 3 in Klein & Smith (2019)
"""
function run_example()
    rng = Random.Xoshiro(1)

    # Simulate the data:
    n = 100
    x, y = simulate_scenario_3(rng, n)
    #x, y = simulate_scenario_3(rng, n)

    order = 4
    dim_basis = 25
    F = B_spline_basis_matrix(x, order, dim_basis)

    # Estimate marginal density and transform data to latent scale:
    My = reshape(y, (length(y), 1))
    Mz_est, kdests = fit_marginals(My, SSVKernel())
    z_est = vec(Mz_est)

    # Create VARModel object:
    J = dim_basis
    K = 1
    M = 1
    model = VARModel(z_est, F, K, J, M, n; df_ξ = 20.0)

    # Fit MCMC posterior
    n_samples, n_adapts = 10, 5_000

    D = LogDensityProblems.dimension(model)
    #metric = DenseEuclideanMetric(D)
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, θ -> logp_joint(model, θ), θ -> logp_and_grad_joint(model, θ))

    #θ_init = rand(rng, Normal(), D)
    θ_init = zeros(Float64, D)

    # Define a leapfrog solver, with the initial step size chosen heuristically
    ϵ_init = find_good_stepsize(hamiltonian, θ_init)
    integrator = Leapfrog(ϵ_init)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(max_depth=8)))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.5, integrator))
    #samples, stats = sample(hamiltonian, kernel, θ_init, n_samples, adaptor, n_adapts; progress=true, (pm_next!) = AdvancedHMC.simple_pm_next!)

    samples = composite_gibbs_mh(rng, model, θ_init, n_samples; progress = true)


    chain = MCMCChains.Chains(samples, get_varsymbols(model))
    describe(chain)

    # Fit variational posterior
    posterior, ELBOs = fitBayesianSVARCopVI(rng, model, 3000, 15)
    
    # Approximate the regression function via Monte Carlo by resampling ϵ
    f = estimate_mean_scenario_3(rng, x; N_mc = 1000)

    # Get new predictions
    newdata = VARModel(z_est, F, K, J, M, n)
    y_pred_VI = predict_response(rng, posterior, kdests, newdata; N_mc = 1000)

    y_pred_MCMC = predict_response(rng, samples[n_adapts+1:end], kdests, newdata)

    # Plotting
    t = LinRange(0, 1, 1001)
    ind = sortperm(x)

    plotdata = VARModel(t, B_spline_basis_matrix(t, order, dim_basis), K, J, M, length(t))
    p = plot(t, estimate_mean_scenario_3(rng, t), label="True mean")
    plot!(p, t, predict_response(rng, posterior, kdests, plotdata), label="VI mean")
    plot!(p, t, predict_response(rng, samples[n_adapts+1:end], kdests, plotdata), label="MCMC mean")

    p = plot(x[ind], f[ind], label="True mean")
    plot!(p, x[ind], y_pred_VI[ind], label="VI mean")
    plot!(p, x[ind], y_pred_MCMC[ind], label="MCMC mean")

    # Something weird is goiing on when using MCMC samples to predict at t.

    # Compute RMSE:
    println(sqrt(mean((f - y_pred_VI).^2)))   # RMSE of model
    println(sqrt(mean((f .- mean(y)).^2))) # RMSE of null model
    println(sqrt(mean((f .- y_pred_MCMC).^2))) # RMSE of predictions based on true model
end

function run_example_bivariate()
    rng = Random.Xoshiro(1)

    # First case h₂(x)
    # Simulate the data:
    n = 10000
    x, y = simulate_scenario_4(rng, n)

    order = 4
    dim_basis = 25
    F = hcat(
        B_spline_basis_matrix(x[:,1], order, dim_basis),
        B_spline_basis_matrix(x[:,2], order, dim_basis)
    )

    # Estimate marginal density and transform data to latent scale:
    My = copy(y)
    Mz_est, kdests = fit_marginals(My, SSVKernel())
    z_est = vec(Mz_est)

    # Create VARModel object:
    K = 2
    J = K*dim_basis
    M = 1
    df_ξ = 1.0
    model = VARModel(z_est, F, K, J, M, n)

    # Fit MCMC posterior
    n_samples, n_adapts = 10_000, 5_000

    D = LogDensityProblems.dimension(model)
    #metric = DenseEuclideanMetric(D)
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, θ -> logp_joint(model, θ), θ -> logp_and_grad_joint(model, θ))

    #θ_init = rand(rng, Normal(), D)
    θ_init = zeros(Float64, D)

    # Define a leapfrog solver, with the initial step size chosen heuristically
    ϵ_init = find_good_stepsize(hamiltonian, θ_init)
    integrator = Leapfrog(ϵ_init)

    # Define NUTS sampler
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(max_depth=8)))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.5, integrator))
    samples, stats = sample(hamiltonian, kernel, θ_init, n_samples, adaptor, n_adapts; progress=true, (pm_next!) = AdvancedHMC.simple_pm_next!)

    chain = MCMCChains.Chains(samples[n_adapts:end], get_varsymbols(model))
    describe(chain)

    # Fit variational posterior
    posterior, ELBOs = fitBayesianSVARCopVI(rng, model, 5000, 15)
    
    # Approximate the regression function via Monte Carlo by resampling ϵ
    f = estimate_mean_scenario_4(rng, x)

    # Get new predictions
    newdata = VARModel(z_est, F, K, J, M, n)
    y_pred_VI = predict_response(rng, posterior, kdests, newdata; N_mc = 500)

    y_pred_MCMC = predict_response(rng, samples[n_adapts+1:end], kdests, newdata)

    # Plotting
    pl = []
    for dim in [1,2]
        xi = x[:,dim]
        ind = sortperm(xi)
        p = plot(xi[ind], f[ind, dim], label="True mean", color="black", lw=3)
        plot!(p, xi[ind], y_pred_VI[ind, dim], label="VI mean", linestyle=:dash)
        #plot!(p, xi[ind], y_pred_MCMC[ind, dim], label="MCMC mean", linestyle=:dash)
        push!(pl, p)
    end
    plot(pl...)

    # Compute RMSE:
    println(sqrt(mean((f - y_pred_VI).^2)))   # RMSE of model
    println(sqrt(mean((f .- mean(y)).^2))) # RMSE of null model
    println(sqrt(mean((f - y_pred_MCMC).^2))) # RMSE of predictions based on true model
end

function simulate_scenario_2(rng::Random.AbstractRNG, n_sim::Int)
    r2 = 0.47

    ϵ = rand(rng, Normal(0.0, r2), n_sim)
    x = rand(rng, Uniform(), n_sim) # nonlinear effect

    # Copula scale:
    z = h2.(x) + ϵ
    u = cdf(Normal(), z)
    y = quantile(Gamma(3, 2), u)
    return x, y
end

function estimate_mean_scenario_2(rng::Random.AbstractRNG, x::AbstractVector; N_mc = 5000)
    r2 = 0.47

    f = Vector{Float64}(undef, length(x))
    e = rand(rng, Normal(0.0, r2), N_mc)
    for i in eachindex(x)
        z = h2(x[i]) .+ e
        u = cdf(Normal(), z)
        y = quantile(Gamma(3, 2), u)
        f[i] = mean(y)
    end
    return f
end

function simulate_scenario_3(rng::Random.AbstractRNG, n_sim::Int)
    r3 = 0.58

    ϵ = rand(rng, Normal(0.0, r3), n_sim)
    x = rand(rng, Uniform(), n_sim) # nonlinear effect

    # Copula scale:
    z = h3.(x) + ϵ
    u = cdf(Normal(), z)
    y = quantile(Gamma(3, 2), u)
    return x, y
end


function estimate_mean_scenario_3(rng::Random.AbstractRNG, x::AbstractVector; N_mc = 5000)
    r3 = 0.58

    f = Vector{Float64}(undef, length(x))
    e = rand(rng, Normal(0.0, r3), N_mc)
    for i in eachindex(x)
        z = h3(x[i]) .+ e
        u = cdf(Normal(), z)
        y = quantile(Gamma(3, 2), u)
        f[i] = mean(y)
    end
    return f
end

function simulate_scenario_4(rng::Random.AbstractRNG, n_sim::Int)
    r2 = 0.47
    r3 = 0.58

    D = Diagonal([r2, r3])
    R = [1.0 0.7; 0.7 1.0]
    V_ϵ = Symmetric(D * R * D)
    ϵ = transpose(rand(rng, MvNormal(zeros(2), V_ϵ), n_sim))
    x = rand(rng, Uniform(), (n_sim, 2)) # nonlinear effect

    # Copula scale:
    z = h_bivariate(x) + ϵ
    u = cdf(Normal(), z)
    y = quantile(Gamma(3, 2), u)
    return x, y
end


function estimate_mean_scenario_4(rng::Random.AbstractRNG, x::AbstractArray; N_mc = 1000)
    r2 = 0.47
    r3 = 0.58
    n = size(x, 1)

    f = Matrix{Float64}(undef, size(x, 1), 2)
    D = Diagonal([r2, r3])
    R = [1.0 0.7; 0.7 1.0]
    V_ϵ = Symmetric(D * R * D)
    e = transpose(rand(rng, MvNormal(zeros(2), V_ϵ), N_mc))
    for i in 1:n
        z = h_bivariate([x[i,1] x[i,2]]) .+ e
        u = cdf(Normal(), z)
        y = quantile(Gamma(3, 2), u)
        f[i,:] = mean(y, dims=1)
    end
    return f
end

h2(x) = sin(10.0*pi*x)
h3(x) = 0.25 * ( pdf(Normal(0.15, 0.05), x) + pdf(Normal(0.6, 0.2), x) )

function h_bivariate(x)
    return hcat(h2.(x[:,1]), h3.(x[:,2]))
end




function test_derivatives()
    rng = Random.Xoshiro(1)

    # First case h₂(x)
    # Simulate the data:
    n = 100
    #x, y = simulate_scenario_2(rng, n)
    x, y = simulate_scenario_3(rng, n)

    order = 4
    dim_basis = 25
    F = B_spline_basis_matrix(x, order, dim_basis)

    # Estimate marginal density and transform data to latent scale:
    My = reshape(y, (length(y), 1))
    Mz_est, kdests = fit_marginals(My, SSVKernel())
    z_est = vec(Mz_est)

    # Create VARModel object:
    J = dim_basis
    K = 1
    M = 1
    df_ξ = 1.0
    model = VARModel(z_est, F, K, J, M, n; df_ξ = df_ξ)

    D = LogDensityProblems.dimension(model)
    θ = rand(rng, Normal(), D)

    β = θ[1:K*J]
    log_ξ = θ[K*J+1:2*K*J]
    log_τ = θ[2*K*J+1:2*K*J+K]
    M_γ = K*M - div(M*(M+1), 2) + M
    γ = θ[2*K*J+K+1:2*K*J+K+M_γ]

    
    f = θ -> ImplicitSVARCop.logp_joint_autodiff(model, θ)
    val1, g1 = f(θ), ForwardDiff.hessian(f, θ)

    obj, g2 = value_and_jacobian(
            ImplicitSVARCop.grad_logp_conditional_ξ_nt,
            AutoForwardDiff(),
            log_ξ,
            Constant(map(exp, log_τ)),
            Constant(β),
            Constant(inv(compute_Σ(γ, K, M))),
            Constant(model.z),
            Constant(model.F),
            Constant(model.F_sq),
            Constant(model.J),
            Constant(model.K),
            Constant(model.Tsubp),
            Constant(model.df_ξ)
        )

    sum(abs.(g1 - g2))
    get_varsymbols(model)

    plot((g1[1:end-1] - g2[1:end-1])./abs.(g1[1:end-1]))
end