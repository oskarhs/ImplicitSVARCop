using Random, Distributions, BSplineKit, StatsPlots, AdvancedHMC, LogDensityProblems, MCMCChains, LinearAlgebra, ForwardDiff, SliceSampling, AbstractMCMC
include(joinpath(@__DIR__, "..", "src", "ImplicitSVARCop.jl"))
using .ImplicitSVARCop

function run_bivariate_var_example()
    rng = Random.Xoshiro(1)

    # First case h₂(x)
    # Simulate the data:
    n = 500
    corr = 0.2
    x, y = simulate_scenario_4(rng, n; corr=corr)

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
    model = VARModel(z_est, F, K, J, M, n; df_ξ=df_ξ)

    # Fit MCMC posterior
    n_samples, n_adapts = 10_000, 2_000

    #D = LogDensityProblems.dimension(model)
    D = 2*K*J+K+1
    θ_init = rand(rng, Normal(), D)

    sampler_ξ = NUTS(0.7)
    sampler_ρ = NUTS(0.7)
    samples = composite_gibbs_abstractmcmc_lkj(rng, model, sampler_ξ, sampler_ρ, θ_init, n_samples; n_adapts=n_adapts, progress=true)

    chain = MCMCChains.Chains(samples[n_adapts:end], get_varsymbols(model))
    describe(chain)

    chain = MCMCChains.Chains(samples[n_adapts:end])
    plot(chain, :param_203)
    mean(tanh.(chain[:param_203].data))
    density(tanh.(chain[:param_203].data))

    # Fit variational posterior
    posterior, ELBOs = fitBayesianSVARCopVI(rng, model, 5000, 15)
    
    # Approximate the regression function via Monte Carlo by resampling ϵ
    f = estimate_mean_scenario_4(rng, x)

    # Get new predictions
    newdata = VARModel(z_est, F, K, J, M, n)
    y_pred_VI = predict_response_plugin(rng, posterior, kdests, newdata; N_mc = 500)

    y_pred_MCMC = predict_response_plugin(rng, samples[n_adapts+1:end], kdests, newdata)

    # Plotting
    pl = []
    for dim in [1,2]
        xi = x[:,dim]
        ind = sortperm(xi)
        p = plot(xi[ind], f[ind, dim], label="True mean", color="black", lw=3)
        plot!(p, xi[ind], y_pred_VI[ind, dim], label="VI mean", linestyle=:dash)
        plot!(p, xi[ind], y_pred_MCMC[ind, dim], label="MCMC mean", linestyle=:dash)
        push!(pl, p)
    end
    plot(pl...)

    # Compute RMSE:
    println(sqrt(mean((f - y_pred_VI).^2)))   # RMSE of model
    println(sqrt(mean((f .- mean(y)).^2))) # RMSE of null model
    println(sqrt(mean((f - y_pred_MCMC).^2))) # RMSE of predictions based on true model
end


function simulate_VARdata(rng::Random.AbstractRNG, n_sim::Int; corr::Real)
    r2 = 0.47 # Perhaps we need to adjust these to get a more favorable SNR ratio.
    r3 = 0.58
    p = 2

    # True parameters
    D = Diagonal([r2, r3])
    R = [1.0 corr; corr 1.0]
    V_ϵ = Symmetric(D * R * D)
    βmat = [
        0.2 0.0;
        0.0 0.3;
        0.1 0.0;
        0.4 0.4
    ]

    ϵ = transpose(rand(rng, MvNormal(zeros(2), V_ϵ), n_sim))
    
    # Simulate from a copula to introduce dependence between covariates
    x = transpose(cdf.(Normal(), rand(rng, MvNormal(zeros(2), [1.0 0.4; 0.4 1.0]), n_sim)))

    # Copula scale:
    z = zeros((n_sim, 2))
    for t in p+1:n_sim
        z[t,:] = transpose(vcat(z[t-2,:], z[t-1,:])) * βmat + hcat(h2(x[t,1]), h3(x[t,2])) + transpose(ϵ[t,:])
    end
    z = z + h_bivariate(x)
    u = cdf(Normal(), z)
    y = quantile(Gamma(3, 2), u)
    return x, y
end


function estimate_mean_VARdata(rng::Random.AbstractRNG, x::AbstractArray; corr::Real, N_mc = 1000)
    r2 = 0.47
    r3 = 0.58
    n = size(x, 1)

    f = Matrix{Float64}(undef, size(x, 1), 2)
    D = Diagonal([r2, r3])
    R = [1.0 corr; corr 1.0]
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
h_bivariate(x) = hcat(h2.(x[:,1]), h3.(x[:,2]))