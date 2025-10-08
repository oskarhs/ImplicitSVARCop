using Random, Distributions, BSplineKit, StatsPlots, AdvancedHMC, LogDensityProblems, MCMCChains, LinearAlgebra, ForwardDiff, AbstractMCMC, DataFrames, StatsBase
include(joinpath(@__DIR__, "..", "src", "ImplicitSVARCop.jl"))
using .ImplicitSVARCop


function example_normaldata()
    #rng = Random.Xoshiro(1)
    rng = Random.default_rng()

    # Simulate the data:
    T = 5_000
    corr = 0.5
    # Order of VAR process
    p = 2

    z = simulate_normal_VARdata(rng, T; corr=corr)
    y = quantile.(Gamma(1, 2), cdf(Normal(), z))
    
    # Estimate marginal density and transform data to latent scale:
    My = copy(y)
    Mz_est, kdests = fit_marginals(My, SSVKernel())
    #Mz_est = z
    z_est = vec(Mz_est)
    #Mz_est = My
    
    # Design matrix (remember, lags should be in descending order when mobing from left to right)
    F = hcat(
        Mz_est[1:T-2,:], # Lag 2, as in paper
        Mz_est[2:T-1,:]  # Lag 1, as in paper
    )

    # Create VARModel object:
    K = 2
    J = K*p
    M = 1
    df_ξ = 1.0
    model = VARModel(vec(Mz_est[p+1:end,:]), F, K, J, M, T-p; df_ξ=df_ξ)

    # Fit MCMC posterior
    n_samples, n_adapts = 10_000, 2_000

    #D = LogDensityProblems.dimension(model)
    D = 2*K*J + K + div(K*(K-1), 2)
    #θ_init = rand(rng, Normal(), D)
    θ_init = zeros(D)

    sampler_ξ = NUTS(0.75)
    sampler_γ = NUTS(0.75)
    samples = composite_gibbs_abstractmcmc_lkj(rng, model, sampler_ξ, sampler_γ, θ_init, n_samples; n_adapts=n_adapts, progress=true)
    chain = Chains(samples[n_adapts+1:end], ImplicitSVARCop.get_varsymbols_lkj(model))
    describe(chain)

    mean(tanh.(chain[Symbol("γ[1]")].data))
    density(tanh.(chain[Symbol("γ[1]")].data))

    posterior, ELBOs = fitBayesianSVARCopVI_lkj(rng, model, 5000, 15)

    y_pred_VI = predict_response_plugin(rng, posterior, kdests, model)

    λ = 0.1

    βhat_r = (transpose(F) * F + λ * I) \ transpose(F) * y[p+1:end,:]
    Σhat_r = 1/(T-p) * transpose(y[p+1:end,:] - F*βhat_r)*(y[p+1:end,:] - F*βhat_r)
    cov2cor(Σhat_r)

    βhat_r = (transpose(F) * F + λ * I) \ transpose(F) * Mz_est[p+1:end,:]
    Σhat_r = 1/(T-p) * transpose(Mz_est[p+1:end,:] - F*βhat_r)*(Mz_est[p+1:end,:] - F*βhat_r)
    cov2cor(Σhat_r)

    y_pred_MCMC = predict_response_plugin(rng, samples, kdests, model)
    y_pred_ridge = F * βhat_r

    println(sqrt(mean((y[p+1:end,:] - y_pred_MCMC).^2))) # Forecast RMSE, MCMC
    println(sqrt(mean((y[p+1:end,:] - y_pred_VI).^2)))
    println(sqrt(mean((y[p+1:end,:] - y_pred_ridge).^2))) # Forecast RMSE, ridge
end

# This only accepts p=2 as a valid argument
function simulate_normal_VARdata(rng::Random.AbstractRNG, T::Int; corr::Real)
    r2 = 1 # Perhaps we need to adjust these to get a more favorable SNR ratio.
    r3 = 1

    p = 2
    
    # True parameters
    Λ = Diagonal([r2, r3])
    R = [1.0 corr; corr 1.0]
    V_ϵ = Symmetric(Λ * R * Λ)
    βmat = [
        0.2 0.0;
        0.0 0.2;
        0.2 0.0;
        0.2 0.2
    ]

    ϵ = transpose(rand(rng, MvNormal(zeros(2), V_ϵ), T))
    
    # Copula scale:
    z = zeros((T, 2))
    z[1,:] = rand(rng, MvNormal(zeros(2), I))
    z[2,:] = rand(rng, MvNormal(zeros(2), I))
    for t in p+1:T
        z[t,:] = transpose(vcat(z[t-2,:], z[t-1,:])) * βmat + transpose(ϵ[t,:])
    end
    return z
end