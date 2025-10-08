using Random, Distributions, BSplineKit, AdvancedHMC, LogDensityProblems, MCMCChains, LinearAlgebra, ForwardDiff, AbstractMCMC, DataFrames, StatsBase, CSV
#include(joinpath(@__DIR__, "..", "src", "ImplicitSVARCop.jl"))
include(joinpath(@__DIR__, "ImplicitSVARCop.jl"))
using .ImplicitSVARCop


function test_correlations()
    #rng = Random.Xoshiro(1)
    rng = Random.default_rng()

    # Data parameters:
    T = 500
    corr = 0.5
    # Order of VAR process
    p = 2

    # Set model/
    K = 2
    J = K*p
    M = 1
    df_ξ = 1.0
    
    # Fit MCMC posterior
    n_samples, n_adapts = 10_000, 2_000

    D = 2*K*J + K + div(K*(K-1), 2)
    #θ_init = rand(rng, Normal(), D)
    θ_init = zeros(D)

    sampler_ξ = NUTS(0.75)
    sampler_γ = NUTS(0.75)

    n_mc = 200
    n_threads = 4
    rngs = [Random.Xoshiro(i) for i in 1:n_threads]

    mean_corr_est = Vector{Float64}(undef, n_mc)
    mean_corr_true = Vector{Float64}(undef, n_mc)
    Base.Threads.@threads for i in 1:n_threads
        rng = rngs[i]
        for j in i:n_threads:n_mc
            z = simulate_normal_VARdata(rng, T; corr=corr)
            y = quantile.(Gamma(1, 2), cdf(Normal(), z))
            
            # Estimate marginal density and transform data to latent scale:
            My = copy(y)
            Mz_est, kdests = fit_marginals(My, SSVKernel())
            Mz_true = copy(z)
            
            # Design matrix (remember, lags should be in descending order when mobing from left to right)
            F = hcat(
                Mz_est[1:T-2,:], # Lag 2, as in paper
                Mz_est[2:T-1,:]  # Lag 1, as in paper
            )
            F_true = hcat(
                Mz_true[1:T-2,:], # Lag 2, as in paper
                Mz_true[2:T-1,:]  # Lag 1, as in paper
            )

            model = VARModel(vec(Mz_est[p+1:end,:]), F, K, J, M, T-p; df_ξ=df_ξ)
            samples = composite_gibbs_abstractmcmc_lkj(rng, model, sampler_ξ, sampler_γ, θ_init, n_samples; n_adapts=n_adapts, progress=true)
            chain = Chains(samples[n_adapts+1:end], ImplicitSVARCop.get_varsymbols_lkj(model))
            mean_corr_est[j] = mean(tanh.(chain[Symbol("γ[1]")].data))

            model = VARModel(vec(Mz_true[p+1:end,:]), F_true, K, J, M, T-p; df_ξ=df_ξ)
            samples = composite_gibbs_abstractmcmc_lkj(rng, model, sampler_ξ, sampler_γ, θ_init, n_samples; n_adapts=n_adapts, progress=true)
            chain = Chains(samples[n_adapts+1:end], ImplicitSVARCop.get_varsymbols_lkj(model))
            mean_corr_true[j] = mean(tanh.(chain[Symbol("γ[1]")].data))
        end
    end

    meandata = DataFrame(:mean_corr_est => mean_corr_est, :mean_corr_true => mean_corr_true)
    CSV.write("corrdata.csv", meandata)
end

test_correlations()