using Random, Distributions, BSplineKit, StatsPlots, AdvancedHMC, LogDensityProblems, MCMCChains, LinearAlgebra, ForwardDiff, SliceSampling, AbstractMCMC
include(joinpath(@__DIR__, "..", "src", "ImplicitSVARCop.jl"))
using .ImplicitSVARCop

function run_bivariate_var_example()
    rng = Random.Xoshiro(1)

    # First case h₂(x)
    # Simulate the data:
    T = 2000
    corr = 0.2
    # Order of VAR process
    p = 2

    x, y = simulate_VARdata(rng, T; corr=corr, p = p)

    order = 4
    dim_basis = 25
    
    # Estimate marginal density and transform data to latent scale:
    My = copy(y)
    Mz_est, kdests = fit_marginals(My, SSVKernel())
    z_est = vec(Mz_est)

    # Design matrix (remember, lags should be in descending order when mobing from left to right)
    F = hcat(
        Mz_est[1:T-2,:], # Lag 2, as in paper
        Mz_est[2:T-1,:], # Lag 1, as in paper
        B_spline_basis_matrix(x[p+1:T,1], order, dim_basis),
        B_spline_basis_matrix(x[p+1:T,2], order, dim_basis)
    )

    # Create VARModel object:
    K = 2
    J = K*dim_basis+K*p
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
    sampler_ρ = NUTS(0.75)
    samples = composite_gibbs_abstractmcmc_lkj(rng, model, sampler_ξ, sampler_ρ, θ_init, n_samples; n_adapts=n_adapts, progress=true)

    chain = MCMCChains.Chains(samples[n_adapts:end], ImplicitSVARCop.get_varsymbols_lkj(model))
    describe(chain)

    mean(tanh.(chain[Symbol("atanh_ρ[1]")].data))

    # Fit variational posterior
    posterior, ELBOs = fitBayesianSVARCopVI(rng, model, 5000, 15)

    posterior, ELBOs = fitBayesianSVARCopVI_lkj(rng, model, 5000, 15)

    t = LinRange(-1+1e-10, 1-1e-10, 1001)
    μ = posterior.μ[end]
    σ = sqrt(cov(posterior)[end,end])
    p = density(tanh.(chain[Symbol("atanh_ρ[1]")].data), label="MCMC", xlims=[0.10, 0.3])

    plot!(p, t, pdf.(Normal(μ, σ), atanh.(t))./(1 .- abs2.(t)), label="VI")

    
    # Approximate the regression function via Monte Carlo by resampling ϵ
    f = estimate_mean_scenario_4(rng, x)

    # Get new predictions
    newdata = VARModel(vec(Mz_est[p+1:end,:]), F, K, J, M, T-p)
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


function run_bivariate_example2()
    rng = Random.Xoshiro(1)

    # First case h₂(x)
    # Simulate the data:
    T = 1000
    corr = 0.4
    # Order of VAR process
    p = 2

    order = 4
    dim_basis = 25

    K = 2
    #J = K*dim_basis+K*p
    J = K*p
    M = 1

    x, y, Mz = simulate_VARdata_truemodel(rng, order, dim_basis, T, J; corr=corr, p = p)
    
    # Estimate marginal density and transform data to latent scale:
    My = copy(y)
    Mz_est, kdests = fit_marginals(My, SSVKernel())
    z_est = vec(Mz_est)

    F = hcat(
        Mz_est[1:T-2,:], # Lag 2, as in paper
        Mz_est[2:T-1,:], # Lag 1, as in paper
        B_spline_basis_matrix(x[p+1:T,1], order, dim_basis),
        B_spline_basis_matrix(x[p+1:T,2], order, dim_basis)
    )
#=     F = hcat(
        Mz_est[1:T-2,:], # Lag 2, as in paper
        Mz_est[2:T-1,:], # Lag 1, as in paper
    ) =#

    # Create VARModel object:
    df_ξ = 1.0
    model = VARModel(vec(Mz_est[p+1:end,:]), F, K, J, M, T-p; df_ξ=df_ξ)

    # Fit MCMC posterior
    n_samples, n_adapts = 10_000, 2_000

    #D = LogDensityProblems.dimension(model)
    D = 2*K*J + K + div(K*(K-1), 2)
    #θ_init = rand(rng, Normal(), D)
    θ_init = zeros(D)

    sampler_ξ = NUTS(0.75)
    sampler_ρ = NUTS(0.75)
    samples = composite_gibbs_abstractmcmc_lkj(rng, model, sampler_ξ, sampler_ρ, θ_init, n_samples; n_adapts=n_adapts, progress=true)

    chain = MCMCChains.Chains(samples[n_adapts:end], ImplicitSVARCop.get_varsymbols_lkj(model))
    describe(chain)

    newdata = VARModel(vec(Mz_est[p+1:end,:]), F, K, J, M, T-p)
    z_pred = predict_linpred_plugin(rng, samples[n_adapts+1:end], kdests, newdata)
    scatter(Mz_est[2:end-1,1], z_pred[:,1])

    cov(Mz_est[2:end-1,1], z_pred[:,1])

    mean(tanh.(chain[Symbol("atanh_ρ[1]")].data))

    # Fit variational posterior
    posterior, ELBOs = fitBayesianSVARCopVI(rng, model, 5000, 15)

    posterior, ELBOs = fitBayesianSVARCopVI_lkj(rng, model, 5000, 15)

    t = LinRange(-1+1e-10, 1-1e-10, 10001)
    μ = posterior.μ[end]
    σ = sqrt(cov(posterior)[end,end])
    pl = density(tanh.(chain[Symbol("atanh_ρ[1]")].data), label="MCMC", xlims=[0.2, 0.6])
    plot!(pl, t, pdf.(Normal(μ, σ), atanh.(t))./(1 .- abs2.(t)), label="VI")

    # Approximate the regression function via Monte Carlo by resampling ϵ
    f = estimate_mean_VARdata(rng, x, Mz[p+1:end,:], K, J, T; corr = corr)

    # Get new predictions
    newdata = VARModel(vec(Mz_est[p+1:end,:]), F, K, J, M, T-p)
    y_pred_VI = predict_response_plugin(rng, posterior, kdests, newdata; N_mc = 500)
    y_pred_MCMC = predict_response_plugin(rng, samples[n_adapts+1:end], kdests, newdata)

    println(sqrt(mean((f - y_pred_VI).^2)))   # RMSE of model
    println(sqrt(mean((f .- mean(y)).^2))) # RMSE of null model
    println(sqrt(mean((f - y_pred_MCMC).^2))) # RMSE of predictions based on true model

    println(sqrt(mean((My[p+1:end,:] - y_pred_VI).^2)))   # RMSE of model
    println(sqrt(mean((My[p+1:end,:] .- mean(y)).^2))) # RMSE of null model
    println(sqrt(mean((My[p+1:end,:] - y_pred_MCMC).^2))) # RMSE of predictions based on true model
    println(sqrt(mean((My[p+1:end,:] - f).^2)))   # RMSE of model

end

# This only accepts p=2 as a valid argument
function simulate_VARdata(rng::Random.AbstractRNG, T::Int; corr::Real, p::Int)
    r2 = 0.47 # Perhaps we need to adjust these to get a more favorable SNR ratio.
    r3 = 0.58
    
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
    
    # Simulate from a copula to introduce dependence between covariates
    x = transpose(cdf.(Normal(), rand(rng, MvNormal(zeros(2), [1.0 0.4; 0.4 1.0]), T)))

    # Copula scale:
    z = zeros((T, 2))
    z[1,:] = rand(rng, MvNormal(zeros(2), I))
    z[2,:] = rand(rng, MvNormal(zeros(2), I))
    for t in p+1:T
        #z[t,:] = transpose(vcat(z[t-2,:], z[t-1,:])) * βmat + hcat(h2(x[t,1]), h3(x[t,2])) + transpose(ϵ[t,:])
        z[t,:] = transpose(vcat(z[t-2,:], z[t-1,:])) * βmat + transpose(ϵ[t,:])
    end
    u = cdf(Normal(), z)
    y = quantile(Gamma(3, 2), u)
    return x, y
end

# This constructs the design matrix after simulating the data (THis is needed to compute S)
# This only accepts p=2 as a valid argument
function simulate_VARdata_truemodel(rng::Random.AbstractRNG, order::Int, dim_basis::Int, T, J; corr::Real, p::Int)
    r2 = 1.0 # Perhaps we need to adjust these to get a more favorable SNR ratio.
    r3 = 1.0
    
    # True parameters
    Λ = Diagonal([r2, r3])
    R = [1.0 corr; corr 1.0]
    V_ϵ = Symmetric(Λ * R * Λ)
    βmat = [
        0.5 0.0;
        0.0 0.5;
        0.5 0.0;
        0.5 0.5
    ]

    ϵ = transpose(rand(rng, MvNormal(zeros(2), V_ϵ), T))
    
    # Simulate from a copula to introduce dependence between covariates
    x = transpose(cdf.(Normal(), rand(rng, MvNormal(zeros(2), [1.0 0.4; 0.4 1.0]), T)))

    # "True" values of ξ^2
    ξ2 = ones(Float64, J*K)

    # Copula scale:
    z = zeros((T, 2))
    z[1,:] = rand(rng, MvNormal(zeros(2), I))
    z[2,:] = rand(rng, MvNormal(zeros(2), I))
    for t in p+1:T
        #= F_t = hcat(z[t-2,:]', z[t-1,:]', B_spline_basis_matrix([x[t,1]], order, dim_basis), B_spline_basis_matrix([x[t,2]], order, dim_basis))
        s_t1 = 1.0 / sqrt( 1.0 + dot( ξ2[1:J], F_t.^2 ) )
        s_t2 = 1.0 / sqrt( 1.0 + dot( ξ2[J+1:2*J], F_t.^2 ) )
        z[t,:] = hcat(s_t1, s_t2) .* (transpose(vcat(z[t-2,:], z[t-1,:])) * βmat + hcat(h2(x[t,1]), h3(x[t,2])) + transpose(ϵ[t,:])) =#
        s_t1 = 1.0 / sqrt( 1.0 + dot( ξ2[1:J], vcat(z[t-2,:], z[t-1,:]).^2 ) )
        s_t2 = 1.0 / sqrt( 1.0 + dot( ξ2[J+1:2*J], vcat(z[t-2,:], z[t-1,:]).^2 ) )
        z[t,:] = hcat(s_t1, s_t2) .* (transpose(vcat(z[t-2,:], z[t-1,:])) * βmat + transpose(ϵ[t,:]))
       
    end
    u = cdf(Normal(), z)
    y = quantile(Gamma(3, 2), u)
    return x, y, z
end


function estimate_mean_VARdata(rng::Random.AbstractRNG, Mz::AbstractArray, x::AbstractArray, K::Int, J::Int, T::Int; corr::Real, N_mc = 1000)
    r2 = 1.0
    r3 = 1.0
    n = size(x, 1)
    p = 2

    ξ2 = ones(Float64, J*K)

    f = Matrix{Float64}(undef, size(x, 1), 2)
    Λ = Diagonal([r2, r3])
    R = [1.0 corr; corr 1.0]
    V_ϵ = Symmetric(Λ * R * Λ)
    e = transpose(rand(rng, MvNormal(zeros(2), V_ϵ), N_mc))
    for t in p+1:T
        F_t = hcat(Mz[t-2,:]', Mz[t-1,:]', h2(x[t,1]), h3(x[t,2]))
        s_t1 = 1.0 / sqrt( 1.0 + dot( ξ2[1:J], F_t.^2 ) )
        s_t2 = 1.0 / sqrt( 1.0 + dot( ξ2[J+1:2*J], F_t.^2 ) )
        z = hcat(s_t1, s_t2) .* (transpose(vcat(Mz[t-2,:], Mz[t-1,:])) * βmat + hcat(h2(x[t,1]), h3(x[t,2])) + e)
        u = cdf(Normal(), z)
        y = quantile(Gamma(3, 2), u)
        f[i,:] = mean(y, dims=1)
    end
    return f
end

h2(x) = sin(10.0*pi*x)
h3(x) = 0.25 * ( pdf(Normal(0.15, 0.05), x) + pdf(Normal(0.6, 0.2), x) )
h_bivariate(x) = hcat(h2.(x[:,1]), h3.(x[:,2]))
