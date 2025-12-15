using Random, DataFrames, CSV, AdvancedHMC
include(joinpath("C:/Users", "yr4426", "Documents", "ImplicitSVARCop", "src", "ImplicitSVARCop.jl"))
using .ImplicitSVARCop

function test_example()
    # Number of threads used for sampling (here: equal to the number of chains)

    df = CSV.read(joinpath("C:/Users","yr4426","Documents", "DE.csv"), DataFrame)

    # Variables of interest:
    responses = [:price, :load]

    # Make new covariate summing :wind and :solar
    preds = [:wind, :solar, :month, :hour, :temperature]

    #= plot(df[!,:price])
    density(df[!,:price])

    plot(df[!,:load])
    density(df[!,:load])

    @show names(df) =#
    df[!,:renew] = df[!,:solar] + df[!,:wind]

    # Responses
    My = Matrix{Float64}(df[!,responses])
    Mz_est, kdests = fit_marginals(My, SSVKernel())

    T = size(df, 1)
    p = 24 # order of process

    renew = Vector{Float64}(df[!,:renew])
    temperature = Vector{Float64}(df[!,:temperature])

    # Normalize covariates to [0,1]:
    renew_min, renew_max = extrema(renew)
    renew_std = (renew .- renew_min) / (renew_max - renew_min)

    temperature_min, temperature_max = extrema(temperature)
    temperature_std = (temperature .- temperature_min) / (temperature_max - temperature_min)

    # Create the design matrix

    # Order and number of spline basis functions for nonlinear effects:
    order = 4
    dim_basis = 10

    F = hcat(
        hcat([Mz_est[lag:T-1-p+lag,:] for lag in 1:p]...),
        B_spline_basis_matrix(renew_std[p+1:end], order, dim_basis),       # assumes data on [0,1]
        B_spline_basis_matrix(temperature_std[p+1:end], order, dim_basis), # assumes data on [0,1]
        df[p+1:T,:hour],
        df[p+1:T,:month]
    )
    
    # Create model object
    M = 1
    df_ξ = 1.0
    K = size(Mz_est, 2)
    J = size(F, 2)
    model = VARModel(vec(Mz_est[p+1:end,:]), F, K, J, M, T-p; df_ξ=df_ξ)

    posterior, ELBOs = fitBayesianSVARCopVI_lkj(rng, model, 2000, 15)


    D = 2*K*J + K + div(K*(K-1), 2)
    θ_init = posterior.μ
    #θ_init = randn(rng, D)

    # Run the sampler
    n_adapts = 1_000
    n_samples = 6_000
    sampler_ξ = NUTS(0.75)
    sampler_γ = NUTS(0.75)

    Random.seed!(1234)
    
    Threads.@threads for i in 1:n_threads
        rng = Random.TaskLocalRNG()
    end

    samples = composite_gibbs_abstractmcmc_lkj(rng, model, sampler_ξ, sampler_γ, θ_init, n_samples; n_adapts=n_adapts, progress=true)


    hcat([Mz_est[lag+1:T-p+lag,:] for lag in 1:p]...)
end