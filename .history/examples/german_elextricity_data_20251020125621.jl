using Random, DataFrames, CSV, AdvancedHMC, MCMCChains, MCMCChainsStorage, HDF5
include(joinpath("C:/Users", "yr4426", "Documents", "ImplicitSVARCop", "src", "ImplicitSVARCop.jl"))
using .ImplicitSVARCop


# Note! The following code fits the model to the entire dataset. As such, there is no test set or similar.
function test_example()
    # Number of threads used for sampling (here: equal to the number of chains)
    n_threads = 2

    df = CSV.read(joinpath("C:/Users","yr4426","Documents", "DE.csv"), DataFrame)

    # Variables of interest:
    responses = [:price, :load]

    # Make new covariate summing :wind and :solar
    preds = [:wind, :solar, :month, :hour, :temperature]

    # Make new covariate summing :wind and :solar
    df[!,:renew] = df[!,:solar] + df[!,:wind]

    # Responses
    My = Matrix{Float64}(df[!,responses])

    # Transform the response variables to the latent normal scale.
    Mz_est, kdests = fit_marginals(My, SSVKernel())

    T = 5_000
    #T = size(df, 1) # number of observations
    p = 24          # order of process

    # Extract spline covariates, as we need to normalize these to the interval [0,1]
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
        B_spline_basis_matrix(renew_std[p+1:T], order, dim_basis),       # assumes data on [0,1]
        B_spline_basis_matrix(temperature_std[p+1:T], order, dim_basis), # assumes data on [0,1]
        df[p+1:T,:hour],
        df[p+1:T,:month]
    )

    # Forgot to dummy code :hour, :month. Fucking idiot.
    
    # Create model object
    M = 1
    df_ξ = 1.0
    K = size(Mz_est, 2)
    J = size(F, 2)
    model = VARModel(vec(Mz_est[p+1:T,:]), F, K, J, M, T-p; df_ξ=df_ξ)

    # Compute the VI posterior to get initial values for MCMC
    # Not only does this reduce the burn-in period of the Markov chain by initializing with a value of the parameter from the typical set,
    # it also helps with numerical stability; whn the parameter vector is randomly initialized the gradients tend to be very large, leading to very large proposed steps.
    # This often ends up causing numerical problems when attempting to find a good initial stepsize.
    Random.seed!(1)

    posterior, ELBOs = fitBayesianSVARCopVI_lkj(Random.default_rng(), model, 1000, 40)

    # Set the initial value of the unconstrained parameter vector equal to the variational posterior mean
    θ_init = posterior.μ

    # Run the sampler
    n_adapts = 2_000
    n_samples = 7_000
    δ = 0.8
    sampler_ξ = NUTS(δ)
    sampler_γ = NUTS(δ)

    # Create vector for storing samples
    samples = Vector{Vector{Vector{Float64}}}(undef, n_threads)
    
    # Run MCMC in parallel
    Threads.@threads for i in 1:n_threads
        rng = Random.TaskLocalRNG()
        samples[i] = composite_gibbs_abstractmcmc_lkj(rng, model, sampler_ξ, sampler_γ, θ_init, n_samples; n_adapts=n_adapts, progress=true)
    end

    # Combine samples from different chains in a MCMCChains.Chains object.
    chains_per_thread = map(samples -> MCMCChains.Chains(samples[n_adapts+1:end], ImplicitSVARCop.get_varsymbols_lkj(model)), samples)
    chains = reduce(chainscat, chains_per_thread)

    plot(ess(chains).nt.ess)
    minimum(ess(chains).nt.ess)
    argmin(ess(chains).nt.ess)
    plot(log.(ess(chains).nt.ess))

    plot(posterior.μ)
    plot(mean(chains).nt.mean)

    plot(chains, Symbol("log_ξ[56]"))
    plot(chains, Symbol("β[56]"))

    plot(chains, Symbol("log_ξ[10]"))


    # Write the chains object to file:
    h5open("an_hdf5_file.h5", "w") do f
        write(f, chains)
    end
end