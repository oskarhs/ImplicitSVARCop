include(joinpath(@__DIR__, "turing_model.jl"))
include(joinpath(@__DIR__, "conditional_beta.jl"))
include(joinpath(@__DIR__, "composite_gibbs.jl"))

function estimate_best_true_model(rng, d_true, x, a)
    N_mc = 200 # number of Monte Carlo samples used to compute the estimate.
    # sample a random row from the chain:
    y_pred = Vector{Float64}(undef, N_mc)
    for j = 1:N_mc
        z = a*x + rand(rng, Normal())                      # first predict on latent scale
        y_pred[j] = quantile(d_true, cdf(Normal(0.0, 1.0/sqrt(1.0-a^2)), z))
    end
    return mean(y_pred)
end

# First example: latent process is AR(1), data process is an inverse gaussian copula with T-distributed marginals.
function test_ar1()
    rng = Xoshiro(1)
    #rng = Random.default_rng()
    p = 1          # order
    Tsubp = 49999   # number of observations
    a = 0.8        # autoregressive parameter

    z = Vector{Float64}(undef, Tsubp+1)
    z[1] = rand(rng, Normal(0.0, 1.0/sqrt(1.0-a^2)))
    for t in 2:Tsubp+1
        z[t] = a*z[t-1] + rand(rng, Normal())
    end
    d_true = Exponential(4.0)
    y = quantile.(d_true, cdf.(Normal(0.0, 1.0/sqrt(1.0-a^2)), z))

    # Estimate the marginals: (here we do it parametrically)
    kdest = fit(UnivariateKDE, y, SSVKernel())
    # Normal inverse copula transformation
    z = quantile.(Normal(0.0, 1.0), cdf.(kdest, y))
    # NB! because of the standardization matrix S we get a different likelihood than the one of the latent normal model.

    F = reshape(z[1:end-1], Tsubp, 1) # need this to be a matrix
    z = z[2:end]

    model = implicit_var(z, F, 1, 1)

    # Compute the maximum likelihood estimate:
    println(maximum_a_posteriori(model)) # this is at the very least consistent with the posterior mean.

    # Simulation from the posterior:
    ad = AutoForwardDiff()
    #ad = AutoReverseDiff(compile=true)
    chain = sample(model, Turing.NUTS(adtype=ad), 1_000)
    describe(chain)
    
    β_chain = chain[Symbol("β[1]")].data
    #τ_chain = chain[Symbol("τ[1]")].data
    τ_chain = fill(1.0, size(β_chain))
    println("Mean β: ", mean(β_chain))
    
    # Generate some more data and check predictions
    z_new = Vector{Float64}(undef, Tsubp+1)
    z_new[1] = rand(rng, Normal(0.0, 1.0/sqrt(1.0-a^2)))
    for t in 2:Tsubp+1
        z_new[t] = a*z_new[t-1] + rand(rng, Normal())
    end
    y_new = quantile.(Exponential(4.0), cdf.(Normal(0.0, 1.0/sqrt(1.0-a^2)), z_new))
    z = quantile.(Normal(0.0, 1.0), cdf.(kdest, y_new))
    F = reshape(z[1:end-1], Tsubp, 1) # need this to be a matrix
    z = z[2:end]

    # get predictions
    y_pred = Vector{Float64}(undef, Tsubp)
    y_best_true = Vector{Float64}(undef, Tsubp)
    for t in 1:Tsubp
        y_pred[t] = predict_obs(rng, kdest, [F[t]], β_chain, τ_chain, 1, 1)[1,1]
        y_best_true[t] = estimate_best_true_model(rng, d_true, z_new[t], a)
    end
    println(sqrt(mean((y_new[2:end] - y_pred).^2)))   # RMSE of model
    println(sqrt(mean((y_new[2:end] .- mean(y)).^2))) # RMSE of null model
    println(sqrt(mean((y_new[2:end] .- y_best_true).^2))) # RMSE of predictions based on true model

    # Okay, so we are doing quite a lot better than the null model, and we are not too far off the true model in terms of RMSE. This is reassuring.
end

test_ar1()

function test_ar1_gibbs()
    rng = Xoshiro(1)
    #rng = Random.default_rng()
    p = 1          # order
    Tsubp = 999   # number of observations
    a = 0.8        # autoregressive parameter

    z = Vector{Float64}(undef, Tsubp+1)
    z[1] = rand(rng, Normal(0.0, 1.0/sqrt(1.0-a^2)))
    for t in 2:Tsubp+1
        z[t] = a*z[t-1] + rand(rng, Normal())
    end
    d_true = Exponential(4.0)
    y = quantile.(d_true, cdf.(Normal(0.0, 1.0/sqrt(1.0-a^2)), z))

    # Estimate the marginals: (here we do it parametrically)
    #marg = fit_mle(Exponential, y)
    kdest = fit(UnivariateKDE, y, SSVKernel())
    # Normal inverse copula transformation
    #z = quantile.(Normal(0.0, 1.0), cdf.(marg, y))
    z = quantile.(Normal(0.0, 1.0), cdf.(kdest, y))
    # NB! because of the standardization matrix S we get a different likelihood than the one of the latent normal model.

    F = reshape(z[1:end-1], Tsubp, 1) # need this to be a matrix
    z = z[2:end]

    F_tilde = F .^2
    
    J = 1; K = 1

    #= P_root = Diagonal(diagm([1.0]))
    inv_Σ = diagm([1.0])
    Σ_β = inv( kron(inv_Σ, F' * F) + P_root * kron(inv_Σ, I(1))* P_root)
    println(Σ_β)

    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_tilde * reshape( diag(inv(P_root)).^2, (J, K) ) ) ) )
    μ_β = Σ_β * kron(I(J), F') * kron(inv_Σ, I(Tsubp)) * inv_S * z
    println(μ_β) =#


    num_samples = 1000
    β_chain = composite_gibbs(z, F, J, K, num_samples)
    println("Mean β: ", mean(β_chain))
    
    # Generate some more data and check predictions
    z_new = Vector{Float64}(undef, Tsubp+1)
    z_new[1] = rand(rng, Normal(0.0, 1.0/sqrt(1.0-a^2)))
    for t in 2:Tsubp+1
        z_new[t] = a*z_new[t-1] + rand(rng, Normal())
    end
    y_new = quantile.(Exponential(4.0), cdf.(Normal(0.0, 1.0/sqrt(1.0-a^2)), z_new))
    z = quantile.(Normal(0.0, 1.0), cdf.(kdest, y_new))
    F = reshape(z[1:end-1], Tsubp, 1) # need this to be a matrix
    z = z[2:end]

    # get predictions
    y_pred = Vector{Float64}(undef, Tsubp)
    y_best_true = Vector{Float64}(undef, Tsubp)
    for t in 1:Tsubp
        y_pred[t] = predict_obs_gibbs(rng, kdest, [F[t]], β_chain, 1, 1)[1,1]
        y_best_true[t] = estimate_best_true_model(rng, d_true, z_new[t], a)
    end
    println(sqrt(mean((y_new[2:end] - y_pred).^2)))   # RMSE of model
    println(sqrt(mean((y_new[2:end] .- mean(y)).^2))) # RMSE of null model
    println(sqrt(mean((y_new[2:end] .- y_best_true).^2))) # RMSE of predictions based on true model

    # Okay, so we are doing quite a lot better than the null model, and we are not too far off the true model in terms of RMSE. This is reassuring.
end

test_ar1_gibbs()