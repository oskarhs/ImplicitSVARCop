include(joinpath(@__DIR__, "..", "src", "turing_model.jl"))

function estimate_best_true_var1_cov(d_true, x, β)
    N_mc = 500 # number of Monte Carlo samples used to compute the estimate.
    # sample a random row from the chain:
    y_pred = Vector{Float64}(undef, N_mc)
    for j = 1:N_mc
        z = dot(x, β) + rand(Normal())                      # first predict on latent scale
        y_pred[j] = quantile(d_true, cdf(Normal(0.0, 1.0/sqrt(1.0-β[1]^2-β[2]^2)), z))
    end
    return mean(y_pred)
end

# Now do a AR(1) process with 1 exogenous covariate
function test_AR1_cov()
    rng = Random.default_rng()
    p = 1          # order
    T = 1_000   # number of observations
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

    marg = fit_mle(Exponential, y)
    z = quantile.(Normal(0.0, 1.0), cdf.(marg, y))

    F = hcat(z[1:end-1], x_exo)
    z = z[2:end]
    J = 2
    K = 1

    model = implicit_var(z, F, J, K)
    #describe(sample(model, Prior(), 1_000))

    # Compute the MAP estimate:
    println(maximum_a_posteriori(model))

    # Simulation from the posterior:
    ad = AutoForwardDiff()
    #ad = AutoEnzyme()
    #ad = AutoReverseDiff(compile=true)
    chain = sample(model, Turing.NUTS(adtype=ad), 1_000)
    describe(chain)
    β_chain = Array(chain[[Symbol("β[1]"), Symbol("β[2]")]])
    τ_chain = Array(chain[[Symbol("τ[1]"), Symbol("τ[2]")]])
    
    # Generate some more data and check predictions
    z_new = Vector{Float64}(undef, Tsubp+1)
    x_exo_new = Vector{Float64}(undef, Tsubp)
    z_new[1] = rand(rng, Normal(0.0, 1.0/sqrt(1.0-β[1]^2-β[2]^2)))
    for t in 2:Tsubp+1
        x_exo_new[t-1] = rand(rng, Normal())
        z_new[t] = β[1]*z_new[t-1] + β[2]*x_exo_new[t-1] + rand(rng, Normal())
    end
    y_new = quantile.(Exponential(4.0), cdf.(Normal(0.0, 1.0/sqrt(1.0-β[1]^2-β[2]^2)), z_new))
    z = quantile.(Normal(0.0, 1.0), cdf.(marg, y_new))
    F = hcat(z[1:end-1], x_exo_new)
    z = z[2:end]

    # get predictions
    y_pred = Vector{Float64}(undef, Tsubp)
    y_best_true = Vector{Float64}(undef, Tsubp)
    for t in 1:Tsubp
        #z_pred_t = mean(F[t] * β_chain)                      # first predict on latent scale
        #y_pred[t] = quantile(marg, cdf.(Normal(), z_pred_t)) # revert to observed scale
        y_pred[t] = predict_obs(marg, F[t,:], β_chain, τ_chain, J, K)[1,1]
        #y_best_true[t] = quantile(Exponential(4.0), cdf.(Normal(0.0, 1.0/sqrt(1.0-a^2)), z_new[t])) # best prediction based on true modely_best_true[t]
        y_best_true[t] = estimate_best_true_var1_cov(d_true, [z_new[t], x_exo_new[t]], β)
    end
    println(sqrt(mean((y_new[2:end] - y_pred).^2)))   # RMSE of model
    println(sqrt(mean((y_new[2:end] .- mean(y)).^2))) # RMSE of null model
    println(sqrt(mean((y_new[2:end] .- y_best_true).^2))) # RMSE of predictions based on true model

    # Again, our approach proves to not be so unreasonable.
end

test_AR1_cov()
