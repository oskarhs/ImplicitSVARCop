"""
    predict_response([rng=Random.default_rng()], posterior_samples::Vector{Vector{Float64}}, kdests::Vector{UnivariateKDE}, model::VARModel

Predict the responses on the observed scale (i.e. the y[t]), from a posterior sample (from MCMC or from the VI posterior)

# Arguments
* `posterior_samples`: Samples from the (variational) posterior distribution
* `kdests`: Vector of Kernel density estimates, with `kdests[k]` corresponding to the marginal of z[:, k].
* `model`: New data, stored as a VARModel object.

# Keyword Arguments
* `N_mc`: Number of Monte Carlo samples drawn to estimate the posterior mean.

# Return
* `y_pred`: A vector of predictions, where `y_pred[t,:]` is the estimated posterior mean corresponding to `model.F[t,:]`
"""
function predict_response(
    rng::Random.AbstractRNG,
    posterior_samples::Vector{<:AbstractVector{<:Real}},
    kdests::Vector{T},
    model::VARModel
) where {T <: UnivariateKDE}
    J = model.J
    K = model.K
    Tsubp = model.Tsubp
    η = 1.0

    ikqfs = Vector{InterpKDEQF}(undef, K) # Vector of interpolated quantile functions
    for k in 1:K
        ikqfs[k] = InterpKDEQF(kdests[k])
    end
    #ikqf = InterpKDEQF(kdest)
    y_pred = Array{Float64}(undef, (i, size(model.F, 1), K)) # so that y[i, t, k, h] is the h-step ahead forecast at time t. (for now, just one-step ahead forecasts)
    for i in eachindex(posterior_samples)
        θ = posterior_samples[i]
        β = θ[1:K*J]
        Mβ = reshape(β, (J, K))
        log_ξ = θ[K*J+1:2*K*J]
        
        # Compute covariance matrix:
        transformed_dist = Bijectors.transformed(LKJCholesky(K, η))
        to_chol = Bijectors.inverse(Bijectors.bijector(LKJCholesky(K, η)))
        L = to_chol(γ).L        

        ξ = map(exp, log_ξ)
        ξ2 = map(abs2, ξ)
        # Compute parameters and so on...
        ξ2mat = reshape(ξ2, (J, K))

        # Then sample z given parameters, for each t.
        for t in 1:Tsubp
            # Generate z:
            ϵ = L * rand(rng, Normal(0, 1), K)
            s_vec = 1.0 ./ sqrt.( 1.0 .+ vec( transpose(@views model.F_sq[t,:]) * ξ2mat ) )
            μ = s_vec .* vec(transpose(@views model.F[t,:]) * Mβ)
            z = μ + s .* ϵ
            for k in 1:K
                y_pred[i, t, k] = quantile(iqkfs[k], cdf(Normal(), z[k]))
            end
            #= for k in 1:K
                #z_t_k = Base.rand(rng, Normal(μ[k], s_vec[k]), N_mc_z)
                z_t_k = μ[k] .+ s_vec[k] * ϵ
                y = quantile(ikqfs[k], cdf(Normal(), z_t_k))
                y_pred[t, k] += mean(y) - cov(ϵ, y) * ϵ_bar # control variates estimator (true mean of ϵ is 0, variance is 1)
            end =#
        end
    end
    return y_pred
end
predict_response(
    posterior_samples::Vector{Vector{Float64}},
    kdests::Vector{T},
    model::VARModel
) where {T <: UnivariateKDE} = predict_response(Random.default_rng(), posterior_samples, kdests, model)

"""
    predict_response([rng=Random.default_rng()], posterior_samples::Vector{Vector{Float64}}, kdests::Vector{UnivariateKDE}, model::VARModel

Predict the responses on the observed scale (i.e. the y[t]) from a posterior sample (from MCMC or from the VI posterior), using a point estimate of the parameters.

This function is significantly faster than `predict_response`, owing to the fact that we only compute the integral over `z` for a single value of the parameter vector.

# Arguments
* `posterior_samples`: Samples from the (variational) posterior distribution
* `kdests`: Vector of Kernel density estimates, with `kdests[k]` corresponding to the marginal of z[:, k].
* `model`: New data, stored as a VARModel object.

# Keyword Arguments
* `N_mc`: Number of Monte Carlo samples drawn to estimate the posterior mean.

# Return
* `y_pred`: A vector of predictions, where `y_pred[t,:]` is the estimated posterior mean corresponding to `model.F[t,:]`
"""
function predict_response_plugin(
    rng::Random.AbstractRNG,
    posterior_samples::Vector{<:AbstractVector{<:Real}},
    kdests::Vector{T},
    model::VARModel
) where {T <: UnivariateKDE}
    N_mc_z = 100
    J = model.J
    K = model.K
    Tsubp = model.Tsubp

    ikqfs = Vector{InterpKDEQF}(undef, K) # Vector of interpolated quantile functions
    for k in 1:K
        ikqfs[k] = InterpKDEQF(kdests[k])
    end
    #ikqf = InterpKDEQF(kdest)
    y_pred = zeros(Float64, (size(model.F, 1), K))
    ϵ = rand(rng, Normal(), N_mc_z)
    ϵ_bar = mean(ϵ)
    β_hat = mean(hcat(posterior_samples...)[1:K*J,:], dims=2)
    βmat = reshape(β_hat, (J, K))
    #ξ2_hat = mean(exp.(2.0*hcat(posterior_samples...)[K*J+1:2*K*J,:]), dims=2)
    #ξ2 = exp.(2.0*hcat(posterior_samples...)[K*J+1:2*K*J,:])
    #ξ2mat = reshape(ξ2_hat, (J, K))
    for t in 1:Tsubp
        #s_vec = mean(1.0 ./ sqrt.( 1.0 .+ vec( transpose(@views model.F_sq[t,:]) * ξ2mat ) ) )
        s_vec = zeros(Float64, K)
        for i in eachindex(posterior_samples)
            ξ2_i = reshape(exp.(2.0*posterior_samples[i][K*J+1:2*K*J]), (J, K))
            s_vec .+= 1/length(posterior_samples) * 1.0 ./ sqrt.( 1.0 .+ vec( transpose(@views model.F_sq[t,:]) * ξ2_i ) )
        end
        μ = s_vec .* vec(transpose(@views model.F[t,:]) * βmat)
        for k in 1:K
            #z_t_k = Base.rand(rng, Normal(μ[k], s_vec[k]), N_mc_z)
            z_t_k = μ[k] .+ s_vec[k] * ϵ
            y = quantile(ikqfs[k], cdf(Normal(), z_t_k))
            y_pred[t, k] += mean(y) - cov(ϵ, y) * ϵ_bar # control variates estimator (true mean of ϵ is 0, variance is 1)
        end
    end
    return y_pred
end
predict_response_plugin(
    posterior_samples::Vector{Vector{Float64}},
    kdests::Vector{T},
    model::VARModel
) where {T <: UnivariateKDE} = predict_response_plugin(Random.default_rng(), posterior_samples, kdests, model)


function predict_linpred_plugin(
    rng::Random.AbstractRNG,
    posterior_samples::Vector{<:AbstractVector{<:Real}},
    kdests::Vector{T},
    model::VARModel
) where {T <: UnivariateKDE}
    N_mc_z = 100
    J = model.J
    K = model.K
    Tsubp = model.Tsubp

    ikqfs = Vector{BandwidthSelectors.InterpKDEQF}(undef, K) # Vector of interpolated quantile functions
    for k in 1:K
        ikqfs[k] = BandwidthSelectors.InterpKDEQF(kdests[k])
    end
    #ikqf = InterpKDEQF(kdest)
    z_pred = zeros(Float64, (size(model.F, 1), K))
    β_hat = mean(hcat(posterior_samples...)[1:K*J,:], dims=2)
    βmat = reshape(β_hat, (J, K))
    #ξ2_hat = mean(exp.(2.0*hcat(posterior_samples...)[K*J+1:2*K*J,:]), dims=2)
    #ξ2 = exp.(2.0*hcat(posterior_samples...)[K*J+1:2*K*J,:])
    #ξ2mat = reshape(ξ2_hat, (J, K))
    for t in 1:Tsubp
        #s_vec = mean(1.0 ./ sqrt.( 1.0 .+ vec( transpose(@views model.F_sq[t,:]) * ξ2mat ) ) )
        s_vec = zeros(Float64, K)
        for i in eachindex(posterior_samples)
            ξ2_i = reshape(exp.(2.0*posterior_samples[i][K*J+1:2*K*J]), (J, K))
            s_vec .+= 1/length(posterior_samples) * 1.0 ./ sqrt.( 1.0 .+ vec( transpose(@views model.F_sq[t,:]) * ξ2_i ) )
        end
        μ = s_vec .* vec(transpose(@views model.F[t,:]) * βmat)
        z_pred[t,:] = μ
    end
    return z_pred
end

"""

Predict the `τ`-quantiles of y at time horizon `h`
"""
function predict_quantiles(
    rng::Random.AbstractRNG,
    h::Int,
    τ::AbstractVector{<:Real},
    posterior_samples::Vector{<:AbstractVector{<:Real}},
    kdests::Vector{T},
    model::VARModel, # This is the new data for which we want to obtain predictions.
    p::Int
) where {T <: UnivariateKDE}
    ikqfs = Vector{BandwidthSelectors.InterpKDEQF}(undef, model.K) # Vector of interpolated quantile functions
    for k in 1:model.K
        ikqfs[k] = BandwidthSelectors.InterpKDEQF(kdests[k])
    end
    η = 1
    to_chol = Bijectors.inverse(Bijectors.bijector(LKJCholesky(model.K, η)))
    quant_pred = Array{Float64}(undef, size(model.F, 1)-h+1, model.K, length(τ)) # (t, k, τ)

    for t in 1:size(model.F, 1)-h+1 # Can only provide forecasts up to observation T, since this we would otherwise not know the values of expogenous covariates.
        #s_vec = mean(1.0 ./ sqrt.( 1.0 .+ vec( transpose(@views model.F_sq[t,:]) * ξ2mat ) ) )
        y_sim = Vector{Float64}(undef, length(posterior_samples))
        for i in eachindex(posterior_samples)
            Mβ = reshape(posterior_samples[i][1:K*J], (model.J, model.K))
            ξ2_i = reshape(exp.(2.0*posterior_samples[i][K*J+1:2*K*J]), (J, K))
            γ_i = posterior_samples[i][2*K*J+K+1:end]
            # Get z tilde first, then transform (μ = F[t,:] * Mβ)
            Σ = Matrix(to_chol(γ_i))
            z_sim_prev = model.F[t,1:K^2*p] # assume that first K^2 * p coefs correspond to past responses.

            for j in 1:h
                μ = vcat(z_sim_prev, model.F[t,K^2*p+1:end]) * Mβ
                s_vec = 1.0 ./ sqrt.( 1.0 .+ vec( transpose(vcat(z_sim_prev.^2, model.F_sq[t+j,K^2*p+1:end])) * ξ2_i ) )
                z_sim_prev = Diagonal(s_vec) * rand(rng, MvNormal(μ, Σ))
            end
            for k in 1:K
                y_sim[i,k] = quantile(ikqfs[k], cdf(Normal(), z_sim))
            end
        end
        for k in 1:K
            quant_pred[t,k,:] = quantile(y_sim[:,k], τ)
        end
    end
    return quant_pred
end