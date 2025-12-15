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
* `y_pred`: A 4-dim array of forecasts 
"""
function forecast(
    rng::Random.AbstractRNG,
    posterior_samples::Vector{<:AbstractVector{<:Real}},
    kdests::Vector{T},
    model::VARModel
) where {T <: UnivariateKDE}
    J = model.J
    K = model.K
    Tsubp = model.Tsubp
    η = 1.0

    to_chol = Bijectors.inverse(Bijectors.bijector(LKJCholesky(K, η)))

    ikqfs = Vector{InterpKDEQF}(undef, K) # Vector of interpolated quantile functions
    for k in 1:K
        ikqfs[k] = InterpKDEQF(kdests[k])
    end
    #ikqf = InterpKDEQF(kdest)
    y_pred = Array{Float64}(undef, (i, size(model.F, 1), K)) # so that y[i, t, k, h] is the h-step ahead forecast of variable k at time t. (for now, just one-step ahead forecasts)
    for i in eachindex(posterior_samples)
        θ = posterior_samples[i]
        β = θ[1:K*J]
        Mβ = reshape(β, (J, K))
        log_ξ = θ[K*J+1:2*K*J]
        
        # Compute covariance matrix:
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
        end
    end
    return y_pred
end
forecast(
    posterior_samples::Vector{Vector{Float64}},
    kdests::Vector{T},
    model::VARModel
) where {T <: UnivariateKDE} = predict_response(Random.default_rng(), posterior_samples, kdests, model)


"""

Predict the `τ`-quantiles of y at time horizon `h`.

Return quant_pred[t, k, h]
"""
function predict_quantiles(
    rng::Random.AbstractRNG,
    h::Int,
    τ::AbstractVector{<:Real},
    posterior_samples::Vector{<:AbstractVector{<:Real}},
    kdests::Vector{T},
    model::VARModel # This is the new data for which we want to obtain predictions.
) where {T <: UnivariateKDE}
    if minimum(τ) < 0.0 || maximum(τ) > 1.0
        throw(ArgumentError("All quantiles must be in the range [0,1]"))
    end
    ikqfs = Vector{BandwidthSelectors.InterpKDEQF}(undef, model.K) # Vector of interpolated quantile functions
    for k in 1:model.K
        ikqfs[k] = BandwidthSelectors.InterpKDEQF(kdests[k])
    end
    y_pred = forecast(rng, posterior_samples, kdests, model) # y[i, t, k, h] (minus the 4th dim for now)
    
    quant_pred = mapslices(x -> quantile(x, τ), y_pred, dims=[1]) # q[τ, t, k, h]

    return quant_pred
end

"""

Compute predictive means
"""
function predict_quantiles(
    rng::Random.AbstractRNG,
    h::Int,
    τ::AbstractVector{<:Real},
    posterior_samples::Vector{<:AbstractVector{<:Real}},
    kdests::Vector{T},
    model::VARModel # This is the new data for which we want to obtain predictions.
) where {T <: UnivariateKDE}
    if minimum(τ) < 0.0 || maximum(τ) > 1.0
        throw(ArgumentError("All quantiles must be in the range [0,1]"))
    end
    ikqfs = Vector{BandwidthSelectors.InterpKDEQF}(undef, model.K) # Vector of interpolated quantile functions
    for k in 1:model.K
        ikqfs[k] = BandwidthSelectors.InterpKDEQF(kdests[k])
    end
    y_pred = forecast(rng, posterior_samples, kdests, model) # y[i, t, k, h] (minus the 4th dim for now)
    
    mean_pred = mapslices(mean, y_pred, dims=[1]) # m[t, k, h]

    return mean_pred
end