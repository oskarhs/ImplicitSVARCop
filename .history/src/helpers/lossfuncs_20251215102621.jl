"""
    pinball_loss(y_obs::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real}, τ::Real)

Compute the pinball loss at target quantile level `τ` for observed data `y_obs` and predicted conditional τ-quantile `y_pred`.

The pinball loss is defined as
```math
    \\sum_{t = p+1}^T \\rho\\big(y_{t} - \\hat{y}_{t}(\\tau)\\big),
```
where ``\\rho_{\\tau}(u) = u\\big(\\tau - I(u < 0)\\big)`` and ``\\hat{y}_{t}(\\tau)`` is the forecasted conditional ``\\tau``-quantile of ``y_t``
"""
function pinball_loss(y_obs_k::AbstractVector{<:Real}, y_pred_k::AbstractVector{<:Real}, τ::Real)
    u = y_obs_k - y_pred_k
    return sum(u .* (τ .- (u .< 0)))
end

function pinball_loss(y_obs::Real, y_pred::Real, τ::Real)
    u = y_obs - y_pred
    return u * (τ - (u < 0))
end

"""
    crps_loss(y_obs::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real})

Compute the continuous ranked probability score for a vector .

We note that the cprs at a given observation can be written as 
```math
    \\int_{0}^1 \\rho_{\\alpha}\\big(y_{t} - \\hat{y}_{t}(\\tau)\\big) \\text{d}\\alpha.
```
"""
function crps_loss(rng::Random.AbstractRNG, y_obs::AbstractMatrix{<:Real}, posterior_samples, kdests, model, h)
    # First estimate the quantiles for many levels τ
    # Then integrate this numerically.
    τ = LinRange(1e-4, 1-1e-4, 1001)
    quant_pred = predict_quantiles(rng, h, τ, posterior_samples, kdests, model) # (t, k, τ)

    # Now use simpsons rule to compute the required integral:
    crps = zeros(Float64, K)
    for k in 1:K
        for t in 1:size(model.F, 1)-h+1
            pinball_losses = Vector{Float64}(undef, length(τ))
            for i in eachindex(τ)
                pinball_losses[i] = pinball_loss(quant_pred[i,t,k], y_obs[t,k], τ[i])
            end
            problem = SampledIntegralProblem(pinball_losses, tau)
            method = SimpsonsRule()
            res = solve(problem, method)
            crps[k] += res.u
        end
    end
    return crps
end

"""
    rmse(rng::Random.AbstractRNG, y_obs::AbstractVector{<:Real}, posterior_samples, kdests, model, h, p)

Compute the rmse on a per-response variable basis.
"""
function rmse(rng::Random.AbstractRNG, y_obs::AbstractVector{<:Real}, posterior_samples, kdests, model, h, p)
    # First estimate the quantiles for many levels τ
    # Then integrate this numerically.
    τ = LinRange(1e-4, 1-1e-4, 1001)
    mean_pred = predict_something(...)

    rmse_vec = zeros(Float64, K)
    for k in 1:K
        rmse_vec[k] = sqrt(sum((y_obs[:,k] - mean_pred[:,k]).^2))
    end
    return rmse_vec
end