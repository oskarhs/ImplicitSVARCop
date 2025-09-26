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
    for i in eachindex(posterior_samples)
        θ = posterior_samples[i]
        β = θ[1:K*J]
        log_ξ = θ[K*J+1:2*K*J]

        ξ = map(exp, log_ξ)
        ξ2 = map(abs2, ξ)
        # Compute parameters and so on...
        ξ2mat = reshape(ξ2, (J, K))

        # Then sample z given parameters, for each t.
        for t in 1:Tsubp
            s_vec = 1.0 ./ sqrt.( 1.0 .+ vec( transpose(@views model.F_sq[t,:]) * ξ2mat ) )
            μ = s_vec .* vec(transpose(@views model.F[t,:]) * reshape(β, J, K))
            for k in 1:K
                #z_t_k = Base.rand(rng, Normal(μ[k], s_vec[k]), N_mc_z)
                z_t_k = μ[k] .+ s_vec[k] * ϵ
                y = quantile(ikqfs[k], cdf(Normal(), z_t_k))
                y_pred[t, k] += mean(y) - cov(ϵ, y) * ϵ_bar # control variates estimator (true mean of ϵ is 0, variance is 1)
            end
        end
    end
    y_pred = y_pred / length(posterior_samples)
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