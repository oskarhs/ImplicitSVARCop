"""
    VIPosterior

Struct representing the multivariate normal variational posterior of θ.

# Fields
* `μ`: Posterior mean of θ.
* `Bfac`: B matrix in the factor covariance matrix expression BB' + Δ²
* `d`: Vector of diagonal entries of Δ in the covariance matrix expression BB' + Δ²
* `J`: Number of regression coefficients per response dimension.
* `K`: Dimension of the response, e.g. z_t has dimension R^K
* `M_γ`: Dimension of γ.
"""
struct VIPosterior{Mat<:AbstractMatrix{<:Real}}
    μ::Vector{Float64}
    Bfac::Mat
    d::Vector{Float64}
    J::Int
    K::Int
    M_γ::Int
end

function VIPosterior(D::Int, N_fac::Int, J::Int, K::Int, M_γ::Int)
    μ = zeros(Float64, D)
    d = fill(1e-1, D)
    Bfac = fill(1e-3, (D, N_fac))
    tril!(Bfac, -1) # sets upper triangle (including diagonal) of Bfac to 0.
    return VIPosterior(μ, Bfac, d, J, K, M_γ)
end

function Base.show(io::IO, posterior::VIPosterior)
    println(io, typeof(posterior))
    println(io, "μ: ", posterior.μ)
    println(io, "Bfac: ", posterior.Bfac)
    println(io, "d: ", posterior.d)
    println(io, "J: ", posterior.J)
    println(io, "K: ", posterior.K)
    println(io, "M_γ: ", posterior.M_γ)
end

"""
    covariance(posterior::VIPosterior)

Return the covariance matrix BB' + Δ² of the variational posterior.
"""
function covariance(posterior::VIPosterior)
    return posterior.Bfac * transpose(posterior.Bfac) + Diagonal(map(abs2, posterior.d))
end

"""
    rand([rng=Random.default_rng()], posterior::VIPosterior)
    rand([rng=Random.default_rng()], posterior::VIPosterior, n::Int)

Return one or several random samples of θ from the variational posterior q(θ; η).

If the sample size `n` is not given, a single sample will be returned as a vector.
If the sample size `n` is given, the result is stored as a dim(θ) × n matrix, so that θ[:, i] represents a single sample.
"""
function Base.rand(rng::Random.AbstractRNG, posterior::VIPosterior)
    w1 = Base.rand(rng, Normal(), size(posterior.Bfac, 2))
    w2 = Base.rand(rng, Normal(), length(posterior.μ))
    θ = posterior.μ + posterior.Bfac * w1 + posterior.d .* w2
    return θ
end
Base.rand(posterior::VIPosterior) = rand(Random.default_rng(), posterior)

function Base.rand(rng::Random.AbstractRNG, posterior::VIPosterior, n::Int)
    if n ≤ 0
        throw(ArgumentError("Number of samples must be a positive number."))
    end
    θs = Matrix{Float64}(undef, length(posterior.μ), n)
    for i in 1:n
        θs[:, i] = rand(rng, posterior)
    end
    return θs
end
Base.rand(posterior::VIPosterior, n::Int) = rand(Random.default_rng(), posterior, n)

"""
    logpdf(posterior::VIPosterior, θ::AbstractVector{<:Real})

Evaluate the (normalized) logpdf of the variational posterior at `θ`.
"""
function Distributions.logpdf(posterior::VIPosterior, θ::AbstractVector{<:Real})
    Bz_deps = θ - posterior.μ
    N_fac = size(posterior.Bfac, 2)
    d2 = 1.0 ./ posterior.d .^2
    Dinv2B = posterior.Bfac .* d2
    DBz_deps = Bz_deps .* d2
    Siginvpart = inv(I(N_fac) + transpose(posterior.Bfac) * Dinv2B)

    Blogdet = logdet(I(N_fac) + transpose(Dinv2B) * posterior.Bfac) + vsum(x -> log(abs2(x)), posterior.d)
    Half2 = Dinv2B * (Siginvpart * (transpose(posterior.Bfac) * DBz_deps))
    quadform = transpose(Bz_deps) * (DBz_deps - Half2)
    logp = - 0.5 * length(posterior.μ) * log(2.0*pi) - 0.5*Blogdet - 0.5 * quadform
    return logp
end

"""
    predict_response([rng=Random.default_rng()], posterior::VIPosterior, kdests::Vector{UnivariateKDE}, model::VARModel; N_mc::Int=200)

Predict the responses on the observed scale (i.e. the y[t]), by sampling from the variational posterior.

# Arguments
* `posterior`: The variational posterior
* `kdests`: Vector of Kernel density estimates, with `kdests[k]` corresponding to the marginal of z[:, k].
* `model`: New data, stored as a VARModel object.

# Keyword Arguments
* `N_mc`: Number of Monte Carlo samples drawn to estimate the posterior mean.

# Return
* `y_pred`: A vector of predictions, where `y_pred[t,:]` is the estimated posterior mean corresponding to `model.F[t,:]`
"""
function predict_response(
    rng::Random.AbstractRNG,
    posterior::VIPosterior,
    kdests::Vector{T},
    model::VARModel;
    N_mc::Int = 500
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
    for _ in 1:N_mc
        θ = Base.rand(rng, posterior)
        β = θ[1:K*J]
        log_ξ = θ[K*J+1:2*K*J]

        ξ = map(exp, log_ξ)
        ξ2 = map(abs2, ξ)
        # Compute parameters and so on...
        ξ2mat = reshape(ξ2, (J, K))
        βmat = reshape(β, J, K)

        # Then sample z given parameters, for each t.
        for t in 1:Tsubp
            s_vec = 1.0 ./ sqrt.( 1.0 .+ vec( transpose(@views model.F_sq[t,:]) * ξ2mat ) )
            μ = s_vec .* vec(transpose(@views model.F[t,:]) * βmat)
            for k in 1:K
                #z_t_k = Base.rand(rng, Normal(μ[k], s_vec[k]), N_mc_z)
                z_t_k = μ[k] .+ s_vec[k] * ϵ
                y = quantile(ikqfs[k], cdf(Normal(), z_t_k))
                y_pred[t, k] += mean(y) - cov(ϵ, y) * ϵ_bar # control variates estimator (true mean of ϵ is 0, variance is 1)
            end
        end
                
    end
    y_pred = y_pred / N_mc
    return y_pred
end
predict_response(posterior::VIPosterior, kdests::Vector{T}, model::VARModel; N_mc::Int = 200) where {T<:UnivariateKDE} = predict_response(Random.default_rng(), posterior, kdests, model; N_mc=N_mc)


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
    posterior_samples::Vector{Vector{Float64}},
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
