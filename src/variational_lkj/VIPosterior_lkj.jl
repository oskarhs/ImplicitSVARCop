"""
    VIPosterior_lkj

Struct representing the multivariate normal variational posterior of θ.

# Fields
* `μ`: Posterior mean of θ.
* `Bfac`: B matrix in the factor covariance matrix expression BB' + Δ²
* `d`: Vector of diagonal entries of Δ in the covariance matrix expression BB' + Δ²
* `J`: Number of regression coefficients per response dimension.
* `K`: Dimension of the response, e.g. z_t has dimension R^K
"""
struct VIPosterior_lkj{Mat<:AbstractMatrix{<:Real}}
    μ::Vector{Float64}
    Bfac::Mat
    d::Vector{Float64}
    J::Int
    K::Int
end

function VIPosterior_lkj(D::Int, N_fac::Int, J::Int, K::Int)
    μ = zeros(Float64, D)
    d = fill(1e-1, D)
    Bfac = fill(1e-3, (D, N_fac))
    tril!(Bfac, 0) # sets upper triangle (not including diagonal) of Bfac to 0.
    return VIPosterior(μ, Bfac, d, J, K)
end

function Base.show(io::IO, posterior::VIPosterior_lkj)
    println(io, typeof(posterior))
    println(io, "μ: ", posterior.μ)
    println(io, "Bfac: ", posterior.Bfac)
    println(io, "d: ", posterior.d)
    println(io, "J: ", posterior.J)
    println(io, "K: ", posterior.K)
end

"""
    cov(posterior::VIPosterior_lkj)

Return the covariance matrix BB' + Δ² of the variational posterior.
"""
function Distributions.cov(posterior::VIPosterior_lkj)
    return posterior.Bfac * transpose(posterior.Bfac) + Diagonal(map(abs2, posterior.d))
end

"""
    rand([rng=Random.default_rng()], posterior::VIPosterior_lkj)
    rand([rng=Random.default_rng()], posterior::VIPosterior_lkj, n::Int)

Return one or several random samples of θ from the variational posterior q(θ; η).

If the sample size `n` is not given, a single sample will be returned as a vector.
If the sample size `n` is given, the result is stored as a dim(θ) × n matrix, so that θ[:, i] represents a single sample.
"""
function Base.rand(rng::Random.AbstractRNG, posterior::VIPosterior_lkj)
    w1 = Base.rand(rng, Normal(), size(posterior.Bfac, 2))
    w2 = Base.rand(rng, Normal(), length(posterior.μ))
    θ = posterior.μ + posterior.Bfac * w1 + posterior.d .* w2
    return θ
end
Base.rand(posterior::VIPosterior_lkj) = rand(Random.default_rng(), posterior)

function Base.rand(rng::Random.AbstractRNG, posterior::VIPosterior_lkj, n::Int)
    if n ≤ 0
        throw(ArgumentError("Number of samples must be a positive number."))
    end
    θs = Matrix{Float64}(undef, length(posterior.μ), n)
    for i in 1:n
        θs[:, i] = rand(rng, posterior)
    end
    return θs
end
Base.rand(posterior::VIPosterior_lkj, n::Int) = rand(Random.default_rng(), posterior, n)

"""
    logpdf(posterior::VIPosterior_lkj, θ::AbstractVector{<:Real})

Evaluate the (normalized) logpdf of the variational posterior at `θ`.
"""
function Distributions.logpdf(posterior::VIPosterior_lkj, θ::AbstractVector{<:Real})
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
    predict_response([rng=Random.default_rng()], posterior::VIPosterior_lkj, kdests::Vector{UnivariateKDE}, model::VARModel; N_mc::Int=200)

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
    posterior::VIPosterior_lkj,
    kdests::Vector{T},
    model::VARModel;
    N_mc::Int = 200
) where {T <: UnivariateKDE}
    θ = rand(rng, posterior, N_mc)
    samples = [θ[:,i] for i in 1:N_mc]
    return predict_response(rng, samples, kdests, model)
end
predict_response(posterior::VIPosterior_lkj, kdests::Vector{T}, model::VARModel; N_mc::Int = 200) where {T<:UnivariateKDE} = predict_response(Random.default_rng(), posterior, kdests, model; N_mc=N_mc)

function predict_response_plugin(
    rng::Random.AbstractRNG,
    posterior::VIPosterior_lkj,
    kdests::Vector{T},
    model::VARModel;
    N_mc::Int = 200
) where {T <: UnivariateKDE}
    θ = rand(rng, posterior, N_mc)
    samples = [θ[:,i] for i in 1:N_mc]
    return predict_response_plugin(rng, samples, kdests, model)
end
predict_response_plugin(posterior::VIPosterior_lkj, kdests::Vector{T}, model::VARModel; N_mc::Int = 200) where {T<:UnivariateKDE} = predict_response_plugin(Random.default_rng(), posterior, kdests, model; N_mc=N_mc)
