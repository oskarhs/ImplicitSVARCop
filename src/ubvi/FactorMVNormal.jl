"""
    FactorMVNormal

Struct representing the multivariate normal variational dist of θ.

# Fields
* `μ`: dist mean of θ.
* `Bfac`: B matrix in the factor covariance matrix expression BB' + Δ²
* `d`: Vector of diagonal entries of Δ in the covariance matrix expression BB' + Δ²
* `J`: Number of regression coefficients per response dimension.
* `K`: Dimension of the response, e.g. z_t has dimension R^K
* `M_γ`: Dimension of γ.
"""
struct FactorMVNormal
    μ::Vector{Float64}
    Bfac::Matrix{Float64}
    d::Vector{Float64}
    J::Int
    K::Int
    M_γ::Int
end

function FactorMVNormal(D::Int, N_fac::Int, J::Int, K::Int, M_γ::Int)
    μ = zeros(Float64, D)
    d = fill(1e-1, D)
    Bfac = fill(1e-3, (D, N_fac))
    tril!(Bfac, -1) # sets upper triangle (including diagonal) of Bfac to 0.
    return FactorMVNormal(μ, Bfac, d, J, K, M_γ)
end

function Base.show(io::IO, dist::FactorMVNormal)
    println(io, typeof(dist))
    println(io, "μ: ", dist.μ)
    println(io, "Bfac: ", dist.Bfac)
    println(io, "d: ", dist.d)
    println(io, "J: ", dist.J)
    println(io, "K: ", dist.K)
    println(io, "M_γ: ", dist.M_γ)
end

"""
    Statistics.cov(dist::FactorMVNormal)

Return the covariance matrix BB' + Δ² of the variational dist.
"""
function Statistics.cov(dist::FactorMVNormal)
    return dist.Bfac * transpose(dist.Bfac) + Diagonal(map(abs2, dist.d))
end

"""
    rand([rng=Random.default_rng()], dist::FactorMVNormal)
    rand([rng=Random.default_rng()], dist::FactorMVNormal, n::Int)

Return one or several random samples of θ from the variational dist q(θ; η).

If the sample size `n` is not given, a single sample will be returned as a vector.
If the sample size `n` is given, the result is stored as a dim(θ) × n matrix, so that θ[:, i] represents a single sample.
"""
function Base.rand(rng::Random.AbstractRNG, dist::FactorMVNormal)
    w1 = Base.rand(rng, Normal(), size(dist.Bfac, 2))
    w2 = Base.rand(rng, Normal(), length(dist.μ))
    θ = dist.μ + dist.Bfac * w1 + dist.d .* w2
    return θ
end
Base.rand(dist::FactorMVNormal) = rand(Random.default_rng(), dist)

function Base.rand(rng::Random.AbstractRNG, dist::FactorMVNormal, n::Int)
    if n ≤ 0
        throw(ArgumentError("Number of samples must be a positive number."))
    end
    θs = Matrix{Float64}(undef, length(dist.μ), n)
    for i in 1:n
        θs[:, i] = rand(rng, dist)
    end
    return θs
end
Base.rand(dist::FactorMVNormal, n::Int) = rand(Random.default_rng(), dist, n)

"""
    logpdf(dist::FactorMVNormal, θ::AbstractVector{<:Real})

Evaluate the (normalized) logpdf of the variational dist at `θ`.
"""
function Distributions.logpdf(dist::FactorMVNormal, θ::AbstractVector{<:Real})
    Bz_deps = θ - dist.μ
    N_fac = size(dist.Bfac, 2)
    d2 = 1.0 ./ dist.d .^2
    Dinv2B = dist.Bfac .* d2
    DBz_deps = Bz_deps .* d2
    Siginvpart = inv(I(N_fac) + transpose(dist.Bfac) * Dinv2B)

    Blogdet = logdet(I(N_fac) + transpose(Dinv2B) * dist.Bfac) + vsum(x -> log(abs2(x)), dist.d)
    Half2 = Dinv2B * (Siginvpart * (transpose(dist.Bfac) * DBz_deps))
    quadform = transpose(Bz_deps) * (DBz_deps - Half2)
    logp = - 0.5 * length(dist.μ) * log(2.0*pi) - 0.5*Blogdet - 0.5 * quadform
    return logp
end

"""
    log_sqrt_pair_integral(dist::FactorMVNormal, comps::Vector{<:FactorMVNormal})

Compute the values of ∫ √φ(x; μ₁, Σ₁) √φ(x; μ₂, Σ₂) dx pairwise for normal distributions with a factor structure. 
"""
function log_sqrt_pair_integral(dist::FactorMVNormal, comps::Vector{<:FactorMVNormal})
    D = length(dist.μ)

    log_integrals = Vector{Float64}(undef, length(comps))
    for i in eachindex(comps)
        Σ = 0.5 * (covariance(dist) + covariance(comps[i]))
        log_integrals[i] = 0.5 * logdet(comps[i]) - 0.25 * logdet(covariance(dist)) - 0.25 * logdet(covariance(comps[i])) - 0.125 * dot((dist.μ) * inv(Σ) * (comps[i].μ))
    end
    return log_integrals
end

"""
Function used to initialize a new component in the VI algorithm.
"""
function params_init(rng::Random.AbstractRNG, comps::Vector{<:FactorMVNormal}, weights::Vector{<:Real}, infl::Real=10.0)
    # For now: assume that the first component has been initialized via the existing VI code.
    m = rand(rng, DiscreteNonParametric(1:k, weights))
    μ = rand(rng, MvNormal(comps[m].μ, infl * covariance(comps[m])))
    Bfac = comps[m].Bfac * rand(rng, LogNormal(0.0, infl). length(comps[m].d))
    d = comps[m].d * rand(rng, LogNormal(0.0, infl). length(comps[m].d))

    return FactorMVNormal(μ, Bfac, d, comps[m].J, comps[m].K, comps[m].M_γ)
end