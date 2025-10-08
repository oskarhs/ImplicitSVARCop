Base.Broadcast.broadcastable(k::UnivariateKDE) = Ref(k)

# Type that stores the interpolated cdf
mutable struct InterpKDECDF{K<:UnivariateKDE, I} <: AbstractKDE
    k::K
    itp::I
end
Base.Broadcast.broadcastable(k::InterpKDECDF) = Ref(k)

"""
   InterpKDECDF(k::UnivariateKDE, it::IT) where {IT<:Interpolations.MonotonicInterpolationType}
   InterpKDECDF(k::UnivariateKDE)

Create an `InterpKDECDF` object to store the (monotone) interpolated cumulative distribution function of `k`, which can be evaluated using the `cdf` method.

Supports the monotone interpolation methods implemented in Interpolations.jl, see the [documentation](https://juliamath.github.io/Interpolations.jl/latest/control/#Monotonic-interpolation)
for the different supported options.
If the interpolation scheme `it` is not explicitly provided, linear interpolation will be used.

# Examples
```julia
x = randn(10^3)
k = kde(x)
ikcdf = InterpKDECDF(k)
cdf(ikcdf, 0.0)
```
"""
function InterpKDECDF(k::UnivariateKDE, it::IT) where {IT<:MonotonicInterpolationType}
    F_grid = cdf_trapz(k)
    F_interp = extrapolate(
        interpolate(k.x, F_grid, it),
        Flat()
    )
    return InterpKDECDF(k, F_interp)
end
InterpKDECDF(k::UnivariateKDE) = InterpKDECDF(k, LinearMonotonicInterpolation())

cdf(ikcdf::InterpKDECDF, x::Real) = ikcdf.itp(x)
cdf(ikcdf::InterpKDECDF, x::AbstractArray{<:Real}) = ikcdf.itp.(x)

"""
    cdf(k::UnivariateKDE, x::Real)

Evaluate the cdf of `k` at `x`.

Letting `m_1, m_2, \\ldots, m_J` denote the midpoints where `k` has been evaluated, the cdf is estimated at the midpoints via 
```math
    \\hat{F}(m_j) = \\frac{\\int_{-\\infty}^{m_j} k(x)\\text{d}x}{\\int_{-\\infty}^{\\infty} k(x)\\text{d}x},
```
where the integrals are replaced with their corresponding trapezoidal approximations.
The cdf is evaluated at a general `x` via linear interpolation of the pairs `\\big(m_j, \\hat{F}(m_j)\\big)`

!!! note
    Each call to `cdf(k, x)` will construct a new interpolation object, which can be quite slow if this method is called repeatedly.
    For better performance in this case, use [`InterpKDECDF`](@ref) instead.
"""
cdf(k::UnivariateKDE, x::Real) = cdf(InterpKDECDF(k), x)
cdf(k::UnivariateKDE, x::AbstractArray{<:Real}) = cdf.(InterpKDECDF(k), x)


# Struct for storing the result of interpolating the quantile function
mutable struct InterpKDEQF{K<:UnivariateKDE, I} <: AbstractKDE
    k::K
    itp::I
end
Base.Broadcast.broadcastable(k::InterpKDEQF) = Ref(k)


"""
   InterpKDEQF(k::UnivariateKDE, it::IT) where {IT<:Interpolations.MonotonicInterpolationType}
   InterpKDEQF(k::UnivariateKDE)

Create an `InterpKDEQF` object to store the (monotone) interpolated quantile function of `k`, which can be evaluated using the `quantile` method.

Supports the monotone interpolation methods implemented in Interpolations.jl, see the [documentation](https://juliamath.github.io/Interpolations.jl/latest/control/#Monotonic-interpolation)
for the different supported options.
If `it` is not explicitly provided, linear interpolation will be used.

# Examples
```julia
x = randn(10^3)
k = kde(x)
ikqf = InterpKDEQF(k)
quantile(ikqf, 0.5)
```
"""
function InterpKDEQF(k::UnivariateKDE, it::IT) where {IT<:MonotonicInterpolationType}
    F_grid = cdf_trapz(k)
    unique_ids = unique(i -> F_grid[i], eachindex(F_grid)) # remove duplicate knots for interpolation
    Q_interp = extrapolate(
        interpolate(F_grid[unique_ids], k.x[unique_ids], it),
        Throw()
    )
    return InterpKDEQF(k, Q_interp)
end
InterpKDEQF(k::UnivariateKDE) = InterpKDEQF(k, LinearMonotonicInterpolation())

function quantile(ikqf::InterpKDEQF, q::Real)
    if q < 0.0 || q > 1.0
        throw(DomainError(q, "Supplied quantile is outside the interval [0, 1]."))
    end
    return ikqf.itp(q)
end
quantile(ikqf::InterpKDEQF, q::AbstractArray{<:Real}) = quantile.(ikqf, q)

"""
    quantile(k::UnivariateKDE, q::Real)

Evaluate the quantile function of `k` at `q`.

The quantile is evaluated by inverting the cdf constructed through linear interpolation. See [`cdf`](@ref) for more details.

!!! note
    Each call to `quantile(k, q)` will construct a new interpolation object, which can be quite slow if this method is called repeatedly.
    For better performance in this case, use [`InterpKDEQF`](@ref) instead.
"""
quantile(k::UnivariateKDE, q::Real) = quantile(InterpKDEQF(k), q)
quantile(k::UnivariateKDE, q::AbstractArray{<:Real}) = quantile(InterpKDEQF(k), q)


# Trapezoidal approximation of F at the points where k was evaluated
function cdf_trapz(k::UnivariateKDE)
    dt = step(k.x)
    F_grid = Vector{Float64}(undef, length(k.x))
    F_grid[1] = 0.0
    for j = 2:length(k.x)
        F_grid[j] = F_grid[j-1] + 0.5 * (k.density[j] + k.density[j-1]) * dt
    end
    F_grid = F_grid / F_grid[end] # normalize so that F is a proper cdf
    return F_grid
end