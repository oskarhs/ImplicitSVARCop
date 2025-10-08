abstract type FixedBandwidthSelector <: AbstractBandwidthSelector end

# All concrete subtypes of FixedBandwidthSelector should implement the bandwidth method, returning the single global bandwidth.

function fit(::Type{UnivariateKDE}, x::AbstractVector{<:Real}, method::FixedBandwidthSelector; npoints::Int=2048)
    h = bandwidth(x, method, npoints=npoints)
    return KernelDensity.kde(x, bandwidth=h, npoints=npoints)
end

function fit(::Type{UnivariateKDE}, x::AbstractVector{<:Real}, method::FixedBandwidthSelector, midpoints::R) where {R<:AbstractRange}
    h = bandwidth(x, method, midpoints)
    return KernelDensity.kde(x, midpoints, bandwidth=h)
end