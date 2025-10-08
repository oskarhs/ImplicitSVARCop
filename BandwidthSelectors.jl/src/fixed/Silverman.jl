struct Silverman <: FixedBandwidthSelector
    scalest::Symbol
end

"""
    Silverman(; scalest::Symbol=:min)
   
Silverman's rule for determining the kernel bandwidth [silverman1986density](@citep).

The bandwidth is computed by minimizing an estimated asymptotic risk, using the normal distribution as a reference.

# Keyword arguments
* `scalest`: Estimate of scale parameter used to compute the roughness of the data-generating density. Valid options are `:std` (standard deviation), `:iqr` (interquartile range) and `:min`, which uses the smaller of the two. Defaults to `:min`.

```julia-repl
julia> x = randn(10^3);

julia> kdest = fit(UnivariateKDE, x, Silverman());
```

!!! note
    This method currently only supports a Normal kernel.

# References
* Silverman (1986). Density Estimation for Statistics and Data Analysis. https://doi.org/10.1201/9781315140919
"""
function Silverman(; scalest::Symbol=:min)
    if !(scalest in [:min, :iqr, :std])
        throw(ArgumentError("Supplied scalest must be one of :min, :iqr and :std"))
    end
    return Silverman(scalest)
end

# For API compatibility
bandwidth(x::AbstractVector{<:Real}, method::Silverman; npoints::Int=2048) = bandwidth_silverman(x, method.scalest)
bandwidth(x::AbstractVector{<:Real}, method::Silverman, midpoints::R) where {R<:AbstractRange} = bandwidth_silverman(x, method.scalest)


function bandwidth_silverman(x::AbstractVector{<:Real}, scalest::Symbol)
    # Compute scale estimate
    σ_hat = if scalest == :std
        std(x)
    elseif scalest == :iqr
        (quantile(x, 0.75) - quantile(x, 0.25)) / 1.349
    else
        min(std(x), (quantile(x, 0.75) - quantile(x, 0.25)) / 1.349)
    end
    
    if σ_hat == 0.0
        throw(DomainError("Scale estimate is 0 for input data."))
    end

    # Compute bandwidth
    n = length(x)
    h_s = 0.9*σ_hat * n^(-0.2)
    return h_s
end