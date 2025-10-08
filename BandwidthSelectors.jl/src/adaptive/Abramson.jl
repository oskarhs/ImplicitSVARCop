struct Abramson{A<:AbstractBandwidthSelector} <: AdaptiveBandwidthSelector
    pilot::A
end

"""
    Abramson(; pilot::A=ISJ()) where {A<:AbstractBandwidthSelector}

An implementation of the variable-bandwidth selector of [abramson1982variation](@citet).

The Abramson kernel estimator is computed as
```
\\hat{f}(x; \\boldsymbol{h}) = \\frac{1}{n} \\sum_{i=1}^n \\frac{1}{h_i}\\phi\\left(\\frac{x - x_i}{h_i}\\right),
```
where ``h_i = h\\sqrt{\\eta / \\hat{f}_P(x_i)}`` for ``h > 0``, a pilot estimate of the density ``\\hat{f}_P`` and ``\\eta = \\prod_{i=1}^n \\hat{f}_P(x_i)``.
The pilot estimate used can be controlled by the user through a corresponding keyword argument. The bandwidth adjustement factor ``h`` is selected based on L2 cross-validation.

# Keyword arguments
* `pilot`: A `BandwidthSelector` used to compute the pilot estimate for the Abramson kde. Default is the [`ISJ`](@ref) method.

```julia-repl
julia> x = randn(10^3);

julia> kdest = fit(UnivariateKDE, x, Abramson());
```

!!! note
    Currently only the Normal kernel is supported.

# References
* Abramson (1982). On bandwidth variation in kernel estimates-a square root law. https://doi.org/10.1214/aos/1176345986
"""
function Abramson(; pilot::A=ISJ()) where {A<:AbstractBandwidthSelector}
    return Abramson(pilot)
end


function fit(::Type{UnivariateKDE}, x::AbstractVector{<:Real}, method::Abramson; npoints::Int=2048)
    n = length(x)
    h0 = KernelDensity.default_bandwidth(x)
    boundary = KernelDensity.kde_boundary(x, h0)
    midpoints = KernelDensity.kde_range(boundary, npoints)
    return fit_abramson(x, midpoints, method.pilot)
end

function fit(::Type{UnivariateKDE}, x::AbstractVector{<:Real}, method::Abramson, midpoints::R) where {R<:AbstractRange}
    return fit_abramson(x, midpoints, method.pilot)
end

function fit_abramson(x::AbstractVector{<:Real}, midpoints::R, pilot::A) where {R<:AbstractRange, A<:AbstractBandwidthSelector}
    n = length(x)

    # Bin data
    counts = KernelDensity.tabulate(x, midpoints).density * step(midpoints) * n

    # Compute pilot estimate:
    k_pilot = fit(UnivariateKDE, x, pilot, midpoints)
    η = exp(mean(log.(pdf(k_pilot, x))))
    f_pilot = k_pilot.density
    bw_adj = sqrt.(η ./ f_pilot) # Abramsons local adjustement of bandwidths

    # First term in l2cv criterion is \\sum_{i\neq j}^n \frac{1}{\sqrt{h_i^2 + h_j^2}} K(\frac{X_i - X_j}{\sqrt{h_i^2 + h_j^2}}) from identity for Gaussian integrals.

    # Objective to be minimized
    l2cv_error = let counts = counts, f_pilot = f_pilot, midpoints = midpoints, bw_adj = bw_adj, counts = counts
        function (h)
            err = 0.0
            loc_bw = h * bw_adj
            for j in eachindex(midpoints)
                Δmidpoints_j = midpoints .- midpoints[j]
                Σbw_j = sqrt.(loc_bw .^2 .+ loc_bw[j]^2)
                err += 1.0 / n^2 * sum(counts[j] * counts .* sign.(abs.(Δmidpoints_j)) .* pdf.(Normal(), Δmidpoints_j ./ Σbw_j))
                err += - 2.0/(n*(n-1)) * sum(counts[j] * counts .* sign.(abs.(Δmidpoints_j)) .* pdf.(Normal(), Δmidpoints_j ./ loc_bw)) # optimize later, of course
            end
            return err
        end
    end

    # Optimize bandwidth:
    a = (last(midpoints) - first(midpoints)) * 1e-12
    b = (last(midpoints) - first(midpoints))
    res = optimize(l2cv_error, a, b, GoldenSection(); rel_tol=1e-6) # can change to the internal Golden Search method at a later point in time
    h_opt = minimizer(res)

    # Compute density estimates at midpoints for the given bandwidth:
    density = Vector{Float64}(undef, length(midpoints))
    loc_bw = h_opt * bw_adj
    for j in eachindex(midpoints)
        Δmidpoints_j = midpoints .- midpoints[j]
        density[j] = 1.0 / n * sum(counts .* pdf.(Normal(), Δmidpoints_j / loc_bw))
    end
    return UnivariateKDE(midpoints, density)
end