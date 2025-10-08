struct ISJ <: FixedBandwidthSelector
    level::Int
end

"""
    ISJ(; level::Int=7)
   
The improved Sheather-Jones bandwidth selector of [botev2010diffusion](@citet).

Determines the optimal bandwidth by solving the equation ``t = \\xi \\gamma^{[l]}(t)``, where ``\\gamma^{[l]}`` is an estimate of a functional of the true density ``f`` using ``l`` stages of functional estimation.
To solve this equation, a simple bisection search is used.
In cases where we are unable to find a bracketing interval, the corresponding plug-in estimator based on a normal reference rule is used to compute the optimal diffusion time instead, as described on page 2931 in [botev2010diffusion](@citet).

# Keyword arguments
* `level`: Integer parameter used to compute the optimal diffusion time via ``t = \\xi \\gamma^{[l]}(t)``, where ``l`` is the supplied level.

```julia-repl
julia> x = randn(10^3);

julia> kdest = fit(UnivariateKDE, x, ISJ());
```

!!! note
    This method only supports a Normal kernel.

# References
* Botev et al. (2010). Kernel density estimation via diffusion. https://doi.org/10.1214/10-AOS799
"""
function ISJ(; level::Int=7)
    if level < 3
        throw(ArgumentError("The supplied value of l must be 3 or greater."))
    end
    return ISJ(level)
end

function bandwidth(x::AbstractVector{<:Real}, method::ISJ; npoints::Int=2048)
    h0 = KernelDensity.default_bandwidth(x)
    boundary = KernelDensity.kde_boundary(x, h0)
    midpoints = KernelDensity.kde_range(boundary, npoints)
    return bandwidth(x, method, midpoints)
end

function bandwidth(x::AbstractVector{<:Real}, method::ISJ, midpoints::R) where {R<:AbstractRange}
    kernel = Normal
    if !(kernel <: Normal)
        throw(ArgumentError("Only the Normal kernel is supported."))
    end
    h_isj = bandwidth_isj(x, midpoints, method.level)
    return h_isj
end

function bandwidth_isj(x::AbstractVector{<:Real}, midpoints::R, level::Int) where {R<:AbstractRange}
    scale = last(midpoints) - first(midpoints)
    n = length(x)
    npoints = length(midpoints)

    # Perform linear binning of the data (multiply by step length to get binprobs)
    probs = KernelDensity.tabulate(x, midpoints).density * step(midpoints)

    # Compute cosine transform and apply the correct normalization.
    ft_dens = dct(probs)[2:end]
    ft_dens *= sqrt(2.0 * npoints)

    # Precompute constants to compute the roughness measures.
    deriv_orders = 1:level
    dfact = cumprod(2 .* deriv_orders .- 1)
    cs = [(1.0 + 2.0^(-(j + 0.5))) * dfact[j] for j in deriv_orders]

    # Function to compute optimal diffusion time using `level` stages of functional estimation, with "pilot" diffusion time `t`.
    ξγ = let ft_dens = ft_dens, level = level, t_dft = (1:(npoints - 1)) * π, cs = cs
        function (t)
            dist = Ref(Normal(0, sqrt(t)))
            ft_j = @. t_dft^level * ft_dens * cf(dist, t_dft)
            roughness = 0.5 * sum(abs2, ft_j)  # roughness (L2-norm) of f^{(j)}
            for j in level-1:-1:2
                t_star = (cs[j] / (3.0*n * (sqrt(0.5*π) * roughness)))^(1/(3 + 2j))
                dist = Ref(Normal(0, t_star))
                @. ft_j = t_dft^j * ft_dens * cf(dist, t_dft)
                roughness = 0.5 * sum(abs2, ft_j)
            end
            return (2.0 * n * (sqrt(π) * roughness))^(-0.4)
        end
    end

    t_l = 0.0
    t_u = 1e-2
    atol = 1e-8

    # Attempt to identify a bracketing interval
    sign_l = sign(t_l - ξγ(t_l))
    sign_u = sign(t_u - ξγ(t_u))
    while t_u ≤ 1.0 && sign_u == sign_l
        t_u *= 1.5
        sign_u = sign(t_u - ξγ(t_u))
    end

    # If we succeed in finding a bracketing interval, solve the equation t = ξγ(t) . Else, return the corresponding plug-in estimator.
    t_opt = if t_u ≤ 1.0 
        bisection_search(t -> t - ξγ(t), t_l, t_u, atol)
    else
        ft_l = prod(range(1, 2*l, 2)) / (sqrt(2.0*π)*std(x)^(2+l))
        K = prod(range(1, 2*l, 2)) / sqrt(2.0*π)   # this does not depend on t, and can as such be precomputed
        c = (1.0 + 0.5^(l+0.5)) / 3.0              # again, this is independent of t 
        t_l = (2*c*K/n/ft_l)^(2/(3+2*l)) 
        ξγ(t_l)
    end

    bw_isj = sqrt(t_opt) * scale
    return bw_isj
end

# Use bisection search to solve the fixed-point equation for the optimal diffusion time.
function bisection_search(fun::F, t_l::Real, t_u::Real, atol::Real) where {F<:Function} # move this to a utility file at some point
    sign_l = sign(fun(t_l))
    sign_u = sign(fun(t_u))
    err = atol + 1.0
    while err > atol
        t_m = 0.5*(t_l + t_u)
        sign_m = sign(fun(t_m))
        if sign_m == 0.0            # we found a solution
            t_u = t_m
            t_l = t_m
        elseif sign_m != sign_l     # set midpoint as new upper bound
            t_u = t_m
            sign_u = sign_m
        else                        # set midpoint as new lower bound
            t_l = t_m
            sign_l = sign_m
        end
        err = t_u - t_l
    end
    t_opt = 0.5 * (t_l + t_u)
    return t_opt
end