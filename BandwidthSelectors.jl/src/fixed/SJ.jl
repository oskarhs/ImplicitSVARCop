# Compute the number of bins using the plug-in bandwidth selector of Sheather and Jones (1991).
# The functions `bkfe` and `dpik` have originally been ported from R code located here https://github.com/cran/KernSmooth/tree/master/R. Distributed under an unlimited license.
struct SJ <: FixedBandwidthSelector
    level::Int
    scalest::Symbol
end

"""
    SJ(; level::Int=2, scalest::Symbol=:min)
   
The plug-in bandwidth selector of [sheather1991reliable](@citet).

The implementation is based on the one of the KernSmooth R package [wand2025kernsmooth](@citet).

# Keyword arguments
* `level`: Number of times functional estimation is applied when computing the plug-in rule. Must be one of [0, 1, 2, 3, 4, 5]. Defaults to 2.
* `scalest`: Estimate of scale parameter used when standardizing data. Valid options are `:std` (standard deviation), `:iqr` (interquartile range) and `:min`, which uses the smaller of the two. Defaults to `:min`.

```julia-repl
julia> x = randn(10^3);

julia> fit(UnivariateKDE, x, SJ());
```

!!! note
    Currently only the Normal kernel is supported.

# References
* Sheather and Jones (1991). A reliable data-based bandwidth selection method for kernel density estimation. https://doi.org/10.1111/j.2517-6161.1991.tb01857.x
* Wand et al. (2025). KernSmooth: Functions for kernel smoothing supporting Wand & Jones (1995). Software version 2.23-26. https://doi.org/10.32614/CRAN.package.KernSmooth
"""
function SJ(; level::Int=2, scalest::Symbol=:min)
    if !(level in 0:5)
        throw(ArgumentError("Supplied level must be one of 0, 1, 2, 3, 4 and 5."))
    end
    if !(scalest in [:min, :iqr, :std])
        throw(ArgumentError("Supplied scalest must be one of :min, :iqr and :std"))
    end
    return SJ(level, scalest)
end

# Kernel estimator of the functionals needed for the plug-in bandwidth
function bkfe(counts::AbstractArray{<:Real}, drv::Int, bw::Real, range_x::Tuple{T, T}) where {T<:Real}
    a, b = range_x
    M = length(counts)
    n = sum(counts)

    δ = (b-a)/(M-1.0)

    # Obtain kernel weights
    tau = 4 + drv
    L = min(Int(fld(tau * bw, δ)), M)

    lvec = 0:L
    arg = lvec .* δ / bw

    kappam = @. 1.0/sqrt(2.0*pi) * exp(-0.5*arg ^ 2) / bw^(drv + 1.0)
    hmold0, hmnew = ones(length(arg)), ones(length(arg))
    hmold1 = arg

    if drv >= 2
        for i in (2:drv)
            hmnew = arg .* hmold1 .- (i - 1) .* hmold0
            hmold0 = hmold1       # Compute mth degree Hermite polynomial
            hmold1 = hmnew        # by recurrence.
        end
    end
    kappam = hmnew .* kappam

    ## Now combine weights and counts to obtain estimate
    ## we need P >= 2L+1L, M: L <= M.
    P = nextpow(2, M + L + 1)
    kappam_pad = vcat(kappam, zeros(P - 2 * L - 1), reverse(kappam[2:end])) # pad counts and weights for fft
    counts_pad = vcat(counts, zeros(P - M))
    rfft_kappam = rfft(kappam_pad)
    rfft_counts = rfft(counts_pad)

    return sum(counts .* (real(irfft(rfft_kappam .* rfft_counts, P)))[1:M]) / (n^2)
end

# Select the bandwidth via the plug-in approach
function dpik(x::AbstractArray{<:Real}, level::Int, scalest::Symbol, kernel::Type{<:ContinuousUnivariateDistribution}, midpoints::R) where {R<:AbstractRange}
    del0 = ifelse(kernel <: Normal, 1.0/((4.0*pi)^(1/10)), (4.5)^(1/5))

    n = length(x)
    a = first(midpoints)
    b = last(midpoints)

    # Bin the data (multiply by step length, n to get "counts")
    counts = KernelDensity.tabulate(x, midpoints).density * n * step(midpoints)

    # Compute the scale estimate
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

    ## Replace input data by standardised data for numerical stability:
    sa = (a-mean(x)) / σ_hat
    sb = (b-mean(x)) / σ_hat

    ψ4_hat = if level == 0
        3.0/(8.0*sqrt(pi))
    elseif level == 1
        alpha = (2.0*(sqrt(2.0))^7/(5.0*n))^(1/7) # bandwidth for ψ4
        bkfe(counts, 4, alpha, (sa, sb)) 
    elseif level == 2
        alpha = (2.0*(sqrt(2.0))^9/(7.0*n))^(1/9) # bandwidth for ψ6
        ψ6_hat = bkfe(counts, 6, alpha, (sa, sb))
        alpha = (-3.0*sqrt(2.0/pi)/(ψ6_hat*n))^(1/7) # bandwidth for ψ4
        bkfe(counts, 4, alpha, (sa, sb))
    elseif level == 3 
        alpha = (2.0*(sqrt(2.0))^11/(9.0*n))^(1/11) # bandwidth for ψ8
        ψ8_hat = bkfe(counts, 8, alpha, (sa, sb))
        alpha = (15.0*sqrt(2.0/pi)/(ψ8_hat*n))^(1/9) # bandwidth for ψ6
        ψ6_hat = bkfe(counts, 6, alpha, (sa, sb))
        alpha = (-3.0*sqrt(2.0/pi)/(ψ6_hat*n))^(1/7) # bandwidth for ψ4
        bkfe(counts, 4, alpha, (sa, sb))
    elseif level == 4
        alpha = (2*(sqrt(2))^13/(11*n))^(1/13) # bandwidth for ψ10
        ψ10_hat = bkfe(counts, 10, alpha, (sa, sb))
        alpha = (-105*sqrt(2/pi)/(ψ10_hat*n))^(1/11) # bandwidth for ψ8
        ψ8_hat = bkfe(counts, 8, alpha, (sa, sb))
        alpha = (15*sqrt(2/pi)/(ψ8_hat*n))^(1/9) # bandwidth for ψ6
        ψ6_hat = bkfe(counts, 6, alpha, (sa, sb))
        alpha = (-3*sqrt(2/pi)/(ψ6_hat*n))^(1/7) # bandwidth for ψ4
        bkfe(counts, 4, alpha, (sa, sb))
    else 
        alpha = (2.0*(sqrt(2.0))^15/(13.0*n))^(1/15) # bandwidth for ψ12
        ψ12_hat = bkfe(counts, 12, alpha, (sa, sb))
        alpha = (945.0*sqrt(2.0/pi)/(ψ12_hat*n))^(1/13) # bandwidth for ψ10
        ψ10_hat = bkfe(counts, 10, alpha, (sa, sb))
        alpha = (-105.0*sqrt(2.0/pi)/(ψ10_hat*n))^(1/11) # bandwidth for ψ8
        ψ8_hat = bkfe(counts, 8, alpha, (sa, sb))
        alpha = (15.0*sqrt(2.0/pi)/(ψ8_hat*n))^(1/9) # bandwidth for ψ6
        ψ6_hat = bkfe(counts, 6, alpha, (sa, sb))
        alpha = (-3.0*sqrt(2.0/pi)/(ψ6_hat*n))^(1/7) # bandwidth for ψ4
        bkfe(counts, 4, alpha, (sa, sb))
    end

    return σ_hat * del0 * (1.0/(ψ4_hat*n))^(1/5) # return computed plug-in bandwidth
end

# Document this function in the base library and extend the kde function to estimate kde for a vector of bandwidths
function bandwidth(x::AbstractVector{<:Real}, method::SJ, midpoints::R) where {R<:AbstractRange}
    kernel = Normal
    if !any([kernel <: krnl for krnl in (Normal, Uniform)])
        throw(ArgumentError(kernel, "Only Normal and Uniform kernels are supported."))
    end
    h_sj = dpik(x, method.level, method.scalest, kernel, midpoints)
    return h_sj
end

function bandwidth(x::AbstractVector{<:Real}, method::SJ; npoints::Int=2048)
    h0 = KernelDensity.default_bandwidth(x)
    boundary = KernelDensity.kde_boundary(x, h0)
    midpoints = KernelDensity.kde_range(boundary, npoints)
    return bandwidth(x, method, midpoints)
end