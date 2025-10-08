"""
    SSVKernel()

The variable-bandwidth selector of [shimazaki2010kernel](@citet).

```julia-repl
julia> x = randn(10^3);

julia> kdest = fit(UnivariateKDE, x, SSVKernel());
```

!!! note
    Currently only the Normal kernel is supported.

# References
* Shimazaki and Shinomoto (2010). Kernel bandwidth optimization in spike rate estimation. https://doi.org/10.1007/s10827-009-0180-4
"""
struct SSVKernel <: AdaptiveBandwidthSelector end

function fit(::Type{UnivariateKDE}, x::AbstractVector{<:Real}, method::SSVKernel; npoints::Int=2048)
    h0 = bandwidth_silverman(x, :min)
    boundary = KernelDensity.kde_boundary(x, h0)
    midpoints = KernelDensity.kde_range(boundary, npoints)
    return fit_ssvkernel(x, midpoints)
end

function fit(::Type{UnivariateKDE}, x::AbstractVector{<:Real}, method::SSVKernel, midpoints::R) where {R<:AbstractRange}
    return fit_ssvkernel(x, midpoints)
end

# Compute the convolution betweeen a vector x and a specified kernel with bandwidth bw
function fftkernel(x::AbstractVector{T}, bw::Real, kernel::Type{<:ContinuousUnivariateDistribution}) where {T<:Real}
    L = length(x)
    Lmax = L .+ 3.0 * bw
    n = 2^(ceil(Int, log2(Lmax)))

    dist = KernelDensity.kernel_dist(kernel, bw)

    # augment vector with zeros s.t. vector length is 0
    if L ≥ n
        xtemp = x[1:n]
    else
        xtemp = vcat(x, zeros(T, n-L))
    end

    ft = rfft(xtemp)

    c = -2.0*pi/n
    for j = 0:length(ft)-1
        ft[j+1] *= cf(dist, j*c)
    end

    y = irfft(ft, n)
    return y[1:L]
end

# Cost function used to determine local bandwidths
function cost_function(y_hist::AbstractVector{<:Real}, n::Int, midpoints::R, optws::AbstractMatrix{<:Real}, local_bws::AbstractVector{<:Real},
                       kernel::Type{<:ContinuousUnivariateDistribution}, g::Real
    ) where {R<:AbstractRange}
    Δm = step(midpoints)
    #Selecting w/W = g bandwidth
    L = length(y_hist)
    optwv = Vector{Float64}(undef, L)
    for k in 1:L
        gs = optws[:,k] ./ local_bws
        if g > maximum(gs)
            optwv[k] = minimum(local_bws)
        elseif g < minimum(gs)
            optwv[k] = maximum(local_bws)
        else
            idx = findlast(gs .>= g)
            optwv[k] = g*local_bws[idx]
        end
    end

    # Nadaraya-Watson kernel regression
    optwp = Vector{Float64}(undef, L)
    dist = KernelDensity.kernel_dist(kernel, 1.0)
    for k in 1:L
        Z = @. pdf(dist, g*(midpoints[k] - midpoints) / optwv)
        optwp[k] =  sum(optwv .* Z)/sum(Z)
    end

    # Baloon estimator (speed optimized)
    idx = y_hist .!= 0
    y_hist_nz = y_hist[idx]
    midpoints_nz = midpoints[idx]

    yv = Vector{Float64}(undef, L)
    for k in 1:L
        dist = KernelDensity.kernel_dist(kernel, optwp[k])
        yv[k] = sum(@. y_hist_nz * Δm * pdf(dist, midpoints[k] - midpoints_nz))
    end
    yv = yv * n / sum(yv * Δm)

    # Cost function of the estimated density
    cg = @. yv^2 - 2.0*yv * y_hist + 2.0/sqrt(2.0*pi) / optwp * y_hist
    Cg = sum(cg .* Δm)

    return Cg, yv, optwp
end

# Evaluate log(1.0 + exp(x)) in a numerically stable manner (avoid overflow)
function log_exp_plus_one(x::Real)
    return x + log(1.0 + exp(-x))
end

# Evaluate log(exp(x) - 1.0) in a numerically stable manner (avoid overflow)
function log_exp_sub_one(x::Real)
    return x + log(1.0 - exp(-x))
end

function fit_ssvkernel(x::AbstractVector{<:Real}, midpoints::R) where {R<:AbstractRange}
    kernel = Normal
    M = 80            # Number of bandwidths examined for optimization.
    n = length(x)

    Δm = step(midpoints)
    xmin, xmax = extrema(x)
    ran = xmax - xmin
    
    # Create a finest histogram
    y_hist = KernelDensity.tabulate(x, midpoints).density * n
    L = length(y_hist)

    # Computing local MISE for each bandwidth
    local_bws = log_exp_plus_one.(LinRange(log_exp_sub_one(maximum(5.0*Δm)), log_exp_sub_one(ran), M))

    c = Matrix{Float64}(undef, M, L)
    for m in 1:M
        y_smooth = fftkernel(y_hist, local_bws[m]/Δm, kernel)
        c[m,:] = @. y_smooth^2 - 2.0*y_smooth * y_hist + 2.0/sqrt(2.0*pi)/local_bws[m]*y_hist
    end
    
    # Smooth local MISE via a Gaussian kernel and record the resulting local optimal bin widths 
    optws = Matrix{Float64}(undef, M, L)
    for m in 1:M
        C_local = Matrix{Float64}(undef, M, L)
        for j in 1:M
            C_local[j,:] = fftkernel(view(c, j,:), local_bws[m]/Δm, kernel)
        end
        j_opt = [argmin(col) for col in eachcol(C_local)]
        optws[m,:] = local_bws[j_opt]
    end

    # Golden Section search for stiffness parameter
    # Selecting a bandwidth w/W = g.
    a = 1e-12
    b = 1.0

    function cost_function_optim(g)
        return cost_function(y_hist, n, midpoints, optws, local_bws, kernel, g)[1]
    end

    res = optimize(cost_function_optim, a, b, GoldenSection(); rel_tol=1e-6) # can change to the internal Golden Search method at a later point in time
    g_opt = minimizer(res)

    _, yv, _ = cost_function(y_hist, n, midpoints, optws, local_bws, kernel, g_opt)

    f_hat = yv / sum(yv * Δm)

    # Return an interpolated KDE object evaluated at t
    return UnivariateKDE(midpoints, f_hat)    
end