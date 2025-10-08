struct DiffusionKernel <: AdaptiveBandwidthSelector
    n_mix::Union{Int,Nothing}
end

"""
    DiffusionKernel(; n_mix::Union{Int,Nothing})
   
The adaptive diffusion kernel estimator of [botev2010diffusion](@citet).

Based on the matlab implementation of [botev2025akde1d](@citet), where the solution to the PDE is approximated using a lower-dimensional finite mixture of normals.
The number of components in the mixture can be controlled through the keyword `n_mix`, with larger values yielding smaller approximation errors at the cost of increased computation.

# Keyword arguments
* `n_mix`: Number of mixture components used to construct the density estimate. Defaults to ``20 + \\lceil n^{1/3} \\rceil`` where ``n`` is the sample size.

```julia-repl
julia> x = randn(10^3);

julia> kdest = fit(UnivariateKDE, x, DiffusionKernel());
```

!!! note
    Currently only the Normal kernel is supported.

# References
* Botev et al. (2010). Kernel density estimation via diffusion. https://doi.org/10.1214/10-AOS799
* Botev (2025). adaptive kernel density estimation in one-dimension. https://de.mathworks.com/matlabcentral/fileexchange/58309-adaptive-kernel-density-estimation-in-one-dimension?s_tid=prof_contriblnk
"""
function DiffusionKernel(; n_mix::Union{Int,Nothing}=nothing)
    if n_mix isa Int
        if n_mix ≤ 0
            throw(DomainError(level, "The number of mixture components must be a positive number."))
        end
    end
    return DiffusionKernel(n_mix)
end

# This is actually a redundant method once a "core" library is implemented.
function fit(::Type{UnivariateKDE}, x::AbstractVector{<:Real}, method::DiffusionKernel; npoints::Int=2048)
    n = length(x)
    n_mix = something(method.n_mix, min(n, 20 + ceil(Int, n^(1/3))))
    h0 = KernelDensity.default_bandwidth(x)
    boundary = KernelDensity.kde_boundary(x, h0)
    midpoints = KernelDensity.kde_range(boundary, npoints)
    return fit_diffusionkernel(x, midpoints, n_mix)
end

function fit(::Type{UnivariateKDE}, x::AbstractVector{<:Real}, method::DiffusionKernel, midpoints::R) where {R<:AbstractRange}
    n = length(x)
    n_mix = something(method.n_mix, min(n, 20 + ceil(Int, n^(1/3))))
    return fit_diffusionkernel(x, midpoints, n_mix)
end

# To do: Implement a version of this that computes a binned approximation instead.
function fit_diffusionkernel(x::AbstractVector{<:Real}, midpoints::R, n_mix::Int) where {R<:AbstractRange}
    n = length(x)

    # Scale input data
    xs = (x .- first(midpoints)) / (last(midpoints) - first(midpoints))

    # Initialization
    δ = 0.2 / n^(1/5)
    μ = quantile(x, LinRange(0.5/n_mix, 1.0-0.5/n_mix, n_mix))
    w = fill(1.0/n_mix, n_mix)
    σ2 = fill(0.5 * δ^2, n_mix)
     
    ent = -Inf
    rtol = 1e-5
    err = rtol + 1.0

    maxiter = 1500
    it = 1
    while err > rtol && it ≤ maxiter
        Eold = ent
        w, μ, σ2, δ, ent = reg_em(w, μ, σ2, δ, xs, n_mix, n)
        err = abs((ent - Eold)/ ent)
    end
    mesh = (midpoints .- first(midpoints)) / (last(midpoints) - first(midpoints))

    f_hat = zeros(Float64, length(midpoints))
    for i in eachindex(mesh)
        for j in eachindex(w)
            f_hat[i] += w[j] * pdf(Normal(μ[j], sqrt(σ2[j])), mesh[i]) 
        end
        f_hat[i] = f_hat[i] / (last(midpoints) - first(midpoints))
    end

    return UnivariateKDE(midpoints, f_hat)    
end

function reg_em(w::AbstractVector{<:Real}, μ::AbstractVector{<:Real}, σ2::AbstractVector{<:Real}, δ::Real, xs::AbstractVector{<:Real}, n_mix::Int, n::Int)
    log_lh = Matrix{Float64}(undef, n, n_mix)
    log_σ = Matrix{Float64}(undef, n, n_mix)

    for j in 1:n_mix
        s = σ2[j]
        xs_cent = (xs .- μ[j]).^2/ s
        xSig = xs_cent / s .+ eps() # for numerical stability
        log_lh[:, j] = -0.5 * xs_cent .- 0.5 * log(s) .+ log(w[j]) .- 0.5 * log(2.0 * pi) .- 0.5 * δ^2 / s
        log_σ[:, j] = view(log_lh, :, j) + log.(xSig) 
    end

    # Max over rows (dim=2), then convert to vector
    maxll = vec(maximum(log_lh, dims=2))
    maxlsig = vec(maximum(log_σ, dims=2))

    # Subtract row max and exponentiate (broadcast)
    p = exp.(log_lh .- maxll)
    p_σ = exp.(log_σ .- maxlsig)

    # Row sums
    density = vec(sum(p, dims=2))
    p_σ_dens = vec(sum(p_σ, dims=2))

    # Log PDFs
    logpdf = log.(density) .+ maxll
    logp_σ_dens = log.(p_σ_dens) .+ maxlsig

    # Normalize p by row sums (broadcast division)
    p = p ./ density

    # loglikelihood
    ent = sum(logpdf)

    # Column sums of normalized p
    w = vec(sum(p, dims=1))

    for j in eachindex(w)
        if w[j] > 0.0
            μ[j] = sum(view(p, :, j) .* xs) / w[j]
            xs_cent = (xs .- μ[j]).^2
            σ2[j] = sum(view(p, :, j) .* xs_cent) / w[j] + δ^2
        end
    end

    w = w / sum(w) # This step shouldnt be necessary
    curv = mean(exp.(logp_σ_dens - logpdf))
    δ = 1.0 / (8.0 * n * sqrt(pi) * curv)^(1/3)
    return w, μ, σ2, δ, ent
end