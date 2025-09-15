"""
    fit_marginals(My::AbstarctMatrix{<:Real}, method::T) where {T<:BandwidthSelectors.AbstractBandWidthSelector}

Fit kernel density estimates to the marginal distributions `My[:,k]` for each k.

# Arguments
* `My`: Matrix of observations.
* `method`: Method used to fit the KDEs.

# Returns
* `Mz_est`: Estimated matrix of latent variables.
* `kdests`: Vector of kernel density estimates, where `kdests[k]` is the KDE fitted to `My[:,k]`
"""
function fit_marginals(My::AbstractMatrix{<:Real}, method::T) where {T<:BandwidthSelectors.AbstractBandwidthSelector}
    # Fit models for the marginals:
    kdests = [fit(UnivariateKDE, My[:, 1], method)]
    for k in 2:size(My, 2)
        push!(kdests, fit(UnivariateKDE, My[:, k], method))
    end

    # Transform data to latent scale through the estimate
    Mz_est = similar(My)
    for k in axes(My, 2)
        Mz_est[:, k] = quantile.(Normal(), cdf.(kdests[k], My[:, k]))
    end
    return Mz_est, kdests
end