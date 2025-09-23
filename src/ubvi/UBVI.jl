struct UBVIState
    Z::Matrix{Float64} # Change this to a lower triangular matrix at a later point
    weights::Vector{Float64}
    comps::Vector{FactorMVNormal}
    logfg::Vector{Float64}
    logfgsum::Float64
end

"""
    compute_weights(state::UBVIState, dist::FactorMVNormal)

Compute the new normalization factors (the Z vectors) for dist.
"""
function update_weights(state::UBVIState, dist::FactorMVNormal)
    # Compute new normalization matrix
    Z_new = exp.(log_sqrt_pair_integral(dist, state.comps))
    Z_old = mixture.Z

    Z = Matrix{Float64}(undef, size(Z_new, 1)+1, size(Z_new, 2)+1)
    Z[1:end-1, 1:end-1] = Z_old
    Z[end,:] = Z_new
    Z[:,end] = Z_new

    # Compute log ⟨f, g⟩

    comps = vcat(state.comps, dist)
    return UBVIState(Z, weights, comps)
end