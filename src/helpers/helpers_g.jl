# Here, the diagonal elements of G are stored last in the vector so that γ = (vecl(G), log(diag(G))).
# Make non-allocating version of the functions here at a later point.

"""
    compute_G(γ::AbstractVector, K::Int, M::Int)

Function used to compute the G matrix in the factor parametrization Γ = GG' + D, where Σ is the correlation matrix corresponding to Γ.

# Arguments
* `γ`: Unconstrained Parameter vector satisfying γ = (vecl(G), log(diag(G)))
* `K`: Dimension of the correlation matrix Σ
* `M`: Second dimension of G, e.g. G is K × M dimensional
"""
function compute_G(γ::AbstractVector, K::Int, M::Int)
    G = zeros(eltype(γ), K, M)

    num_lower = K*M - M*(M + 1) ÷ 2
    idx = 1             # for lower triangle values
    diag_idx = num_lower + 1  # start of diagonal entries in γ

    for j in 1:M
        for i in j+1:K
            idx = (j - 1)*K - (j - 1)*j ÷ 2 + (i - j)
            G[i, j] = γ[idx]
        end
        diag_idx = num_lower + j
        G[j, j] = exp(γ[diag_idx])
    end

    return G
end


"""
    compute_Σ(γ::AbstractVector, K::Int, M::Int)

Function used to compute the correlation matrix Σ from the unconstrained parameter vector γ.

# Arguments
* `γ`: Unconstrained Parameter vector satisfying γ = (vecl(G), log(diag(G)))
* `K`: Dimension of the correlation matrix Σ
* `M`: Second dimension of G, e.g. G is K × M dimensional
"""
function compute_Σ(γ::AbstractVector, K::Int, M::Int) # Keep in mind that the original function here takes in transformed parameters.
    G = compute_G(γ, K, M)
    G2_rowsum = sum(G .^2, dims = 2)
    
    G_tilde = G ./ sqrt.(1.0 .+ G2_rowsum)                      # element-wise division, broadcasting denom along columns

    D_tilde = Diagonal(1.0 ./ (1.0 .+ G2_rowsum)[:, 1])
    Σ = G_tilde * transpose(G_tilde) + D_tilde
    return Symmetric(Σ)
end


"""
    partial_deriv_Σ_g03(γ::AbstractVector, K::Int, M::int)

Function calculating the derivative of Σ(γ) with respect to γ.

# Arguments
* `γ`: Unconstrained Parameter vector satisfying γ = (vecl(G), log(diag(G)))
* `K`: Dimension of the correlation matrix Σ
* `M`: Second dimension of G, e.g. G is K × M dimensional
"""
function deriv_Σ_g03(γ::AbstractVector, K::Int, M::Int) # this yields the same result as R!
    G = compute_G(γ, K, M)
    
    total_len = K*M - M*(M + 1) ÷ 2
    diag_start = total_len + 1

    G_sqnorm_p1 = Vector{Float64}(undef, K)
    for k in 1:K
        G_sqnorm_p1[k] = 1.0 + sum(abs2, view(G, k, :))
    end

    compute_row = let G = G, G_sqnorm_p1 = G_sqnorm_p1, K = K, M = M
        function (i)
            itemp = i - K*(cld(i, K) - 1)
            jtemp = cld(i, K)

            Di = zeros(Float64, total_len + M)
            idx = 1

            if itemp != jtemp
                for n in 1:M
                    for m in (n + 1):K
                        Di[idx] = partial_deriv_Σ_G2(G, G_sqnorm_p1, m, n, itemp, jtemp)
                        #Di[idx] = partial_deriv_Σ_G2(G, m, n, itemp, jtemp)
                        idx += 1
                    end
                end
            end

            for m in 1:M
                lmm = log(G[m, m])
                Di[diag_start + m - 1] = partial_deriv_Σ_l(lmm, G, G_sqnorm_p1, m, itemp, jtemp)
                #Di[diag_start + m - 1] = partial_deriv_Σ_l(lmm, G, m, itemp, jtemp)
            end

            return Di
        end
    end

    deriv = Matrix{eltype(γ)}(undef, K^2, total_len + M)
    for i in 1:K^2 # This can be multithreaded, but it is already fairly quick to evaluate.
        deriv[i, :] .= compute_row(i)
    end
    return deriv
end


"""
    partial_deriv_Σ_G2(G::AbstractMatrix, m::Int, n::Int, i::Int, j::Int)

Function calculating the partial derivative of Σ(γ)_{ij} wrt g_{mn} for n ≤ m
"""
function partial_deriv_Σ_G2(G::AbstractMatrix, m::Int, n::Int, i::Int, j::Int) # write out the loops here later
    d = zero(eltype(G)) # we can assume i != j here as this has already been checked
    if i == m
        d = @views (G[j, n] - (dot(G[m, :], G[j, :])) / (1.0 + sum(abs2, G[m, :])) * G[m, n]) /
            (sqrt(1.0 + sum(abs2, G[m, :])) * sqrt(1.0 + sum(abs2, G[j, :])))
    elseif j == m
        d = @views (G[i, n] - (dot(G[m, :], G[i, :])) / (1.0 + sum(abs2, G[m, :])) * G[m, n]) /
            (sqrt(1.0 + sum(abs2, G[m, :])) * sqrt(1.0 + sum(abs2, G[i, :])))
    end
    return d
end


"""
    partial_deriv_Σ_G(G::AbstractMatrix, m::Int, n::Int, i::Int, j::Int)

Function calculating the partial derivative of Σ(γ)_{ij} wrt g_{mn} for n ≤ m
"""
function partial_deriv_Σ_G(G::AbstractMatrix, m::Int, n::Int, i::Int, j::Int) # write out the loops here later
    d = zero(eltype(G))
    if i != j && i == m
        #d = @views (G[j, n] - (sum(G[m, :] .* G[j, :])) * inv(1.0 + sum(abs2, G[m, :])) * G[m, n]) /
        #    (sqrt(1.0 + sum(abs2, G[m, :])) * sqrt(1.0 + sum(abs2, G[j, :])))
        d = @views (G[j, n] - (dot(G[m, :], G[j, :])) / (1.0 + sum(abs2, G[m, :])) * G[m, n]) /
            (sqrt(1.0 + sum(abs2, G[m, :])) * sqrt(1.0 + sum(abs2, G[j, :])))
    elseif i != j && j == m
        #d = @views (G[i, n] - (sum(G[m, :] .* G[i, :])) * inv(1.0 + sum(abs2, G[m, :])) * G[m, n]) /
        #    (sqrt(1.0 + sum(abs2, G[m, :])) * sqrt(1.0 + sum(abs2, G[i, :])))
        d = @views (G[i, n] - (dot(G[m, :], G[i, :])) / (1.0 + sum(abs2, G[m, :])) * G[m, n]) /
            (sqrt(1.0 + sum(abs2, G[m, :])) * sqrt(1.0 + sum(abs2, G[i, :])))
    end
    return d 
end

function partial_deriv_Σ_G2(G::AbstractMatrix, G_sqnorm_p1::AbstractVector, m::Int, n::Int, i::Int, j::Int) # write out the loops here later
    d = zero(eltype(G)) # we can assume i != j here as this has already been checked
    if i == m
        #d = @views (G[j, n] - (sum(G[m, :] .* G[j, :])) * inv(1.0 + sum(abs2, G[m, :])) * G[m, n]) /
        #    (sqrt(1.0 + sum(abs2, G[m, :])) * sqrt(1.0 + sum(abs2, G[j, :])))
        d = @views (G[j, n] - (dot(G[m, :], G[j, :])) / (G_sqnorm_p1[m]) * G[m, n]) /
            (sqrt(G_sqnorm_p1[m]) * sqrt(G_sqnorm_p1[j]))
    elseif j == m
        #d = @views (G[i, n] - (sum(G[m, :] .* G[i, :])) * inv(1.0 + sum(abs2, G[m, :])) * G[m, n]) /
        #    (sqrt(1.0 + sum(abs2, G[m, :])) * sqrt(1.0 + sum(abs2, G[i, :])))
        d = @views (G[i, n] - (dot(G[m, :], G[i, :])) / (G_sqnorm_p1[m]) * G[m, n]) /
            (sqrt(G_sqnorm_p1[m]) * sqrt(G_sqnorm_p1[i]))
    end
    return d
end

function partial_deriv_Σ_G(G::AbstractMatrix, G_sqnorm_p1::AbstractVector, m::Int, n::Int, i::Int, j::Int) # write out the loops here later
    d = zero(eltype(G))
    if i != j && i == m
        #d = @views (G[j, n] - (sum(G[m, :] .* G[j, :])) * inv(1.0 + sum(abs2, G[m, :])) * G[m, n]) /
        #    (sqrt(1.0 + sum(abs2, G[m, :])) * sqrt(1.0 + sum(abs2, G[j, :])))
        d = @views (G[j, n] - (dot(G[m, :], G[j, :])) / (G_sqnorm_p1[m]) * G[m, n]) /
            (sqrt(G_sqnorm_p1[m]) * sqrt(G_sqnorm_p1[j]))
    elseif i != j && j == m
        #d = @views (G[i, n] - (sum(G[m, :] .* G[i, :])) * inv(1.0 + sum(abs2, G[m, :])) * G[m, n]) /
        #    (sqrt(1.0 + sum(abs2, G[m, :])) * sqrt(1.0 + sum(abs2, G[i, :])))
        d = @views (G[i, n] - (dot(G[m, :], G[i, :])) / (G_sqnorm_p1[m]) * G[m, n]) /
            (sqrt(G_sqnorm_p1[m]) * sqrt(G_sqnorm_p1[i]))
    end
    return d 
end


"""
    partial_deriv_Σ_l(lmm::Real, G::AbstractMatrix, m::Int, i::Int, j::Int)

Function calculating the partial derivative of Σ(γ)_{ij} wrt l_{mm} for n ≤ m
"""
function partial_deriv_Σ_l(lmm::Real, G::AbstractMatrix, m::Int, i::Int, j::Int)
    return exp(lmm) * partial_deriv_Σ_G(G, m, m, i, j)
end

function partial_deriv_Σ_l(lmm::Real, G::AbstractMatrix, G_sqnorm_p1::AbstractVector, m::Int, i::Int, j::Int)
    return exp(lmm) * partial_deriv_Σ_G(G, G_sqnorm_p1, m, m, i, j)
end