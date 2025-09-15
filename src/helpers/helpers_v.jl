"""
    compute_Σ_from_v(v::AbstractVector, δ::Real)

Function to compute the correlation matrix Σ(v) from the unconstrained parameter vector v.

# Arguments
* `v`: Unconstrained parameter vector.
* `δ`: Tolerance for iterative procedure use to compute Σ.

# Return
* `Σ`: The correlation matrix.
"""
function compute_Σ_from_v(v::AbstractVector, δ::Real)
    p = floor(Int, 0.5 * (1.0 + sqrt(1.0 + 8.0*length(v))))
    x0 = zeros(Float64, p)
    A = Matrix{Float64}(undef, p, p)
    # Set non-diagonal entries of A to those of v
    ind = 1
    for i in 1:p
        for j in i+1:p
            A[i,j] = v[ind]
            A[j,i] = v[ind]
            ind += 1
        end
    end
    B = Symmetric(A)
    tol = sqrt(0.5 * (1.0 + sqrt(1.0 + 8.0*length(v)))) * δ
    err = tol + 1.0
    while err > tol
        for i = 1:p
            B[i,i] = x0[i] # setting diagonal entries in Symmetric matrix is ok
        end
        x = x0 - log.(diag(exp(B)))
        err = sum((x - x0).^2) |> sqrt
        x0 = copy(x)
    end
    for i = 1:p
        B[i,i] = x0[i] # setting diagonal entries in Symmetric matrix is ok
    end
    Σ = Symmetric(exp(B))
    for i = 1:p
        Σ[i,i] = 1.0 # setting diagonal entries in Symmetric matrix is ok
    end
    return Σ
end


"""
    compute_Dl(K::Int)

Function calculating a matrix Dl s.t. vec(Σ) = vec(I(K)) + Dl * vecl(Σ)

# Arguments
* `K`: Dimension of the correlation matrix Σ
"""
function compute_Dl(K::Int) # This should perhaps return a sparse matrix in the future, each row has at most a single nonzero entry
    Dl = zeros(Float64, K^2, div(K*(K-1), 2))
    for k in 1:K
        for q in 1:K
            i = (k-1)*K + q
            if k < q
                Dl[i, (k - 1)*K - div(k*(k + 1), 2) + q] = 1
            elseif k > q
                Dl[i, (q - 1)*K - div(q*(q + 1), 2) + k] = 1
            end
        end
    end
    return sparse(Dl) # return sparse matrix for more efficient Matrix-vector operations down the line
end

"""
    deriv_vecΣ_veclogΣ(v::AbstractVector, δ::Real)

Computes the derivative of vec(Σ) with respect to vec(log(Σ)) for Σ = Σ(v)

# Arguments
* `v`: Unconstrained parameter vector
* `δ`: Tolerance for iterative procedure use to compute Σ when calling `compute_Σ_from_v`.
"""
function deriv_vecΣ_veclogΣ(v::AbstractVector, δ::Real) # checked against corresponding R function
    K = floor(Int, 0.5 * (1.0 + sqrt(1.0 + 8.0*length(v))))
    Σ = compute_Σ_from_v(v, δ) # this is unecessary, we call this multiple times so we can pass it as an argument instead
    log_Σ = log(Σ)
    eig_log_Σ = eigen(log_Σ, sortby=-)
    Q = eig_log_Σ.vectors
    λ = eig_log_Σ.values
    Xi_vec = Vector{Float64}(undef, K^2)

    for i in 1:K
        for j in 1:K
            Xi_vec[(i-1)*K + j] = ifelse(
                λ[i] ≈ λ[j],
                exp(λ[i]),
                (exp(λ[i]) - exp(λ[j])) / (λ[i]-λ[j])
            )
        end
    end
    QkronQ = kron(Q, Q)
    A = QkronQ * Diagonal(Xi_vec) * transpose(QkronQ)
    return A
end


"""
    compute_El(K::Int)

Function calculating an elimination matrix El s.t. vecl(M) = El * vec(M).

# Arguments
* `K`: Dimension of the square matrix M.

# Returns
* `El`: The K(K-1)/2 × K² elimination matrix stored in sparse array format.
"""
function compute_El(K::Int)
    El = zeros(Float64, div(K*(K-1),2 ), K^2)
    k = 1
    l = 1
    for j in 1:K-1
        Ej = zeros(Float64, K-j, K)
        Ej[:, j+1:K] = I(K-j)
        El[k : (k + (K - j) - 1), l : (l + K - 1)] = Ej
        k = k + K - j
        l = l + K
    end
    return sparse(El) # Consider making thjs sparse
end

"""
    compute_Km(m::Int, n::Int)

Function calculating the commutation matrix of dimension m × n.

# Arguments
* `m, n`: Dimension of the commutator matrix

# Returns
* `Km`: The mn × mn commutator matrix stored in sparse array format.

# References 
Se the "Commutation matrix" article on [Wikipedia](https://en.wikipedia.org/wiki/Commutation_matrix).
"""
function compute_Km(m::Int, n::Int)
    K = zeros(Float64, m*n, m*n)
    for i in 1:m
        for j in 1:n
            K[i + m*(j-1), j + n*(i - 1)] = 1
        end
    end
    return sparse(K)
end


"""
    compute_Eu(K::Int)

Function calculating an elimination matrix Eu s.t. vecl(M') = Eu * vec(M)

# Arguments
* `K`: Dimensional of the square matrix M

# Returns
* `Eu`: The K² × K(K-1)/2 elimination matrix stored in a sparse array format.
"""
compute_Eu(K::Int) = compute_El(K) * compute_Km(K, K) # result is automatically sparse

"""
    compute_Ed(K::Int)

Function calculating an elimination matrix Ed s.t. diag(M) = Ed * vec(M)

# Arguments
* `K`: Dimension of the square matrix M

# Returns:
* `Ed`: The K × K² elimination matrix stored in a sparse array format.
"""
function compute_Ed(K::Int)
    Ed = zeros(Float64, K, K^2)
    for i in 1:K
        Ed[i, 1 + (i - 1)*(K + 1)] = 1.0
    end
    return sparse(Ed)
end

"""
    deriv_veclΣ_v(v::AbstractVector, δ::Real)

Function calculating the derivative of vecl(Σ) with respect to v.

# Arguments
* `v`: Unconstrained parameter vector.
* `δ`: Tolerance for iterative procedure use to compute Σ.

# Return
* The K(K-1)/2 × K(K-1)/2 derivative matrix.
"""
function deriv_veclΣ_v(v::AbstractVector, δ::Real)
    K = floor(Int, 0.5 * (1.0 + sqrt(1.0 + 8.0*length(v))))
    A = deriv_vecΣ_veclogΣ(v, δ) 
    El = compute_El(K)
    Eu = compute_Eu(K)
    Ed = compute_Ed(K)

    #El * (I(K^2) - A * transpose(Ed) * inv(Ed * A * transpose(Ed))* Ed) * A * transpose(El + Eu)
    ret = El * (I(K^2) - A * transpose(Ed) * (Ed * A * transpose(Ed) \ Ed)) * A * transpose(El + Eu)
    return ret
end
