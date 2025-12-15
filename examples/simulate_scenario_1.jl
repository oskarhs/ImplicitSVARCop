using Random, Distributions, LinearAlgebra

function check_stationarity(B::NTuple{p, <:AbstractMatrix{<:Real}}) where {p}
    # Compute the VAR(1)-matrix corresponding to the VAR(p) process
    K = size(B[1], 1)
    A = Matrix{Float64}(undef, (K*p, K*p))
    for lag in 1:p
        ind = (lag-1)*K+1:lag*K
        A[1:p, ind] = B[lag]
    end
    eig = eigen(A)
    return maximum(abs.(eig.values)) < 1
end

function simulate_scenario_1(rng, T)
    p = 3
    K = 2
    # Generate parameter matrices:
    not_stationary = true
    while not_stationary
        B = Tuple(rand(rng, Normal(0, 0.1), (K, K)) for lag in 1:p)
        not_stationary = check_stationarity(B)
    end
    Σ = rand(rng, InverseWishart(K+1, I(2)))

    My = Matrix{Float64}(undef, (T, K))
    My[1:p, :] .= rand(rng, Normal(), (p, K))
    for t in p+1:T
        My[t, :] = B[1] * y[t-1] + B[2] * y[t-2] + B[3] * y[t-3] + rand(rng, MvNormal(zeros(K), Σ))
    end
    return My
end