"""
    logp_conditional_γ(γ::AbstractVector, inv_Σ::AbstractMatrix, β::AbstractArray, P_root::AbstractArray, inv_S::AbstractArray, z::AbstractArray, F::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)

Function for computing the log-conditional density log p(γ | ⋯) with respect to γ.

# Arguments
* `γ`: Unconstrained parameter vector that parametrizes the covariance matrix.
* `inv_Σ`: The inverse of the covariance matrix corresponding to γ, e.g. inv_Σ = inv(Σ(v)).
"""
function logp_conditional_γ(γ::AbstractVector, inv_Σ::AbstractMatrix, β::AbstractVector, P_root::AbstractArray, inv_S::AbstractArray, z::AbstractArray, F::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)
    a_γ = 3.0
    b_γ = 1.0

    C = cholesky(inv_Σ).L

    logp = 0.0

    # Contribution from p(g0)
    f = let b_γ = b_γ
        x -> log(1.0 + abs(x)/b_γ)
    end
    logp -= (a_γ + 1.0) * sum(f, γ)

    # Contribution from p(β | Σ)
    logp += J * logdet(C)
    temp1 = reshape(P_root * β, (J, K)) * C
    logp -= 0.5*sum(abs2, temp1)

    # contribution from likelihood
    logp += Tsubp * logdet(C)
    XB = F * reshape(β, (J, K))
    temp2 = (reshape(inv_S * z, (Tsubp, K)) - XB) * C
    logp -= 0.5*sum(abs2, temp2)
    
    return logp
end

function logp_conditional_γ_autodiff(γ::AbstractVector, β::AbstractVector, P_root::AbstractArray, inv_S::AbstractArray, z::AbstractArray, F::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)
    Σ = compute_Σ(γ, K, M)
    inv_Σ = inv(Σ)
    return logp_conditional_γ(γ, inv_Σ, β, P_root, inv_S, z, F, M, J, K, Tsubp)
end

"""
    grad_logp_conditional_γ(γ::AbstractVector, inv_Σ::AbstractMatrix, β::AbstractArray, P_root::AbstractArray, inv_S::AbstractArray, z::AbstractArray, F::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)

Function for computing the gradient of the log-conditional density log p(γ | ⋯) with respect to γ.

# Arguments
* `γ`: Unconstrained parameter vector that parametrizes the covariance matrix.
* `inv_Σ`: The inverse of the covariance matrix corresponding to γ, e.g. inv_Σ = inv(Σ(v)).
"""
function grad_logp_conditional_γ(γ::AbstractVector, inv_Σ::AbstractMatrix, β::AbstractArray, P_root::AbstractArray, inv_S::AbstractArray, z::AbstractArray, F::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)
    a_γ = 3.0
    b_γ = 1.0

    Dldv = deriv_Σ_g03(γ, K, M)
    temp_0 = sum(Dldv .* vec(inv_Σ), dims=1)
    temp_02 = kron(inv_Σ, inv_Σ) * Dldv

    # veclG is the first K*M - M*(M - 1) ÷ 2 elements of γ
    # elli are the last M
    num_nondiag = K*M - div(M*(M+1), 2)
    grad_log_dens_γ = similar(γ)
    for i in 1:num_nondiag # non-diagonal elements
        abs_γ_i = abs(γ[i])
        grad_log_dens_γ[i] = -(a_γ + 1) * (γ[i] / ((b_γ + abs_γ_i) * abs_γ_i))
    end
    for i in num_nondiag+1:num_nondiag+M # diagonal elements
        exp_γ_i = exp(γ[i])
        grad_log_dens_γ[i] = -(a_γ + 1.0) * ( exp_γ_i / (b_γ + exp_γ_i) - 1.0 / (a_γ + 1.0) )
    end

    # Contribution from p(β |γ, ξ)
    temp_1 = -0.5*K*temp_0 # from logdeterminant of Σ(v)
    t1 = reshape(P_root * β, (J, K))
    temp_2 = 0.5 * (vec(t1' * t1)') * temp_02

    grad_log_dens_β = vec(temp_1 + temp_2)

    # Contribution from likelihood
    temp_3 = -0.5*Tsubp * temp_0 # from logdeterminant of Σ(v)
    XB = F * reshape(β, (J, K))
    temp = reshape(inv_S * z, (Tsubp, K)) - XB # this is recomputed for a lot of the derivatives, can likely reduce the number of times this is computed by passing it as an arg
    temp_4 = 0.5 * (vec(temp' * temp)') * temp_02
    grad_log_dens_y = vec(temp_3 + temp_4)

    grad = grad_log_dens_β + grad_log_dens_γ + grad_log_dens_y
    return grad
end


function grad_logp_conditional_γ_unpacked(γ::AbstractVector, inv_Σ::AbstractMatrix, P_rootβrs::AbstractArray,
                                          Mlik::AbstractArray, M::Int, J::Int, K::Int, Tsubp::Int)
    a_γ = 3.0
    b_γ = 1.0

    Dldv = deriv_Σ_g03(γ, K, M)
    temp_0 = sum(Dldv .* vec(inv_Σ), dims=1)
    temp_02 = kron(inv_Σ, inv_Σ) * Dldv

    # veclG is the first K*M - M*(M - 1) ÷ 2 elements of γ
    # elli are the last M
    num_nondiag = K*M - div(M*(M+1), 2)
    grad_log_dens_γ = similar(γ)
    for i in 1:num_nondiag # non-diagonal elements
        abs_γ_i = abs(γ[i])
        grad_log_dens_γ[i] = -(a_γ + 1) * (γ[i] / ((b_γ + abs_γ_i) * abs_γ_i))
    end
    for i in num_nondiag+1:num_nondiag+M # diagonal elements
        exp_γ_i = exp(γ[i])
        grad_log_dens_γ[i] = -(a_γ + 1.0) * ( exp_γ_i / (b_γ + exp_γ_i) - 1.0 / (a_γ + 1.0) )
    end

    # Contribution from p(β |γ, ξ)
    temp_1 = -0.5*K*temp_0 # from logdeterminant of Σ(γ)
    t1 = P_rootβrs
    temp_2 = 0.5 * (vec(t1' * t1)') * temp_02

    grad_log_dens_β = vec(temp_1 + temp_2)

    # Contribution from likelihood
    temp_3 = -0.5*Tsubp * temp_0 # from logdeterminant of Σ(γ)
    temp_4 = 0.5 * transpose(vec(transpose(Mlik) * Mlik)) * temp_02
    grad_log_dens_y = vec(temp_3 + temp_4)

    grad = grad_log_dens_β + grad_log_dens_γ + grad_log_dens_y
    return grad
end

