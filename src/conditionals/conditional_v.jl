function logp_conditional_v(v::AbstractArray, inv_Σ::AbstractArray, σ_v2::Real, β::AbstractArray, inv_S::AbstractArray, P_root::AbstractArray, z::AbstractArray, F::AbstractArray, J::Int, K::Int, Tsubp::Int)
    logp = 0.0

    # contribution from prior of v | σ_v
    logp -= 0.5/σ_v2 * sum(abs2, v)

    # contribution from prior of β | Σ
    C = cholesky(inv_Σ).L # this will be moved out eventually
    logp += J * logdet(C) # maybe this was the problem
    temp1 = vec(reshape(P_root * β, (J, K)) * C)
    logp -= 0.5*sum(abs2, temp1)

    # contribution from likelihood
    logp += Tsubp * logdet(C)
    XB = F * reshape(β, (J, K))
    temp2 = vec( (reshape(inv_S * z, (Tsubp, K)) - XB) * C )
    logp -= 0.5*sum(abs2, temp2)
    return logp
end


function grad_logp_conditional_v(v::AbstractArray, inv_Σ::AbstractArray, σ_v2::Real, β::AbstractArray, inv_S::AbstractArray, P_root::AbstractArray, z::AbstractArray, F::AbstractArray, Dl::AbstractArray, J::Int, K::Int, Tsubp::Int)
    δ = 1e-6
    #sqPvec = reshape(P_root, (J, K))
    sqPvec = diag(P_root)

    dv = deriv_veclΣ_v(v, δ) # This function has to be written later
    Dldv = Dl * dv
    temp_0 = sum(Dldv .* vec(inv_Σ), dims=1)
    temp_02 = kron(inv_Σ, inv_Σ) * Dldv # this is a vector of size length(v)

    #= dv <- deriv_veclSigma_v(nu, delta)
    Dldv <- Dl%*%dv
    temp_0 <- colsums((Dldv)*c(Sigmainv))
    temp_02 <- akron(Sigmainv,Sigmainv)%*%Dldv =#

    temp_1 = -0.5*K*temp_0 # from logdeterminant of Σ(v)
    t1 = reshape(P_root * β, (J, K)) # a bit unsure about this one
    temp_2 = 0.5 * (vec(t1' * t1)') * temp_02 # also not sure about this one

    grad_log_dens_β = vec(temp_1 + temp_2)

#=     temp_1 <- -q/2*temp_0
    t1 <- matrix(sqPvec*c(Mbeta),ncol=p)
    temp_2 <- 0.5*as.vector((crossprod(t1,t1)))%*%temp_02
    grad_log_dens_beta <- temp_1 + temp_2 =#

    grad_log_dens_v = -v / σ_v2

    temp_3 = -0.5*Tsubp * temp_0 # from logdeterminant of Σ(v)
    XB = F * reshape(β, (J, K))
    temp = reshape(inv_S * z, (Tsubp, K)) - XB # this is recomputed for a lot of the derivatives, can likely reduce the number of times this is computed by passing it as an arg
    temp_4 = 0.5 * (vec(temp' * temp)') * temp_02
    grad_log_dens_y = vec(temp_3 + temp_4)

#=     temp_3 <- -n/2*temp_0 
    temp <- Msvecinvz - amatr(F,Mbeta)
    temp_4 <- 0.5*as.vector(crossprod(temp,temp))%*%temp_02
    grad_log_dens_y <- temp_3 + temp_4 =#

    grad = grad_log_dens_β + grad_log_dens_v + grad_log_dens_y
    return grad
end


function test_autodiff()
    rng = Random.Xoshiro(1)
    K = 4
    J = 20
    Tsubp = 9990

    log_ξ = rand(rng, Normal(), K*J)
    β = rand(rng, Normal(), K*J)
    v = rand(rng, Normal(), div(K*(K-1), 2))
    inv_Σ = inv(compute_Σ_from_v(v, 1e-6))
    σ_v2 = rand(rng, InverseGamma(1e-3, 1e-3))
    #Σ = Symmetric(rand(rng, LKJ(K, 1)))
    #inv_Σ = inv(Σ)

    P_root = Diagonal(@. 1.0 / exp(log_ξ))

    F = rand(rng, Normal(), (Tsubp, J))
    z = rand(rng, Normal(), Tsubp*K)
    F_tilde = F .^2
    inv_S = Diagonal( sqrt.( 1.0 .+ vec( F_tilde * reshape( diag(inv(P_root)) .^2, (J, K) ) ) ) )

    Dl = compute_Dl(K)

    grad_logp_conditional_v(v, inv_Σ, σ_v2, β, inv_S, P_root, z, F, Dl, J, K, Tsubp)

    logp_conditional_v(v, inv_Σ, σ_v2, β, inv_S, P_root, z, F, J, K, Tsubp)

    # Can test derivative with Optim. If it is correct, we can expect it to yield similar results to NelderMead()
    function obj(v)
        inv_Σ = Symmetric(inv(compute_Σ_from_v(v, 1e-14)))
        #println(minimum(eigen(inv_Σ)))
        println(v)
        logp_conditional_v(v, inv_Σ, σ_v2, β, inv_S, P_root, z, F, J, K, Tsubp)
    end
    v0 = rand(rng, Normal(), length(v))
    optimize(obj, v0) # Running this eventualy


    function grad!(G, v)
        G = grad_logp_conditional_v(v, inv_Σ, σ_v2, β, inv_S, P_root, z, F, Dl, J, K, Tsubp)
    end

    optimize(obj, )

    # Can't use autodiff here due to the iteration required. At least, a custom rule for that step is required.
    
    #= logp_conditional_v_cl = let β = β, inv_S = inv_S, P_root = P_root, z = z, F = F, J = J, K = K, Tsubp = Tsubp
        v -> logp_conditional_v_autodiff(v, β, inv_S, P_root, z, F, J, K, Tsubp)
    end =#

    #ForwardDiff.gradient(logp_conditional_v_cl, v) |> println
end