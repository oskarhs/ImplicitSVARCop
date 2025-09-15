"""
    get_initial_ϵ([rng::Random.AbstractRNG,] η::AbstractVector, logp_cond::T, grad_logp_cond::S) where {T, S}

Compute the initial step length for the HMCDA algorithm as in Algorithm 4 in Hoffman and Gelman (2014).

# Arguments
- `η`: Unconstrained parameter vector.
* `logp_cond`: The target log-conditional density
* `grad_logp_cond`: Derivative of the target log-conditional density

# Returns
* `ϵ_0`: Initial step length for HMCDA.
"""
function get_initial_ϵ(rng::Random.AbstractRNG, η::AbstractVector, logp_cond::T, grad_logp_cond::S) where {T, S}
    ϵ = 1.0
    dim = length(η)
    r = rand(rng, Normal(), dim)

    ∇logp_η = grad_logp_cond(η)
    r_tilde = r + 0.5 * ϵ * ∇logp_η

    η_tilde = η + ϵ * r_tilde
        
    ∇logp_η = grad_logp_cond(η_tilde)
    r_tilde = r + 0.5 * ϵ * ∇logp_η

    while any(isinf.(∇logp_η)) || any(isnan.(∇logp_η))
        ϵ *= 0.5
        ∇logp_η = grad_logp_cond(η)
        r_tilde = r + 0.5 * ϵ * ∇logp_η

        η_tilde = η + ϵ * r_tilde
        
        ∇logp_η = grad_logp_cond(η_tilde) # When we loop around, the first gradient computation of the loop is redundant. Remove this later, but keep for now for enhanced readability.
    end

    log_ratio = logp_cond(η_tilde) - 0.5 * dot(r_tilde, r_tilde) - (logp_cond(η) - 0.5 * dot(r, r))
    a = 2.0 * (exp(log_ratio) > 0.5) - 1.0
    #while (exp(logp_cond(θ_tilde) - 0.5 * dot(r_tilde, r_tilde)) / exp(logp_cond(θ) - 0.5 * dot(r, r)))^a > 2^(-a)
    while a * log_ratio > -a * log(2.0)
        ϵ = 2.0^a * ϵ
        ∇logp_η = grad_logp_cond(η)
        r_tilde = r + 0.5 * ϵ * ∇logp_η

        η_tilde = η + ϵ * r_tilde
        
        ∇logp_η = grad_logp_cond(η_tilde)
        r_tilde = r_tilde + 0.5 * ϵ * ∇logp_η

        log_ratio = logp_cond(η_tilde) - 0.5 * dot(r_tilde, r_tilde) - (logp_cond(η) - 0.5 * dot(r, r))
    end
    return ϵ
end



function nuts_step(rng::Random.AbstractRNG, η::AbstractVector, δ, logp_cond::T, grad_logp_cond::S, ϵ, ϵ_bar, ϵ0, H_bar, m, Madapt, max_tree_depth=5) where {T<:Function, S<:Function}
    """
        - η         : current model parameter
        - δ         : desirable average accept rate
        - logp_cond : target density
        - grad_logp_cond : gradient of target density
        - m         : iteration number
        - Madapt    : number of samples for step size adaptation
        - verbose   : whether to show log
    """

    function build_tree(rng, η, r, u, v, j, ϵ, η0, r0)
        """
        - η   : model parameter
        - r   : momentum variable
        - u   : slice variable
        - v   : direction ∈ {-1, 1}
        - j   : depth of tree
        - ϵ   : leapfrog step size
        - η0  : initial model parameter
        - r0  : initial mometum variable
        """
        Δ_max = 1000.0
        if j == 0
            # Base case - take one leapfrog step in the direction v.
            #θ′, r′ = leapfrog(θ, r, v * ϵ)
            ∇logp_η = grad_logp_cond(η)
            r′ = r + 0.5 * v * ϵ * ∇logp_η

            η′ = η + v * ϵ * r′
            
            ∇logp_η = grad_logp_cond(η′)
            r′ = r′ + 0.5 * v * ϵ * ∇logp_η

            # NOTE: this trick prevents the log-joint or its gradient from being infinte
            while any(isinf.(∇logp_η)) || any(isnan.(∇logp_η))
                ϵ = ϵ * 0.5
                #θ′, r′ = leapfrog(θ, r, v * ϵ)
                ∇logp_η = grad_logp_cond(η)
                r′ = r + 0.5 * v * ϵ * ∇logp_η

                η′ = η + v * ϵ * r′
                
                ∇logp_η = grad_logp_cond(η′)
                r′ = r′ + 0.5 * v * ϵ * ∇logp_η
            end
            ham = logp_cond(η′) - 0.5 * dot(r′, r′)
            n′ = u <= exp(ham)
            #s′ = u < exp(Δ_max + logp_cond(η′) - 0.5 * dot(r′, r′))
            s′ = ham > (log(u) - Δ_max)
            return η′, r′, η′, r′, η′, n′, s′, min(1.0, exp(ham - logp_cond(η0) + 0.5 * dot(r0, r0))), 1 # we can remove a loglik evaluation here, but do this later.
        else
            # Recursion - build the left and right subtrees.
            ηm, rm, ηp, rp, η′, n′, s′, α′, n′_α = build_tree(rng, η, r, u, v, j - 1, ϵ, η0, r0)
            if s′ == 1
                if v == -1
                    ηm, rm, _, _, η′′, n′′, s′′, α′′, n′′_α = build_tree(rng, ηm, rm, u, v, j - 1, ϵ, η0, r0)
                else
                    _, _, ηp, rp, η′′, n′′, s′′, α′′, n′′_α = build_tree(rng, ηp, rp, u, v, j - 1, ϵ, η0, r0)
                end
                if rand(rng, Uniform()) < n′′ / (n′ + n′′)
                    η′ = η′′
                end
                α′ = α′ + α′′
                n′_α = n′_α + n′′_α
                s′ = s′′ & (dot(ηp - ηm, rm) >= 0) & (dot(ηp - ηm, rp) >= 0)
                n′ = n′ + n′′
            end
            return ηm, rm, ηp, rp, η′, n′, s′, α′, n′_α
        end
    end

    μ, γ, t_0, κ = log(10.0 * ϵ0), 0.05, 10, 0.75

    r0 = rand(rng, Normal(), length(η))
    u = rand(rng, Uniform()) * exp(logp_cond(η) - 0.5 * dot(r0, r0)) # Note: θ^{m-1} in the paper corresponds to
                                                #       `θs[m]` in the code
    #θm, θp, rm, rp, j, θs[m + 1], n, s = θs[m], θs[m], r0, r0, 0, θs[m], 1, 1
    ηm, ηp, rm, rp, j, η_new, n, s = η, η, r0, r0, 0, η, 1, 1
    α, n_α = NaN, NaN

    η_new = η
    η_new = η
    while s == 1 && j < max_tree_depth
        v = rand([-1, 1])
        if v == -1
            ηm, rm, _, _, η′, n′, s′, α, n_α = build_tree(rng, ηm, rm, u, v, j, ϵ, η, r0)
        else
            _, _, ηp, rp, η′, n′, s′, α, n_α = build_tree(rng, ηp, rp, u, v, j, ϵ, η, r0)
        end
        if s′ == 1
            if rand(rng, Uniform()) < min(1.0, n′ / n)
                η_new = η′
            end
        end
        n = n + n′
        s = s′ & (dot(ηp - ηm, rm) >= 0) & (dot(ηp - ηm, rp) >= 0)
        j = j + 1
    end

    if m <= Madapt
    # NOTE: H_bar goes to negative when δ - α / n_α < 0
        H_bar = (1.0 - 1.0 / (m + t_0)) * H_bar + 1.0 / (m + t_0) * (δ - α / n_α)
        ϵ = exp(μ - sqrt(m) / γ * H_bar)
        ϵ_bar = exp(m^float(-κ) * log(ϵ) + (1.0 - m^float(-κ)) * log(ϵ_bar))
    else
        ϵ = ϵ_bar
    end
    #if verbose println("[NUTS] sampling complete with final apated ϵ = $ϵ") end

    return η_new, ϵ, ϵ_bar, H_bar
end