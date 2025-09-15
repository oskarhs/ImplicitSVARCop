# Helper struct for running covariance estimation
mutable struct RunningCovariance
    mean::Vector{Float64}
    M2::Matrix{Float64}
    n::Int
end

function RunningCovariance(dims::Int)
    RunningCovariance(zeros(dims), zeros(dims, dims), 0)
end

function update!(rc::RunningCovariance, x::Vector{Float64})
    rc.n += 1
    delta = x .- rc.mean
    rc.mean .+= delta ./ rc.n
    delta2 = x .- rc.mean
    rc.M2 .+= delta * delta2'
end

function covariance(rc::RunningCovariance)
    rc.n > 1 ? rc.M2 ./ (rc.n - 1) : I * 1e-3  # return jittered I if too few samples
end

# Your nuts_step with mass matrix adaptation
function nuts_step2(rng::Random.AbstractRNG, η::AbstractVector, δ, logp_cond::Function, grad_logp_cond::Function,
                  ϵ, ϵ_bar, ϵ0, H_bar, m, Madapt,
                  metric::AbstractMatrix{Float64}, inv_metric::AbstractMatrix{Float64},
                  running_cov::RunningCovariance;
                  max_tree_depth=5)

    # Leapfrog step using inverse mass matrix
    function leapfrog(η, r, vϵ)
        # Half step momentum
        r_new = r .+ 0.5 .* vϵ .* grad_logp_cond(η)
        # Full step position
        η_new = η .+ vϵ .* (inv_metric * r_new)
        # Half step momentum
        r_new .+= 0.5 .* vϵ .* grad_logp_cond(η_new)
        return η_new, r_new
    end

    function build_tree(rng, η, r, u, v, j, ϵ, η0, r0)
        Δ_max = 1000.0
        if j == 0
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
            ham = logp_cond(η′) - 0.5 * dot(r′, inv_metric, r′)
            n′ = u <= exp(ham)
            s′ = ham > (log(u) - Δ_max)
            α′ = min(1.0, exp(ham - (logp_cond(η0) - 0.5 * dot(r0, inv_metric, r0))))
            return η′, r′, η′, r′, η′, n′, s′, α′, 1
        else
            ηm, rm, ηp, rp, η′, n′, s′, α′, n′_α = build_tree(rng, η, r, u, v, j-1, ϵ, η0, r0)
            if s′ == 1
                if v == -1
                    ηm, rm, _, _, η′′, n′′, s′′, α′′, n′′_α = build_tree(rng, ηm, rm, u, v, j-1, ϵ, η0, r0)
                else
                    _, _, ηp, rp, η′′, n′′, s′′, α′′, n′′_α = build_tree(rng, ηp, rp, u, v, j-1, ϵ, η0, r0)
                end
                if rand(rng) < n′′ / (n′ + n′′)
                    η′ = η′′
                end
                α′ += α′′
                n′_α += n′′_α
                s′ = s′ && s′′ && (dot(ηp - ηm, rm) >= 0) && (dot(ηp - ηm, rp) >= 0)
                n′ += n′′
            end
            return ηm, rm, ηp, rp, η′, n′, s′, α′, n′_α
        end
    end

    # Sample momentum with current metric
    L = cholesky(metric).L
    z = randn(rng, length(η))
    r0 = L * z

    # Slice variable
    u = rand(rng) * exp(logp_cond(η) - 0.5 * dot(r0, inv_metric * r0))

    ηm, ηp, rm, rp, j, η_new, n, s = η, η, r0, r0, 0, η, 1, 1
    α, n_α = 0.0, 0.0

    while s == 1 && j < max_tree_depth
        v = rand(rng, [-1, 1])
        if v == -1
            ηm, rm, _, _, η′, n′, s′, α′, n′_α = build_tree(rng, ηm, rm, u, v, j, ϵ, η, r0)
        else
            _, _, ηp, rp, η′, n′, s′, α′, n′_α = build_tree(rng, ηp, rp, u, v, j, ϵ, η, r0)
        end
        if s′ == 1
            if rand(rng) < min(1.0, n′ / n)
                η_new = η′
            end
        end
        n += n′
        s = s′ && (dot(ηp - ηm, rm) >= 0) && (dot(ηp - ηm, rp) >= 0)
        α += α′
        n_α += n′_α
        j += 1
    end

    # Mass matrix adaptation via running covariance during warmup
    if m <= Madapt
        #update!(running_cov, η_new)
        # Update metric every 50 iterations for stability
        #new_metric = covariance(running_cov) + 1e-3 * I
        #new_inv_metric = inv(new_metric)
        # Update passed metric matrices in-place
        #metric = new_metric
        #inv_metric = new_inv_metric

        # Dual averaging for step size
        μ, γ, t_0, κ = log(10.0 * ϵ0), 0.05, 10, 0.75
        H_bar = (1.0 - 1.0 / (m + t_0)) * H_bar + 1.0 / (m + t_0) * (δ - α / n_α)
        ϵ = exp(μ - sqrt(m)/γ * H_bar)
        ϵ_bar = exp(m^(-κ) * log(ϵ) + (1.0 - m^(-κ)) * log(ϵ_bar))
    else
        ϵ = ϵ_bar
    end

    return η_new, ϵ, ϵ_bar, H_bar, metric, inv_metric
end
