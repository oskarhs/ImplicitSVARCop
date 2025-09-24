# THIS IS NOT NEEDED SO LONG AS THE FACTOR VI PRIOR IS INCLUDED
#= """
    ADADELTAState

Struct storing the current state of the ADADELTA algorithm.
"""
struct ADADELTAState
    ρ::Float64 
    ϵ::Float64
    Eδ2_μ::Vector{Float64}
    Eg2_μ::Vector{Float64}
    Eδ2_B::Matrix{Float64}
    Eg2_B::Matrix{Float64}
    Eδ2_d::Vector{Float64}
    Eg2_d::Vector{Float64}
end =#


"""
    adadelta_step(
        rng::Random.AbstractRNG,
        model::VARModel, 
        μ::AbstractVector{<:Real}, 
        Bfac::AbstractMatrix{<:Real},
        d::AbstractVector{<:Real}, 
        N_fac::Int,
        Siginvpart::AbstractMatrix{<:Real},
        ADA::ADADELTAState
    )

Function for executing one step of the adadelta algorithm.

# Arguments
* `rng`: Seed used for random variate generation
* `model`: VARModel object
* `μ`: Current value of mean vector in VI parameterization
* `Bfac`: Current value of the factor matrix in the covariance BB' + Δ²
* `d`: Vector of diagonal entries of Δ in the expression BB' + Δ²
* `N_fac`: First dimension of the matrix B
* `Siginvpart`: Current value of the inverse covariance matrix inv(BB' + Δ²)
* `ADA`: Object holding the updated state of the ADADELTA optimizer.
"""
function adadelta_step_lkj(
    rng::Random.AbstractRNG,
    model::VARModel,
    μ::AbstractVector{<:Real}, 
    Bfac::AbstractMatrix{<:Real},
    d::AbstractVector{<:Real}, 
    N_fac::Int,
    Siginvpart::AbstractMatrix{<:Real},
    ADA::ADADELTAState
)
    D = length(μ)
    ρ = ADA.ρ
    ϵ = ADA.ϵ

    oldEδ2_μ = ADA.Eδ2_μ
    oldEg2_μ = ADA.Eg2_μ

    oldEδ2_B = ADA.Eδ2_B
    oldEg2_B = ADA.Eg2_B

    oldEδ2_d = ADA.Eδ2_d
    oldEg2_d = ADA.Eg2_d

    w1 = Base.rand(rng, Normal(), N_fac)
    w2 = Base.rand(rng, Normal(), D)
    
    θ = μ + Bfac * w1 + d .* w2

    L_μ, L_B, L_d, logp = grad_and_logp_elbo_lkj(model, θ, Bfac, d, w1, w2, Siginvpart) # get gradients and logdensity

    # Set upper triangle of L_B (not including diagonal) to 0:
    tril!(L_B, 0)

    # update μ
    newEg2_μ = ρ*oldEg2_μ + (1.0 - ρ)*L_μ .^2
    change_δ_μ = @. sqrt(oldEδ2_μ + ϵ) / sqrt(newEg2_μ + ϵ)*L_μ
    μ = μ + change_δ_μ
    newEδ2_μ = ρ * oldEδ2_μ + (1.0 - ρ) * change_δ_μ .^2

    # update B
    newEg2_B = ρ * oldEg2_B + (1.0 - ρ) * L_B .^2
    change_δ_B = @. sqrt(oldEδ2_B + ϵ)/sqrt(newEg2_B + ϵ)*L_B
    Bfac = Bfac + change_δ_B # How do we ensure that Bfac remains lower triangular here?
    newEδ2_B = ρ * oldEδ2_B + (1.0 - ρ) * change_δ_B .^2

    # update d
    newEg2_d = ρ*oldEg2_d + (1.0 - ρ)*L_d .^2
    change_δ_d = @. sqrt(oldEδ2_d + ϵ)/sqrt(newEg2_d + ϵ)*L_d
    d = d + change_δ_d
    newEδ2_d = ρ * oldEδ2_d + (1.0 - ρ) * change_δ_d .^2

    d[abs.(d) .<= 1e-4] .= 1e-4

    # Update ADADELTA state:
    ADA = ADADELTAState(ρ, ϵ, newEδ2_μ, newEg2_μ, newEδ2_B, newEg2_B, newEδ2_d, newEg2_d)

    Bz_deps = Bfac * w1 + d .* w2

    d2 = 1.0 ./ d .^2
    Dinv2B = Bfac .* d2
    DBz_deps = Bz_deps .* d2
    Siginvpart = inv(I(N_fac) + transpose(Bfac) * Dinv2B)

    Blogdet = logdet(I(N_fac) + transpose(Dinv2B) * Bfac) + vsum(x -> log(abs2(x)), d)
    Half2 = Dinv2B * (Siginvpart * (transpose(Bfac) * DBz_deps))
    quadform = transpose(Bz_deps) * (DBz_deps - Half2)
    LowerB = logp + 0.5 * D * log(2.0*pi) + 0.5*Blogdet + 0.5 * quadform

    return LowerB, μ, Bfac, d, ADA, Siginvpart, L_μ, L_B, L_d
end