"""
    fitBayesianSVARCopVI_lkj([rng=Random.default_rng()], model::VARModel, n_iter::Int, N_fac::Int; progress=true)

    Fit the BayesianSVAR model to a VARModel object using variational inference.

# Arguments
* `model`: VARModel object.
* `n_iter`: Number of iterations
* `N_fac`: First dimension of B in the factor covariance matrix expression BB' + Δ²

# Keyword Arguments
* `progress`: If set to true, a simple progressbar is shown.
"""
function fitBayesianSVARCopVI_lkj(rng::Random.AbstractRNG, model::VARModel, n_iter::Int, N_fac::Int; progress=true)
    J = model.J
    K = model.K
    D = 2*K*J + K + 1

    # Initialize variational parameters:
    μ = zeros(Float64, D)
    d = fill(1e-1, D)
    Bfac = fill(1e-3, (D, N_fac))
    tril!(Bfac, 0) # sets upper triangle (not including diagonal) of Bfac to 0.

    # Initialize ADADELTA state
    Eδ2_μ = zeros(Float64, length(μ))
    Eg2_μ = zeros(Float64, length(μ))

    Eδ2_B = zeros(Float64, size(Bfac))
    Eg2_B = zeros(Float64, size(Bfac))

    Eδ2_d = zeros(Float64, length(d))
    Eg2_d = zeros(Float64, length(d))

    ρ = 0.95
    ϵ = 1e-6

    ADA = ADADELTAState(ρ, ϵ, Eδ2_μ, Eg2_μ, Eδ2_B, Eg2_B, Eδ2_d, Eg2_d)

    # Compute inverse of factor matrix
    d2 = 1.0 ./ d .^2
    Dinv2B = Bfac .* d2
    Siginvpart = inv(I(N_fac) + transpose(Bfac) * Dinv2B)

    # Store values of lower bound
    ELBOs = Vector{Float64}(undef, n_iter)
    μ_c = zeros(Float64, length(μ))
    Bfac_c = zeros(Float64, size(Bfac))
    d_c = zeros(Float64, length(d))
    n_avg = 0

    # Set up progressmeter
    pm = progress ? Progress(n_iter; desc="Optimizing ELBO", barlen=31) : nothing

    # Perform SGA iterations
    for it in 1:n_iter
        ELBOs[it], μ, Bfac, d, ADA, Siginvpart, L_μ, L_B, L_d = adadelta_step_lkj(rng, model, μ, Bfac, d, N_fac, Siginvpart, ADA)
        if !isnothing(pm)
            next!(pm; showvalues=[("ELBO", ELBOs[it])])
        end
        # Parameter averaging strategy
        if it ≥ 0.9 * n_iter
            μ_c += μ
            Bfac_c += Bfac
            d_c += d
            n_avg += 1
        end
    end
    μ_c = μ_c / n_avg
    Bfac_c = Bfac_c / n_avg
    d_c = d_c / n_avg
    
    return VIPosterior_lkj(μ_c, Bfac_c, d_c, J, K), ELBOs
end

# Fallback version in case no seed was explicitly provided
fitBayesianSVARCopVI(model::VARModel, n_iter::Int, N_fac::Int) = fitBayesianSVARCopVI(Random.default_rng(), model, n_iter, N_fac)