"""
    VARModel

Struct holding var model parameters.

A model object can be constructed via `model = VARModel(z, F, K, J, M, Tsubp)`.

# Fields
* `z`: Latent transformed data stored in array format
* `F`: Design matrix of dimension (T-p)×J 
* `F_sq`: Designed matrix with all elements squared, e.g. `F_sq = F .^2`
* `K`: Dimension of response variable.
* `J`: Number of predictors per response variable.
* `M`: Number of columns in the G matrix used to parameterize the covariance
* `Tsubp`: Number of observations minus the orer of the var model.
"""
struct VARModel{A<:AbstractVector{<:Real}, B<:AbstractMatrix{<:Real}}
    z::A
    F::B
    F_sq::B
    K::Int
    J::Int
    M::Int
    Tsubp::Int
    a_γ::Float64
    b_γ::Float64
    df_ξ::Float64
    #α::Float64

    function VARModel(z::A, F::B, K::Int, J::Int, M::Int, Tsubp::Int; a_γ::Float64 = 3.0, b_γ::Float64 = 1.0, df_ξ::Float64 = 1.0) where {A<:AbstractVector{<:Real}, B<:AbstractMatrix{<:Real}}
        # Check that dimensions are in order:
        length(z) != K*Tsubp && throw(ArgumentError("Length of z does not equal K*Tsubp."))
        size(F, 1) != Tsubp && throw(ArgumentError("First dimension of F does not equal Tsubp."))
        size(F, 2) != J && throw(ArgumentError("Second dimension of F does not equal J."))

        F_sq = F .^2
        #a_γ = 3.0
        #b_γ = 1.0
        #df = 4
        return new{A, B}(z, F, F_sq, K, J, M, Tsubp, a_γ, b_γ, df_ξ)
    end
end

# Wrap functions for computing full logdensity and gradient in the LogDensityProblems API
function LogDensityProblems.logdensity(model::VARModel, θ::AbstractVector)
    return logp_joint(model, θ)
end

function LogDensityProblems.dimension(model::VARModel)
    K = model.K
    J = model.J
    M = model.M
    M_γ = K*M - div(M*(M+1), 2) + M
    return 2*K*J+K + M_γ
end
LogDensityProblems.capabilities(::Type{<:VARModel}) = LogDensityProblems.LogDensityOrder{1}() # Indicates that handwritten derivatives are available

#= function LogDensityProblems.logdensity_and_gradient(model::Pigeons.BufferedAD{VARModel}, θ::AbstractVector)
    logp, grad_θ = logp_and_grad_joint(model, θ)
    return logp, grad_θ
end =#

#Pigeons.initialization(model::VARModel, ::Random.AbstractRNG, ::Int) = zeros(LogDensityProblems.dimension(model))


"""
    get_varsymbols(model::VARModel)

Function to get the vector of variable names (as symbols) from a VARModel object. Each entry corresponds to the variable of the same index in θ.
"""
function get_varsymbols(model::VARModel)
    K = model.K
    J = model.J
    M = model.M
    M_γ = K*M - div(M*(M+1), 2) + M

    varnames = Vector{Symbol}(undef, 2*K*J+K + M_γ)
    K = model.K
    J = model.J
    M = model.M
    M_γ = K*M - div(M*(M+1), 2) + M

    ind0 = 0
    ind1 = K*J
    for i in ind0+1:ind1
        varnames[i] = Symbol("β[$(i-ind0)]")
    end
    ind0 = ind1
    ind1 = 2*K*J
    for i in ind0+1:2*K*J
        varnames[i] = Symbol("log_ξ[$(i-ind0)]")
    end
    ind0 = ind1
    ind1 = 2*K*J+K
    for i in ind0 + 1:ind1
        varnames[i] = Symbol("log_τ[$(i-ind0)]")
    end
    ind0 = ind1
    ind1 = 2*K*J+K + M_γ
    for i in ind0 + 1:ind1
        varnames[i] = Symbol("γ[$(i-ind0)]")
    end
    return varnames
end

"""
    get_varsymbols_lkj(model::VARModel)

Function to get the vector of variable names (as symbols) from a VARModel object. Each entry corresponds to the variable of the same index in θ.
"""
function get_varsymbols_lkj(model::VARModel)
    K = model.K
    J = model.J
    M = model.M

    varnames = Vector{Symbol}(undef, 2*K*J+K + div(K*(K-1), 2))
    K = model.K
    J = model.J

    ind0 = 0
    ind1 = K*J
    for i in ind0+1:ind1
        varnames[i] = Symbol("β[$(i-ind0)]")
    end
    ind0 = ind1
    ind1 = 2*K*J
    for i in ind0+1:2*K*J
        varnames[i] = Symbol("log_ξ[$(i-ind0)]")
    end
    ind0 = ind1
    ind1 = 2*K*J+K
    for i in ind0 + 1:ind1
        varnames[i] = Symbol("log_τ[$(i-ind0)]")
    end
    ind0 = ind1
    ind1 = 2*K*J+K + div(K*(K-1), 2)
    for i in ind0 + 1:ind1
        varnames[i] = Symbol("γ[$(i-ind0)]")
    end
    return varnames
end