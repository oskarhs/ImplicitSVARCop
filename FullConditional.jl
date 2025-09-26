using Random, Distributions, AbstractMCMC, LogDensityProblems, LinearAlgebra, MacroTools

"""
    FullConditional <: AbstractMCMC.AbstractSampler

Sample exactly from the target log conditional density.

This sampler expects the user to implement a `rand(rng, problem)` for the target logdensity to generate exact samples from the full conditional distribution.

# Examples
```julia
struct MyIsotropicNormal
    dim::Int64
end
LogDensityProblems.dimension(model::MyIsotropicNormal) = model.dim

# This has to be implemented for the LogDensityProblems interface, use [`@fullconditional`](@ref) to skip the implementation of these methods.
LogDensityProblems.logdensity(::MyIsotropicNormal, x) = 0.0                                    # Since we don't use this for sampling, we can set an arbitrary value here
LogDensityProblems.capabilities(::Type{MyIsotropicNormal}) = LogDensityProblems.LogDensityOrder{0}() # Set order to 0 so we don't have to implement derivatives too.

# Implement rand method:
Base.rand(rng::Random.AbstractRNG, model::MyIsotropicNormal) = rand(rng, MvNormal(zeros(model.dim), Diagonal(ones(model.dim))))

# Wrap model in a LogDensityModel and sample:
model = AbstractMCMC.LogDensityModel(MyIsotropicNormal(2))
samples = AbstractMCMC.sample(Random.default_rng(), model, FullConditional(), 10^3; init_params=zeros(2))
```
"""
struct FullConditional <: AbstractMCMC.AbstractSampler end

struct FullConditionalState{T}
    params::T
end

AbstractMCMC.getparams(state::FullConditionalState) = state.params
AbstractMCMC.getparams(::AbstractMCMC.AbstractModel, state::FullConditionalState) = state.params

AbstractMCMC.setparams!!(state::FullConditionalState, params) = FullConditionalState(params)
AbstractMCMC.setparams!!(::AbstractMCMC.AbstractModel, state::FullConditionalState, params) = FullConditionalState(params)

function AbstractMCMC.step(rng::Random.AbstractRNG, model::AbstractMCMC.AbstractModel, fullcond::FullConditional; kwargs...)
    # Just call rand(rng, model.logdensity) and call it a day
    newparams = Base.rand(rng, model.logdensity)
    return newparams, FullConditionalState(newparams)
end
AbstractMCMC.step(rng::Random.AbstractRNG, model::AbstractMCMC.AbstractModel, fullcond::FullConditional, state::FullConditionalState; kwargs...) = AbstractMCMC.step(rng, model, fullcond; kwargs...)
#= struct MyIsotropicNormal
    dim::Int64
end
LogDensityProblems.logdensity(::MyIsotropicNormal, x) = 0.0
LogDensityProblems.dimension(model::MyIsotropicNormal) = model.dim
LogDensityProblems.capabilities(::Type{MyIsotropicNormal}) = LogDensityProblems.LogDensityOrder{0}()

model = AbstractMCMC.LogDensityModel(MyIsotropicNormal(2))

samples = AbstractMCMC.sample(Random.default_rng(), model, FullConditional(), 100000; init_params=zeros(2)); =#



function _fullconditionalize(ex)
    dict = MacroTools.splitstructdef(ex)
    struct_name = dict[:name]

    func1 = quote 
        LogDensityProblems.logdensity(::$struct_name, x) = 0.0
    end
    func2 = quote
        LogDensityProblems.capabilities(::Type{$struct_name}) = LogDensityProblems.LogDensityOrder{0}()
    end
    return func1, func2
end

"""
    @fullconditional

Convenience macro to make a conditional distribution object conform to the LogDensityProblems API.

Since `LogDensityProblems.logdensity` and `LogDensityProblems.capabilities` do not have to be implemented when sampling from the exact full conditional densities,
this macro can be used to automatically implement these methods.
In particular, using this macro is equivalent to manually setting `LogDensityProblems.logdensity(::MyIsotropicNormal, x) = 0.0` and
`LogDensityProblems.capabilities(::Type{MyIsotropicNormal}) = LogDensityProblems.LogDensityOrder{0}()`. This macro should only be used when the sampler is 

# Example
```julia
@fullconditional struct MyIsotropicNormal
    dim::Int64
end
LogDensityProblems.dimension(model::MyIsotropicNormal) = model.dim

# Implement rand method:
Base.rand(rng::Random.AbstractRNG, model::MyIsotropicNormal) = rand(rng, MvNormal(zeros(model.dim), Diagonal(ones(model.dim))))

# Wrap model in a LogDensityModel and sample:
model = AbstractMCMC.LogDensityModel(MyIsotropicNormal(2)) # note that this would have thrown an error if we did not use the macro
samples = AbstractMCMC.sample(Random.default_rng(), model, FullConditional(), 10^3; init_params=zeros(2))
```
"""
macro fullconditional(ex)
    func1, func2 = _fullconditionalize(ex)
    return esc(Expr(:block, ex, func1, func2))
end

#= @fullconditional struct MyNewIsotropicNormal
    dim::Int64
end
LogDensityProblems.dimension(model::MyNewIsotropicNormal) = model.dim
Base.rand(rng::Random.AbstractRNG, model::MyNewIsotropicNormal) = rand(rng, MvNormal(zeros(model.dim), Diagonal(ones(model.dim))))
model = AbstractMCMC.LogDensityModel(MyNewIsotropicNormal(3))

samples = AbstractMCMC.sample(Random.default_rng(), model, FullConditional(), 100000; init_params=zeros(2));

x1 = [samples[i][1] for i in eachindex(samples)]
density(x1) =#