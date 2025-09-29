# Need to write a macro. But what should it do?
# It is important that the user is consistent with naming of variables. Otherwise, we have no clue as to which input to put where when creating new structs.
# User also has to link each variable to its respective conditional distribution.


struct GibbsModel{N,V<:NTuple{N,AbstractVector{<:VarName}},L<:NTuple{N,Any}}
    varnames::V
    models::L
end

# Create a wrapper so that we can do something akin to @capture, @insert

struct SequentialGibbs{N,V<:NTuple{N,AbstractVector{<:VarName}},A<:NTuple{N,Any}} <: AbstractMCMC.AbstractSampler
    varnames::V
    samplers::A

    function Gibbs(varnames, samplers)
        if length(varnames) != length(samplers)
            throw(ArgumentError("Number of varnames and samplers must match."))
        end

        for spl in samplers # Replace with check that setparams!!, getparams and step have been implemented for all samplers.
            if !isgibbscomponent(spl)
                msg = "All samplers must be valid Gibbs components, $(spl) is not."
                throw(ArgumentError(msg))
            end
        end

        samplers = tuple(map(wrap_in_sampler, samplers)...)
        varnames = tuple(map(to_varname_list, varnames)...)
        return new{length(samplers),typeof(varnames),typeof(samplers)}(varnames, samplers)
    end
end

function SequentialGibbs(algs::Pair...)
    return SequentialGibbs(map(first, algs), map(last, algs))
end

# This has to store sampler states for each sampler
struct GibbsState{S}
    vi::V
    states::S # States of component samplers
end

# Use getparams here
AbstractMCMC.getparams(gibbs_state::GibbsState) = mapreduce(AbstractMCMC.getparams, vcat, gibbs_state.states)
AbstractMCMC.getparams(model::AbstractMCMC.AbstractModel, gibbs_state::GibbsState) = mapreduce(st -> AbstractMCMC.getparams(model, st), vcat, gibbs_state.states)

# Use setparams!! here
AbstractMCMC.setparams!!(gibbs_state::GibbsState, params) = mapreduce(AbstractMCMC.setparams!!, vcat, gibbs_state.states, params)

# Need some logic here mapping params to states
function AbstractMCMC.setparams!!(gibbs_state::GibbsState, params)
    for component_state in eachindex(gibbs_state.states)
        new_component_state = AbstractMCMC.setparams!!(gibbs_state.states[i], params)
    end
end # NB! This is not super important for now

AbstractMCMC.setparams!!(model::AbstractMCMC.AbstractModel, gibbs_state::GibbsState, params) = mapreduce(st -> AbstractMCMC.setparams!!(model, st), vcat, gibbs_state.states, params)

# Make a GibbsModel type 
function AbstractMCMC.step(rng::Random.AbstractRNG, model::GibbsModel, sampler::SequentialGibbs; kwargs...)
    # Just call rand(rng, model.logdensity) and call it a day
    newparams = []
    # Here: use a for loop, make new conditional model for each variable.
    # This last point is actually nontrivial. We know that model to use.
    # something like `gibbs_model.models[i](gibbs_model.signature[i])`
    newparams = Base.rand(rng, model.logdensity)
    return newparams, FullConditionalState(newparams)
end
function AbstractMCMC.step(rng::Random.AbstractRNG, model::GibbsModel, sampler::SequentialGibbs, state::FullConditionalState; kwargs...) 
    AbstractMCMC.step(rng, model, sampler; kwargs...)
end