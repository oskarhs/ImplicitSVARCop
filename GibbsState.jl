# Need to write a macro. But what should it do?
# It is important that the user is consistent with naming of variables. Otherwise, we have no clue as to which input to put where when creating new structs.
# User also has to link each variable to its respective conditional distribution.

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

struct GibbsState{S}
    vi::V
    states::S
end

# Use getparams here
AbstractMCMC.getparams(state::GibbsState) = state.params
AbstractMCMC.getparams(::AbstractMCMC.AbstractModel, state::GibbsState) = FullConditionalState.params

# Use setparams!! here
AbstractMCMC.setparams!!(state::FullConditionalState, params) = FullConditionalState(params)
AbstractMCMC.setparams!!(::AbstractMCMC.AbstractModel, state::FullConditionalState, params) = FullConditionalState(params)