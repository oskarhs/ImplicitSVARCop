using AdvancedHMC, AbstractMCMC, SliceSampling, Distributions, LogDensityProblems, StatsPlots, Random, MCMCChains, Distributions, LinearAlgebra, ForwardDiff

"""
This script should work with any sampler that implements AbstractMCMC.setparams!! and AbstractMCMC.getparams in addition to AbstractMCMC.step.
AFAIK all samplers from the Turing ecosystem implement the latter, but not all implement the former.
For example, SliceSampling.GibbsPolarSlice does not currently implement AbstractMCMC.setparams!!, so it cannot be used to sample the conditional of y in this script.

The example shown off here is a variant of Neal's Funnel, where the number of "variance parameters" is equal to the number of "regression parameters". The model is as follows:
      y_i ~ Normal(0, 3^2),
x_i | y_i ~ Normal(0, exp(y_i)),
independently for i = 1,2, …, dim. In total, there are 2*dim random variables.

In this example, we take dim = 10.
"""

δ       = 0.8                  # target acceptance rate
sampler = AdvancedHMC.NUTS(δ)  # define a NUTS sampler with target acceptance rate δ

# Struct representing the target conditional distribution for the NUTS sampler.
struct NealsFunnel
    x::Vector{Float64}
    dim::Int
end

# Dimension of the funnel in our test example
dim = 10

# The target logdensity. Note that in this case, the full log-conditional cannot be simplified
function logdensity(x, y)
    return sum(-0.5 .* y .- 0.5*(x.^2) ./ exp.(y) .- (y.^2) ./ 18.0)
end

# The following methods starting with "LogDensityProblems" have to be implemented in order to use the samplers from the Turing ecosystem.
# This is the dimension of the conditional density p(y | x)
LogDensityProblems.dimension(model::NealsFunnel) = dim

# The following tells the sampler that we do not provided a "handwritten" gradient.
# When using gradient-based samplers, this means that the gradients will be computed using AutoDiff, which we do here in this example.
# LogDensityProblems.capabilities(::Type{<:NealsFunnel}) = LogDensityProblems.LogDensityOrder{0}()

# If you want to use a handwritten gradient, comment out the above and uncomment the lines below, which provide the gradient analytically
LogDensityProblems.capabilities(::Type{<:NealsFunnel}) = LogDensityProblems.LogDensityOrder{1}() # Indicate that we provide a gradient (derivative of order 1)
function LogDensityProblems.logdensity_and_gradient(model::NealsFunnel, y) # this should return a tuple, e.g. logdensity, grad
    logp = logdensity(x, y)
    grad = -0.5 .+ 0.5 * x.^2 .* exp.(-y) .- y / 9
    return logp, grad
end

# Wrap the target logdensity within the struct
LogDensityProblems.logdensity(model::NealsFunnel, y) = logdensity(model.x, y) # This is the conditional log p(y | x), up to an additive constant wrt y.

# Initialize sampler
n_samples = 50_000
n_adapts  = 1_000
rng       = Random.Xoshiro(1)                         # Set seed
samples   = Matrix{Float64}(undef, 2*dim, n_samples)  # Matrix to store samples

# Initialize variables:
x_init = rand(rng, Normal(), dim)
y_init = rand(rng, Normal(), dim)

# First iteration of Gibbs sampler. This is done outside the loop as we have to initialize the state of the NUTS sampler by calling a different version of AbstractMCMC.step
x = rand(rng, MvNormal(zeros(dim), Diagonal(exp.(y_init))))

# Hi
const MetaSliceSamplers = Union{
    SliceSampling.GibbsState,
    SliceSampling.HitAndRunState
}

AbstractMCMC.getparams(state::MetaSliceSamplers) = state.transition.params

# Initialize the state of the slice sampler
Cond = AbstractMCMC.LogDensityModel(NealsFunnel(x, dim))                          # Initialize a conditional density object
transition, state = AbstractMCMC.step(rng, Cond, sampler; initial_params=y_init, n_adapts=n_adapts)  # Carry out one iteration of the sampler in order to initialize the state
y = AbstractMCMC.getparams(state)
samples[:,1] = vcat(x, y)

for i in 2:n_samples
    # Sample x using a Gibbs step:
    x = rand(rng, MvNormal(zeros(dim), Diagonal(exp.(y))))

    # Update logdensity value in the GibbsState (requires creating a new GibbsState as the struct is immutable)
    Cond = AbstractMCMC.LogDensityModel(NealsFunnel(x, dim))

    # Update the state based on the new conditional distribution
    state = AbstractMCMC.setparams!!(Cond, state, y)

    # Sample y using the specified sampler:
    transition, state = AbstractMCMC.step(rng, Cond, sampler, state; n_adapts=n_adapts)

    # Retrieve the parameters from the updated state:
    y = AbstractMCMC.getparams(state)

    # Record samples
    samples[:,i] = vcat(x, y)
end

p = plot()
for i in 11:20
    density!(p, samples[i,:], label="")
end
#xlims!(p, -20.0, 20.0)
plot!(p, LinRange(-20, 20, 1001), Normal(0.0, 3.0), color=:black, label="True marginal density")
p

plot(chn)

chn = Chains([samples[:,i] for i in axes(samples, 2)])
describe(chn[1001:end])

scatter(chn[:param_1], chn[:param_11])

# So our sampler works fine. Lets now embed it within a Gibbs sampler