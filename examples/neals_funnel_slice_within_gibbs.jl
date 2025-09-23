using AbstractMCMC, SliceSampling, Distributions, LogDensityProblems, StatsPlots, Random, MCMCChains, Distributions, LinearAlgebra

#slice_sampler   = GibbsPolarSlice(2.0, max_proposals = 1000000)
#slice_sampler   = RandPermGibbs(SliceSteppingOut(0.5, max_proposals=10^6))
slice_sampler   = RandPermGibbs(SliceDoublingOut(0.5))
n_samples       = 10000

struct NealsFunnel
    x::Vector{Float64}
    dim::Int
end

dim = 10
df = 10.0

#= function logdensity(x, y)
    return sum(-0.5 .* y .- 0.5*(x.^2) ./ exp.(y) .- (y.^2) ./ 18.0)
end =#
function logdensity(x, y)
    return sum(-0.5 .* y .- 0.5*(x.^2) ./ exp.(y)) - 0.5*(df+1.0)*sum(log.(1.0 .+ y.^2/df))
end
LogDensityProblems.logdensity(model::NealsFunnel, y) = logdensity(model.x, y)
LogDensityProblems.dimension(model::NealsFunnel) = dim
LogDensityProblems.capabilities(::Type{<:NealsFunnel}) = LogDensityProblems.LogDensityOrder{0}()

# Let us generate one sample
# Initialize sampler
n_samples = 10^5
rng = Random.Xoshiro(1)
samples = Matrix{Float64}(undef, 2*dim, n_samples)

x_init = rand(rng, Normal(), dim)
y_init = rand(rng, Normal(), dim)

x = rand(rng, MvNormal(zeros(dim), Diagonal(exp.(y_init))))

# Initialize the state of the slice sampler
transition, state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(NealsFunnel(x, dim)), slice_sampler; initial_params=y_init)
y = transition.params
samples[:,1] = vcat(x, y)

for i in 2:n_samples
    # Sample x using a Gibbs step:
    x = rand(rng, MvNormal(zeros(dim), Diagonal(exp.(y))))

    # Update logdensity value in the GibbsState (requires creating a new GibbsState as the struct is immutable)
    Cond = AbstractMCMC.LogDensityModel(NealsFunnel(x, dim))
    state = AbstractMCMC.setparams!!(Cond, state, y)

    # Sample y using a slice sampler:
    transition, state = AbstractMCMC.step(rng, Cond, slice_sampler, state)
    y = transition.params

    # Record samples
    samples[:,i] = vcat(x, y)
end

p = plot()
for i in 11:20
    density!(p, samples[i,:], label="")
end
xlims!(p, -20.0, 20.0)
plot!(p, LinRange(-20, 20, 1001), TDist(df), color=:black)
p


chn = Chains([samples[:,i] for i in axes(samples, 2)])
describe(chn)

scatter(chn[:param_3], chn[:param_13])
scatter(chn[:param_2], chn[:param_4])

# So our sampler works fine. Lets now embed it within a Gibbs sampler