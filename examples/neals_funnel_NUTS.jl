using AbstractMCMC, AdvancedHMC, Distributions, LogDensityProblems, StatsPlots, Random, MCMCChains, Distributions, LinearAlgebra, ForwardDiff

δ       = 0.8                  # target acceptance rate
sampler = AdvancedHMC.NUTS(δ)  # define a NUTS sampler with target acceptance rate δ

struct NealsFunnel
    dim::Int
end

dim = 10

function logdensity(x)
    return sum(-0.5 .* x[dim+1:end] .- 0.5*(x[1:dim].^2) ./ exp.(x[dim+1:end]) .- (x[dim+1:end].^2) ./ 18.0)
end
LogDensityProblems.capabilities(::Type{<:NealsFunnel}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.logdensity(model::NealsFunnel, x) = logdensity(x)
LogDensityProblems.dimension(model::NealsFunnel) = 2*dim

# Initialize sampler
n_samples = 50_000
n_adapts  = 1_000
rng       = Random.Xoshiro(1)                         # Set seed
samples   = Matrix{Float64}(undef, 2*dim, n_samples)  # Matrix to store samples

x_init = randn(rng, 2*dim)

Target = NealsFunnel(dim)
metric = DiagEuclideanMetric(2*dim)
hamiltonian = Hamiltonian(metric, Target)

# Define a leapfrog solver, with the initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, x_init)
integrator = Leapfrog(initial_ϵ)

kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
samples, stats = sample(
    hamiltonian, kernel, x_init, n_samples, adaptor, n_adapts; progress=true, (pm_next!) = AdvancedHMC.simple_pm_next!
)

chain_nuts = Chains(samples)
plot(chain_nuts[1001:end])
describe(chain_nuts)

scatter(chain_nuts[:param_3], chain_nuts[:param_13])


samples1 = hcat(samples...)

p = plot()
for i in 11:20
    density!(p, samples1[i,:], label="")
end
p