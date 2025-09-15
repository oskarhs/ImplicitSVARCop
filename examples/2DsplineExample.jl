using BSplineKit, Random, Distributions, MCMCChains, AdvancedHMC
import LogDensityProblems

# Example with bivariate response data
n = 10_000

rng = Random.Xoshiro(1)

# Simulate covariate data
x1 = rand(rng, n) # nonlinear effect
x2 = rand(rng, n) # nonlinear effect
x3 = rand(rng, n) # linear effect

b = BSplineBasis(BSplineOrder(4), LinRange(0, 1, 11))

b1 = b.(x1)
b2 = b.(x2)

b1_matrix = zeros(Float64, n, length(b))
b2_matrix = zeros(Float64, n, length(b))

# Get B-spline basis matrix
for i in eachindex(b1)
    j1, val1 = b1[i]
    ind1 = j1:-1:j1-3
    b1_matrix[i, ind1] .= val1

    j2, val2 = b2[i]
    ind2 = j2:-1:j2-3
    b2_matrix[i, ind2] .= val2
end

# Correlation matrix for latent responses
Σ = Float64[1.0 0.8; 0.8 1.0]

# Create latent responses
Mz = Matrix{Float64}(undef, n, 2)
mean_Mz1 = mean(exp.(sin.(x1)) - x2.^3 - 0.5*x3) # technically we could replace these with analytical expressions
mean_Mz2 = mean(log.(x1 .+ 0.5) + cos.(2.0*x2) + 1.5 * x3)
for i in 1:n
    Mz[i, 1] = exp(sin(x1[i])) - x2[i]^3 - 0.5 * x3[i] - mean_Mz1           # demean variables
    Mz[i, 2] = log(x1[i]+0.5) + cos(2.0*x2[i]) + 1.5 * x3[i] - mean_Mz2     # demean variables
    Mz[i, :] .+= rand(MvNormal([0.0, 0.0], Σ))
end

# Transform data
d_true = Gamma(3.0, 3.0)
My = quantile.(d_true, cdf.(Normal(), Mz))

# Fit models for the marginals:
kdests = [fit(UnivariateKDE, My[:, 1], SSVKernel())]
for k in 2:size(My, 2)
    push!(kdests, fit(UnivariateKDE, My[:, k], SSVKernel()))
end

# Transform data to latent scale through the estimate
Mz_est = similar(Mz)
for k in axes(My, 2)
    Mz_est[:, k] = quantile.(Normal(), cdf.(kdests[k], My[:, k]))
end
z_est = vec(Mz_est)

# Design matrix
F = hcat(b1_matrix, b2_matrix, x3)

J = size(F, 2)
K = 2
M = 1 # NB! factor prior is overparameterized in this case
Tsubp = n

model = VARModel(z_est, F, K, J, M, Tsubp)

posterior, ELBOs = fitBayesianSVARCopVI(rng, model, 3000, 10)

D = LogDensityProblems.dimension(model)

# Number of samples and adaptation steps
# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 1_000, 500

# Define a Hamiltonian system
#metric = DiagEuclideanMetric(D)
metric = DenseEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, θ -> logp_joint(model, θ), θ -> logp_and_grad_joint(model, θ))

initial_θ = rand(rng, Normal(), D)    

# Define a leapfrog solver, with the initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)

# Define NUTS sampler
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(max_depth=8)))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.7, integrator))

# Run the sampler
@time samples, stats = sample(
    hamiltonian, kernel, initial_θ, n_samples, adaptor, n_adapts; progress=true, (pm_next!) = AdvancedHMC.simple_pm_next!
)

chain = MCMCChains.Chains(samples[n_adapts+1:end], get_varsymbols(model))
describe(chain)

plot(chain[Symbol("log_ξ[6]")])

plot(autocor(chain[Symbol("β[18]")]))

# Plot basis expansions
β_matrix = Matrix{Float64}(undef, n_samples, 54)
for k in 1:n_samples
    β_matrix[k,:] = samples[k][1:54]
end
β_hat = mean(β_matrix, dims=1)
S = Spline(b, β_hat[1:13])

t = LinRange(0.0, 1.0, 1001)
p = plot(t, S.(t))
plot!(p, t, exp.(sin.(t)).-1.0)


i = 5

marginalscatter(chain[Symbol("β[$i]")], chain[Symbol("log_ξ[$i]")], xlabel="β[$i]", ylabel="log_ξ[$i]")

plot(MCMCChains.Chains(samples[1:100], get_varsymbols(model)))


plot(chain[Symbol("log_ξ[$i]")])


density(vec(chain[Symbol("log_ξ[9]")].data))