module BandwidthSelectors

import Distributions: Normal, Uniform, pdf, cdf, cf, ContinuousUnivariateDistribution
import Interpolations: MonotonicInterpolationType, LinearMonotonicInterpolation, interpolate, extrapolate, Flat, Throw
import FFTW: rfft, irfft, fft, ifft, dct, idct
import Optim: optimize, GoldenSection, minimizer, summary
import Statistics: mean, quantile, std
import StatsAPI: fit

import KernelDensity: UnivariateKDE, InterpKDE, AbstractKDE
import KernelDensity
export UnivariateKDE, InterpKDE

abstract type AbstractBandwidthSelector end

"""
    bandwidth(x::AbstractVector{<:Real}, method::FixedBandwidthSelector, midpoints::R)
    bandwidth(x::AbstractVector{<:Real}, method::FixedBandwidthSelector; npoints::Int=2048)

Compute the optimal bandwidth for a dataset according to a specified rule for selecting a fixed bandwidth.

```julia-repl
julia> x = randn(10^3);

julia> h = bandwidth(x, Silverman());

julia> k = kde(x, bandwidth = h);

julia> k == fit(UnivariateKDE, x, Silverman())
true
```
"""
function bandwidth end

include(joinpath("fixed", "FixedBandwidthSelector.jl"))
include(joinpath("fixed", "Silverman.jl"))
include(joinpath("fixed", "SJ.jl"))
include(joinpath("fixed", "ISJ.jl"))


abstract type AdaptiveBandwidthSelector <: AbstractBandwidthSelector end

include(joinpath("adaptive", "SSVKernel.jl"))
include(joinpath("adaptive", "DiffusionKernel.jl"))

export fit, bandwidth, SSVKernel, SJ, DiffusionKernel, ISJ, Silverman

include("cdf_and_quantile.jl")
export cdf, quantile, pdf
export InterpKDECDF, InterpKDEQF

end # end module