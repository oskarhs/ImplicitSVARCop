# BandwidthSelectors.jl Documentation

BandwidthSelectors.jl is a Julia package for automatic bandwidth selection in univariate kernel density estimation.

## Introduction

Kernel density estimators (KDEs) are a much-used class of density estimators. Contrary to parametric density estimators, KDEs rely on less stringent assumptions on the data-generating density, making them a robust and popular tool for density estimation and data visualization, particularly in low-dimensional applications.

Although the greater flexibility of KDEs is an appealing property, the quality of the resulting density estimates is highly sensitive to the value of the bandwidth, which in practice has to be chosen based on the observed sample. This issue has lead to the development of many different proposals for selecting in a data-based manner. The purpose of BandwidthSelectors.jl is to provide easy access to state-of-the-art bandwidth selection algorithms. In addition to providing smoothing parameter selection for classical fixed-bandwidth kernel estimators, we also provide an implementation of several adaptive procedures, where the bandwidth is allowed to vary together with the input point.

## Quick start
In the following, we show how BandwidthSelectors can be used to construct a KDE using the Sheather-Jones method to select the bandwidth. To fit a kernel estimate to univariate data, we use the [`fit`](@ref) method, which follows the syntax `fit(UnivariateKDE, data, method)`, as shown in the following code snippet:
```julia
using BandwidthSelectors
x = randn(10^5)
k = fit(UnivariateKDE, x, SJ())  # fit a KDE to x using the Sheather-Jones method
```
BandwidthSelectors.jl is built on top of [KernelDensity.jl](https://juliastats.org/KernelDensity.jl/stable/), and the call to fit will return an object of type `UnivariateKDE`, as is returned by the default `kde` method provided in that package. `UnivariateKDE` objects provide a compact summary of the kernel estimate by storing the values of the density estimate evaluated on a regular grid. The evaluation grid can be accessed through `k.x` and the corresponding density values through `k.density`, which allows one to quickly plot the resulting density estimates, for instance using Plots.jl:
```julia
import Plots
Plots.plot(k.x, k.density)
```
To compute the kernel estimate or the corresponding cdf or quantile function at an arbitrary points, see WIP.

For fixed-bandwidth algorithms such as the Sheather-Jones method, the chosen bandwidth itself may also be of interest. The syntax for computing the bandwidth of a given method is `bandwidth(data, method)`, as shown below:
```julia
bw = bandwidth(x, SJ())
```