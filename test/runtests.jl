using Test
using Jutul
using JutulDarcyAD
using Flux
using MultiComponentFlash
using Printf
using Random
using LinearAlgebra

Random.seed!(2023)

include("test_utils.jl")

include("test_gradient.jl")

include("test_conversion.jl")

include("test_jutulState.jl")
include("test_jutulForce.jl")
include("test_jutulSource.jl")
include("test_jutulModel.jl")
include("test_jutulModeling.jl")
