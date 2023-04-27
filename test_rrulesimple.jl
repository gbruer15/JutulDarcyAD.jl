using Test
using Jutul
using JutulDarcy
using JutulDarcyRules
using Flux
using Printf
using Random
using LinearAlgebra

Random.seed!(2023)

include("test/test_utils.jl")
include("simulate_rrulesimple.jl")


M, M0, q, q1, q2, state0, state1, tstep = test_config();

## set up modeling operator
S = jutulModeling(M0, tstep)
S_true = jutulModeling(M, tstep)
op1 = make_op(S, state1.state, q1)

## simulation
x = log.(KtoTrans(CartesianMesh(M), M.K))
x0 = log.(KtoTrans(CartesianMesh(M0), M0.K))
ϕ = S.model.ϕ

using UnicodePlots
mesh_2d = CartesianMesh(M0.n[1:2:3], M0.d[1:2:3])
function display_on_mesh(arr)
    arr_2d = reshape(arr, mesh_2d.dims)
    a = heatmap(arr_2d, xfact = mesh_2d.deltas[1], yfact=mesh_2d.deltas[2])
    display(a)
end
function display_states_simple(states)
    arr = reshape(state0[:Saturations][1, :], mesh_2d.dims)
    a = heatmap(arr, xfact = mesh_2d.deltas[1], yfact=mesh_2d.deltas[2])
    display(a)
    for i = 1:length(states)
        println(i)
        display_on_mesh(states[i][:Saturations][1, :])
    end
end
states_true_simple = S_true(x, ϕ, q1)
states_orig_simple = S(x0, ϕ, q1)


println("states")
display_states_simple(dict(states_true_simple))

println("states0")
display_states_simple(dict(states_orig_simple))

results_simple = op_simple(exp.(x0), ϕ)

println("results.states - states_true")
display_states_simple(dict(jutulStates(results.states) - states_true_simple))


misfit_orig(x0, ϕ, states, q) = 0.5 * norm(S(x0, ϕ, q) - states).^2
misfit(x0, ϕ, states::T, op) where T = 0.5 * norm(T(op(exp.(x0), ϕ).states) - states).^2

dx = randn(MersenneTwister(2023), length(x0))
dx = dx/norm(dx) * norm(x0)/5.0

dϕ = randn(MersenneTwister(2023), length(ϕ))
dϕ = dϕ/norm(dϕ) * norm(ϕ)/2.75e11



@show misfit_orig(x0, ϕ, states_true_simple, q1)
@show misfit(x0, ϕ, states_true_simple, op1)

g1_orig = gradient(()->misfit_orig(x0, ϕ, states_true_simple, q1), Flux.params(x0, ϕ))
g1 = gradient(()->misfit(x0, ϕ, states_true_simple, op1), Flux.params(x0, ϕ))

@testset "Taylor-series gradient test of simple jutulModeling" begin
    grad_test(x0->misfit(x0, ϕ, states_true_simple, op1), x0, dx, g1[x0])
    grad_test(ϕ->misfit(x0, ϕ, states_true_simple, op1), ϕ, dϕ, g1[ϕ], h0=2e1, maxiter=12)
end

@testset "Taylor-series gradient test of simple jutulModeling" begin
    grad_test(x0->misfit(x0, ϕ, states_true_simple, op1), x0, dx, g1_orig[x0])
    grad_test(ϕ->misfit(x0, ϕ, states_true_simple, op1), ϕ, dϕ, g1_orig[ϕ], h0=2e1, maxiter=12)
end

