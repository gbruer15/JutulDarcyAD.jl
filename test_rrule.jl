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
include("simulate_rrule_wrapper.jl")


M, M0, q, q1, q2, state0, state1, tstep = test_config();

## set up modeling operator
S = jutulModeling(M0, tstep)
S_true = jutulModeling(M, tstep)

## simulation
x = log.(KtoTrans(CartesianMesh(M), M.K))
x0 = log.(KtoTrans(CartesianMesh(M0), M0.K))
ϕ = S.model.ϕ

op = make_op(S, state0.state, q)

using UnicodePlots
mesh_2d = CartesianMesh(M0.n[1:2:3], M0.d[1:2:3])
function display_on_mesh(arr)
    arr_2d = reshape(arr, mesh_2d.dims)
    a = heatmap(arr_2d, xfact = mesh_2d.deltas[1], yfact=mesh_2d.deltas[2])
    display(a)
end
function display_states(states)
    arr = reshape(state0[:Reservoir][:Saturations][1, :], mesh_2d.dims)
    a = heatmap(arr, xfact = mesh_2d.deltas[1], yfact=mesh_2d.deltas[2])
    display(a)
    for i = 1:length(states)
        println(i)
        display_on_mesh(states[i][:Reservoir][:Saturations][1, :])
    end
end
states_true = S_true(x, ϕ, q)
states_orig = S(x0, ϕ, q)
results = op(exp.(x0), ϕ)

println("states")
display_states(dict(states_true))

println("states0")
display_states(dict(states_orig))

println("results.states - states_true")
display_states(dict(jutulStates(results.states) - states_true))

println("results.states - states_orig")
display_states(dict(jutulStates(results.states) - states_orig))

# result, sim_pullback = ChainRulesCore.rrule(simulate!, op.sim, op.tstep; forces = op.forces, op.kwargs...);
# dnew_state = deepcopy(result.states);
# dresult = (states=dnew_state,);
# _, dsim, _ = sim_pullback(dresult);
# g = dsim.storage.state0

misfit_orig(x0, ϕ, states, q) = 0.5 * norm(S(x0, ϕ, q) - states).^2
misfit(x0, ϕ, states::T, op) where T = 0.5 * norm(T(op(exp.(x0), ϕ).states) - states).^2

dx = randn(MersenneTwister(2023), length(x0))
dx = dx/norm(dx) * norm(x0)/5.0

dϕ = randn(MersenneTwister(2023), length(ϕ))
dϕ = dϕ/norm(dϕ) * norm(ϕ)/2.75e11

g_orig = gradient(()->misfit_orig(x0, ϕ, states, q), Flux.params(x0, ϕ))
g = gradient(()->misfit(x0, ϕ, states, op), Flux.params(x0, ϕ))

@show misfit_orig(x0, ϕ, states, q)
@show misfit(x0, ϕ, states, op)

@testset "Taylor-series gradient test of jutulModeling with wells" begin
    grad_test(x0->misfit(x0, ϕ, states, op), x0, dx, g[x0])
    grad_test(ϕ->misfit(x0, ϕ, states, op), ϕ, dϕ, g[ϕ], h0=2e1, maxiter=12)
end



