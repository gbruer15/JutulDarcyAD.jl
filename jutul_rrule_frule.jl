using Jutul
using JutulDarcy
using JutulDarcyRules
using ChainRulesCore
using LinearAlgebra
import Zygote
using Zygote.Forward: @tangent, zerolike

Jutul.linear_operator(op::Jutul.LinearOperator) = op

jutulState{T}(s::jutulState{T}) where T = return deepcopy(s)

function ChainRulesCore.rrule(::Type{jutulState{T}}, d::Dict) where T
    obj = jutulState(d)
    pullback(Δobj::jutulState{T}) = NoTangent(), Δobj
    return obj, pullback
end

function ChainRulesCore.rrule(::Type{jutulSimpleState{T}}, d::Dict) where T
    obj = jutulSimpleState(d)
    pullback(Δobj::jutulSimpleState{T}) = NoTangent(), Δobj
    return obj, pullback
end

function ChainRulesCore.rrule(::Type{jutulState{T}}, s::jutulSimpleState{T}) where T
    obj = deepcopy(s)
    pullback(Δobj::jutulSimpleState{T}) = NoTangent(), Δobj
    return obj, pullback
end

function ChainRulesCore.rrule(::Type{jutulState{T}}, s::jutulState{T}) where T
    obj = deepcopy(s)
    pullback(Δobj::jutulState{T}) = NoTangent(), Δobj
    return obj, pullback
end

function ChainRulesCore.rrule(::Type{jutulStates{T}}, states::Vector{jutulState{T}}) where T
    obj = jutulStates(states)
    pullback(Δobj::jutulStates{T}) = NoTangent(), Δobj.states
    pullback(Δobj::Vector{T}) = NoTangent(), obj(Δobj).states
    return obj, pullback
end

function ChainRulesCore.rrule(::Type{jutulSimpleStates{T}}, states::Vector{jutulSimpleState{T}}) where T
    obj = jutulSimpleStates(states)
    pullback(Δobj::jutulSimpleStates{T}) = NoTangent(), Δobj.states
    pullback(Δobj::Vector{T}) = NoTangent(), obj(Δobj).states
    return obj, pullback
end

# function ChainRulesCore.rrule(::Type{jutulState{T}}, state::Dict) where T
#     pullback(Δobj) = NoTangent(), Δobj.state
#     return jutulState(state), pullback
# end

ChainRulesCore.@opt_out ChainRulesCore.rrule(::typeof(vec), ::jutulStates{T}) where T
ChainRulesCore.@opt_out ChainRulesCore.rrule(::typeof(norm), ::jutulStates{T}) where T
ChainRulesCore.@opt_out ChainRulesCore.rrule(::Any, ::jutulStates{T}) where T

function setup_gradient_storage(model; adjoint = true,
                                    state0 = Jutul.setup_state(model),
                                    parameters = Jutul.setup_parameters(model),
                                    targets = Jutul.parameter_targets(model),
                                    linear_solver = Jutul.select_linear_solver(model, mode = adjoint ? :adjoint : :tlm, rtol = 1e-8),
                                    kwarg...)
    primary_model = adjoint ? Jutul.adjoint_model_copy(model) : model

    # Standard model for: ∂Fₙᵀ / ∂uₙ
    forward_sim = Simulator(primary_model, state0 = deepcopy(state0), parameters = deepcopy(parameters), mode = :forward, extra_timing = nothing)

    # Same model, but adjoint for: ∂Fₙᵀ / ∂uₙ₋₁
    backward_sim = Simulator(primary_model, state0 = deepcopy(state0), parameters = deepcopy(parameters), mode = :reverse, extra_timing = nothing)

    # Create parameter model for ∂Fₙ / ∂p
    parameter_model = Jutul.adjoint_parameter_model(model, targets)

    # Note that primary is here because the target parameters are now the primaries for the parameter_model
    parameter_map, = Jutul.variable_mapper(parameter_model, :primary, targets = targets; kwarg...)

    # Transfer over parameters and state0 variables since many parameters are now variables
    state0_p = Jutul.swap_variables(state0, parameters, parameter_model, variables = true)
    parameters_p = Jutul.swap_variables(state0, parameters, parameter_model, variables = false)
    parameter_sim = Simulator(parameter_model, state0 = deepcopy(state0_p), parameters = deepcopy(parameters_p), mode = :sensitivities, extra_timing = nothing)

    # Create buffer for linear solve.
    n_pvar = Jutul.number_of_degrees_of_freedom(model)
    λ_buffer = zeros(n_pvar)

    rhs_buffer = Jutul.vector_residual(forward_sim.storage.LinearizedSystem)
    dx_buffer = forward_sim.storage.LinearizedSystem.dx_buffer
    n_var = length(dx_buffer)
    if length(rhs_buffer) != n_var
        rhs_buffer = zeros(n_var)
    end
    return (forward = forward_sim,
            backward = backward_sim,
            parameter = parameter_sim,
            parameter_map = parameter_map,
            linear_solver = linear_solver,
            λ_buffer = λ_buffer,
            dx_buffer = dx_buffer,
            rhs_buffer = rhs_buffer,
        )
end

get_parameters_dict(sim, model::JutulModel) = Dict{Symbol, Any}(pairs(sim.storage.parameters))

function get_parameters_dict(sim, model::MultiModel)
    parameters = Dict{Symbol, Any}()
    for k in Jutul.submodel_symbols(model)
        parameters[k] = Dict{Symbol, Any}(pairs(sim.storage[k].parameters))
    end
    return parameters
end

get_state0_dict(sim, model::JutulModel) = Dict{Symbol,Any}(pairs(sim.storage.state0))

function get_state0_dict(sim, model::MultiModel)
    state0 = Dict{Symbol, Any}()
    for k in Jutul.submodel_symbols(model)
        state0[k] = Dict{Symbol, Any}(pairs(sim.storage.state0[k]))
    end
    return state0
end

function ChainRulesCore.rrule(::typeof(simulate!), sim::JutulSimulator, timesteps::AbstractVector;
        forces = setup_forces(sim.model),
        kwarg...
    )
    # Run simulation.
    result = simulate!(sim, timesteps; forces=forces, kwarg...)

    # Pop some kwargs that are only for the simulate! call. I may want to pass the rest of the kwargs as adjoint kwargs.
    # config = pop!(kwarg, :config, nothing)
    # initialize = pop!(kwarg, :initialize, true)
    # restart = pop!(kwarg, :restart, nothing)
    # state0 = pop!(kwarg, :state0, nothing)
    # parameters = pop!(kwarg, :parameters, nothing)
    # start_date = pop!(kwarg, :start_date, nothing)

    model = sim.model
    state0 = deepcopy(get_state0_dict(sim, model))
    parameters = deepcopy(get_parameters_dict(sim, model))
    states = result.states

    # Set up adjoint things.
    adjoint_storage = setup_gradient_storage(model; adjoint = true, state0 = state0, parameters = parameters);

    parameter_sim = adjoint_storage.parameter
    backward_sim = adjoint_storage.backward
    forward_sim = adjoint_storage.forward

    all_forces = forces
    n_primary = Jutul.number_of_degrees_of_freedom(model)
    n_param = Jutul.number_of_degrees_of_freedom(parameter_sim.model)

    function pullback(dresult::Union{NamedTuple, Tangent})
        return pullback(dresult.states)
    end
    function pullback(dresult::Vector{Tangent})
        @assert length(dresult) == 1
        return pullback(dresult[1])
    end
    function pullback(dstates)
        # At each timestep, uᵢ is a function uᵢ(uᵢ₋₁, p), which is
        # determined by the implicit equation Fᵢ(uᵢ, uᵢ₋₁, p) = 0.
        # Here, p is any parameter given to the simulator.

        # We need to multiply ∂J/∂uᵢ by the adjoint Jacobian (duᵢ/duᵢ₋₁)ᵀ
        # at each timstep 1 through N.
        # The contribution to dJ/dm is (∂F/∂p') (∂F/∂uᵢ' \ ∂J/∂uᵢ) for each uₙ.
        N = length(timesteps)
        global ∂J_∂u_dict = states
        @assert N == length(∂J_∂u_dict)

        # Set gradient to zero before solve starts
        global dJ_dp = zeros(n_param)
        λ = adjoint_storage.λ_buffer
        global λ_mine = λ
        λ_b = zeros(length(λ))
        global λ_b_mine = λ_b
        rhs = adjoint_storage.rhs_buffer
        global rhs_mine = rhs
        dx = adjoint_storage.dx_buffer
        global dx_mine = dx
        global dJ_du₀ = zeros(n_primary)
        global ∂J_∂u_orig = [convert_primary_tangents_to_array(d, model) for d in ∂J_∂u_dict]
        global ∂J_∂u = [zeros(size(d)) for d in ∂J_∂u_orig]
        for i in 1:N
            Jutul.adjoint_transfer_canonical_order!(∂J_∂u[i], ∂J_∂u_orig[i], forward_sim.model, to_canonical=false)
        end
        for i in N:-1:1
            fn = deepcopy
            if i == 1
                s0 = fn(state0)
                dJ_duᵢ₋₁ = dJ_du₀
            else
                s0 = fn(states[i-1])
                dJ_duᵢ₋₁ = ∂J_∂u[i-1]
            end
            s = fn(states[i])

            dt = timesteps[i]
            t = sum(timesteps[1:i-1])

            global ∂J_∂uᵢ = ∂J_∂u[i]
            global state0 = s0
            global state = s

            # 1. Compute ∂F/∂uᵢ.
            forces = Jutul.forces_for_timestep(forward_sim, all_forces, timesteps, i)
            Jutul.adjoint_reassemble!(forward_sim, state, state0, dt, forces, t)
            global f_lsys = forward_sim.storage.LinearizedSystem
            global ∂F_∂uᵢᵀ = Jutul.linear_operator(f_lsys; skip_red=true)

            # 2. Compute adjoint solution λ = ∂F/∂uᵢ' \ ∂J/∂uᵢ.
            # Note: the negative sign is handled by the linear_solve function.
            # global λ = -∂F_∂uᵢᵀ \ ∂J_∂uᵢ
            global lsolve = adjoint_storage.linear_solver
            f_lsys.r_buffer .= ∂J_∂uᵢ
            linear_solve!(f_lsys, lsolve, forward_sim.model, forward_sim.storage)

            Jutul.adjoint_transfer_canonical_order!(λ, f_lsys.dx_buffer, forward_sim.model, to_canonical = true)
            # λ .= f_lsys.dx_buffer


            # 3a. Compute ∂F/∂uᵢ₋₁.
            Jutul.adjoint_reassemble!(backward_sim, state, state0, dt, forces, t)
            global ∂F_∂uᵢ₋₁ᵀ = Jutul.linear_operator(backward_sim.storage.LinearizedSystem; skip_red=true)

            # 3b. Compute contribution to dJ/duᵢ₋₁.
            Jutul.adjoint_transfer_canonical_order!(λ_b, λ, forward_sim.model, to_canonical = false)
            global dJ_duᵢ₋₁ .+= ∂F_∂uᵢ₋₁ᵀ * λ_b

            # 4a. Compute ∂F/∂p.
            Jutul.adjoint_reassemble!(parameter_sim, state, state0, dt, forces, t)
            global ∂F_∂pᵀ = Jutul.linear_operator(parameter_sim.storage.LinearizedSystem)

            # 4b. Compute contribution to dJ/dp.
            global dJ_dp .+= ∂F_∂pᵀ * λ
        end

        Jutul.rescale_sensitivities!(dJ_dp, adjoint_storage.parameter.model, adjoint_storage.parameter_map)
        @assert all(isfinite, dJ_dp)
        @assert all(isfinite, dJ_du₀)

        dparameters = Jutul.store_sensitivities(parameter_sim.model, dJ_dp, adjoint_storage.parameter_map)
        dstate0 = convert_primary_tangents_to_dict(dJ_du₀, model)
        dsim_storage = JutulStorage()
        dsim_storage[:parameters] = dparameters
        dsim_storage[:state0] = dstate0
        dsim = Simulator(model, dsim_storage)
        return NoTangent(), dsim, NoTangent()
    end
    return result, pullback
end

function get_dofs(state, pkey, p::JutulVariables, model::JutulModel)
    state_val = state[pkey]
    return state_val
end
function get_dofs(state, pkey, p::JutulDarcy.Saturations, model::JutulModel)
    state_val = state[pkey]
    return state_val[1, :]
end
function get_dofs(arr, p::JutulVariables, model::JutulModel)
    return arr
end
function get_dofs(arr, p::JutulDarcy.Saturations, model::JutulModel)
    return [arr'; (1 .- arr)']
end

get_models(model) = Dict(nothing=>model)
get_models(model::MultiModel) = model.models


function convert_primary_variables_to_array(state, model::JutulModel)
    layout = Jutul.matrix_layout(model.context)
    cell_major = Jutul.is_cell_major(layout)
    if cell_major
        error("Didn't implement this yet")
    end

    n = Jutul.number_of_degrees_of_freedom(model)
    array = zeros(n)

    offset = 0
    for (mkey, m) in pairs(get_models(model))
        mstate = isnothing(mkey) ? state : state[mkey]
        primary = Jutul.get_primary_variables(m)
        for (pkey, p) in primary
            n = Jutul.number_of_degrees_of_freedom(m, p)
            rng = (1:n) .+ offset
            arr_p = view(array, rng)
            arr_p .= get_dofs(mstate, pkey, p, m)
            offset += n
        end
    end
    return array
end

function convert_primary_variables_to_dict(array, model::JutulModel)
    layout = Jutul.matrix_layout(model.context)
    cell_major = Jutul.is_cell_major(layout)
    if cell_major
        error("Didn't implement this yet")
    end

    n = Jutul.number_of_degrees_of_freedom(model)
    state = Dict{Symbol, Any}()

    offset = 0
    for (mkey, m) in pairs(get_models(model))
        mstate = Dict{Symbol, Any}()
        primary = Jutul.get_primary_variables(m)
        for (pkey, p) in primary
            n = Jutul.number_of_degrees_of_freedom(m, p)
            rng = (1:n) .+ offset
            arr_p = view(array, rng)
            mstate[pkey] = get_dofs(arr_p, p, m)
            offset += n
        end
        if isnothing(mkey)
            state = mstate
        else
            state[mkey] = mstate
        end
    end
    return state
end

function get_dof_tangents(arr, p::JutulDarcy.Saturations, model::JutulModel)
    return [arr'; .- arr']
end
function get_dof_tangents(arr, p::JutulVariables, model::JutulModel)
    return arr
end

function convert_primary_tangents_to_dict(array, model::JutulModel)
    layout = Jutul.matrix_layout(model.context)
    cell_major = Jutul.is_cell_major(layout)
    if cell_major
        error("Didn't implement this yet")
    end

    n = Jutul.number_of_degrees_of_freedom(model)
    state = Dict{Symbol, Any}()

    offset = 0
    for (mkey, m) in pairs(get_models(model))
        mstate = Dict{Symbol, Any}()
        primary = Jutul.get_primary_variables(m)
        for (pkey, p) in primary
            n = Jutul.number_of_degrees_of_freedom(m, p)
            rng = (1:n) .+ offset
            arr_p = view(array, rng)
            mstate[pkey] = get_dof_tangents(arr_p, p, m)
            offset += n
        end
        if isnothing(mkey)
            state = mstate
        else
            state[mkey] = mstate
        end
    end
    return state
end


function convert_primary_tangents_to_array(state, model::JutulModel)
    layout = Jutul.matrix_layout(model.context)
    cell_major = Jutul.is_cell_major(layout)
    if cell_major
        error("Didn't implement this yet")
    end

    n = Jutul.number_of_degrees_of_freedom(model)
    array = zeros(n)

    offset = 0
    for (mkey, m) in pairs(get_models(model))
        mstate = isnothing(mkey) ? state : state[mkey]
        primary = Jutul.get_primary_variables(m)
        for (pkey, p) in primary
            n = Jutul.number_of_degrees_of_freedom(m, p)
            rng = (1:n) .+ offset
            arr_p = view(array, rng)
            arr_p .= get_dofs(mstate, pkey, p, m)
            offset += n
        end
    end
    return array
end

isexpr = Zygote.Forward.isexpr

using IRTools
using Zygote.Forward

function Zygote.instrument(ir::IRTools.IR)
  pr = IRTools.Pipe(ir)
  for (v, st) in pr
    println()
    @show v st
    ex = st.expr
    if isexpr(ex, :foreigncall, :isdefined)
      continue
    elseif isexpr(ex, :enter, :leave)
      error("""try/catch is not supported.
            Refer to the Zygote documentation for fixes.
            https://fluxml.ai/Zygote.jl/latest/limitations
            """)
    elseif isexpr(ex, :(=))
      @assert ex.args[1] isa GlobalRef
      pr[v] = xcall(Zygote, :global_set, QuoteNode(ex.args[1]), ex.args[2])
    else
      ex = Zygote.instrument_new!(pr, v, ex)
      ex = Zygote.instrument_literals!(pr, v, ex)
      ex = Zygote.instrument_global!(pr, v, ex)
    end
  end
  ir = Zygote.Forward.finish(pr)
  # GlobalRefs can turn up in branch arguments
  for b in Zygote.Forward.blocks(ir), br in Zygote.Forward.branches(b), i in 1:length(Zygote.Forward.arguments(br))
    (ref = Zygote.Forward.arguments(br)[i]) isa GlobalRef || continue
    Zygote.Forward.arguments(br)[i] = push!(b, Zygote.Forward.xcall(Zygote, :unwrap, QuoteNode(ref), ref))
  end
  return ir
end

function Zygote.Forward.dual(ir)
  args = copy(Zygote.Forward.arguments(ir))
  dx = Zygote.Forward.argument!(ir, at = 1)
  Δs = Dict()
  for bl in Zygote.Forward.blocks(ir)[2:end], arg in copy(Zygote.Forward.arguments(bl))
    Δs[arg] = Zygote.Forward.argument!(bl, insert = false)
  end
  pr = Zygote.Forward.Pipe(ir)
  partial(x::Zygote.Forward.Variable) = Δs[x]
  partial(x) = push!(pr, Zygote.Forward.xcall(Forward, :zerolike, x))
  partial(v, x::Zygote.Forward.Variable) = Δs[x]
  partial(v, x) = insert!(pr, v, Zygote.Forward.xcall(Forward, :zerolike, x))
  for (i, x) in enumerate(args)
    if i == length(args) && ir.meta.method.isva
      Δs[x] = push!(pr, Zygote.Forward.ntail(dx, i-1))
    else
      Δs[x] = push!(pr, Zygote.Forward.xcall(:getindex, dx, i))
    end
  end
  Zygote.Forward.branches(pr) do br
    args = Zygote.Forward.arguments(br)
    if Zygote.Forward.isreturn(br)
      args[1] = push!(pr, Zygote.Forward.xcall(:tuple, args[1], partial(args[1])))
    else
      for arg in copy(args)
        push!(args, partial(arg))
      end
    end
    br
  end
@show pr
  for (v, st) in pr
    println()
    println()
    @show pr
    println()
    st = Zygote.Forward.instrument!(pr, v, st)
    @show v
    @show st
    @show st.expr
    if isexpr(st.expr, :meta, :inbounds, :loopinfo)
      Δs[v] = nothing
    elseif isexpr(st.expr, :boundscheck) ||
           (isexpr(st.expr, :call) && st.expr.args[1] == GlobalRef(Base, :not_int)) ||
           (isexpr(st.expr, :call) && st.expr.args[1] == GlobalRef(Core, :(===))) ||
           (isexpr(st.expr, :call) && st.expr.args[1] == GlobalRef(Main, :(===)))
      Δs[v] = false
    elseif isexpr(st.expr, :call)
        # The current line of code is v = some_function(st.expr.args...), where st.expr calls a function.
        # We want to replace that line with the following:
        #   dargs = tuple(partial(v, st.expr.args)...)
        #   result = _pushforward(dargs, st.expr.args...)
        #   v = result[1]
        #   grad = result[2]
        @show st.expr.args
      dargs = insert!(pr, v, Zygote.Forward.xcall(:tuple, partial.((v,), st.expr.args)...))
        @show dargs
      result = insert!(pr, v, Zygote.Forward.stmt(st, expr = Zygote.Forward.xcall(Forward, :_pushforward, dargs, st.expr.args...)))
        @show result
      pr[v] = Zygote.Forward.xcall(:getindex, result, 1)
      Δs[v] = push!(pr, Zygote.Forward.xcall(:getindex, result, 2))
    elseif !isexpr(st.expr)
      Δs[v] = push!(pr, Zygote.Forward.xcall(Forward, :zerolike, v))
    else
      error("Unsupported $(st.expr.head) expression: $(st.expr)")
    end
  end
  ir = Zygote.Forward.finish(pr)
  return ir
end

# @tangent Base.structdiff(a, b) = Base.structdiff((;a...), b), (ȧ, ḃ) -> Base.structdiff((;ȧ...), b)
@tangent function Base.structdiff(a, b)
    sd = Base.structdiff((;a...), b)
    forw(ȧ, ḃ) = Base.structdiff((;ȧ...), b)
    forw(ȧ::Nothing, ḃ) = nothing
    forw(ȧ, ḃ::Nothing) = ȧ
    forw(ȧ::Nothing, ḃ::Nothing) = nothing
    return sd, forw
end

Zygote.Forward.@dynamo function _pushforward(_, x...)
    global ir = Zygote.Forward.IR(x...)
    ir === nothing && return :(error("non-differentiable function $(args[2])"))
    ir = Zygote.instrument(ir)
    ir.meta.code.inlineable = true
    return Zygote.Forward.dual(ir)
end

using MacroTools: @capture, @q, shortdef
named = Zygote.Forward.named
typeless = Zygote.Forward.typeless
isvararg = Zygote.Forward.isvararg
using Base: tail

drop(x, n) = n == 0 ? x : :(tail($(drop(x, n-1))))
drop(n) = x -> drop(x, n)

tangent = Zygote.Forward.tangent

function Zygote.Forward.gradm(ex)
  @capture(shortdef(ex), (name_(args__) = body_) |
                         (name_(args__) where {Ts__} = body_)) || error("Need a function definition")
  global body = body
  global kw = length(args) > 1 && isexpr(args[1], :parameters) ? esc(popfirst!(args)) : nothing
  isclosure = isexpr(name, :(::)) && length(name.args) > 1
  global f, T = isexpr(name, :(::)) ?
    (length(name.args) == 1 ? (esc(gensym()), esc(name.args[1])) : esc.(name.args)) :
    (esc(gensym()), :(Core.Typeof($(esc(name)))))
  global kT = :(Core.kwftype($T))
  Ts === nothing && (global Ts = [])
  global args = named.(args)
  global argnames = Any[typeless(arg) for arg in args]
  !isempty(args) && isvararg(args[end]) && (argnames[end] = :($(argnames[end])...,))
  global args = esc.(args)
  global argnames = esc.(argnames)
  global Ts = esc.(Ts)
  global fargs = kw === nothing ? [:($f::$T), args...] : [kw, :($f::$T), args...]
  global dropg  = isclosure ? identity : drop(1)
  global dropkw = isclosure ?  drop(2) : drop(3)
  # adj = @q @inline Zygote.Forward.tangent($(fargs...)) where $(Ts...) = $(esc(body))
  global adj = @q @inline Zygote.Forward.tangent($(fargs...)) where $(Ts...) = $(esc(body))

  # TODO: I AM HERE for frule. I may need to implement a tangent function for the kwfunc of this function.
  # kwfargs = kw === nothing ? [:($f::$T), args...] : [kw, :($f::$T), args...]

  quote
    $adj
    @inline function Zygote.Forward._pushforward(partials, $f::$T, $(args...); kwargs...) where $(Ts...)
      y, forw = tangent($f, $(argnames...); kwargs...)
      return y, forw($(dropg(:partials))...)
      # return y, forw($(:partials)...)
    end
    # When keywords are passed to any function, they are lowered to a different form.
    # f(args...; kwargs...) becomes kwfunc(f)::kwftype(typeof(f))(kwargs::NamedTuple, f, args...).
    # I.e., it becomes a function of type kwftype(typeof(f)) that takes all the kwargs as the first arg.
    # We want to define the pushforward for that lowered form. It should call the same tangent function, but pass in the kwargs, too.
    # The partials for the kwarg call will include two extra arguments for the kwfunc(f) and the kwargs tuple, so we will drop those. 

    # Problem: the pushforward isn't allowed to call kwfunc, because that is a foreign call.
    @inline function Zygote.Forward._pushforward(partials, kwf::$kT, kw, $f::$T, $(args...)) where $(Ts...)
      y, forw = tangent(kw, $f, $(argnames...); kwargs...)
      # y, forw = kwf(kw, $f, $(argnames...))
      return y, forw($(dropkw(:partials))...)
    end
    nothing
  end
end

# _pushforward(::Tuple{Nothing, Simulator, Nothing},
#     ::Jutul.var"#simulate!##kw",
#     ::Base.Pairs{Symbol, Any, NTuple{5, Symbol}, NamedTuple{(:state0, :forces, :info_level, :max_timestep_cuts, :end_report), Tuple{Dict{Symbol, Any}, NamedTuple{(:sources, :bc), Tuple{Vector{SourceTerm{Int64, Float64, Tuple{Float64, Float64}}}, Nothing}}, Int64, Int64, Bool}}},
#     ::typeof(simulate!),
#     ::Simulator,
#     ::Vector{Float64})


Zygote.Forward.pushforward(f, x...) = (ẋ...) -> _pushforward((zerolike(f), ẋ...), f, x...)[2]
Zygote.Forward.pushforward(f, x...; kw...) = (ẋ...) -> _pushforward((nothing, nothing, zerolike(f), ẋ...), Core.kwfunc(f), (;kw...), f, x...)[2]

import Zygote

@tangent function Zygote.Forward.literal_getfield(t, ::Val{i}) where i
  y = getfield(t, i)
  forw(ṫ, _) = getfield(ṫ, i)
  forw(ṫ::Nothing, _) = nothing
  return y, forw
end

@tangent function Zygote.Forward.literal_getindex(t, ::Val{i}) where i
  y = getindex(t, i)
  forw(ṫ, _) = getindex(ṫ, i)
  forw(ṫ::Nothing, _) = nothing
  return y, forw
end

function Zygote.Forward.zerolike(x::Dict)
  length(x) == 0 ? nothing : Dict((k,zerolike(v)) for (k,v) in x)
end
Zygote.Forward.zerolike(x::Number) = zero(x)
Zygote.Forward.zerolike(x::Tuple) = zerolike.(x)

# @generated function Zygote.Forward.zerolike(x::T) where T
#   length(fieldnames(T)) == 0 ? nothing :
#   :(NamedTuple{$(fieldnames(T))}(($(map(f -> :(zerolike(x.$f)), fieldnames(T))...),)))
# end

function Zygote.Forward.zerolike(x::T) where T
    fns = fieldnames(T)
    if length(fns) == 0
        return nothing
    end
    zeroed_fields = map(f -> zerolike(getfield(x, f)), fns)
    return NamedTuple{fns}((zeroed_fields...,))
end


@tangent  function simulate!(sim::JutulSimulator, timesteps::AbstractVector;
        forces = setup_forces(sim.model),
        kwarg...
    )
    # Run simulation.
    @show length(kwarg)
    state0 = kwarg[:state0] #|| Dict{Any,Any}(pairs(deepcopy(sim.storage.state0)))
    result = simulate!(sim, timesteps; forces=forces, kwarg...)

    # Pop some kwargs that are only for the simulate! call. I may want to pass the rest of the kwargs as adjoint kwargs.
    # config = pop!(kwarg, :config, nothing)
    # initialize = pop!(kwarg, :initialize, true)
    # restart = pop!(kwarg, :restart, nothing)
    # state0 = pop!(kwarg, :state0, nothing)
    # parameters = pop!(kwarg, :parameters, nothing)
    # start_date = pop!(kwarg, :start_date, nothing)

    model = sim.model
    state0 = Dict{Any,Any}(pairs(deepcopy(sim.storage.state0)))
    parameters = Dict{Any,Any}(pairs(deepcopy(sim.storage.parameters)))
    states = result.states

    # Set up adjoint things.
    gradient_storage = setup_gradient_storage(model; adjoint = false, state0 = state0, parameters = parameters)

    parameter_sim = gradient_storage.parameter
    backward_sim = gradient_storage.backward
    forward_sim = gradient_storage.forward

    all_forces = forces
    n_primary = Jutul.number_of_degrees_of_freedom(model)
    n_param = Jutul.number_of_degrees_of_freedom(parameter_sim.model)

    dstates = []
    function pushforward(dsimulate!::Nothing, dsim, dtimesteps::Nothing)
        start_timestamp = Jutul.now()
        # At each timestep, uᵢ is a function uᵢ(uᵢ₋₁, p), which is
        #  determined by the implicit equation Fᵢ(uᵢ, uᵢ₋₁, p) = 0.
        #  Here, p is any parameter given to the simulator.
        # We need to compute duᵢ at each timstep 1 through N.
        #  duᵢ = duᵢ/duᵢ₋₁ * duᵢ₋₁ + duᵢ/dp * dp, where 
        #     duᵢ/duᵢ₋₁ = - ∂F/∂uᵢ \ ∂F/∂uᵢ₋₁ and duᵢ/dp = - ∂F/∂uᵢ \ ∂F/∂p.

        dparameters = dsim.storage.parameters
        dp = isnothing(dparameters) ? nothing : vectorize_variables(parameter_sim.model, parameters)

        dstate0 = dsim.storage.state0
        du₀ = isnothing(dstate0) ? nothing : convert_primary_tangents_to_array(dstate0, model)

        # TODO: Need to scale or unscale the parameters.
        # Jutul.store_sensitivities(parameter_sim.model, dJ_dp, gradient_storage.parameter_map)
        # Jutul.rescale_sensitivities!(dJ_dp, gradient_storage.parameter.model, gradient_storage.parameter_map)

        N = length(timesteps)
        duᵢ₋₁ = du₀
        for i in 1:N
            fn = deepcopy
            if i == 1
                s0 = fn(state0)
            else
                s0 = fn(states[i-1])
            end
            s = fn(states[i])

            dt = timesteps[i]
            t = sum(timesteps[1:i-1])

            state0 = s0
            state = s
            contributions = []

            if !isnothing(dp)
                # 1a. Compute ∂F/∂p * dp.
                Jutul.adjoint_reassemble!(parameter_sim, state, state0, dt, forces, t)
                ∂F_∂p = Jutul.linear_operator(parameter_sim.storage.LinearizedSystem)
                ∂F_p = ∂F_∂p * dp
                push!(contributions, ∂F_p)
            end

            if !isnothing(duᵢ₋₁)
                # 1b. Compute ∂F/∂uᵢ₋₁ * duᵢ₋₁.
                Jutul.adjoint_reassemble!(backward_sim, state, state0, dt, forces, t)
                global ∂F_∂uᵢ₋₁ = Jutul.linear_operator(backward_sim.storage.LinearizedSystem)
                ∂F_uᵢ₋₁ = ∂F_∂uᵢ₋₁ * duᵢ₋₁
                push!(contributions, ∂F_uᵢ₋₁)
            end

            # 2. Compute change in F: ∂F/∂p * dp + ∂F/∂uᵢ₋₁ * duᵢ₋₁.
            ∂F = sum(contributions)

            # 3a. Compute ∂F/∂uᵢ.
            forces = Jutul.forces_for_timestep(forward_sim, all_forces, timesteps, i)
            Jutul.adjoint_reassemble!(forward_sim, state, state0, dt, forces, t)
            global ∂F_∂uᵢ = Jutul.linear_operator(forward_sim.storage.LinearizedSystem)

            # 3b. Transform the change in F to a change in uᵢ.
            global duᵢ = -∂F_∂uᵢ \ ∂F

            dstate = convert_primary_tangents_to_dict(duᵢ, model)
            push!(dstates, dstate)

            duᵢ₋₁ = duᵢ
        end

        return Jutul.SimResult(dstates, Vector{Any}(nothing, N), start_timestamp)
    end
    return result, pushforward
end


