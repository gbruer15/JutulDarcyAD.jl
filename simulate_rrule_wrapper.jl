using JutulDarcyRules: day, visCO2, visH2O, ρCO2, ρH2O, setup_well_model, get_Reservoir_state, setup_reservoir_simulator, simulate!, jutulStates, optimization_config, loss_per_step, setup_parameter_optimization, force
import ChainRulesCore
include("jutul_rrule_frule.jl")

struct JutulOperator1
    sim
    forces
    tstep
    kwargs
end
JutulOperator = JutulOperator1


function (op::JutulOperator)(state::Dict; dt=op.tstep[1])
    result = simulate!(op.sim, [dt]; state0 = state, forces = op.forces, op.kwargs...)
    return result.states[1]
end
function ChainRulesCore.rrule(op::JutulOperator, state::Dict; dt=op.tstep[1])
    result, sim_pullback = ChainRulesCore.rrule(simulate!, op.sim, [dt]; state0 = state, forces = op.forces, op.kwargs...)
    function pullback(dnew_state)
        dresult = (states=[dnew_state],)
        _, dsim, _ = sim_pullback(dresult)
        return NoTangent(), dsim.storage.state0
    end
    return result.states[1], pullback
end

function make_op(S, state0, f::Union{jutulForce{D, N}, jutulVWell{D, N}}) where {D,N}
    tstep = day * S.tstep
    model, parameters, state0, forces = setup_well_model(S.model, f, tstep; visCO2=visCO2, visH2O=visH2O, ρCO2=ρCO2, ρH2O=ρH2O)
    kwargs = (info_level=0, max_timestep_cuts = 1000, state0=state0)
    op = JutulOperator(sim, forces, tstep, kwargs);
    return op
end

function make_op(S, state0, f)
    tstep = day * S.tstep
    forces = JutulDarcyRules.source(S.model, f; ρCO2=ρCO2)
    model = JutulDarcyRules.simple_model(S.model)
    parameters = setup_parameters(model, PhaseViscosities = [visCO2, visH2O]);
    sim = Simulator(model, state0 = deepcopy(state0), parameters = parameters)
    kwargs = (info_level=0, max_timestep_cuts = 1000, state0=state0)
    op = JutulOperator(sim, forces, tstep, kwargs);
    return op
end


function (S::jutulModeling{D, T})(LogTransmissibilities::AbstractVector{T}, ϕ::AbstractVector{T}, f::Union{jutulForce{D, N}, jutulVWell{D, N}};
    state0=nothing, visCO2::T=T(visCO2), visH2O::T=T(visH2O),
    ρCO2::T=T(ρCO2), ρH2O::T=T(ρH2O), info_level::Int64=-1) where {D, T, N}

    Transmissibilities = exp.(LogTransmissibilities)

    ### set up simulation time
    tstep = day * S.tstep

    ### set up simulation configurations
    model, parameters, state0_, forces = setup_well_model(S.model, f, tstep; visCO2=visCO2, visH2O=visH2O, ρCO2=ρCO2, ρH2O=ρH2O);

    model.models.Reservoir.data_domain[:porosity] = ϕ
    parameters[:Reservoir][:Transmissibilities] = Transmissibilities
    parameters[:Reservoir][:FluidVolume] .= prod(S.model.d) .* ϕ

    isnothing(state0) || (state0_[:Reservoir] = get_Reservoir_state(state0))

    ### simulation
    sim, config = setup_reservoir_simulator(model, state0_, parameters);
    kwargs = (config=config, info_level=info_level, max_timestep_cuts = 1000)
    # op = JutulOperator(sim, forces, tstep, kwargs);

    # result = op(Transmissibilities, ϕ)
    # output = jutulStates(result.states)


    states, report = simulate!(sim, tstep; forces = forces, kwargs...);
    output = jutulStates(states)
    return output
end

function set_transmissibilities_porosity!(op::JutulOperator, Transmissibilities::AbstractVector{T}, ϕ::AbstractVector{T}) where {T}
    return set_transmissibilities_porosity!(op, Transmissibilities, ϕ, op.sim.model)
end

function set_transmissibilities_porosity!(op::JutulOperator, Transmissibilities::AbstractVector{T}, ϕ::AbstractVector{T}, ::Any) where {T}
    op.sim.model.data_domain[:porosity] = ϕ
    volumes = op.sim.model.data_domain[:volumes]
    parameters = Dict{Symbol, Any}(
        :Transmissibilities => Transmissibilities,
        :FluidVolume =>  volumes .* ϕ,
    )
    p = op.sim.storage.parameters
    for pk in keys(p)
        if !haskey(parameters, pk)
            parameters[pk] = p[pk]
        end
    end
    return parameters
end

function set_transmissibilities_porosity!(op::JutulOperator, Transmissibilities::AbstractVector{T}, ϕ::AbstractVector{T}, ::MultiModel) where {T}
    op.sim.model.models.Reservoir.data_domain[:porosity] = ϕ
    volumes = op.sim.model.models.Reservoir.data_domain[:volumes]
    parameters = Dict{Symbol, Any}(
        :Reservoir => Dict{Symbol, Any}(
            :Transmissibilities => Transmissibilities,
            :FluidVolume =>  volumes .* ϕ,
        )
    )
    parameters = Jutul.setup_parameters(op.sim.model, parameters)
    return parameters
    for (k, m) in pairs(op.sim.model.models)
        p = op.sim.storage[k].parameters
        if !haskey(parameters, k)
            parameters[k] = Dict(pairs(p))
            continue
        end
        for pk in keys(p)
            if !haskey(parameters[k], pk)
                parameters[k][pk] = p[pk]
            end
        end
    end
    return parameters
end

function (op::JutulOperator)(Transmissibilities::AbstractVector{T}, ϕ::AbstractVector{T}, model) where {T}
    parameters = set_transmissibilities_porosity!(op, Transmissibilities, ϕ, model)
    result = simulate!(op.sim, op.tstep; forces = op.forces, parameters = parameters, op.kwargs...)
    return result
end

function (op::JutulOperator)(Transmissibilities::AbstractVector{T}, ϕ::AbstractVector{T}) where {T}
    return op(Transmissibilities, ϕ, op.sim.model)
end

reservoir_parameters(sim::JutulSimulator, model) = sim.storage.parameters
reservoir_parameters(sim::JutulSimulator, model::MultiModel) = sim.storage.parameters[:Reservoir]

function ChainRulesCore.rrule(op::JutulOperator, Transmissibilities::AbstractVector{T}, ϕ::AbstractVector{T}) where {T}
    parameters = set_transmissibilities_porosity!(op, Transmissibilities, ϕ)
    result, sim_pullback = ChainRulesCore.rrule(simulate!, op.sim, op.tstep; forces = op.forces, parameters = parameters, op.kwargs...)
    function pullback(dnew_state)
        dresult = (states=[dnew_state],);
        _, dsim, _ = sim_pullback(dresult);

        dJ_dp = reservoir_parameters(dsim, op.sim.model)
        global dJ_dp_d = dJ_dp
        volumes = reservoir_model(op.sim.model).data_domain[:volumes]

        dTransmissibilities = dJ_dp[:Transmissibilities]
        dϕ = dJ_dp[:FluidVolume] .* volumes
        return NoTangent(), dTransmissibilities, dϕ
    end
    return result, pullback
end

