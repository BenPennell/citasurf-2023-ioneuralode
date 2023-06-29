using OrdinaryDiffEq, Optimization, OptimizationOptimisers
using ComponentArrays, Lux, Plots, StableRNGs, JSON

### HYPERPARAMETERS
RNG_SEED::Int = 1112
NETWORK_SIZE::Int = 25
LEARNING_RATE::Float64 = 0.01
ITERATIONS::Int = 200

rng = StableRNG(RNG_SEED);

### DATA
data = JSON.parsefile("./Saha++/RecfastData.json"); # He, a, H, T4
original_data = [ Float64.(data[s]) for s in ("H", "He", "T4") ];

norm_consts = first.(maximum!.([[1.,]], original_data));
training_data = original_data ./ norm_consts;

asteps = data["a"];
aspan = (first(asteps), last(asteps));

characteristic_ascale = 1 / (aspan[2] - aspan[1])

### NETWORK
network_u = Lux.Chain(Lux.Dense(4, NETWORK_SIZE, tanh), 
                        Lux.Dense(NETWORK_SIZE, NETWORK_SIZE, tanh), 
                        Lux.Dense(NETWORK_SIZE, NETWORK_SIZE, tanh),
                        Lux.Dense(NETWORK_SIZE, NETWORK_SIZE, tanh),
                        Lux.Dense(NETWORK_SIZE, 3));

p, st = Lux.setup(rng, network_u);

function ude!(du::AbstractArray{T}, u, p, t) where {T}
    u[4] = t # add time dependance to dummy time slot
    û = network_u(u, p, st)[1] .* T(characteristic_ascale) # Scale to datascale
    du[1:3] = u[1:3] .+ û # Ignore dummy time slot
end

u0 = push!(first.(training_data), 0);
ivp = ODEProblem{true}(ude!, u0, aspan, p);

function probe_network(p)
    solution = solve(remake(ivp, p=p), Tsit5(), saveat=asteps)
    promoted = [ solution[s,:] for s in 1:1:3 ]
    return promoted
end

function loss_series(network, training)
    return sum(abs, (network .- training))
end

function loss(p)
    return sum(loss_series.(probe_network(p), training_data))
end

### LEARNING RATE DECAY
step_iterations = 100
decay_max = 1000
decay_rate = 500
α = log(10) / 300
decay(x) = exp(-α * x)
decay_amounts = decay.(0:step_iterations:decay_max);
learning_rates = LEARNING_RATE .* decay_amounts
plot(learning_rates)

learning_rates
### TRAINING
losses = Float64[];

callback = function (p, l)
    push!(losses, l)
    println("Current loss after $(length(losses)) iterations: $(losses[end])")
    return false
end

optf = Optimization.OptimizationFunction((x, p) -> loss(x), Optimization.AutoForwardDiff());

function train_network(optf, p0, rates, step_iters)
    ps = p0
    result = nothing
    for rate in rates
        println("Learning rate: $(rate)")
        optprob = Optimization.OptimizationProblem(optf, ps)
        result = Optimization.solve(optprob, ADAM(rate), callback=callback, maxiters=step_iters)
        ps = result.u
    end

    return result
end

result_node = train_network(optf, ComponentVector{Float64}(p), learning_rates, step_iterations);

### EXTRAS
function check_network(result)
    network_data = probe_network(result.u)
    plt = plot(asteps, training_data[1], label = "Training Hydrogen",
                        title="Neural Network for Recfast", xlabel="Scale Factor")
    scatter!(plt, asteps, network_data[1], label = "Network Hydrogen")

    plot!(plt, asteps, training_data[2], label = "Training Helium",)
    scatter!(plt, asteps, network_data[2], label = "Network Helium")

    plot!(plt, asteps, training_data[3], label = "Training Temperautre",)
    scatter!(plt, asteps, network_data[3], label = "Network Temperature")
    display(Plots.plot(plt))
end

function plot_loss(values)
    plt = plot(1:1:length(values), values, yscale=:log10, 
                            title="Loss for Saha recombination training", xlabel="Iteration", ylabel="Loss")
    display(Plots.plot(plt))
end

check_network(result_node)
plot_loss(losses)