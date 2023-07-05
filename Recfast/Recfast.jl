using OrdinaryDiffEq, Optimization, OptimizationOptimisers, OptimizationOptimJL, SciMLSensitivity
using ComponentArrays, Lux, Plots, StableRNGs, JSON
using Zygote, StaticArrays

### HYPERPARAMETERS
RNG_SEED::Int = 1112
NETWORK_SIZE::Int = 20
LEARNING_RATE::Float64 = 0.005
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

function ude(u, p, t)
    û = network_u(SA[u[1], u[2], u[3], t], p, st)[1] .* [characteristic_ascale, characteristic_ascale, characteristic_ascale]  # Scale to datascale
    du1 = u[1] + û[1]
    du2 = u[2] + û[2]
    du3 = u[3] + û[3]
    du4 = 0
    return SA[du1, du2, du3, du4]
end

u0 = first.(training_data)
ivp = ODEProblem{false}(ude, SA[u0[1], u0[2], u0[3], 0.], aspan, p);

function probe_network(p)
    solution = solve(remake(ivp, p=p), Tsit5(), saveat=asteps, sensealg=QuadratureAdjoint(autojacvec=ZygoteVJP()))
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
step_iterations = 200
step_size = 0.3
step_max = 2π
decay(x) = cos(x % π/2) # Only get first quadrant
decay_amounts = decay.(0:step_size:step_max);
learning_rates = LEARNING_RATE .* decay_amounts
plot(learning_rates)

### TRAINING
losses = Float64[];

callback = function (p, l)
    push!(losses, l)
    if (length(losses) - 1) % 25 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

optf = Optimization.OptimizationFunction((x, p) -> loss(x), Optimization.AutoForwardDiff());

function train_network(optf, p0, rates, step_iters)
    ps = p0
    result = nothing
    for rate in rates
        println("Learning rate: $(rate)")
        optprob = Optimization.OptimizationProblem(optf, ps)
        result = Optimization.solve(optprob, ADAM(rate), callback=callback, maxiters=step_iters);
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
savefig("RecfastRecombination.png")
plot_loss(losses)