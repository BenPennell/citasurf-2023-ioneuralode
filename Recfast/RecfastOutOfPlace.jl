using OrdinaryDiffEq, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Lux, Plots, StableRNGs, JSON
using Zygote, StaticArrays

### HYPERPARAMETERS
RNG_SEED::Int = 1112
NETWORK_SIZE::Int = 25
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

### SHOW ME THE MONEY vvvv
function ude!(u, p, t)
    û = network_u(SA[u[1], u[2], u[3], t], p, st)[1] .* characteristic_ascale # Scale to datascale
    du = SA[u[1:3] .+ û]
    return du
end

u0 = SA[push!(first.(training_data), 0)];
ivp = ODEProblem{false}(ude, u0, aspan, p);
sol = solve(ivp, Tsit5(), saveat=asteps)
### DAMN IT ^^^^

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

### TRAINING
losses = [];
callback = function (p, l)
    push!(losses, l)
    println("Current loss after $(length(losses)) iterations: $(losses[end])")
    return false
end

optf = Optimization.OptimizationFunction((x, p) -> loss(x), Optimization.AutoForwardDiff());
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p));
result_node1 = Optimization.solve(optprob, ADAM(LEARNING_RATE), callback=callback, maxiters=ITERATIONS);
optprob2 = remake(optprob,u0 = result_node1.u);
result_node2 = Optimization.solve(optprob2, ADAM(0.0003), callback=callback, maxiters=300);
optprob3 = remake(optprob,u0 = result_node2.u);
result_node3 = Optimization.solve(optprob3, BFGS(initial_stepnorm=0.0001), callback=callback, allow_f_increases = false);

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

check_network(result_node3)
plot_loss(losses)