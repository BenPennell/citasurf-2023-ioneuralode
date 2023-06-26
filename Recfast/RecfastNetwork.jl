using OrdinaryDiffEq, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Lux, Plots, StableRNGs, JSON

### HYPERPARAMETERS
RNG_SEED::Int = 1111
NETWORK_SIZE::Int = 25

rng = StableRNG(RNG_SEED);

### IDK
function characteristic_length(series)
    return (maximum!([1.,], series)[1] - minimum!([1.,], series)[1])
end

function norm_const(series)
    return abs(maximum!([1.,], series)[1])
end

### DATA
data = JSON.parsefile("./Saha++/RecfastData.json"); # He, a, H, T4

original_data = [Float64.(data["H"]), Float64.(data["He"]), Float64.(data["T4"])]; # There must be a better way to get Float 64s

# The idea here is that we will train on normalized data
# Re-multiply by norm_consts to get original data
norm_consts = norm_const.(original_data);
training_data = original_data ./ norm_consts;

asteps = data["a"];
aspan = (first(asteps), last(asteps));

characteristic_ascale = 1 / characteristic_length(asteps)

### THE NETWORK
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
    promoted = [solution[1,:], solution[2,:], solution[3,:]] # How can I do this better?
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
# Speed of the network is really reliant on these learning rates. This is all just chosen via trial and error, and I don't like that
result_node1 = Optimization.solve(optprob, ADAM(0.01), callback=callback, maxiters=100);
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