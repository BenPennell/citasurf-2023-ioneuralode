using OrdinaryDiffEq, Optimization, OptimizationOptimisers
using ComponentArrays, Lux, Plots, StableRNGs

### HYPERPARAMETERS
RNG_SEED::Int = 1111
SAMPLE_SIZE::Int = 20
NETWORK_SIZE::Int = 20
LEARNING_RATE::Float64 = 0.01
ITERATION_COUNT::Int = 500

rng = StableRNG(RNG_SEED);

### COSMOLOGICAL PARAMETERS
Tᵧ::Float64 = 0.000273 # kelvin * 10^4
ΩBHSQUARED::Float64 = 0.024

α = 5.8e15/ΩBHSQUARED # Constant term in Saha

### INITIAL/FINAL CONDITIONS
z₀::Float64 = 1575.
z₁::Float64 = 1140.

### THE DATA
a(z) = 1/(1 + z)
β(T) = α * T^(-1.5) * exp(-15.8/T) # Righthand side of Saha
T₄(a) = Tᵧ * (1/a) # kelvin * 10^4

function xₑ(a) 
    # Solving for Xₑ using quadratic formula
    T = T₄(a)
    β₄ = β(T)
    return (0.5) * (-β₄ + (β₄^2 + 4*β₄)^(0.5))
end

aspan = (a(z₀), a(z₁));
asteps = range(aspan[1], aspan[2], length=SAMPLE_SIZE);
training_xₑ = Array(xₑ.(asteps));

scaling = 1 / (aspan[2] - aspan[1]) # for scaling the network output

### THE NETWORK
network_u = Lux.Chain(Lux.Dense(2, NETWORK_SIZE, tanh), 
                        Lux.Dense(NETWORK_SIZE, NETWORK_SIZE, tanh), 
                        Lux.Dense(NETWORK_SIZE, NETWORK_SIZE, tanh),
                        Lux.Dense(NETWORK_SIZE, NETWORK_SIZE, tanh),
                        Lux.Dense(NETWORK_SIZE, NETWORK_SIZE, tanh),
                        Lux.Dense(NETWORK_SIZE, 1));

p, st = Lux.setup(rng, network_u);

function ude!(du::AbstractArray{T}, u, p, t) where {T}
    û = network_u([u[1], t], p, st)[1] * T(scaling)
    du[1] = u[1] + û[1]
end

ivp = ODEProblem{true}(ude!, [training_xₑ[1], 0], aspan, p);

function probe_network(p)
    return Array((solve(remake(ivp, p=p), Tsit5(), saveat=asteps))[1, :])
end

function loss(p)
    return sum(abs2, probe_network(p) .- training_xₑ)
end

### TRAINING
losses = [];
callback = function (p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

optf = Optimization.OptimizationFunction((x, p) -> loss(x), Optimization.AutoForwardDiff());
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p));
result_node = Optimization.solve(optprob, ADAM(LEARNING_RATE), callback=callback, maxiters=ITERATION_COUNT); # Done!

### EXTRAS
function check_network(result)
    plt = plot(asteps, training_xₑ, label = "Training xₑ",
                        title="Neural Network for Saha Recombination", xlabel="Scale Factor", ylabel="xₑ")
    scatter!(plt, asteps, probe_network(result.u), label = "Network xₑ")
    display(Plots.plot(plt))
end

function plot_loss(values)
    plt = plot(1:1:length(values), values, yscale=:log10, 
                            title="Loss for Saha recombination training", xlabel="Iteration", ylabel="Loss")
    display(Plots.plot(plt))
end

check_network(result_node)
plot_loss(losses)