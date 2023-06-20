using OrdinaryDiffEq
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Lux, Plots, StableRNGs
gr()

# Set a random seed for reproducible behaviour
rng = StableRNG(1111)

### THE DATA
SAMPLE_SIZE = 20
Tᵧ = 0.000273f0 # kelvin * 10^4
α = Float32(5.8e15/0.024) # Constant term in Saha
T₀ = 0.4305f0

z(T) = T/Tᵧ - 1.f0 # kelvin * 10^4
β(T) = α * T^(-1.5f0) * exp(-15.8f0/T) # Righthand side of Saha

T₄(a) = Tᵧ * (1.f0/a) # kelvin * 10^4

function xₑ(a) 
    # Solving for Xₑ using quadratic formula
    T = T₄(a)
    β₄ = β(T)
    return Float64((0.5f0) * (-β₄ + (β₄^2.f0 + 4.f0*β₄)^(0.5f0)))
end

function xₑseries(aseries)
    return Array(xₑ.(aseries))
end

### BATCHES OF DATA
const BATCH_COUNT = 5

u0_series = [1, 0.001] .* rand(2, BATCH_COUNT)

aspan_series = [u0_series[2,:], 3e-4 .+ u0_series[2,:]]
astep_series = range.(aspan_series[1], aspan_series[2], length=SAMPLE_SIZE)

training_series = xₑseries.(astep_series)

### NETWORK
const NETWORK_SIZE = 20

rbf(x) = exp.(-(x .^ 2))

network_u = Lux.Chain(Lux.Dense(2, NETWORK_SIZE, rbf), Lux.Dense(NETWORK_SIZE, NETWORK_SIZE, rbf), 
                        Lux.Dense(NETWORK_SIZE, NETWORK_SIZE, rbf), Lux.Dense(NETWORK_SIZE, 2));

p, st = Lux.setup(rng, network_u);

function ude!(du, u, p, t)
    u[2] = t
    û = network_u(u, p, st)[1]
    du[1] = u[1] + û[1]
    du[2] = 1
end

ivp = ODEProblem{true}(ude!, u0_series[:,1], (aspan_series[1][1], aspan_series[2][1]), p);

function probe_network(p, index)
    updated_ivp = remake(ivp, u0=u0_series[:,index], p=p, 
                            tspan=(aspan_series[1][index], aspan_series[2][index]))
    return Array((solve(updated_ivp, Tsit5(), saveat=astep_series[index]))[1, :])
end

function loss_example(p, index)
    network_xₑ = probe_network(p, index)
    return sum(abs2, network_xₑ .- training_series[index])
end

function loss(p)
    indices = 1:1:BATCH_COUNT
    return sum(loss_example.([p], indices))
end

### TRAIN THAT 
losses = [];

## Documentation callback
callback = function (p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

optf = Optimization.OptimizationFunction((x, p) -> loss(x), Optimization.AutoForwardDiff());
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p));

res1 = Optimization.solve(optprob, ADAM(10.0), callback = callback, maxiters = 1000)

function check_network()
    plt = plot(asteps, training_xₑ, label = "Training xₑ")
    scatter!(plt, asteps, probe_network(res1.u), label = "Network xₑ")
    display(Plots.plot(plt))
end

check_network()