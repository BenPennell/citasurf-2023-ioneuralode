using OrdinaryDiffEq, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Lux, Plots, StableRNGs
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

aspan = (Float64(1f0/(1f0+z(T₀))), Float64(1f0/(1f0+1140.f0)));
asteps = range(aspan[1], aspan[2], length=SAMPLE_SIZE);

training_xₑ = Array(xₑ.(asteps)); # Training xₑ

### NETWORK
NETWORK_SIZE = 20

increase(x) = x * 100

network_u = Lux.Chain(Lux.Dense(2, NETWORK_SIZE, tanh), 
                        Lux.Dense(NETWORK_SIZE, NETWORK_SIZE, tanh), 
                        Lux.Dense(NETWORK_SIZE, 1, increase));

p, st = Lux.setup(rng, network_u);

function ude!(du, u, p, t)
    û = network_u([u[1], t], p, st)[1]
    du[1] = u[1] + û[1]
end

ivp = ODEProblem{true}(ude!, [0.99, 0], aspan, p);

function probe_network(p)
    updated_ivp = remake(ivp, p=p)
    return Array((solve(updated_ivp, Tsit5(), saveat=asteps))[1, :])
end

function loss(p)
    network_xₑ = probe_network(p)
    return sum(abs2, network_xₑ .- training_xₑ)
end

### TRAIN THAT 
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

result_node1 = Optimization.solve(optprob, ADAM(100.0), callback=callback, maxiters=500);

optprob2 = remake(optprob,u0 = result_node1.u);

result_node2 = Optimization.solve(optprob2, ADAM(0.1), callback=callback, maxiters=1000);

optprob3 = remake(optprob2, u0 = result_node2.u);

result_node3 = Optimization.solve(optprob3, BFGS(initial_stepnorm=0.005), 
                                    callback=callback, allow_f_increases = false);  

function check_network(result)
    plt = plot(asteps, training_xₑ, label = "Training xₑ")
    scatter!(plt, asteps, probe_network(result.u), label = "Network xₑ")
    display(Plots.plot(plt))
end

check_network(result_node1)