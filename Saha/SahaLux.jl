# SciML Tools
using OrdinaryDiffEq, SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Lux, Zygote, Plots, StableRNGs
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

aspan = (Float64(1f0/(1f0+z(T₀))), Float64(1f0/(1f0+1140.f0)));
asteps = range(aspan[1], aspan[2], length=SAMPLE_SIZE);

training_xₑ = Array(xₑ.(asteps)); # Training xₑ

### NETWORK
NETWORK_SIZE = 5

rbf(x) = exp.(-(x .^ 2))

network_u = Lux.Chain(Lux.Dense(2, NETWORK_SIZE, rbf), Lux.Dense(NETWORK_SIZE, NETWORK_SIZE, rbf), 
                        Lux.Dense(NETWORK_SIZE, NETWORK_SIZE, rbf), Lux.Dense(NETWORK_SIZE, 2))

p, st = Lux.setup(rng, network_u);

function ude!(du, u, p, t)
    u[2] = t
    û = network_u(u, p, st)[1]
    du[1] = u[1] + û[1]
    du[2] = 1
end

ivp = ODEProblem{true}(ude!, [0.99, 0], aspan, p);

## ----> CAN IT SOLVE SOMETHING?
sol = solve(ivp, Tsit5(), saveat=asteps)

function probe_network(p)
    updated_ivp = remake(ivp, p=p)
    return Array((solve(updated_ivp, Tsit5(), saveat=asteps))[1, :])
end

function loss(p)
    network_xₑ = probe_network(p)
    return sum(abs2, network_xₑ .- training_xₑ)
end

## ----> DOES THE GRADIENT WORK????
Zygote.gradient(loss, ComponentVector{Float64}(p))

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

adtype = Optimization.AutoZygote();
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p));

res1 = Optimization.solve(optprob, ADAM(100.0, (0.6, 0.8)), callback = callback, maxiters = 3000)

function check_network()
    plt = plot(asteps, training_xₑ, label = "Training xₑ")
    scatter!(plt, asteps, probe_network(res1.u), label = "Network xₑ")
    display(Plots.plot(plt))
end

check_network()
## OLD CALLBACK
```
callback = function (ps, test_loss, test_output; doplot=true)
    # Plot at every training step
    println(test_loss)
    push!(loss_values, test_loss)
    if doplot
        plt = plot(asteps, training_xₑ, label = "Training xₑ", title=test_loss)
        scatter!(plt, asteps, test_output, label = "Network xₑ")

        display(Plots.plot(plt))
    end
    return false
end
```