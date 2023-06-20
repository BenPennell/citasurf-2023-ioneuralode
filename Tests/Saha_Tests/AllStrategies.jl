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
BATCH_COUNT::Int = 5

a₀ = Float64(1f0/(1f0+z(T₀)))
a₁ = Float64(1f0/(1f0+1140.f0))
da = a₁ - a₀

target_u0 = [0.99, a₀]
target_aspan = (Float64(1f0/(1f0+z(T₀))), Float64(1f0/(1f0+1140.f0)));
target_asteps = range(target_aspan[1], target_aspan[2], length=SAMPLE_SIZE);
target_xₑ = Array(xₑ.(target_asteps));

u0_series = target_u0 .+ ([1e-2, 1e-5] .* rand(2, BATCH_COUNT))

aspan_series = [u0_series[2,:], da .+ u0_series[2,:]]
astep_series = range.(aspan_series[1], aspan_series[2], length=SAMPLE_SIZE)

training_series = xₑseries.(astep_series)

### NETWORK
NETWORK_SIZE::Int = 25

rbf(x) = exp.(-(x .^ 2))

const network_u = Lux.Chain(Lux.Dense(2, NETWORK_SIZE, rbf), Lux.Dense(NETWORK_SIZE, NETWORK_SIZE, rbf), 
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
    λ = 1e-7
    return sum(loss_example.([p], indices)) + λ * sum(abs2, p) # ADD THE WEIGHT DECAY TERM!
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

res1 = Optimization.solve(optprob, ADAM(10.0, (0.6, 0.8)), callback = callback, maxiters = 200)

optprob2 = remake(optprob,u0 = res1.u)

result_neuralode2 = Optimization.solve(optprob2,
                                        ADAM(0.1),
                                        callback=callback,
                                        maxiters=1000);

optprob3 = remake(optprob2, u0 = result_neuralode2.u)

result_neuralode3 = Optimization.solve(optprob3,
                                        ADAM(0.01),
                                        callback=callback,
                                        maxiters=1000);

optprob4 = remake(optprob3, u0 = result_neuralode3.u)

result_neuralode4 = Optimization.solve(optprob4,
                                        ADAM(0.001),
                                        callback=callback,
                                        maxiters=600);                                  

optprob5 = remake(optprob4, u0 = result_neuralode4.u)

result_neuralode5 = Optimization.solve(optprob5,
                                        BFGS(initial_stepnorm=0.005),
                                        callback=callback,
                                        allow_f_increases = false);  

function check_network(result)
    updated_ivp = remake(ivp, u0=target_u0, p=result.u, 
                            tspan=target_aspan)
    networkdata = Array(solve(updated_ivp, Tsit5(), saveat=target_asteps))
    plt = plot(target_asteps, target_xₑ, label = "Training xₑ", title=losses[length(losses)])
    plot!(plt, target_asteps, networkdata[1,:], label = "Network xₑ")
    display(Plots.plot(plt))
end

check_network(result_neuralode5)
savefig("Saha_DataBatching_WeightDecay_10-1000_01-2000_001-1000_0001-600_BFGS")
### Loss Plot
function draw_loss(values)
    plt = plot(1:1:length(values), values, yscale=:log10)
    display(Plots.plot(plt))
end

draw_loss(losses)
savefig("Saha_DB_WD_losses")
