using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, ComponentArrays, OptimizationOptimisers
rng = Random.default_rng()

## Create the test data
params = [1.5, 1.0, 3.0, 1.0]
u₀ = [1.0, 1.0]
time_range = [0.0, 10.0]
datasize = 150
timesteps = range(time_range[1], time_range[2], length=datasize)

function lotka_volterra(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
end

ivp = ODEProblem(lotka_volterra, [1.0, 1.0], (0.0, 10.0), params);
training_data = Array(solve(ivp, Tsit5(), saveat=timesteps));
print(size(training_data))
## Set up the Network
network = Lux.Chain(Lux.Dense(2, 5, tanh),
                    Lux.Dense(5, 2));
ps, st = Lux.setup(rng, network);
ps_array = ComponentArray(ps);
probe_network = NeuralODE(network, time_range, Tsit5(), saveat=timesteps)

function test_network(ps)
    Array(probe_network(u₀, ps, st)[1])
end
  
function loss(ps)
      network_output = test_network(ps)
      loss = sum(abs2, training_data .- network_output)
      return loss, network_output
end

## Callback function
callback = function (ps, test_loss, test_output; doplot=false)
    println(test_loss)
    if doplot
        plt = scatter(timesteps, training_data[2,:], label = "Real data")
        scatter!(plt, timesteps, test_output[2,:], label = "Neural Network")
        scatter!(timesteps, training_data[1,:], label = "Real data")
        scatter!(plt, timesteps, test_output[1,:], label = "Neural Network")
        display(plot(plt))
    end
    return false
end

## Approximate ADAM solution
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ps_array)

result_neuralode = Optimization.solve(optprob,
                                       ADAM(0.05),
                                       callback = callback,
                                       maxiters = 300);

## Precise BFGS solution
optprob2 = remake(optprob,u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
                                        Optim.BFGS(initial_stepnorm=0.01),
                                        callback=callback,
                                        allow_f_increases = false);

callback(result_neuralode2.u, loss(result_neuralode2.u)...; doplot=true)
savefig("network_output")