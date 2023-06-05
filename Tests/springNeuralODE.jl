using OrdinaryDiffEq, Flux, DiffEqFlux, DiffEqSensitivity, Zygote, RecursiveArrayTools, Plots, Optimization, OptimizationOptimisers, Optimization, OptimizationOptimJL
#### Set Up Data
SAMPLE_COUNT = 12
ITERATION_COUNT = 10000

du₀ = 1.f0
u₀ = 5.f0
u0 = Float32[du₀; u₀]

timespan = (0.f0, 10.f0)
timesteps = range(timespan[1], timespan[2], length=SAMPLE_COUNT)

function acceleration(ddx, x, p, t)
    k, m = 2.f0, 1.f0
    return ddx = -k*x/m
end

ivp = SecondOrderODEProblem{false}(acceleration, du₀, u₀, timespan)
solution_set = solve(ivp, Tsit5(), saveat=timesteps)
solution_positions = Array(solution_set[2, :])
solution_velocities = Array(solution_set[1, :])

#### Set Up Network
network = Flux.Chain(Dense(2, 5, tanh),
                    Dense(5, 2))
nODE = NeuralODE(network, timespan, Tsit5(), saveat=timesteps)
new_values = nODE(u0, nODE.p)

function probe_network(p)
    probe = nODE(u0, p)
    return Array(probe[2, :]), Array(probe[1, :])
end

function loss(p)
    network_positions, network_velocities = probe_network(p)
    loss = sum(abs2, solution_positions .- network_positions) + sum(abs2, solution_velocities .- network_velocities)
    return loss, (network_positions, network_velocities)
end

#### Train it!
callback = function (ps, test_loss, test_output; doplot=true)
    println(test_loss)
    if doplot
        plt = plot(timesteps, solution_positions, label = "Real position")
        plot!(plt, timesteps, test_output[1], label = "Network position")
        plot!(timesteps, solution_velocities, label = "Real velocity")
        plot!(plt, timesteps, test_output[2], label = "Network velocity")
        display(Plots.plot(plt))
    end
    return false
end

adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, nODE.p)

result_neuralode = Optimization.solve(optprob,
                                       OptimizationOptimisers.ADAM(0.05),
                                       callback = callback,
                                       maxiters = 500);

optprob2 = remake(optprob,u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
                                        BFGS(initial_stepnorm=0.01),
                                        callback=callback,
                                        allow_f_increases = false);

callback(result_neuralode2.u, loss(result_neuralode2.u)...; doplot=true)
savefig("Spring_Output_Network")


#### CREATE FINE SOLUTION
# notice that I've also extended the domain (0, 20) to see how it holds up beyond (0, 10)
finespan = (0.f0, 20.f0)
finesteps = range(finespan[1], finespan[2], length=1000)
finevp = SecondOrderODEProblem{false}(acceleration, du₀, u₀, finespan)
fine_set = solve(finevp, Tsit5(), saveat=finesteps)
fine_positions = Array(fine_set[2, :])
fine_velocities = Array(fine_set[1, :])

fine_nODE = NeuralODE(network, finespan, Tsit5(), saveat=finesteps)
fine_network_solution = fine_nODE(u0, result_neuralode2.u)

plt = plot(finesteps, fine_positions, label = "Real position")
plot!(plt, finesteps, fine_network_solution[2, :], label = "Network position")
plot!(finesteps, fine_velocities, label = "Real velocity")
plot!(plt, finesteps, fine_network_solution[1, :], label = "Network velocity")
display(Plots.plot(plt))
savefig("Spring_Output_Network_Fine_Extended")