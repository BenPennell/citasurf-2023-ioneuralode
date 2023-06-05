using Plots, DiffEqFlux, Flux, DifferentialEquations, Optimization, OptimizationOptimisers, OptimizationOptimJL

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
    return (0.5f0) * (-β₄ + (β₄^2.f0 + 4.f0*β₄)^(0.5f0))
end

aspan = (1f0/(1f0+z(T₀)), 1f0/(1f0+1140.f0));
asteps = range(aspan[1], aspan[2], length=SAMPLE_SIZE);

training_xₑ = Array(xₑ.(asteps)); # Training xₑ

### THE NETWORK
NETWORK_SIZE = 16
ITERATION_COUNT = 1000
u0 = 0.99f0

network = Flux.Chain(Dense(2, NETWORK_SIZE, tanh),
                    Dense(NETWORK_SIZE, NETWORK_SIZE, tanh),
                    Dense(NETWORK_SIZE, 1),
                    first);
prams, re = Flux.destructure(network);
promoted_network(u, p, t) = re(p)([u, t]);
nODE = ODEProblem{false}(promoted_network, u0, aspan, prams);

function probe_network(p)
    # Returns the 'time' series for xₑ, T₄
    return Array(solve(nODE, Tsit5(), p=p, saveat=asteps))
end

function loss(p)
    # Loss function to minimize
    network_xₑ = probe_network(p)
    loss = sum(abs2, network_xₑ .- training_xₑ)
    return loss, network_xₑ
end

loss_values = []

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

### THE SOLVER
#res = DiffEqFlux.sciml_train(loss, prams, ADAM(0.1), cb=callback, maxiters=1000)
adtype = Optimization.AutoZygote();
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, prams);

result_neuralode = Optimization.solve(optprob,
                                        Flux.Optimisers.RAdam(100.0, (0.6, 0.8)),
                                        callback = callback,
                                        maxiters = 2000);

callback(result_neuralode.u, loss(result_neuralode.u)...; doplot=true)
savefig("Saha_Output")