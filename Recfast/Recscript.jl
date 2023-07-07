using Lux, OrdinaryDiffEq, Optimization, OptimizationOptimisers, SciMLSensitivity
using ComponentArrays, StableRNGs, JSON, ArgParse, StaticArrays, Dates, JLD2

function main(args)
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--NAME"
            nargs = "?"
            arg_type = String
            default = "Recfast_Network"
        "--INPUT"
            nargs = '?'
            arg_type = String
            default = "./RecfastData.json"
        "--SEED"
            nargs = '?'
            arg_type = Int
            default = 1112
        "--WIDTH"
            nargs = '?'
            arg_type = Int
            default = 20
        "--RATE"
            nargs = '?'
            arg_type = Float64
            default = 0.005
        "--DECAY"
            nargs = 4
            arg_type = Float64
            default = [0, 0.1, 3π, 200] # Cosine Schedule: Min, Step Size, Max, Step Iterations
        "--OUTPUT"
            nargs = '?'
            arg_type = String
            default = "./output"
    end

    ar = parse_args(s)
    rng = StableRNG(ar["SEED"]);

    # Where the output is saved to
    out_folder = "$(ar["OUTPUT"])/$(ar["NAME"])"
    try
        mkdir(out_folder)
    catch
    end

    ### DATA
    data = JSON.parsefile(ar["INPUT"]); # He, a, H, T4
    original_data = [ Float64.(data[s]) for s in ("H", "He", "T4") ];

    norm_consts = first.(maximum!.([[1.,]], original_data));
    training_data = original_data ./ norm_consts;

    asteps = data["a"];
    aspan = (first(asteps), last(asteps));

    characteristic_ascale = 1 / (aspan[2] - aspan[1])

    ### NETWORK
    network_u = Lux.Chain(Lux.Dense(4, ar["WIDTH"], tanh), 
                            Lux.Dense(ar["WIDTH"], ar["WIDTH"], tanh), 
                            Lux.Dense(ar["WIDTH"], ar["WIDTH"], tanh),
                            Lux.Dense(ar["WIDTH"], ar["WIDTH"], tanh),
                            Lux.Dense(ar["WIDTH"], 3));
    p, st = Lux.setup(rng, network_u);

    function ude(u, p, t)
        û = network_u(SA[u[1], u[2], u[3], t], p, st)[1] .* characteristic_ascale  # Scale to datascale
        du1 = u[1] + û[1]
        du2 = u[2] + û[2]
        du3 = u[3] + û[3]
        du4 = 0
        return SA[du1, du2, du3, du4]
    end

    u0 = first.(training_data)
    ivp = ODEProblem{false}(ude, SA[u0[1], u0[2], u0[3], 0.], aspan, p);

    function probe_network(p)
        solution = solve(remake(ivp, p=p), Tsit5(), saveat=asteps, sensealg=QuadratureAdjoint(autojacvec=ZygoteVJP()))
        promoted = [ solution[s,:] for s in 1:1:3 ]
        return promoted
    end

    function loss_series(network, training)
        return sum(abs, (network .- training))
    end

    function loss(p)
        return sum(loss_series.(probe_network(p), training_data))
    end

    ### LEARNING RATE SCHEDULE
    decay(x) = cos(x % π/2) # Only get first quadrant
    decay_amounts = decay.(ar["DECAY"][1]:ar["DECAY"][2]:ar["DECAY"][3]);
    learning_rates = ar["RATE"] .* decay_amounts

    ### TRAINING
    losses = Float64[];
    checkpoint_losses = Float64[]

    callback = function (p, l)
        push!(losses, l)
        return false
    end

    optf = Optimization.OptimizationFunction((x, p) -> loss(x), Optimization.AutoForwardDiff());
    function train_network(optf, p0, rates, step_iters)
        ps = p0
        for rate in rates
            optprob = Optimization.OptimizationProblem(optf, ps)
            result = Optimization.solve(optprob, ADAM(rate), callback=callback, maxiters=step_iters);
            ps = result.u
            jldsave("$(out_folder)/$(ar["NAME"])_$(length(losses))_checkpoint"; ps)
            push!(checkpoint_losses, last(losses))
        end
    end

    train_network(optf, ComponentVector{Float64}(p), learning_rates, Int(ar["DECAY"][4]));

    ### SAVING
    logfile = open("$(out_folder)/$(ar["NAME"])_log.txt", "w")
    write(logfile, """
                    NAME: $(ar["NAME"])
                    SOURCE: $(ar["INPUT"])
                    TIMESTAMP: $(now())
                    SEED: $(ar["SEED"])
                    WIDTH: $(ar["WIDTH"])
                    MAX LEARNING RATE: $(ar["RATE"])
                    SCHEDULE: ($(join(ar["DECAY"], ",")))
                    RECORD LOSS: $(minimum!([1.,], losses)[1])
                    __CHECKPOINT LOSSES__\n$(join(checkpoint_losses, "\n"))
                    """)
    close(logfile)

    lossfile = open("./output/$(ar["NAME"])/$(ar["NAME"])_loss.txt","w")
    write(lossfile, join(losses, "\n"))
    close(lossfile)
    
end

main(ARGS)