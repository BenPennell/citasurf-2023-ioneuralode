base_directory = "/fs/lustre/scratch/bpennel"
dump_directory = "run_from_here"
data_directory = "./github/Recfast/RecfastData.json"
script_directory = "./github/Recfast/Recscript.jl"
project_directory = "./github/rep"
outfolder_directory = "./Recfast_Batches"

rands = 1001:1:1500
sizes = [5, 20, 60]

for seed in rands
    for size in sizes
        name = "$(size)_$(seed)"

        PBS_script = """
        #PBS -l nodes=1:ppn=2
        #PBS -l mem=8gb
        #PBS -l walltime=12:00:00
        #PBS -r n
        #PBS -j oe
        #PBS -q starq

        cd $(base_directory)
        julia -t 2 --project=$(project_directory) $(script_directory) --INPUT $(data_directory) --SIZE $(size) --SEED $(seed) --DECAY $(decay) --OUTPUT $(outfolder_directory) --NAME $(name)
                    """
        scriptfile = joinpath(base_directory, dump_directory, name)
        open(scriptfile, "w") do file write(file, PBS_script) end
        run(`qsub $scriptfile`)
    end          
end