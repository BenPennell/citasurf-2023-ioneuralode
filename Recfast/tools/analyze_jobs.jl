using Plots

outfolder = "/fs/lustre/scratch/bpennel/Recfast_Batches"
sizes = ["5", "20", "40"]

folders = readdir(outfolder)

total_data = [[],[],[]]

for folder in folders
    loss_dir = "$(outfolder)/$(folder)/$(folder)_loss.txt"
    for i in 1:1:3
        if startswith(folder, sizes[i])
            file = open(loss_dir, "r")
            data = (parse.(Float64, readlines(file)))
            push!(total_data[i], data)
            close(file)
        end
    end
end

Plots.plot(total_data[1])
Plots.plot(total_data[2])
Plots.plot(total_data[3])