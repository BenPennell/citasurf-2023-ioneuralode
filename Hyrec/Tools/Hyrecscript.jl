using ArgParse

### What This is expecting is HYREC-2/Hyrecscript to contain this file, and this directory also has a subdirectory /tests

quantities = ["h", "T0", "Omega_b", "Omega_cb", "Omega_k", "w0-wa", "Nmnu", "mnu1", "mnu2", "mnu3", "Y_He", "Neff", "Alpha_ratio", "me_ratio", "pann", "pann_halo", "ann_z", "ann_zmax", "ann_zmin", "ann_var", "ann_z_halo", "on_the_spot", "decay", "Mpbh", "fpbh"]
standard_values = ["0.70", "2.7255", "0.0494142797907188", "0.31242079216478097", "0", "-1 0", "1", "0.06", "0", "0", "0.245", "3.046", "1", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0"]

function main(args)

    s = ArgParseSettings()

    @add_arg_table! s begin
        "--name"
            nargs = 1
            arg_type = String
        "--modify"
            nargs = '*'
            arg_type = String
    end

    parsed_args = parse_args(args, s)
    split_args = split.(parsed_args["modify"], "=")
    indeces = findfirst.(isequal.(first.(split_args)),[quantities])
    standard_values[indeces] = last.(split_args)
    input_name = "$(parsed_args["name"][1])"
    open("./tests/$(input_name).dat", "w") do file write(file, join(standard_values, "\n")) end

    pbsstring = """
    cd /home/bpennel/Documents/HYREC-2
    ./hyrec < ./Hyrecscript/tests/$(input_name).dat"""
    open("./tests/$(input_name).sh", "w") do file write(file, pbsstring) end
    run(`bash $(input_name).sh`)
end

main(ARGS)