# using Pkg 
# Pkg.activate("/resnick/groups/wugroup/zblood/CMBLensing.jl")
using CMBLensing
using ArgParse
using NPZ

#load in the task id and the f, phi combo id
function parse_commandline()

    settings = ArgParseSettings()

    @add_arg_table settings begin
        "--seed"
            arg_type = Int
        "--map_size"
            arg_type = Int
    end

    return parse_args(settings)
end

args = parse_commandline()
seed = args["seed"]
map_size = args["map_size"]

#We will use the following settings for load_sim()
θpix  = 2.5        # pixel size in arcmin #TODO make these arguments that can be passed to the slurm script
Nside = map_size      # number of pixels per side in the map
pol   = :I       # type of data to use (can be :T, :P, or :TP)
T     = Float64  # data type (Float32 is ~2 as fast as Float64);

#run load_sim() using the given seed and settings
(;f, f̃, ϕ, ds) = load_sim(
    seed = seed,
    θpix = θpix,
    T = T,
    Nside = Nside,
    pol = pol
)

#store the data into its own specific seed folder
file_path = pwd() * "/julia_generated_data/map_size_$(map_size)_seed_$seed"
mkpath(file_path)

#store all the necesary arrays and fields
println("writing simulation data to storage...\n")
npzwrite(file_path * "/cf_diagonal.npz", ds.Cf.op.diag)
npzwrite(file_path * "/cf_lensed_diagonal.npz", ds.Cf̃.diag)
npzwrite(file_path * "/cphi_diagonal.npz", ds.Cϕ.op.diag)
npzwrite(file_path * "/nphi_diagonal.npz", ds.Nϕ.diag)
npzwrite(file_path * "/cn_diagonal.npz", ds.Cn̂.diag)
npzwrite(file_path * "/m_diagonal.npz", transpose(ds.M̂.diag.arr))
npzwrite(file_path * "/b_diagonal.npz", transpose(ds.B̂.diag.arr))
npzwrite(file_path * "/f_array.npz", transpose(f.arr))
npzwrite(file_path * "/f_map_array.npz", transpose(Map(f).arr))
npzwrite(file_path * "/phi_array.npz", transpose(ϕ.arr))
npzwrite(file_path * "/phi_map_array.npz", transpose(Map(ϕ).arr))
npzwrite(file_path * "/f_lensed_array.npz", transpose(ds.M̂.diag.arr .* ds.B̂.diag.arr .* f̃[:Il]))
npzwrite(file_path * "/f_lensed_no_b_no_m_array.npz", transpose(f̃[:Il]))
npzwrite(file_path * "/f_lensed_map_array.npz", transpose(Map(f̃).arr))
npzwrite(file_path * "/data_array.npz", transpose(ds.d.arr))
npzwrite(file_path * "/f_lambda_array.npz", f.λ_rfft)
npzwrite(file_path * "/phi_lambda_array.npz", ϕ.λ_rfft)
npzwrite(file_path * "/data_lambda_array.npz", ds.d.λ_rfft)
npzwrite(file_path * "/cf_lambda_array.npz", ds.Cf.op.diag.λ_rfft)
npzwrite(file_path * "/d_lambda_array.npz", ds.D.op.diag.λ_rfft)
npzwrite(file_path * "/cphi_lambda_array.npz", ds.Cϕ.op.diag.λ_rfft)
npzwrite(file_path * "/cn_lambda_array.npz", ds.Cn.diag.λ_rfft)
npzwrite(file_path * "/m_lambda_array.npz", ds.M̂.diag.λ_rfft)
npzwrite(file_path * "/b_lambda_array.npz", ds.B̂.diag.λ_rfft)
npzwrite(file_path * "/cf_lensed_lambda_array.npz", ds.Cf̃.diag.λ_rfft)
npzwrite(file_path * "/f_lensed_lambda_array.npz", f̃.λ_rfft)

#matrices needed for the mixed parametrization
npzwrite(file_path * "/d_diagonal.npz", ds.D.op.diag)
npzwrite(file_path * "/g_diagonal.npz", ds.G.op.diag)

open(file_path * "/pix_width.txt", "w") do io
    pix_width = ds.d.metadata.Δx
    println(io, pix_width)
end
open(file_path * "/num_pixels.txt", "w") do io
    Nx = ds.d.metadata.Nx
    Ny = ds.d.metadata.Ny
    num_pixels = Nx * Ny
    println(io, num_pixels)
end

