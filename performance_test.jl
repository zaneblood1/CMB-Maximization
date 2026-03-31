#imports
# using Pkg #NOTE if this is not here everything will break!!!!!!!
# Pkg.activate("/resnick/groups/wugroup/zblood/CMBLensing.jl/Project.toml")
using CMBLensing #TODO send help ticket / ask for advice from the Resnick HPC team...
using ArgParse #Can specify which architecture
using NPZ

#load in the task id and the f, phi combo id
function parse_commandline()

    settings = ArgParseSettings()

    @add_arg_table settings begin
        "--map_size"
            arg_type = Int
        "--seed"
            arg_type = Int
        "--trial"
            arg_type = Int
    end

    return parse_args(settings)
end

args = parse_commandline()
trial = args["trial"]
map_size = args["map_size"]
#NOTE we use f_and_phi_id as a seed for load sim. Ideally it would be cooler to be able to use the 
#already generated julia data, but MAP_joint() takes in a ds; object as input and I do not
#know how to easily serialize / deserialize that since it requires casting to a bunch of 
#Marius' special object types
seed = args["seed"]

#We will use the following settings for load_sim()
θpix  = 2.5      # pixel size in arcmin
Nside = map_size # number of pixels per side in the map
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

#run the algorithm once initially to get the uncached time
start_time = time()
fJ, phiJ = MAP_joint(ds, nsteps = 30, progress = false);
end_time = time()
uncached_time = end_time - start_time

#get the current working directory for writing to
cwd = pwd()

#store the uncached time for this (f, phi) combo in its own proper directory
mkpath(cwd * "/performance_results/julia_results/map_size_$(map_size)_seed_$seed/uncached_times")
open(cwd * "/performance_results/julia_results/map_size_$(map_size)_seed_$seed/uncached_times/uncached_time_$trial.txt", "w") do io
    write(io, string(uncached_time))
end

#now run the algorithm a second time to get the cache time
start_time = time()
_, _ = MAP_joint(ds, nsteps = 30, progress = false);
end_time = time()
cached_time = end_time - start_time

#store the cached time for this (f, phi) combo in its own proper directory
mkpath(cwd * "/performance_results/julia_results/map_size_$(map_size)_seed_$seed/cached_times")
open(cwd * "/performance_results/julia_results/map_size_$(map_size)_seed_$seed/cached_times/cached_time_$trial.txt", "w") do io
    write(io, string(cached_time))
end

#store the learned / estimated f and phi from python to compare with what julia found
mkpath(cwd * "/performance_results/julia_results/map_size_$(map_size)_seed_$seed/learned_fields/temperature")
mkpath(cwd * "/performance_results/julia_results/map_size_$(map_size)_seed_$seed/learned_fields/lensing_potential")
npzwrite(cwd * "/performance_results/julia_results/map_size_$(map_size)_seed_$seed/learned_fields/temperature/fJ_$trial.npz", fJ.arr)
npzwrite(cwd * "/performance_results/julia_results/map_size_$(map_size)_seed_$seed/learned_fields/lensing_potential/phiJ_$trial.npz", phiJ.arr)
