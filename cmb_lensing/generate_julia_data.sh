#!/bin/bash

#SBATCH --time=00:05:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on
#SBATCH --ntasks=4 #number of processor cores / tasks
#SBATCH --mail-user=zblood@caltech.edu #mail updates to this address
#SBATCH --mail-type=BEGIN,END,FAIL #mail updates on begin, end, or failure

#call the package activation helper before anything else and ensure this finishes
#before calling any of the simulation data
PID=$(sbatch --parsable julia_package_activation.sh)

num_seeds=1 #20
#loop over map size values
for map_size in 128; do # 256 512 1024; do
    #loop over number of seeds
    for ((seed=1; seed<=num_seeds; seed++)); do
        sbatch --dependency=afterok:$PID generate_julia_data_helper.sh $map_size $seed
    done
done