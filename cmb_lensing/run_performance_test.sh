#!/bin/bash

#SBATCH --time=00:10:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on
#SBATCH --ntasks=4 #number of processor cores / tasks
#SBATCH --mail-user=zblood@caltech.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on begin, end, or failure

#the total number of seeds (f, phi) pairs from load_sim() and the
#total number of trials per (f, phi, map_size) triplet combo
num_seeds=1
num_trials=1

#call the precompilation helper and wait until this finishes to call the other scripts
PID=$(sbatch --parsable julia_package_activation.sh)

#loop over map size values
for map_size in 128; do # 256 512 1024; do
    #loop over number of seeds
    for ((seed=1; seed<=num_seeds; seed++)); do
        #loop over number of trials per (f, phi, map size) tuple
        for ((trial=1; trial<=num_trials; trial++)); do
            #spawn a slurm job from a helper script that takes in (seed_id, map_size, trial_id)
            #and make sure this file writes to the proper folders... Do this for both julia and python
            sbatch --dependency=afterok:$PID julia_performance_helper.sh $map_size $seed $trial
            sbatch --dependency=afterok:$PID python_performance_helper.sh $map_size $seed $trial
        done
    done
done