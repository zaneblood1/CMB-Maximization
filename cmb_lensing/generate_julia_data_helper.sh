#!/bin/bash

#SBATCH --time=48:00:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on
#SBATCH --ntasks=4 #number of processor cores / tasks
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH --mail-user=zblood@caltech.edu #mail updates to this address
#SBATCH --mail-type=BEGIN,END,FAIL #mail updates on begin, end, or failure

#use the conda initializer located in your own bashrc
source /home/zblood/.bashrc 
conda init
#activate your own specific conda
conda activate myenv 

#call the julia script to generate the load_sim() data using the slurm array task id as the seed
#copy the master Project.toml into a temporary folder with the current process ID pass the ID as an argument
#to the julia code call below and have it activate its own Project.toml
julia /resnick/groups/wugroup/zblood/cmb_lensing/generate_julia_data_bash.jl --map_size $1 --seed $2
#once the julia code is finished running, delete the temporary directory...