#!/bin/bash

#SBATCH --time=00:10:00 #wall-time / max run time before termination in the format hh:mm:ss
#SBATCH --nodes=1 #i.e. the number of machines to run on
#SBATCH --ntasks=4 #number of processor cores / tasks
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH --mail-user=zblood@caltech.edu #mail updates to this address
#SBATCH --mail-type=FAIL #mail updates on begin, end, or failure

#use the conda initializer in your own bashrc
source /home/zblood/.bashrc 
conda init
#activate your own specific conda
conda activate myenv
#call the julia performance timing script for a specific (f, phi) combo and a specific trial number
julia /resnick/groups/wugroup/zblood/cmb_lensing_reorganized/julia_package_activation.jl