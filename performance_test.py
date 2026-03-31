#imports
from imports import *
from functions import *
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--map_size", type = int)
parser.add_argument("--seed", type = int)
parser.add_argument("--trial", type = int)
args = parser.parse_args()

#initialize the data set object and load the julia computed data for the specifc f and phi combo
data_set_folder = os.getcwd() + f"/julia_generated_data/map_size_{args.map_size}_seed_{args.seed}/"
data_set = load_data_set(data_set_folder)

#run the algorithm once initially to get the uncached time
start_time = time.time()
fP, phiP, _ = run_gradient_descent_v3(data_set, 30)
end_time = time.time()
total_time = end_time - start_time

#store the uncached time for this (f, phi) combo in its own proper directory
os.makedirs(os.getcwd() + f"/performance_results/python_results/map_size_{args.map_size}_seed_{args.seed}/uncached_times", exist_ok = True)
np.savetxt(os.getcwd() + f"/performance_results/python_results/map_size_{args.map_size}_seed_{args.seed}/uncached_times/uncached_time_{args.trial}.txt", \
           np.array([total_time]))

#now run the algorithm a second time to get the cache time
start_time = time.time()
fP, phiP, _ = run_gradient_descent_v3(data_set, 30) #engine is JIT-ted, constant step size alpha = 1
end_time = time.time()
total_time = end_time - start_time

#store the cached time for this (f, phi) combo in its own proper directory
os.makedirs(os.getcwd() + f"/performance_results/python_results/map_size_{args.map_size}_seed_{args.seed}/cached_times", exist_ok = True)
np.savetxt(os.getcwd() + f"/performance_results/python_results/map_size_{args.map_size}_seed_{args.seed}/cached_times/cached_time_{args.trial}.txt", \
           np.array([total_time]))

#store the learned / estimated f and phi from python to compare with what julia found
os.makedirs(os.getcwd() + f"/performance_results/python_results/map_size_{args.map_size}_seed_{args.seed}/learned_fields/temperature", exist_ok = True)
os.makedirs(os.getcwd() + f"/performance_results/python_results/map_size_{args.map_size}_seed_{args.seed}/learned_fields/lensing_potential", exist_ok = True)
np.savez(os.getcwd() + f"/performance_results/python_results/map_size_{args.map_size}_seed_{args.seed}/learned_fields/temperature/fP_{args.trial}.npz", fP)
np.savez(os.getcwd() + f"/performance_results/python_results/map_size_{args.map_size}_seed_{args.seed}/learned_fields/lensing_potential/phiP_{args.trial}.npz", phiP)
