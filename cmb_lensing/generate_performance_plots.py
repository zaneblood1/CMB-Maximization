from imports import *
from functions import *
import shutil

#remove the results directory if already there and add a clean new one
#otherwise, just add a clean new dir
results_path = "/home/zane-blood/Desktop/cmb_lensing_reorganized/performance_results"
if os.path.exists(results_path):
    shutil.rmtree(results_path)
os.makedirs(results_path, exist_ok = True)
os.makedirs(results_path + "/cached_time_plots/", exist_ok = True)
os.makedirs(results_path + "/cached_time_plots/map_size_128/", exist_ok = True)
os.makedirs(results_path + "/cached_time_plots/map_size_256/", exist_ok = True)
os.makedirs(results_path + "/cached_time_plots/map_size_512/", exist_ok = True)
os.makedirs(results_path + "/cached_time_plots/map_size_1024/", exist_ok = True)

os.makedirs(results_path + "/uncached_time_plots/", exist_ok = True)
os.makedirs(results_path + "/uncached_time_plots/map_size_128/", exist_ok = True)
os.makedirs(results_path + "/uncached_time_plots/map_size_256/", exist_ok = True)
os.makedirs(results_path + "/uncached_time_plots/map_size_512/", exist_ok = True)
os.makedirs(results_path + "/uncached_time_plots/map_size_1024/", exist_ok = True)

os.makedirs(results_path + "/julia_pred_vs_julia_sim_diff/", exist_ok = True)
os.makedirs(results_path + "/julia_pred_vs_julia_sim_diff/map_size_128/", exist_ok = True)
os.makedirs(results_path + "/julia_pred_vs_julia_sim_diff/map_size_256/", exist_ok = True)
os.makedirs(results_path + "/julia_pred_vs_julia_sim_diff/map_size_512/", exist_ok = True)
os.makedirs(results_path + "/julia_pred_vs_julia_sim_diff/map_size_1024/", exist_ok = True)

os.makedirs(results_path + "/python_pred_vs_julia_sim_diff/", exist_ok = True)
os.makedirs(results_path + "/python_pred_vs_julia_sim_diff/map_size_128/", exist_ok = True)
os.makedirs(results_path + "/python_pred_vs_julia_sim_diff/map_size_256/", exist_ok = True)
os.makedirs(results_path + "/python_pred_vs_julia_sim_diff/map_size_512/", exist_ok = True)
os.makedirs(results_path + "/python_pred_vs_julia_sim_diff/map_size_1024/", exist_ok = True)

os.makedirs(results_path + "/julia_vs_python_diff/", exist_ok = True)
os.makedirs(results_path + "/julia_vs_python_diff/map_size_128/", exist_ok = True)
os.makedirs(results_path + "/julia_vs_python_diff/map_size_256/", exist_ok = True)
os.makedirs(results_path + "/julia_vs_python_diff/map_size_512/", exist_ok = True)
os.makedirs(results_path + "/julia_vs_python_diff/map_size_1024/", exist_ok = True)


num_seeds = 10
num_trials = 10
trials_list = list(range(1, num_trials + 1))
map_sizes = [128, 256, 512, 1024]

uncached_julia_avg_times = np.array([])
uncached_julia_stds = np.array([])
uncached_python_avg_times = np.array([])
uncached_python_stds = np.array([])
cached_julia_avg_times = np.array([])
cached_julia_stds = np.array([])
cached_python_avg_times = np.array([])
cached_python_stds = np.array([])
phi_diffs_jp_avg = np.array([])
phi_diffs_js_avg = np.array([])
phi_diffs_ps_avg = np.array([])
f_diffs_jp_avg = np.array([])
f_diffs_js_avg = np.array([])
f_diffs_ps_avg = np.array([])
phi_diffs_jp_std = np.array([])
phi_diffs_js_std = np.array([])
phi_diffs_ps_std = np.array([])
f_diffs_jp_std = np.array([])
f_diffs_js_std = np.array([])
f_diffs_ps_std = np.array([])

for map_size in map_sizes:

    all_uncached_julia_times = np.array([])
    all_uncached_python_times = np.array([])
    all_cached_julia_times = np.array([])
    all_cached_python_times = np.array([])
    all_phi_diffs_jp = np.array([])
    all_phi_diffs_js = np.array([])
    all_phi_diffs_ps = np.array([])
    all_f_diffs_jp = np.array([])
    all_f_diffs_js = np.array([])
    all_f_diffs_ps = np.array([])

    for seed in range(1, num_seeds + 1):

        uncached_julia_times = np.array([])
        uncached_python_times = np.array([])
        cached_julia_times = np.array([])
        cached_python_times = np.array([])
        phi_diffs_jp = np.array([])
        phi_diffs_js = np.array([])
        phi_diffs_ps = np.array([])
        f_diffs_jp = np.array([])
        f_diffs_js = np.array([])
        f_diffs_ps = np.array([])

        for trial in range(1, num_trials + 1):

            uncached_julia_path = f"/home/zane-blood/Desktop/performance_results/julia_results/map_size_{map_size}_seed_{seed}/uncached_times/uncached_time_{trial}.txt"
            cached_julia_path = f"/home/zane-blood/Desktop/performance_results/julia_results/map_size_{map_size}_seed_{seed}/cached_times/cached_time_{trial}.txt"
            uncached_python_path = f"/home/zane-blood/Desktop/performance_results/python_results/map_size_{map_size}_seed_{seed}/uncached_times/uncached_time_{trial}.txt"
            cached_python_path = f"/home/zane-blood/Desktop/performance_results/python_results/map_size_{map_size}_seed_{seed}/cached_times/cached_time_{trial}.txt"
            
            if os.path.exists(uncached_julia_path):
                with open(uncached_julia_path, "r") as f:
                    uncached_julia_time = float(f.read())
                    uncached_julia_times = np.append(uncached_julia_times, uncached_julia_time)
                    all_uncached_julia_times = np.append(all_uncached_julia_times, uncached_julia_time)
            if os.path.exists(cached_julia_path):
                with open(cached_julia_path, "r") as f:
                    cached_julia_time = float(f.read())
                    cached_julia_times = np.append(cached_julia_times, cached_julia_time)
                    all_cached_julia_times = np.append(all_cached_julia_times, cached_julia_time)
            if os.path.exists(uncached_python_path):
                with open(uncached_python_path, "r") as f:
                    uncached_python_time = float(f.read())
                    uncached_python_times = np.append(uncached_python_times, uncached_python_time)
                    all_uncached_python_times = np.append(all_uncached_python_times, uncached_python_time)
            if os.path.exists(cached_python_path):
                with open(cached_python_path, "r") as f:
                    cached_python_time = float(f.read())
                    cached_python_times = np.append(cached_python_times, cached_python_time)
                    all_cached_python_times = np.append(all_cached_python_times, cached_python_time)

            #load the true (f, phi) pair for each map and seed combo from the julia_generated_data folder
            #also load the julia predictions and the python predictions
            #using this data, make plots of julia diffed against python and julia diffed against ground truth simulation 
            f_sim_path = f"/home/zane-blood/Desktop/julia_generated_data/map_size_{map_size}_seed_{seed}/f_array.npz"
            all_exist = True
            if os.path.exists(f_sim_path):
                f_sim = jfft.irfft2(precision_load(f_sim_path))
            else:
                all_exist = False

            phi_sim_path = f"/home/zane-blood/Desktop/julia_generated_data/map_size_{map_size}_seed_{seed}/phi_array.npz" 
            if os.path.exists(phi_sim_path):
                phi_sim = jfft.irfft2(precision_load(phi_sim_path))
            else:
                all_exist = False

            f_python_predict_path = f"/home/zane-blood/Desktop/performance_results/python_results/map_size_{map_size}_seed_{seed}/learned_fields/temperature/fP_{trial}.npz"
            if os.path.exists(f_python_predict_path):
                f_python_predict = jfft.irfft2(np.load(f_python_predict_path)["arr_0"])
            else:
                all_exist = False

            phi_python_predict_path = f"/home/zane-blood/Desktop/performance_results/python_results/map_size_{map_size}_seed_{seed}/learned_fields/lensing_potential/phiP_{trial}.npz" 
            if os.path.exists(phi_python_predict_path):
                phi_python_predict = jfft.irfft2(np.load(phi_python_predict_path)["arr_0"])   
            else:
                all_exist = False

            f_julia_predict_path = f"/home/zane-blood/Desktop/performance_results/julia_results/map_size_{map_size}_seed_{seed}/learned_fields/temperature/fJ_{trial}.npz"
            if os.path.exists(f_julia_predict_path):
                f_julia_predict = jfft.irfft2(precision_load(f_julia_predict_path).T)
            else:
                all_exist = False
                
            phi_julia_predict_path = f"/home/zane-blood/Desktop/performance_results/julia_results/map_size_{map_size}_seed_{seed}/learned_fields/lensing_potential/phiJ_{trial}.npz" 
            if os.path.exists(phi_julia_predict_path):
                phi_julia_predict = (precision_load(phi_julia_predict_path).T)
            else:
                all_exist = False
            
            if all_exist:
                phi_diffs_jp = np.append(phi_diffs_jp, percent_diff_2d(phi_julia_predict, phi_python_predict))
                phi_diffs_js = np.append(phi_diffs_js, percent_diff_2d(phi_sim, phi_julia_predict))
                phi_diffs_ps = np.append(phi_diffs_ps, percent_diff_2d(phi_sim, phi_python_predict))
                f_diffs_jp = np.append(f_diffs_jp, percent_diff_2d(f_julia_predict, f_python_predict))
                f_diffs_js = np.append(f_diffs_js, percent_diff_2d(f_sim, f_julia_predict))
                f_diffs_ps = np.append(f_diffs_ps, percent_diff_2d(f_sim, f_python_predict))
                all_phi_diffs_jp = np.append(all_phi_diffs_jp, percent_diff_2d(phi_julia_predict, phi_python_predict))
                all_phi_diffs_js = np.append(all_phi_diffs_js, percent_diff_2d(phi_sim, phi_julia_predict))
                all_phi_diffs_ps = np.append(all_phi_diffs_ps, percent_diff_2d(phi_sim, phi_python_predict))
                all_f_diffs_jp = np.append(all_f_diffs_jp, percent_diff_2d(f_julia_predict, f_python_predict))
                all_f_diffs_js = np.append(all_f_diffs_js, percent_diff_2d(f_sim, f_julia_predict))
                all_f_diffs_ps = np.append(all_f_diffs_ps, percent_diff_2d(f_sim, f_python_predict))

        plt.figure()
        plt.plot(list(range(1, len(uncached_julia_times) + 1)), uncached_julia_times, color = "blue", label = "Julia")
        plt.plot(list(range(1, len(uncached_python_times) + 1)), uncached_python_times, color = "red", label = "Python")
        plt.legend()
        plt.xlabel("Trial Number")
        plt.ylabel("Duration in Seconds")
        plt.title(f"Uncached Time Comparison Map Size {map_size}, Seed {seed}")
        plt.savefig(os.getcwd() + f"/performance_results/uncached_time_plots/map_size_{map_size}/Uncached Time Comparison Map Size {map_size}, Seed {seed}")
        plt.close()

        plt.figure()
        plt.plot(list(range(1, len(cached_julia_times) + 1)), cached_julia_times, color = "blue", label = "Julia")
        plt.plot(list(range(1, len(cached_python_times) + 1)), cached_python_times, color = "red", label = "Python")
        plt.legend()
        plt.xlabel("Trial Number")
        plt.ylabel("Duration in Seconds")
        plt.title(f"Cached Time Comparison Map Size {map_size}, Seed {seed}")
        plt.savefig(os.getcwd() + f"/performance_results/cached_time_plots/map_size_{map_size}/Cached Time Comparison Map Size {map_size}, Seed {seed}")
        plt.close()

        plt.figure()
        plt.plot(list(range(1, len(phi_diffs_jp) + 1)), phi_diffs_jp, color = "blue", label = "Phi Diff")
        plt.plot(list(range(1, len(f_diffs_jp) + 1)), f_diffs_jp, color = "red", label = "F Diff")
        plt.legend()
        plt.xlabel("Trial Number")
        plt.ylabel("Fractional Difference")
        plt.title(f"Julia v.s. Python Fractional Difference Map Size {map_size}, Seed {seed}")
        plt.savefig(os.getcwd() + f"/performance_results/julia_vs_python_diff/map_size_{map_size}/Julia vs Python Fractional Difference Map Size {map_size}, Seed {seed}")
        plt.close()

        plt.figure()
        plt.plot(list(range(1, len(phi_diffs_js) + 1)), phi_diffs_js, color = "blue", label = "Phi Diff")
        plt.plot(list(range(1, len(f_diffs_js) + 1)), f_diffs_js, color = "red", label = "F Diff")
        plt.legend()
        plt.xlabel("Trial Number")
        plt.ylabel("Fractional Difference")
        plt.title(f"Julia v.s. Simulation Fractional Difference Map Size {map_size}, Seed {seed}")
        plt.savefig(os.getcwd() + f"/performance_results/julia_pred_vs_julia_sim_diff/map_size_{map_size}/Julia vs Simulation Fractional Difference Map Size {map_size}, Seed {seed}")
        plt.close()

        plt.figure()
        plt.plot(list(range(1, len(phi_diffs_ps) + 1)), phi_diffs_ps, color = "blue", label = "Phi Diff")
        plt.plot(list(range(1, len(f_diffs_ps) + 1)), f_diffs_ps, color = "red", label = "F Diff")
        plt.legend()
        plt.xlabel("Trial Number")
        plt.ylabel("Fractional Difference")
        plt.title(f"Python v.s. Simulation Fractional Difference Map Size {map_size}, Seed {seed}")
        plt.savefig(os.getcwd() + f"/performance_results/python_pred_vs_julia_sim_diff/map_size_{map_size}/Python vs Simulation Fractional Difference Map Size {map_size}, Seed {seed}")
        plt.close()

    uncached_julia_avg_times = np.append(uncached_julia_avg_times, np.mean(all_uncached_julia_times))
    uncached_julia_stds = np.append(uncached_julia_stds, np.std(all_uncached_julia_times))
    uncached_python_avg_times = np.append(uncached_python_avg_times, np.mean(all_uncached_python_times))
    uncached_python_stds = np.append(uncached_python_stds, np.std(all_uncached_python_times))
    cached_julia_avg_times = np.append(cached_julia_avg_times, np.mean(all_cached_julia_times))
    cached_julia_stds = np.append(cached_julia_stds, np.std(all_cached_julia_times))
    cached_python_avg_times = np.append(cached_python_avg_times, np.mean(all_cached_python_times))
    cached_python_stds = np.append(cached_python_stds, np.std(all_cached_python_times))

    phi_diffs_jp_avg = np.append(phi_diffs_jp_avg, np.mean(all_phi_diffs_jp))
    phi_diffs_js_avg = np.append(phi_diffs_js_avg, np.mean(all_phi_diffs_js))
    phi_diffs_ps_avg = np.append(phi_diffs_ps_avg, np.mean(all_phi_diffs_ps))
    f_diffs_jp_avg = np.append(f_diffs_jp_avg, np.mean(all_f_diffs_jp))
    f_diffs_js_avg = np.append(f_diffs_js_avg, np.mean(all_f_diffs_js))
    f_diffs_ps_avg = np.append(f_diffs_ps_avg, np.mean(all_f_diffs_ps))
    phi_diffs_jp_std = np.append(phi_diffs_jp_std, np.std(all_phi_diffs_jp))
    phi_diffs_js_std = np.append(phi_diffs_js_std, np.std(all_phi_diffs_js))
    phi_diffs_ps_std = np.append(phi_diffs_ps_std, np.std(all_phi_diffs_ps))
    f_diffs_jp_std = np.append(f_diffs_jp_std, np.std(all_f_diffs_jp))
    f_diffs_js_std = np.append(f_diffs_js_std, np.std(all_f_diffs_js))
    f_diffs_ps_std = np.append(f_diffs_ps_std, np.std(all_f_diffs_ps))

plt.figure()
plt.errorbar(map_sizes, uncached_julia_avg_times, yerr = uncached_julia_stds, color = "blue", label = "Julia", marker = "o", markersize = 5)
plt.errorbar(map_sizes, uncached_python_avg_times, yerr = uncached_python_stds, color = "red", label = "Python", marker = "o", markersize = 5)
plt.legend()
plt.xlabel("Map Size")
plt.ylabel("Average Duration in Seconds")
plt.title(f"Average Uncached Time Comparison")
plt.savefig(os.getcwd() + f"/performance_results/uncached_time_plots/Average Uncached Time Comparison")
plt.close()

plt.figure()
plt.errorbar(map_sizes, cached_julia_avg_times, yerr = cached_julia_stds, color = "blue", label = "Julia", marker = "o", markersize = 5)
plt.errorbar(map_sizes, cached_python_avg_times, yerr = cached_python_stds, color = "red", label = "Python", marker = "o", markersize = 5)
plt.legend()
plt.xlabel("Map Size")
plt.ylabel("Average Duration in Seconds")
plt.title(f"Average Cached Time Comparison")
plt.savefig(os.getcwd() + f"/performance_results/cached_time_plots/Average Cached Time Comparison")
plt.close()

plt.figure()
plt.plot(map_sizes, cached_julia_avg_times - uncached_julia_avg_times, color = "blue", label = "Julia", marker = "o", markersize = 10)
plt.plot(map_sizes, cached_python_avg_times - uncached_python_avg_times, color = "red", label = "Python", marker = "o", markersize = 10)
plt.legend()
plt.xlabel("Map Size")
plt.ylabel("Average Duration in Seconds")
plt.title(f"Average Uncached Minus Cached Time Difference")
plt.savefig(os.getcwd() + f"/performance_results/Cached vs Uncached Time Difference")
plt.close()

plt.figure()
plt.errorbar(map_sizes, phi_diffs_jp_avg, yerr = phi_diffs_jp_std, color = "blue", label = "Phi", marker = "o", markersize = 5)
plt.errorbar(map_sizes, f_diffs_jp_avg, yerr = f_diffs_jp_std, color = "red", label = "F", marker = "o", markersize = 5)
plt.legend()
plt.xlabel("Map Size")
plt.ylabel("Average Fractional Difference")
plt.title(f"Average Fractional Difference Julia v.s. Python")
plt.savefig(os.getcwd() + f"/performance_results/julia_vs_python_diff/Average Fractional Difference Julia vs Python")
plt.close()

plt.figure()
plt.errorbar(map_sizes, phi_diffs_js_avg, yerr = phi_diffs_js_std, color = "blue", label = "Phi", marker = "o", markersize = 5)
plt.errorbar(map_sizes, f_diffs_js_avg, yerr = f_diffs_js_std, color = "red", label = "F", marker = "o", markersize = 5)
plt.legend()
plt.xlabel("Map Size")
plt.ylabel("Average Fractional Difference")
plt.title(f"Average Fractional Difference Julia v.s. Simulation")
plt.savefig(os.getcwd() + f"/performance_results/julia_pred_vs_julia_sim_diff/Average Fractional Difference Julia vs Simulation")
plt.close()

plt.figure()
plt.errorbar(map_sizes, phi_diffs_ps_avg, yerr = phi_diffs_ps_std, color = "blue", label = "Phi", marker = "o", markersize = 5)
plt.errorbar(map_sizes, f_diffs_ps_avg, yerr = f_diffs_ps_std, color = "red", label = "F", marker = "o", markersize = 5)
plt.legend()
plt.xlabel("Map Size")
plt.ylabel("Average Fractional Difference")
plt.title(f"Average Fractional Difference Python v.s. Simulation")
plt.savefig(os.getcwd() + f"/performance_results/python_pred_vs_julia_sim_diff/Average Fractional Difference Python vs Simulation")
plt.close()

plt.figure()
plt.errorbar(map_sizes, phi_diffs_ps_avg - phi_diffs_js_avg, color = "blue", label = "Phi", marker = "o", markersize = 5)
plt.errorbar(map_sizes, f_diffs_ps_avg - f_diffs_js_avg, color = "red", label = "F", marker = "o", markersize = 5)
plt.legend()
plt.xlabel("Map Size")
plt.ylabel("Difference in Fractional Difference")
plt.title(f"Difference in Simulation Fractional Difference: Python Minus Julia")
plt.savefig(os.getcwd() + f"/performance_results/Difference in Simulation Fractional Difference: Python Minus Julia")
plt.close()