#imports
from imports import *
from functions import *
from simulate import *
import time
#from jax import config
#config.update("jax_log_compiles", True)
#config.update("jax_disable_jit", True)

file_path_to_folder = "/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/"
i_data_set = load_data_set(file_path_to_folder)
eb_data_set = load_eb_data_set(file_path_to_folder)
ieb_data_set = load_ieb_data_set(file_path_to_folder)

start_time = time.time()
i_predict_I, phi_predict_I, _ = run_gradient_descent_v3(i_data_set, num_steps = 30)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time for intensity only = {total_time}") #NOTE about 2 min 50

start_time = time.time()
e_predict_P, b_predict_P, phi_predict_P, _, _ = run_gradient_descent_eb(eb_data_set, num_steps = 30)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time for polarization only = {total_time}") #NOTE about 4 min

start_time = time.time()
i_predict_IP, e_predict_IP, b_predict_IP, phi_predict_IP = run_gradient_descent_ieb(ieb_data_set, num_steps = 30)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time for intensity AND polarization = {total_time}")

#TODO next steps are to compare accuracy of I, P, and IP all together... aiming for something in the 1e-4 to 1e-5 range
#and also to compute the cross correlation...

i_ground_I = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/fJ_T_I.npz")
phi_ground_I = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/phiJ_I.npz")
i_sim_I = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/i_field_I.npz")
phi_sim_I = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/phi_I.npz")

e_ground_P = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/fJ_E_P.npz")
b_ground_P = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/fJ_B_P.npz")
phi_ground_P = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/phiJ_P.npz")
e_sim_P = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/e_field_P.npz")
b_sim_P = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/b_field_P.npz")
phi_sim_P = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/phi_P.npz")

i_ground_IP = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/fJ_T_IP.npz")
e_ground_IP = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/fJ_E_IP.npz")
b_ground_IP = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/fJ_B_IP.npz")
phi_ground_IP = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/phiJ_IP.npz")
i_sim_IP = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/i_field_IP.npz")
e_sim_IP = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/e_field_IP.npz")
b_sim_IP = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/b_field_IP.npz")
phi_sim_IP = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/phi_IP.npz")

print("T Julia MLE v.s. Python MLE Fractional Diff (I Only) = " + str(percent_diff_2d(jfft.irfft2(i_ground_I), jfft.irfft2(i_predict_I))))
print("Phi Julia MLE v.s. Python MLE Fractional Diff (I Only) = " + str(percent_diff_2d(jfft.irfft2(phi_ground_I), jfft.irfft2(phi_predict_I))))
print("T Julia MLE v.s. Julia Sim Fractional Diff (I Only) = " + str(percent_diff_2d(jfft.irfft2(i_sim_I), jfft.irfft2(i_ground_I))))
print("Phi Julia MLE v.s. Julia Sim Fractional Diff (I Only) = " + str(percent_diff_2d(jfft.irfft2(phi_sim_I), jfft.irfft2(phi_ground_I))))
print("T Python MLE v.s. Julia Sim Fractional Diff (I Only) = " + str(percent_diff_2d(jfft.irfft2(i_sim_I), jfft.irfft2(i_predict_I))))
print("Phi Python MLE v.s. Julia Sim Fractional Diff (I Only) = " + str(percent_diff_2d(jfft.irfft2(phi_sim_I), jfft.irfft2(phi_predict_I))))

print("E Julia MLE v.s. Python MLE Fractional Diff (P only) = " + str(percent_diff_2d(jfft.irfft2(e_ground_P), jfft.irfft2(e_predict_P))))
print("B Julia MLE v.s. Python MLE Fractional Diff (P only) = " + str(percent_diff_2d(jfft.irfft2(b_ground_P), jfft.irfft2(b_predict_P))))
print("Phi Julia MLE v.s. Python MLE Fractional Diff (P only) = " + str(percent_diff_2d(jfft.irfft2(phi_ground_P), jfft.irfft2(phi_predict_P))))
print("E Julia MLE v.s. Julia Sim Fractional Diff (P only) = " + str(percent_diff_2d(jfft.irfft2(e_sim_P), jfft.irfft2(e_ground_P))))
print("B Julia MLE v.s. Julia Sim Fractional Diff (P only) = " + str(percent_diff_2d(jfft.irfft2(b_sim_P), jfft.irfft2(b_ground_P))))
print("Phi Julia MLE v.s. Julia Sim Fractional Diff (P only) = " + str(percent_diff_2d(jfft.irfft2(phi_sim_P), jfft.irfft2(phi_ground_P))))
print("E Python MLE v.s. Julia Sim Fractional Diff (P only) = " + str(percent_diff_2d(jfft.irfft2(e_sim_P), jfft.irfft2(e_predict_P))))
print("B Python MLE v.s. Julia Sim Fractional Diff (P only) = " + str(percent_diff_2d(jfft.irfft2(b_sim_P), jfft.irfft2(b_predict_P))))
print("Phi Python MLE v.s. Julia Sim Fractional Diff (P only) = " + str(percent_diff_2d(jfft.irfft2(phi_sim_P), jfft.irfft2(phi_predict_P))))

print("T Julia MLE v.s. Python MLE Fractional Diff (IP Both) = " + str(percent_diff_2d(jfft.irfft2(i_ground_IP), jfft.irfft2(i_predict_IP))))
print("E Julia MLE v.s. Python MLE Fractional Diff (IP Both) = " + str(percent_diff_2d(jfft.irfft2(e_ground_IP), jfft.irfft2(e_predict_IP))))
print("B Julia MLE v.s. Python MLE Fractional Diff (IP Both) = " + str(percent_diff_2d(jfft.irfft2(b_ground_IP), jfft.irfft2(b_predict_IP))))
print("Phi Julia MLE v.s. Python MLE Fractional Diff (IP Both) = " + str(percent_diff_2d(jfft.irfft2(phi_ground_IP), jfft.irfft2(phi_predict_IP))))
print("T Julia MLE v.s. Julia Sim Fractional Diff (IP Both) = " + str(percent_diff_2d(jfft.irfft2(i_sim_IP), jfft.irfft2(i_ground_IP))))
print("E Julia MLE v.s. Julia Sim Fractional Diff (IP Both) = " + str(percent_diff_2d(jfft.irfft2(e_sim_IP), jfft.irfft2(e_ground_IP))))
print("B Julia MLE v.s. Julia Sim Fractional Diff (IP Both) = " + str(percent_diff_2d(jfft.irfft2(b_sim_IP), jfft.irfft2(b_ground_IP))))
print("Phi Julia MLE v.s. Julia Sim Fractional Diff (IP Both) = " + str(percent_diff_2d(jfft.irfft2(phi_sim_IP), jfft.irfft2(phi_ground_IP))))
print("T Python MLE v.s. Julia Sim Fractional Diff (IP Both) = " + str(percent_diff_2d(jfft.irfft2(i_sim_IP), jfft.irfft2(i_predict_IP))))
print("E Python MLE v.s. Julia Sim Fractional Diff (IP Both) = " + str(percent_diff_2d(jfft.irfft2(e_sim_IP), jfft.irfft2(e_predict_IP))))
print("B Python MLE v.s. Julia Sim Fractional Diff (IP Both) = " + str(percent_diff_2d(jfft.irfft2(b_sim_IP), jfft.irfft2(b_predict_IP))))
print("Phi Python MLE v.s. Julia Sim Fractional Diff (IP Both) = " + str(percent_diff_2d(jfft.irfft2(phi_sim_IP), jfft.irfft2(phi_predict_IP))))

print("Done!")