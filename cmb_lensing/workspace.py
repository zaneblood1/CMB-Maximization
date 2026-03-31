#imports
from imports import *
from functions import *
from simulate import *
import time

#initialize the data set object
data_set = {}

data, lensed_temp, unlensed_temp, phi, cn, cf, cfl, cphi, m, b, d_matrix, g, nphi = \
        load_sim(N = 256, theta_pix = 2, uk_arcmin_t = 10, seed = None, lmax = 17000)
data = jfft.rfft2(data)

np.savez(os.getcwd() + "/julia_maximization_debug/" + "unlensed_temp.npz", unlensed_temp)
data_set["unlensed_temp"] = unlensed_temp
np.savez(os.getcwd() + "/julia_maximization_debug/" + "phi.npz", phi)
data_set["phi"] = phi
np.savez(os.getcwd() + "/julia_maximization_debug/" + "data.npz", data)
data_set["data"] = data
np.savez(os.getcwd() + "/julia_maximization_debug/" + "cn.npz", cn)
data_set["cn"] = cn
np.savez(os.getcwd() + "/julia_maximization_debug/" + "cf.npz", cf)
data_set["cf"] = cf
np.savez(os.getcwd() + "/julia_maximization_debug/" + "cphi.npz", cphi)
data_set["cphi"] = cphi
np.savez(os.getcwd() + "/julia_maximization_debug/" + "m.npz", m)
data_set["m"] = m
np.savez(os.getcwd() + "/julia_maximization_debug/" + "b.npz", b)
data_set["b"] = b
np.savez(os.getcwd() + "/julia_maximization_debug/" + "d_matrix.npz", d_matrix)
data_set["d_matrix"] = d_matrix
np.savez(os.getcwd() + "/julia_maximization_debug/" + "g.npz", g)
data_set["g"] = g
np.savez(os.getcwd() + "/julia_maximization_debug/" + "nphi.npz", nphi)
data_set["nphi"] = nphi

fourier_weights = 2 * jnp.ones(129,)
fourier_weights = fourier_weights.at[0].set(1)
fourier_weights = fourier_weights.at[-1].set(1)
np.savez(os.getcwd() + "/julia_maximization_debug/" + "fourier_weights.npz", fourier_weights)
data_set["fourier_weights"] = fourier_weights

pix_width = jnp.deg2rad(2/ARCMIN_PER_DEGREE)
with open(os.getcwd() + "/julia_maximization_debug/" + "pix_width.txt", "w") as f:
    f.write(str(pix_width))
data_set["pix_width"] = pix_width

num_pixels = 256**2
with open(os.getcwd() + "/julia_maximization_debug/" + "num_pixels.txt", "w") as f:
    f.write(str(num_pixels))
data_set["num_pixels"] = num_pixels

start_time = time.time()
f_predict, phi_predict, _ = run_gradient_descent_v3(data_set, num_steps = 30, save_animation = True)
end_time = time.time()
total_time = end_time - start_time
print("Done!")

#NOTE F Python v.s. Python Sim Percent Diff ~ 0.10443548, Phi Python v.s. Python Sim Percent Diff ~ 0.450749524
#     F Julia v.s. Python Sim Percent Diff ~ 0.10062930, Phi Julia v.s. Python Sim Percent Diff ~ 0.4320026
#     F Julia v.s. F Python Percent Diff ~ 0.0177680890624, Phi Julia v.s. Phi Python Percent Diff ~ 0.140196 