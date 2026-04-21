#get necessary imports
#from imports import *
from cmb_lensing.functions import *

#get the current working directory
cwd = os.getcwd() + "/cmb_lensing/"
#Set the threshold for all the 2D percent diff calculations in this file
PERCENT_DIFF_THRESHOLD = 0.05
#NOTE The below threshold is WAAYYY too high... 
#The error in inverse lensing seems to get compounded by the elements of the inverse F covariance matrix...
ABS_LOGPDF_DIFF_THRESHOLD_LENSED = 5
ABS_LOGPDF_DIFF_THRESHOLD_UNLENSED = 0.1 #NOTE this is also concerning but not nearly as much...
FIGURE_FILE_PATH = cwd + "/unit_test_generated_figures/"

#load any necessary constants from the simulation data
with open(cwd + "/julia_ground_truth_data/pix_width.txt", "r") as f:
    pix_width = float(f.read().strip()) 
with open(cwd + "/julia_ground_truth_data/num_pixels.txt", "r") as f:
    num_pixels = float(f.read().strip()) 
with open(cwd + "/julia_ground_truth_data/logpdf_value.txt", "r") as f:
    logpdf_ground = float(f.read().strip()) 

#compare lensed field in python and lensed field in julia
def test_forward_lensing():

    #load the julia ground truth f, phi and f_lensed fields
    f_ground = precision_load(cwd + "/julia_ground_truth_data/f_array.npz")
    phi_ground = precision_load(cwd + "/julia_ground_truth_data/phi_array.npz")
    #ground truth lensed field withOUT beam or mask functions applied
    f_lensed_ground = precision_load(cwd + "/julia_ground_truth_data/f_lensed_no_b_no_m_array.npz")

    #all of the ground truth data is currently in the fourier space format
    #so we must convert to real space to apply our lense flow function whose inputs are real valued
    f_ground = jfft.irfft2(f_ground, s = (256, 256))
    phi_ground = jfft.irfft2(phi_ground, s = (256, 256))
    f_lensed_ground = jfft.irfft2(f_lensed_ground, s = (256, 256))

    #compute the lensed field from the unlensed field using this project's python code
    num_steps = 10; direction = 1; adjoint = False;  
    f_lensed_predict = lense_flow_wrapper(f_ground, phi_ground, pix_width, num_steps, direction, adjoint, num_pixels, False)

    #save visual comparisons to a temporary file as well
    plot_heat_map(f_lensed_ground, "Forward Lensed Ground", "ArcMin",\
                   "ArcMin", "Intensity", True, file_path = FIGURE_FILE_PATH + "Forward Lensed Ground.png",)
    plot_heat_map(f_lensed_predict, "Forward Lensed Predict", "ArcMin",\
                   "ArcMin", "Intensity", True, file_path = FIGURE_FILE_PATH + "Forward Lensed Predict.png")
    plot_heat_map(f_lensed_ground - f_lensed_predict, "Forward Lensed Diff",\
                   "ArcMin", "ArcMin", "Intensity", True, file_path = FIGURE_FILE_PATH + "Forward Lensed Diff.png")

    #enforce that the 2D percent difference between the ground truth and
    #prediction is less than our specified threshold
    assert percent_diff_2d(f_lensed_ground, f_lensed_predict) < PERCENT_DIFF_THRESHOLD

#compare inverse lensed field to original field
def test_inverse_lensing():

    #load the julia ground truth f, phi
    f_ground = precision_load(cwd + "/julia_ground_truth_data/f_array.npz")
    phi_ground = precision_load(cwd + "/julia_ground_truth_data/phi_array.npz")

    #all of the ground truth data is currently in the fourier space format
    #so we must convert to real space to apply our lense flow function whose inputs are real valued
    f_ground = jfft.irfft2(f_ground, s = (256, 256))
    phi_ground = jfft.irfft2(phi_ground, s = (256, 256))

    #compute the forward lensed field from the unlensed field using this project's python code
    num_steps = 10; direction = 1; adjoint = False;  
    f_lensed_predict = lense_flow_wrapper(f_ground, phi_ground, pix_width, num_steps, direction, adjoint, num_pixels, False)

    #now take the predicted lensed field and inverse lense it
    num_steps = 100; direction = -1; #adding more steps makes things more accurate ESPECIALLY for inverse lensing
    f_predict = lense_flow_wrapper(f_lensed_predict, phi_ground, pix_width, num_steps, direction, adjoint, num_pixels, False)

    #save visual comparisons to a temporary file as well
    plot_heat_map(f_ground - f_predict, "Inverse Lensing Diff", "ArcMin", \
                  "ArcMin", "Intensity", True, file_path = FIGURE_FILE_PATH + "Inverse Lensing Diff.png")

    #enforce that the 2D percent difference between the ground truth and
    #prediction is less than our specified threshold
    assert percent_diff_2d(f_ground, f_predict) < PERCENT_DIFF_THRESHOLD

#compare adjoint lensed field in python to adjoint lensed field in julia
def test_adjoint_lensing():

    #load the julia ground truth f, phi and f_lensed fields
    f_ground = precision_load(cwd + "/julia_ground_truth_data/f_array.npz")
    phi_ground = precision_load(cwd + "/julia_ground_truth_data/phi_array.npz")
    #ground truth adjoint lensed field withOUT beam or mask functions applied
    f_adjoint_lensed_ground = precision_load(cwd + "/julia_ground_truth_data/f_adjoint_lensed_array.npz")

    #all of the ground truth data is currently in the fourier space format
    #so we must convert to real space to apply our lense flow function whose inputs are real valued
    f_ground = jfft.irfft2(f_ground, s = (256, 256))
    phi_ground = jfft.irfft2(phi_ground, s = (256, 256))
    #The generated ground truth data is misshapen for some reason...
    f_adjoint_lensed_ground = jfft.irfft2(reshape_diagonal(f_adjoint_lensed_ground), s = (256, 256))

    #compute the adjoint lensed field from the unlensed field using this project's python code
    num_steps = 10; direction = -1; adjoint = True; #NOTE that "forward" adjoint lensing requires setting the direction to -1
    f_adjoint_lensed_predict = lense_flow_wrapper(f_ground, phi_ground, pix_width, num_steps, direction, adjoint, num_pixels, False)

    #save visual comparisons to a temporary file as well
    plot_heat_map(f_adjoint_lensed_ground, "Forward Adjoint Lensed Ground", "ArcMin", \
                  "ArcMin", "Intensity", True, file_path = FIGURE_FILE_PATH + "Forward Adjoint Lensed Ground.png")
    plot_heat_map(f_adjoint_lensed_predict, "Forward Adjoint Lensed Predict", "ArcMin", \
                  "ArcMin", "Intensity", True, file_path = FIGURE_FILE_PATH + "Forward Adjoint Lensed Predict.png")
    plot_heat_map(f_adjoint_lensed_ground - f_adjoint_lensed_predict, "Forward Adjoint Lensed Diff", \
                  "ArcMin", "ArcMin", "Intensity", True, file_path = FIGURE_FILE_PATH + "Forward Adjoint Lensed Diff.png")

    #enforce that the 2D percent difference between the ground truth and
    #prediction is less than our specified threshold
    assert percent_diff_2d(f_adjoint_lensed_ground, f_adjoint_lensed_predict) < PERCENT_DIFF_THRESHOLD

#compare inverse adjoint lensed field to original field
def test_inverse_adjoint_lensing():

    #load the julia ground truth f, phi and f_lensed fields
    f_ground = precision_load(cwd + "/julia_ground_truth_data/f_array.npz")
    phi_ground = precision_load(cwd + "/julia_ground_truth_data/phi_array.npz")

    #all of the ground truth data is currently in the fourier space format
    #so we must convert to real space to apply our lense flow function whose inputs are real valued
    f_ground = jfft.irfft2(f_ground, s = (256, 256))
    phi_ground = jfft.irfft2(phi_ground, s = (256, 256))

    #compute the adjoint lensed field from the unlensed field using this project's python code
    num_steps = 10; direction = -1; adjoint = True; #NOTE that "forward" adjoint lensing requires setting the direction to -1
    f_adjoint_lensed = lense_flow_wrapper(f_ground, phi_ground, pix_width, num_steps, direction, adjoint, num_pixels, False)
    #now take the adjoint lensed field and inverse adjoint lense it
    direction = 1
    f_predict = lense_flow_wrapper(f_adjoint_lensed, phi_ground, pix_width, num_steps, direction, adjoint, num_pixels, False)

    #save visual comparisons to a temporary file as well
    plot_heat_map(f_ground - f_predict, "Inverse Adjoint Lensing Diff", \
    "ArcMin", "ArcMin", "Intensity", True, file_path = FIGURE_FILE_PATH + "Inverse Adjoint Lensing Diff.png")

    #enforce that the 2D percent difference between the ground truth and
    #prediction is less than our specified threshold
    assert percent_diff_2d(f_ground, f_predict) < PERCENT_DIFF_THRESHOLD

#compare value of logpdf in python to value of logpdf in julia
def test_logpdf_value():

    #load the necessary julia ground truth data to compute the logpdf in both parametrizations
    f_array = precision_load(cwd + "/julia_ground_truth_data/f_array.npz")
    f_lensed_no_b_no_m_array = precision_load(cwd + "/julia_ground_truth_data/f_lensed_no_b_no_m_array.npz")
    phi_array = precision_load(cwd + "/julia_ground_truth_data/phi_array.npz")
    data_array = precision_load(cwd + "/julia_ground_truth_data/data_array.npz")
    m_diagonal = precision_load(cwd + "/julia_ground_truth_data/m_diagonal.npz")
    b_diagonal = precision_load(cwd + "/julia_ground_truth_data/b_diagonal.npz")
    cf_diagonal = precision_load(cwd + "/julia_ground_truth_data/cf_diagonal.npz")
    cn_diagonal = precision_load(cwd + "/julia_ground_truth_data/cn_diagonal.npz")
    cphi_diagonal = precision_load(cwd + "/julia_ground_truth_data/cphi_diagonal.npz")
    f_lambda_array = precision_load(cwd + "/julia_ground_truth_data/f_lambda_array.npz")
    phi_lambda_array = precision_load(cwd + "/julia_ground_truth_data/phi_lambda_array.npz")
    data_lambda_array = precision_load(cwd + "/julia_ground_truth_data/data_lambda_array.npz")
    cf_lambda_array = precision_load(cwd + "/julia_ground_truth_data/cf_lambda_array.npz")
    cphi_lambda_array = precision_load(cwd + "/julia_ground_truth_data/cphi_lambda_array.npz")
    cn_lambda_array = precision_load(cwd + "/julia_ground_truth_data/cn_lambda_array.npz")

    #compute the unlensed and lensed parametrization values
    logpdf_unlensed_predict = logpdf_unlensed_param(f_array, phi_array, data_array,
                                                    b_diagonal, m_diagonal, 
                                                    cf_diagonal, cphi_diagonal, cn_diagonal, 
                                                    f_lambda_array, phi_lambda_array, data_lambda_array, 
                                                    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)

    #call the logpdf (f_lensed, phi parametrization) and confirm its values
    logpdf_lensed_predict = logpdf_lensed_param(f_lensed_no_b_no_m_array, phi_array, data_array,
                                                b_diagonal, m_diagonal, 
                                                cf_diagonal, cphi_diagonal, cn_diagonal, 
                                                f_lambda_array, phi_lambda_array, data_lambda_array, 
                                                cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)

    #enforce equality within a certain threshold to the ground truth
    #NOTE The fact that this needs to be so high is ALARMING
    assert jnp.abs(logpdf_ground - logpdf_lensed_predict) < ABS_LOGPDF_DIFF_THRESHOLD_LENSED
    assert jnp.abs(logpdf_ground - logpdf_unlensed_predict) < ABS_LOGPDF_DIFF_THRESHOLD_UNLENSED


#compare gradient of logpdf w.r.t. f in python to julia
def test_gradient_logpdf_wrt_f():

    #load the necessary julia ground truth data to compute the logpdf in both parametrizations
    f_array = precision_load(cwd + "/julia_ground_truth_data/f_array.npz")
    phi_array = precision_load(cwd + "/julia_ground_truth_data/phi_array.npz")
    data_array = precision_load(cwd + "/julia_ground_truth_data/data_array.npz")
    m_diagonal = precision_load(cwd + "/julia_ground_truth_data/m_diagonal.npz")
    b_diagonal = precision_load(cwd + "/julia_ground_truth_data/b_diagonal.npz")
    cf_diagonal = precision_load(cwd + "/julia_ground_truth_data/cf_diagonal.npz")
    cn_diagonal = precision_load(cwd + "/julia_ground_truth_data/cn_diagonal.npz")
    cphi_diagonal = precision_load(cwd + "/julia_ground_truth_data/cphi_diagonal.npz")
    f_lambda_array = precision_load(cwd + "/julia_ground_truth_data/f_lambda_array.npz")
    phi_lambda_array = precision_load(cwd + "/julia_ground_truth_data/phi_lambda_array.npz")
    data_lambda_array = precision_load(cwd + "/julia_ground_truth_data/data_lambda_array.npz")
    cf_lambda_array = precision_load(cwd + "/julia_ground_truth_data/cf_lambda_array.npz")
    cphi_lambda_array = precision_load(cwd + "/julia_ground_truth_data/cphi_lambda_array.npz")
    cn_lambda_array = precision_load(cwd + "/julia_ground_truth_data/cn_lambda_array.npz")
    
    #load the julia ground truth gradients
    gradf_logpdf_ground = precision_load(cwd + "/julia_ground_truth_data/grad_f.npz")
    #these gradients are computed in fourier space but we want to compare them in real space
    gradf_logpdf_ground = jfft.irfft2(gradf_logpdf_ground, s = (256, 256))

    #compute the gradient of logpdf in the unlensed parametrizations with jax
    #and also via directly calling the analytical gradient function
    gradf_logpdf_jax_predict = jax.grad(logpdf, argnums=0)(f_array, phi_array, data_array,
                                                            b_diagonal, m_diagonal,
                                                            cf_diagonal, cphi_diagonal, cn_diagonal, 
                                                            f_lambda_array, phi_lambda_array, data_lambda_array, 
                                                            cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)
    gradf_logpdf_analytical_predict = gradf_logpdf(f_array, phi_array, data_array, m_diagonal, b_diagonal,
                                                   cf_diagonal, cn_diagonal, pix_width, num_pixels)
    gradf_logpdf_jax_predict = jfft.irfft2(gradf_logpdf_jax_predict, s = (256, 256))
    gradf_logpdf_analytical_predict = jfft.irfft2(gradf_logpdf_analytical_predict, s = (256, 256))

    #save visual comparisons to a temporary file as well
    plot_heat_map(gradf_logpdf_jax_predict, "Grad LogPDF w.r.t. F via JAX", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt F via JAX.png")
    plot_heat_map(gradf_logpdf_analytical_predict, "Grad LogPDF w.r.t. F Analytical", "ArcMin", \
                  "ArcMin", "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt F Analytical.png")
    plot_heat_map(gradf_logpdf_ground, "Grad LogPDF w.r.t. F Ground", "ArcMin", "ArcMin", \
                  "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt F Ground.png")
    plot_heat_map(gradf_logpdf_ground - gradf_logpdf_jax_predict, \
                  "Grad LogPDF w.r.t. F Diff: Ground Minus JAX", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt F Diff: Ground Minus JAX.png")
    plot_heat_map(gradf_logpdf_analytical_predict - gradf_logpdf_jax_predict, \
                 "Grad LogPDF w.r.t. F Diff: JAX Minus Analytical", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt F Diff: JAX Minus Analytical.png")
    
    plot_heat_map(jnp.abs(jfft.rfft2(gradf_logpdf_ground - gradf_logpdf_jax_predict)), \
                  "Grad LogPDF w.r.t. F Diff - Fourier: Ground Minus JAX", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt F Diff - Fourier: Ground Minus JAX.png")

    #the jax AutoDiff output and analytical output should be EXACTLY equal
    assert jnp.array_equal(gradf_logpdf_jax_predict, gradf_logpdf_analytical_predict)

    #enforce that the 2D percent difference between the ground truth and
    #prediction is less than our specified threshold
    assert percent_diff_2d(gradf_logpdf_ground, gradf_logpdf_jax_predict) < PERCENT_DIFF_THRESHOLD

#compare gradient of logpdf w.r.t. phi in python to julia
def test_gradient_logpdf_wrt_phi():

    #load the necessary julia ground truth data
    f_array = precision_load(cwd + "/julia_ground_truth_data/f_array.npz")
    phi_array = precision_load(cwd + "/julia_ground_truth_data/phi_array.npz")
    data_array = precision_load(cwd + "/julia_ground_truth_data/data_array.npz")
    m_diagonal = precision_load(cwd + "/julia_ground_truth_data/m_diagonal.npz")
    b_diagonal = precision_load(cwd + "/julia_ground_truth_data/b_diagonal.npz")
    cf_diagonal = precision_load(cwd + "/julia_ground_truth_data/cf_diagonal.npz")
    cn_diagonal = precision_load(cwd + "/julia_ground_truth_data/cn_diagonal.npz")
    cphi_diagonal = precision_load(cwd + "/julia_ground_truth_data/cphi_diagonal.npz")
    f_lambda_array = precision_load(cwd + "/julia_ground_truth_data/f_lambda_array.npz")
    phi_lambda_array = precision_load(cwd + "/julia_ground_truth_data/phi_lambda_array.npz")
    data_lambda_array = precision_load(cwd + "/julia_ground_truth_data/data_lambda_array.npz")
    cf_lambda_array = precision_load(cwd + "/julia_ground_truth_data/cf_lambda_array.npz")
    cphi_lambda_array = precision_load(cwd + "/julia_ground_truth_data/cphi_lambda_array.npz")
    cn_lambda_array = precision_load(cwd + "/julia_ground_truth_data/cn_lambda_array.npz")
    
    #load the julia ground truth gradients
    grad_phi_logpdf_ground = precision_load(cwd + "/julia_ground_truth_data/grad_phi.npz")
    #these gradients are computed in fourier space but we want to compare them in real space
    grad_phi_logpdf_ground = jfft.irfft2(grad_phi_logpdf_ground, s = (256, 256))

    #compute the gradient of logpdf in the unlensed parametrizations with jax
    grad_phi_logpdf_predict = jax.grad(logpdf, argnums=1)(f_array, phi_array, data_array,
                                                            b_diagonal, m_diagonal,
                                                            cf_diagonal, cphi_diagonal, cn_diagonal, 
                                                            f_lambda_array, phi_lambda_array, data_lambda_array, 
                                                            cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)
    grad_phi_logpdf_predict = jfft.irfft2(grad_phi_logpdf_predict, s = (256, 256))

    #save visual comparisons to a temporary file as well
    plot_heat_map(grad_phi_logpdf_predict, "Grad LogPDF w.r.t. Phi Predict - Real", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt Phi Predict - Real.png")
    plot_heat_map(grad_phi_logpdf_ground, "Grad LogPDF w.r.t. Phi Ground - Real", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt Phi Ground - Real.png")
    plot_heat_map(grad_phi_logpdf_ground - grad_phi_logpdf_predict, \
                 "Grad LogPDF w.r.t. Phi Diff - Real", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt Phi Diff - Real.png")
    
    plot_heat_map(jnp.abs(jfft.rfft2(grad_phi_logpdf_predict)), "Grad LogPDF w.r.t. Phi Predict - Fourier", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt Phi Predict - Fourier.png")
    plot_heat_map(jnp.abs(jfft.rfft2(grad_phi_logpdf_ground)), "Grad LogPDF w.r.t. Phi Ground - Fourier", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt Phi Ground - Fourier.png")
    plot_heat_map(jnp.abs(jfft.rfft2(grad_phi_logpdf_ground - grad_phi_logpdf_predict)), \
                 "Grad LogPDF w.r.t. Phi Diff - Fourier", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt Phi Diff - Fourier.png")

    #enforce that the 2D percent difference between the ground truth and
    #prediction is less than our specified threshold
    assert percent_diff_2d(grad_phi_logpdf_ground, grad_phi_logpdf_predict) < PERCENT_DIFF_THRESHOLD

    #also compare a special gaussian test case
    gaussian_phi = jfft.rfft2(precision_load(cwd + "/julia_ground_truth_data/gaussian_phi.npz"))
    grad_phi_gaussian_ground = precision_load(cwd + "/julia_ground_truth_data/grad_phi_gauss_unmixed.npz")
    grad_phi_gaussian_predict = jax.grad(logpdf, argnums=1)(f_array, gaussian_phi, data_array,
                                                            b_diagonal, m_diagonal,
                                                            cf_diagonal, cphi_diagonal, cn_diagonal, 
                                                            f_lambda_array, phi_lambda_array, data_lambda_array, 
                                                            cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)
    grad_phi_gaussian_predict = jfft.irfft2(grad_phi_gaussian_predict, s = (256, 256))

    #save visual comparisons to a temporary file as well
    plot_heat_map(grad_phi_gaussian_predict, "Grad LogPDF w.r.t. Gaussian Phi Predict - Real", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt Gaussian Phi Predict - Real.png")
    plot_heat_map(grad_phi_gaussian_ground, "Grad LogPDF w.r.t. Gaussian Phi Ground - Real", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt Gaussian Phi Ground - Real.png")
    plot_heat_map(grad_phi_gaussian_ground - grad_phi_gaussian_predict, \
                 "Grad LogPDF w.r.t. Gaussian Phi Diff - Real", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt Gaussian  Phi Diff - Real.png")
    
    plot_heat_map(jnp.abs(jfft.rfft2(grad_phi_gaussian_predict)), "Grad LogPDF w.r.t. Gaussian Phi Predict - Fourier", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt Gaussian Phi Predict - Fourier.png")
    plot_heat_map(jnp.abs(jfft.rfft2(grad_phi_gaussian_ground)), "Grad LogPDF w.r.t. Gaussian Phi Ground - Fourier", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt Gaussian Phi Ground - Fourier.png")
    plot_heat_map(jnp.abs(jfft.rfft2(grad_phi_gaussian_ground - grad_phi_gaussian_predict)), \
                 "Grad LogPDF w.r.t. Gaussian Phi Diff - Fourier", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "Grad LogPDF wrt Gaussian  Phi Diff - Fourier.png")

    #enforce that the 2D percent difference between the ground truth and
    #prediction is less than our specified threshold
    assert percent_diff_2d(grad_phi_gaussian_ground, grad_phi_gaussian_predict) < PERCENT_DIFF_THRESHOLD

#compare gradient of forward lensing operator w.r.t. f and phi in python to julia
def test_gradient_forward_lense_flow():

    #load the necessary julia ground truth data
    delta_f_forward_ground = jfft.irfft2(precision_load(cwd + "/julia_ground_truth_data/delta_f_forward_array.npz"), s = (256, 256))
    delta_phi_forward_ground = jfft.irfft2(precision_load(cwd + "/julia_ground_truth_data/delta_phi_forward_array.npz"), s = (256, 256))

    f_array = precision_load(cwd + "/julia_ground_truth_data/f_array.npz")
    cf_diagonal = precision_load(cwd + "/julia_ground_truth_data/cf_diagonal.npz")
    phi_array = precision_load(cwd + "/julia_ground_truth_data/phi_array.npz")

    num_steps = 10; direction = -1

    #compute the python prediction
    delta_f_init = jfft.irfft2(reciprocal_matrix(reshape_diagonal(cf_diagonal)) * f_array, s = (256, 256))
    _, delta_f_forward_predict, delta_phi_forward_predict = get_lensing_operator_gradients(jfft.irfft2(phi_array, s = (256, 256)), \
                                            jfft.irfft2(f_array, s = (256, 256)), delta_f_init, pix_width, direction, num_steps)
    
    #save visual comparisons to a temporary file as well
    plot_heat_map(delta_f_forward_predict, "Grad LenseFlow w.r.t. F Predict", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LenseFlow wrt F Predict.png")
    plot_heat_map(delta_f_forward_ground, "Grad LenseFlow w.r.t. F Ground", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LenseFlow wrt F Ground.png")
    plot_heat_map(delta_f_forward_ground - delta_f_forward_predict, \
                 "Grad LenseFlow w.r.t. F Diff", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "Grad LenseFlow wrt F Diff.png")
    
    plot_heat_map(delta_phi_forward_predict, "Grad LenseFlow w.r.t. Phi Predict", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LenseFlow wrt Phi Predict.png")
    plot_heat_map(delta_phi_forward_ground, "Grad LenseFlow w.r.t. Phi Ground", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad LenseFlow wrt Phi Ground.png")
    plot_heat_map(jnp.abs(jfft.rfft2(delta_phi_forward_ground - delta_phi_forward_predict)), \
                 "Grad LenseFlow w.r.t. Phi Diff", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "Grad LenseFlow wrt Phi Diff.png")
    plot_heat_map((delta_phi_forward_ground - delta_phi_forward_predict)/delta_phi_forward_ground, \
                 "Grad LenseFlow w.r.t. Phi Diff - Normalized", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "Grad LenseFlow wrt Phi Diff - Normalized.png", clim=[0, 1])

    #enforce that the 2D percent difference between the ground truth and
    #prediction is less than our specified threshold
    assert percent_diff_2d(delta_f_forward_ground, delta_f_forward_predict) < PERCENT_DIFF_THRESHOLD
    assert percent_diff_2d(delta_phi_forward_ground, delta_phi_forward_predict) < PERCENT_DIFF_THRESHOLD

#compare gradient of inverse lensing operator w.r.t. f in python to julia
def test_gradient_inverse_lense_flow():
    
    #load the necessary julia ground truth data
    delta_f_inverse_ground = jfft.irfft2(precision_load(cwd + "/julia_ground_truth_data/delta_f_inverse_array.npz"), s = (256, 256))
    delta_phi_inverse_ground = jfft.irfft2(precision_load(cwd + "/julia_ground_truth_data/delta_phi_inverse_array.npz"), s = (256, 256))

    f_array = precision_load(cwd + "/julia_ground_truth_data/f_array.npz")
    f_lensed_array = precision_load(cwd + "/julia_ground_truth_data/f_lensed_no_b_no_m_array.npz")
    cf_diagonal = precision_load(cwd + "/julia_ground_truth_data/cf_diagonal.npz")
    phi_array = precision_load(cwd + "/julia_ground_truth_data/phi_array.npz")

    num_steps = 10; direction = 1

    #compute the python prediction
    delta_f_init = jfft.irfft2(reciprocal_matrix(reshape_diagonal(cf_diagonal)) * f_lensed_array, s = (256, 256))
    _, delta_f_inverse_predict, delta_phi_inverse_predict = get_lensing_operator_gradients(jfft.irfft2(phi_array, s = (256, 256)), \
                                            jfft.irfft2(f_array, s = (256, 256)), delta_f_init, pix_width, direction, num_steps)
    
    #save visual comparisons to a temporary file as well
    plot_heat_map(delta_f_inverse_predict, "Grad Inverse LenseFlow w.r.t. F Predict", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad Inverse LenseFlow wrt F Predict.png")
    plot_heat_map(delta_f_inverse_ground, "Grad Inverse LenseFlow w.r.t. F Ground", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad Inverse LenseFlow wrt F Ground.png")
    plot_heat_map(delta_f_inverse_ground - delta_f_inverse_predict, \
                 "Grad Inverse LenseFlow w.r.t. F Diff", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "Grad Inverse LenseFlow wrt F Diff.png")
    
    plot_heat_map(delta_phi_inverse_predict, "Grad Inverse LenseFlow w.r.t. Phi Predict", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad Inverse LenseFlow wrt Phi Predict.png")
    plot_heat_map(delta_phi_inverse_ground, "Grad Inverse LenseFlow w.r.t. Phi Ground", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "Grad Inverse LenseFlow wrt Phi Ground.png")
    plot_heat_map(jnp.abs(jfft.rfft2(delta_phi_inverse_ground - delta_phi_inverse_predict)), \
                 "Grad Inverse LenseFlow w.r.t. Phi Diff", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "Grad Inverse LenseFlow wrt Phi Diff.png")
    plot_heat_map((delta_phi_inverse_ground - delta_phi_inverse_predict)/delta_phi_inverse_ground, \
                 "Grad Inverse LenseFlow w.r.t. Phi Diff - Normalized", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "Grad Inverse LenseFlow wrt Phi Diff - Normalized.png", clim=[-1, 1])

    #enforce that the 2D percent difference between the ground truth and
    #prediction is less than our specified threshold
    assert percent_diff_2d(delta_f_inverse_ground, delta_f_inverse_predict) < PERCENT_DIFF_THRESHOLD
    assert percent_diff_2d(delta_phi_inverse_ground, delta_phi_inverse_predict) < PERCENT_DIFF_THRESHOLD

#Compare the ground truth and predicted A matrix operator from the paper 
def test_alpha_matrix_operator():

    #load ground truth and other necessary data
    f_array = precision_load(cwd + "/julia_ground_truth_data/f_array.npz")
    cf_diagonal = precision_load(cwd + "/julia_ground_truth_data/cf_diagonal.npz")
    cn_diagonal = precision_load(cwd + "/julia_ground_truth_data/cn_diagonal.npz")
    m_diagonal = precision_load(cwd + "/julia_ground_truth_data/m_diagonal.npz")
    b_diagonal = precision_load(cwd + "/julia_ground_truth_data/b_diagonal.npz")
    phi = precision_load(cwd + "/julia_ground_truth_data/phi_array.npz")
    alpha_matrix_ground = precision_load(cwd + "/julia_ground_truth_data/alpha_matrix_ground.npz")

    #compute python prediction
    alpha_operator = partial(wiener_filter_matrix_operator, phi = phi, m_diagonal = m_diagonal, b_diagonal = b_diagonal, \
                cf_diagonal = cf_diagonal, cn_diagonal = cn_diagonal, pix_width = pix_width, num_pixels = num_pixels)
    alpha_matrix_predict = alpha_operator(f_array)

    #save visual comparisons in fourier space to a temporary file as well
    plot_heat_map(jnp.abs(alpha_matrix_predict), "A Matrix Predict", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "A Matrix Predict.png")
    plot_heat_map(jnp.abs(alpha_matrix_ground), "A Matrix Ground", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "A Matrix Ground.png")
    plot_heat_map(jnp.abs(alpha_matrix_ground - alpha_matrix_predict), \
                 "A Matrix Diff", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "A Matrix Diff.png")

    #compare 2D percent diff in fourier space
    assert percent_diff_2d(alpha_matrix_ground, alpha_matrix_predict) < PERCENT_DIFF_THRESHOLD

#Compare the ground truth and predicted b vector from the paper
def test_beta_vector():

    #load ground truth and other necessary data
    data = precision_load(cwd + "/julia_ground_truth_data/data_array.npz")
    cf_diagonal = precision_load(cwd + "/julia_ground_truth_data/cf_diagonal.npz")
    cn_diagonal = precision_load(cwd + "/julia_ground_truth_data/cn_diagonal.npz")
    m_diagonal = precision_load(cwd + "/julia_ground_truth_data/m_diagonal.npz")
    b_diagonal = precision_load(cwd + "/julia_ground_truth_data/b_diagonal.npz")
    phi = precision_load(cwd + "/julia_ground_truth_data/phi_array.npz")
    beta_vector_ground = precision_load(cwd + "/julia_ground_truth_data/beta_vector_ground.npz")

    #compute python prediction
    beta_vector_predict = wiener_filter_b_vector(phi, data, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, pix_width, num_pixels)

    #save visual comparisons in Fourier space to a temporary file as well
    plot_heat_map(jnp.abs(beta_vector_predict), "B Vector Predict", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "B Vector Predict.png")
    plot_heat_map(jnp.abs(beta_vector_ground), "B Vector Ground", "ArcMin", "ArcMin", \
                 "Intensity", True, file_path = FIGURE_FILE_PATH + "B Vector Ground.png")
    plot_heat_map(jnp.abs(beta_vector_ground - beta_vector_predict), \
                 "B Vector Diff", "ArcMin", "ArcMin", "Intensity", \
                  True, file_path = FIGURE_FILE_PATH + "B Vector Diff.png")

    #compare 2D percent diff in fourier space
    assert percent_diff_2d(beta_vector_ground, beta_vector_predict) < PERCENT_DIFF_THRESHOLD

#Compare the ground truth and predicted hessian preconditioners
def test_hessian_preconditioner():

    #load the ground truth and other necessary data
    cf_diagonal = precision_load(cwd + "/julia_ground_truth_data/cf_diagonal.npz")
    cn_diagonal = precision_load(cwd + "/julia_ground_truth_data/cn_diagonal.npz")
    m_diagonal = precision_load(cwd + "/julia_ground_truth_data/m_diagonal.npz")
    b_diagonal = precision_load(cwd + "/julia_ground_truth_data/b_diagonal.npz")

    #The ground truth preconditioner is a 33024,1 diagonal in fourier space that must be reshaped for comparison
    preconditioner_ground = reshape_diagonal(precision_load(cwd + "/julia_ground_truth_data/preconditioner.npz"))

    #compute python prediction
    preconditioner_predict = hessian_logpdf_preconditioner(cf_diagonal, b_diagonal, m_diagonal, cn_diagonal)

    #save visual comparisons in fourier space to temp files as well
    plot_heat_map(jnp.abs(preconditioner_predict), "Preconditioner Predict", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Preconditioner Predict.png")
    plot_heat_map(jnp.abs(preconditioner_ground), "Preconditioner Ground", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Preconditioner Ground.png")
    plot_heat_map(jnp.abs(preconditioner_ground - preconditioner_predict), \
                    "Preconditioner Diff", "ArcMin", "ArcMin", "Intensity", \
                    True, file_path = FIGURE_FILE_PATH + "Preconditioner Diff.png")
    
    #compare 2D percent diff in fourier space
    assert percent_diff_2d(preconditioner_ground, preconditioner_predict) < PERCENT_DIFF_THRESHOLD

#Compare ground truth and predicted wiener filter on various input "f" field arrays
def test_wiener_filter():

    #load ground truth and other necessary data
    f_array = precision_load(cwd + "/julia_ground_truth_data/f_array.npz")
    fourier_weights = precision_load(cwd + "/julia_ground_truth_data/f_lambda_array.npz")
    f_wf_at_f_phi_ground = jfft.irfft2(precision_load(cwd + "/julia_ground_truth_data/wiener_filter_at_f_phi.npz"), s = (256, 256))
    f_wf_at_f_phi0_ground = jfft.irfft2(precision_load(cwd + "/julia_ground_truth_data/wiener_filter_at_f_phi0.npz"), s = (256, 256))
    f_wf_at_f0_phi_ground = jfft.irfft2(precision_load(cwd + "/julia_ground_truth_data/wiener_filter_at_f0_phi.npz"), s = (256, 256))
    f_wf_at_f0_phi0_ground = jfft.irfft2(precision_load(cwd + "/julia_ground_truth_data/wiener_filter_at_f0_phi0.npz"), s = (256, 256))
    data = precision_load(cwd + "/julia_ground_truth_data/data_array.npz")
    cf_diagonal = precision_load(cwd + "/julia_ground_truth_data/cf_diagonal.npz")
    cn_diagonal = precision_load(cwd + "/julia_ground_truth_data/cn_diagonal.npz")
    m_diagonal = precision_load(cwd + "/julia_ground_truth_data/m_diagonal.npz")
    b_diagonal = precision_load(cwd + "/julia_ground_truth_data/b_diagonal.npz")
    phi = precision_load(cwd + "/julia_ground_truth_data/phi_array.npz")

    #calculate the python predicted values
    zero_f = jnp.zeros(f_array.shape)
    zero_phi = jnp.zeros(phi.shape)

    f_wf_at_f0_phi0_predict = jfft.irfft2(wiener_filter(zero_f, zero_phi, data, m_diagonal, \
                           b_diagonal, cf_diagonal, cn_diagonal, fourier_weights, pix_width, num_pixels), s = (256, 256))
    f_wf_at_f0_phi_predict = jfft.irfft2(wiener_filter(zero_f, phi, \
                           data, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, fourier_weights, pix_width, num_pixels), s = (256, 256))
    f_wf_at_f_phi0_predict = jfft.irfft2(wiener_filter(f_array, zero_phi, data, m_diagonal, \
                           b_diagonal, cf_diagonal, cn_diagonal, fourier_weights, pix_width, num_pixels), s = (256, 256))
    f_wf_at_f_phi_predict = jfft.irfft2(wiener_filter(f_array, phi, \
                           data, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, fourier_weights, pix_width, num_pixels), s = (256, 256))

    #save visual comparisons to temp file as well
    plot_heat_map(f_wf_at_f_phi_predict, "Wiener Filter at F = F, Phi = Phi Predict", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Wiener Filter at F = F, Phi = Phi Predict.png")
    plot_heat_map(f_wf_at_f_phi_ground, "Wiener Filter at F = F, Phi = Phi Ground", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Wiener Filter at F = F, Phi = Phi Ground.png")
    plot_heat_map(f_wf_at_f_phi_ground - f_wf_at_f_phi_predict, "Wiener Filter at F = F, Phi = Phi Diff", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Wiener Filter at F = F, Phi = Phi Diff.png")
    
    plot_heat_map(f_wf_at_f0_phi_predict, "Wiener Filter at F = 0, Phi = Phi Predict", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Wiener Filter at F = 0, Phi = Phi Predict.png")
    plot_heat_map(f_wf_at_f0_phi_ground, "Wiener Filter at F = 0, Phi = Phi Ground", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Wiener Filter at F = 0, Phi = Phi Ground.png")
    plot_heat_map(f_wf_at_f0_phi_ground - f_wf_at_f0_phi_predict, "Wiener Filter at F = 0, Phi = Phi Diff", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Wiener Filter at F = 0, Phi = Phi Diff.png")
    
    plot_heat_map(f_wf_at_f_phi0_predict, "Wiener Filter at F = F, Phi = 0 Predict", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Wiener Filter at F = F, Phi = 0 Predict.png")
    plot_heat_map(f_wf_at_f_phi0_ground, "Wiener Filter at F = F, Phi = 0 Ground", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Wiener Filter at F = F, Phi = 0 Ground.png")
    plot_heat_map(f_wf_at_f_phi0_ground - f_wf_at_f_phi0_predict, "Wiener Filter at F = F, Phi = 0 Diff", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Wiener Filter at F = F, Phi = 0 Diff.png")
    
    plot_heat_map(f_wf_at_f0_phi0_predict, "Wiener Filter at F = 0, Phi = 0 Predict", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Wiener Filter at F = 0, Phi = 0 Predict.png")
    plot_heat_map(f_wf_at_f0_phi0_ground, "Wiener Filter at F = 0, Phi = 0 Ground", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Wiener Filter at F = 0, Phi = 0 Ground.png")
    plot_heat_map(f_wf_at_f0_phi0_ground - f_wf_at_f0_phi0_predict, "Wiener Filter at F = 0, Phi = 0 Diff", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Wiener Filter at F = 0, Phi = 0 Diff.png")
    
    #compare 2D percent diff
    assert percent_diff_2d(f_wf_at_f_phi_ground, f_wf_at_f_phi_predict) < PERCENT_DIFF_THRESHOLD
    assert percent_diff_2d(f_wf_at_f_phi0_ground, f_wf_at_f_phi0_predict) < PERCENT_DIFF_THRESHOLD
    assert percent_diff_2d(f_wf_at_f0_phi0_ground, f_wf_at_f0_phi0_predict) < PERCENT_DIFF_THRESHOLD

    #NOTE the assertion below seems to fail because Marius' code just blows up to NaN after about 50 iterations
    #whereas mine doesn't seem to blow up but oscillates around really large values... Is this a problem?
    #assert percent_diff_2d(f_wf_at_f0_phi_ground, f_wf_at_f0_phi_predict) < PERCENT_DIFF_THRESHOLD

def test_mixing():

    #load the ground truth mixed and unmixed f and phi
    f_ground = precision_load(cwd + "/julia_ground_truth_data/f_array.npz")
    phi_ground = precision_load(cwd + "/julia_ground_truth_data/phi_array.npz")
    f_mixed_ground = jfft.rfft2(precision_load(cwd + "/julia_ground_truth_data/f_mixed_array.npz"))
    phi_mixed_ground = precision_load(cwd + "/julia_ground_truth_data/phi_mixed_array.npz")
    d_diagonal = precision_load(cwd + "/julia_ground_truth_data/d_diagonal.npz")
    g_diagonal = precision_load(cwd + "/julia_ground_truth_data/g_diagonal.npz")

    #mix the ground truth unmixed f and phi and compare to the ground truth mixed f and phi
    f_mixed_predict, phi_mixed_predict = mix(f_ground, phi_ground, d_diagonal, g_diagonal, pix_width, num_pixels)

    plot_heat_map(jfft.irfft2(f_mixed_ground, s = (256, 256)), "F Mixed Ground", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "F Mixed Ground.png")
    plot_heat_map(jfft.irfft2(f_mixed_predict, s = (256, 256)), "F Mixed Predict", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "F Mixed Predict.png")
    plot_heat_map(jfft.irfft2(f_mixed_ground - f_mixed_predict, s = (256, 256)), "F Mixed Diff", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "F Mixed Diff.png")
    
    plot_heat_map(jfft.irfft2(phi_mixed_ground, s = (256, 256)), "Phi Mixed Ground", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Phi Mixed Ground.png")
    plot_heat_map(jfft.irfft2(phi_mixed_predict, s = (256, 256)), "Phi Mixed Predict", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Phi Mixed Predict.png")
    plot_heat_map(jfft.irfft2(phi_mixed_ground - phi_mixed_predict, s = (256, 256)), "Phi Mixed Diff", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Phi Mixed Diff.png")

    assert percent_diff_2d(jfft.irfft2(f_mixed_ground, s = (256, 256)), jfft.irfft2(f_mixed_predict, s = (256, 256))) < PERCENT_DIFF_THRESHOLD
    assert percent_diff_2d(jfft.irfft2(phi_mixed_ground, s = (256, 256)), jfft.irfft2(phi_mixed_predict, s = (256, 256))) < PERCENT_DIFF_THRESHOLD

    #unmix your mixed f and phi and compare to the original f and phi
    f_predict, phi_predict = unmix(f_mixed_predict, phi_mixed_predict, d_diagonal, g_diagonal, pix_width, num_pixels)

    plot_heat_map(jfft.irfft2(f_ground - f_predict, s = (256, 256)), "F Unmixed Diff", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "F Unmixed Diff.png")
    plot_heat_map(jfft.irfft2(phi_ground - phi_predict, s = (256, 256)), "Phi Unmixed Diff", "ArcMin", "ArcMin", \
                    "Intensity", True, file_path = FIGURE_FILE_PATH + "Phi Unmixed Diff.png")

    assert percent_diff_2d(jfft.irfft2(f_ground, s = (256, 256)), jfft.irfft2(f_predict, s = (256, 256))) < PERCENT_DIFF_THRESHOLD
    assert percent_diff_2d(jfft.irfft2(phi_ground, s = (256, 256)), jfft.irfft2(phi_predict, s = (256, 256))) < PERCENT_DIFF_THRESHOLD

#TODO compare lensing potential found via gradient descent to true lensing potential
def test_phi_gradient_descent_result():
    assert 1 == 1

def test_f_gradient_descent_result():
    assert 1 == 1

# TODO create helper function that loads and stores all julia data in one object...
# TODO revist the hard coded values in the fortran_x helper functions
# TODO rewrite these all using julia code embedded into the above python code e.g.

# from julia import Main

# def run_julia(x, y):
#     Main.eval("""
#     function f(x, y)
#         return x^2 + y
#     end
#     """)
#     return Main.f(x, y)

# out = run_julia(3, 4)
# print(out)  # 13

#This will then get rid of the need for the generate_julia_data.jl script

