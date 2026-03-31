from imports import *
from functions import *
from constants import *
from juliacall import Main as julia
from juliacall import Pkg

#TODO 1. implement my own auto and cross - correlation functions
#2. Implement a function which calls load_sim many times to get multiple auto-spectra
#and then average the auto-spectra from Python and compare to those of Julia (should use multi-threading)
#3. Plot those using Kimmy's analysis formula
#4. If all goes well start working on polarizations? 

#NOTE for the time being I would only recommend running this with the default params...
#noise level, theta_pix, N, seed, and lmax could all be changed comfortably but more
#validation needs to be done before other params can be set
def load_sim(N, theta_pix, seed = None, uk_arcmin_t = 3, H0 = None, 
            ombh2 = 0.0224567, 
            omch2 = 0.118489, 
            cosmomc_theta = 0.0104098,
            r = 0.2,
            mnu = 0.06, 
            tau = 0.055, 
            As = np.exp(3.043) * 1e-10, 
            nt = -0.2/8, #i.e. -r/8
            ns = 0.968602,
            lmax = 17000,
            k_pivot = 0.002,
            Alens = 1,
            nphi_fac = 2):

    #the lower-valued lmax_prime is used to get Dl's and Cl's from
    #camb and then we linearly extrapolate in log-log space higer ell
    #Dl's and Cl's using the higher original lmax value
    lmax_prime = min(lmax, 5000)
    #first generate the camb parameters object
    pars = camb.set_params(
        H0 = H0, 
        ombh2 = ombh2, 
        omch2 = omch2, 
        cosmomc_theta = cosmomc_theta,
        r = r,
        mnu = mnu, 
        tau = tau, 
        As = As,
        nt = nt, 
        ns = ns,
        lmax = lmax_prime,
        pivot_scalar = k_pivot,
        pivot_tensor = k_pivot,
        Alens = Alens)
    
    pars.max_l_tensor = 2*lmax_prime
    pars.max_eta_k_tensor = 4*lmax_prime
    pars.WantScalars = True
    pars.WantTensors = True
    pars.DoLensing = True
    pars.set_nonlinear_lensing(True)

    #calculate results for these parameters
    results = camb.get_results(pars)

    #get the Dl's from camb
    power_spectra = results.get_cmb_power_spectra(pars, lmax = lmax_prime - 1, CMB_unit = "muK")

    #temperature Dl's
    dl_tt_scalar = jnp.asarray(power_spectra["unlensed_scalar"][:,0])
    dl_tt_tensor = jnp.asarray(power_spectra["tensor"][:,0])
    dl_tt_total = jnp.asarray(power_spectra["total"][:,0])
    cl_tt_scalar = dl2cl(dl_tt_scalar, lmax, lmax_prime)
    cl_tt_tensor = dl2cl(dl_tt_tensor, lmax, lmax_prime)
    cl_tt_total = dl2cl(dl_tt_total, lmax, lmax_prime)

    #lensing potential Dl's
    dl_pp = jnp.asarray(results.get_lens_potential_cls(lmax = lmax_prime - 1)[:,0])
    cl_pp = dl2cl(dl_pp, lmax, lmax_prime, is_phi = True)

    #compute a meshgrid of fourier modes and also
    #return the pixel width in radians
    ls, d = gen_fourier_grid(N, theta_pix)

    #calculate the noise Cl's
    cl_n = noise_cls(lmax_prime, uk_arcmin_t)

    #given the Cl's from each type of field,
    #generate these fields using Gaussian statistics
    ell = jnp.arange(2, lmax).astype(jnp.float64)
    cphi = covar_matrix_from_cls(N, d, ls, ell, cl_pp, origin_value = 0)
    phi = field_from_covar(N, cphi, seed)

    #the cf covariance matrix is the sum of the tensor and scalar matrices
    cf_scalar = covar_matrix_from_cls(N, d, ls, ell, cl_tt_scalar, origin_value = 0)
    cf_tensor = covar_matrix_from_cls(N, d, ls, ell, cl_tt_tensor, origin_value = 0)
    cf = cf_scalar + cf_tensor
    #the lensed cf i.e. "cfl" is also needed for the quadratic estimate
    cfl = covar_matrix_from_cls(N, d, ls, ell, cl_tt_total, origin_value = 0)
    unlensed_temp = field_from_covar(N, cf, seed)
    #the lensed field is just found by lensing the unlensed field
    lensed_temp = lense_flow(unlensed_temp, phi, d, 10, 1, False, N*N, False)

    #compute the mask and beam which are needed to simulate the data field
    l_cutoff = 3000
    m = get_mask(l_cutoff, N, d, ls)
    b = get_beam(N, d, ls, lmax_prime)

    #the data field is M * B * L * f + n where n ~ N(0, Cn) i.e. "white noise"
    ell_prime = jnp.arange(2, lmax_prime)
    cn = covar_matrix_from_cls(N, d, ls, ell_prime, cl_n, origin_value = 0)
    #NOTE the sum of seeds below is necessary to get rid of possible possible correlations
    #between lensed_temp and white noise... In fact, I should probably re-write
    #the whole seed generating code to start with a given seed and then continuosly
    #update itself for the next seed method
    white_noise = field_from_covar(N, cn, seed + np.random.randint(0, 2**31))
    sum_total = jfft.irfft2(m * b * jfft.rfft2(lensed_temp))
    data = sum_total + white_noise

    #the D matrix is used in mixing and map estimation...
    d_matrix = get_d_matrix(cf, cn)

    #NOTE this seems to be working for the most part... I.e. comparing Nphi
    #from julia and python directly gives huge percent difference because of numerical precision?
    #But... We only ever use Nphi^-1 which is a lot close (~ 0.02% difference) and in practice
    #this is added to Cphi^-1 so the quantity we really care about is hessian = Cphi^-1 + Nphi^-1
    #and the error here is on the order of ~ 1e-5
    nphi = quadratic_estimate(cn, cf, cfl, m, b, d) / nphi_fac

    #MAP estimate doesn't use G so set it to 1 for the time being
    g = jnp.ones(d_matrix.shape)

    #return all the generated fiels and their covariance matrices...
    results = {}
    results["data"] = data
    results["lensed_temp"] = lensed_temp
    results["unlensed_temp"] = unlensed_temp
    results["phi"] = phi
    results["cn"] = cn
    results["cf"] = cf
    results["cfl"] = cfl
    results["cphi"] = cphi
    results["m"] = m
    results["b"] = b
    results["d"] = d_matrix
    results["g"] = g
    results["nphi"] = nphi
    results["white_noise"] = white_noise
    results["sum_total"] = sum_total
    return results

#TODO use the following style of code to re-write the RNG code parts in the simulation method
# # 1. Convert the integer seed to a JAX PRNGKey
# key = jax.random.PRNGKey(master_seed)

# # 2. Split the key into N independent sub-keys
# # This ensures field_1 always gets keys[0], field_2 gets keys[1], etc.
# keys = jax.random.split(key, n_fields)

def quadratic_estimate(cn, cf, cfl, m, b, pix_width):

    tf = m * b 
    sigma_temp_total = tf**2 * cfl + cn
    ct = cf
    qe_sum = np.zeros(ct.shape, dtype = jnp.float64)

    #loop over the types of derivatives e.g. 
    #(0, 0) --> d^2/dx^2
    #(0, 1) = (1, 0) --> d^2/dxdy
    #(1, 1) --> d^2/dy^2
    for (type_1, type_2) in [(0, 0), (1, 1), (0, 1), (1, 0)]: #, (0, 1), (1, 0), (1, 1)]:
        qe_norm_at_position = get_qe_norm_at_position(tf, ct, sigma_temp_total, pix_width, type_1, type_2)
        qe_norm_at_position = get_specific_derivative(qe_norm_at_position, pix_width, type_1, type_2)
        qe_sum += abs(qe_norm_at_position)

    return jnp.nan_to_num(1 / qe_sum, nan = 0, posinf = 0, neginf = 0)

def get_qe_norm_at_position(tf, ct, sigma_temp_total, pix_width, type_1, type_2):
    result = jfft.irfft2(qe_leg(tf**2*ct**2/sigma_temp_total, pix_width, type_1, type_2))*jfft.irfft2(qe_leg(tf**2/sigma_temp_total, pix_width)) \
            + jfft.irfft2(qe_leg(tf**2*ct/sigma_temp_total, pix_width, type_1))*jfft.irfft2(qe_leg(tf**2*ct/sigma_temp_total, pix_width, type_2))
    return jfft.rfft2(result)

def qe_leg(field, pix_width, type_1 = None, type_2 = None):

    #start by converting nan to 0 in case we quit early and 
    #just return the field
    field = jnp.nan_to_num(field, nan = 0)

    #compute the total number of derivatives we will take which is 
    #use to raise the laplacian to the correct power
    num_derivatives = int(isinstance(type_1, int)) + int(isinstance(type_2, int))

    #for zero derivatives just return the field
    if num_derivatives == 0:
        return field
    
    #compute the specific derivative term
    qe_leg_value = get_specific_derivative(field, pix_width, type_1, type_2)
    return jnp.nan_to_num(qe_leg_value, nan = 0)

def laplacian_2d(field, pix_width):
    Nx, _ = field.shape
    Ny = Nx
    kx = 2 * jnp.pi * jfft.fftfreq(Nx, pix_width)
    ky = 2 * jnp.pi * jfft.rfftfreq(Ny, pix_width)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    laplacian = -KX**2-KY**2
    return laplacian

def get_specific_derivative(field, pix_width, type_1 = None, type_2 = None):

    #0th order derivative of a field is the field
    if type_1 is None and type_2 is None:
        return field
    
    #derivative calculations should be done in fourier space
    #for this specific QE Lef calculation
    x, y, xx, xy, yy = get_fourier_derivatives(field, pix_width)
    first_derivatives = [x, y]
    second_derivatives = [xx, yy, xy]

    #1st order derivatives
    if type_2 is None:
        return first_derivatives[type_1]
    #similar second order derivatives
    elif type_1 == type_2:
        return second_derivatives[type_1]
    #mixed second order derivatives
    else:
        return second_derivatives[2]

def get_beam(N, d, ls, lmax_prime, beam_fwhm = 0):
    ell_prime = jnp.arange(2, lmax_prime).astype(jnp.float64)
    b = jnp.sqrt(beam_cls(beam_fwhm, ell_prime))
    beam = covar_matrix_from_cls(N, d, ls, ell_prime, b, origin_value = 0, rescale = False, use_linear_interpolation = True, fill_value = 1.0)
    return beam

def get_mask(l_cutoff, N, d, ls):
    screen_cls = low_pass(l_cutoff)
    ell = jnp.arange(2, len(screen_cls)).astype(jnp.float64)
    #note that we always disregard the first two Cl's
    mask = covar_matrix_from_cls(N, d, ls, ell, screen_cls[2:], origin_value = 1, rescale = False)
    return mask

def dl2cl(dl_xx, lmax, lmax_prime, is_phi = False):

    #create lists of ell and ell primed values
    ell = jnp.asarray(list(range(2, lmax))).astype(jnp.float64) #2, 3, 4, ..., lmax - 1
    ell_prime = jnp.asarray(list(range(2, lmax_prime))).astype(jnp.float64) #2, 3, 4, ..., lmax_prime - 1

    #convert Dl's to Cl's with special logic for Phi Cl's
    if is_phi:
        #cl_pp is rescaled by ~ 1/l^4 specifically whereas the other
        #fields are rescaled by ~ 1/l^2
        cl_xx = dl_xx[2:] * 2 * jnp.pi / ell_prime**4
    else:
        #NOTE we disregard the l = 0 and l = 1 modes
        cl_xx = dl_xx[2:] * 2 * jnp.pi / (ell_prime*(ell_prime + 1))

    #interpolate and extrapolate the Cl's to higher ell values
    if jnp.all(cl_xx > 0): #if we can safely do it, use log-log interpolation
        cl_xx = jnp.interp(jnp.log(ell), jnp.log(ell_prime), jnp.log(cl_xx), left="extrapolate", right="extrapolate")
        cl_xx = jnp.exp(cl_xx)
    else: #otherwise use standard linear interpolation
        cl_xx = jnp.interp(ell, ell_prime, cl_xx, left=0.0, right=0.0)

    #return the resulting Cl's
    return cl_xx

def gen_fourier_grid(N, theta_pix):
    #1 deg = 60' so convert arcmins per pixel to deg per pixel and then deg per pixel to rad per pixel
    d = jnp.deg2rad(theta_pix / ARCMIN_PER_DEGREE)
    #create an array of real frequencies in the x-direction lx[] using fft.rfreq
    lx = 2 * jnp.pi * jfft.rfftfreq(N, d) #d is the spacing in radians per pixel
    #create an array of fully complex frequencies in the y-direction ly[] using fft.freq
    ly = 2 * jnp.pi * jfft.fftfreq(N, d)
    #create a mesh-grid of these frequencies
    lx, ly = jnp.meshgrid(lx, ly) #i.e. repeat lx for length of ly and vice versa
    #find the magnitude of total l at each point in this meshgrid l = sqrt{lx^2+ly^2}
    ls =  jnp.sqrt(lx**2 + ly**2)
    return ls, d

def field_from_covar(N, covar_matrix, seed = None):
    #make results reproduceable if so desired
    if seed is not None:
        key = jax.random.key(seed)
    else:
        key = jax.random.key(np.random.randint(0, 2**31))
    #real and imaginary Gaussian fields both with mean 0, variance 1
    real_dist = jax.random.normal(key, shape=(N, N//2 + 1))
    imag_dist = 1j * jax.random.normal(key, shape=(N, N//2 + 1))
    #Rescale the variance using the fact that Var(c*X) = c^2 * Var(X)
    field = jnp.sqrt(covar_matrix/2) * (real_dist + imag_dist)
    #now transform back to real space
    field = jfft.irfft2(field, norm = "ortho")
    return field

def covar_matrix_from_cls(N, d, ls, ell, cls, origin_value = None, rescale = True, use_linear_interpolation = False, fill_value = None):
    #interpolate the cls on the grid of fourier modes
    if jnp.all(cls > 0) and use_linear_interpolation == False: #use log-log interpolation if all cls greater than zero and override not specified
        ls_safe = ls.flatten()
        #avoid divide by zero errors using a small-epsilon
        ls_safe = jnp.where(ls_safe == 0, 1e-10, ls_safe)
        result = jnp.interp(jnp.log(ls_safe), jnp.log(ell), jnp.log(cls), left="extrapolate", right="extrapolate")
        result = jnp.exp(result).reshape((N, N//2+1))
    else: #otherwise use linear-linear interpolation with a default fill_value of 0.0
        fill_value = fill_value if fill_value is not None else 0.0
        result = jnp.interp(ls.flatten(), ell, cls, left = fill_value, right = fill_value)
        result = result.reshape((N, N//2+1))
    #replace value at origin with custom value if specified otherwise use numerical value
    if origin_value is not None:
        result = result.at[0, 0].set(origin_value)
    #return the interpolated result rescaled by radians per pixel squared if rescale is specified True
    if rescale:
        result = result / d**2 
    #otherwise, return the unscaled value
    return result

#Cl's associated with atmospheric noise (i.e. so-called 1/f noise) and the beam transfer function
def noise_cls(lmax_prime, uk_arcmin_t, beam_fwhm = 0, l_knee = 100, alpha_knee = 3):
    ell_prime = jnp.asarray(list(range(2, lmax_prime))) #2, 3, 4, ..., lmax_prime - 1
    bls = beam_cls(beam_fwhm, ell_prime)
    nls = 1 + (l_knee/ell_prime)**alpha_knee
    cnls = jnp.deg2rad(uk_arcmin_t/ARCMIN_PER_DEGREE)**2 * nls / bls
    return jnp.nan_to_num(cnls, nan = 0.0, posinf = 0.0, neginf = 0.0)

#so-called beam transfer function which essentially filters off high-ell
def beam_cls(beam_fwhm, ell):
    bls = jnp.exp(-ell**2 * jnp.deg2rad(beam_fwhm/ARCMIN_PER_DEGREE)**2 / BEAM_TRANSFER_SCALAR)
    return bls

#create a jax array from 0 to 1 in the range of length which increases 
#in the same functional form as a cosine wave
def cos_ramp_up(length):
    result = (jnp.array([jnp.cos(x) for x in jnp.linspace(jnp.pi, 0, length)]) + 1)/2
    return result

#create a jax array from 1 to 0 in the range of length which decreases 
#in the same functional form as a cosine wave
def cos_ramp_down(length):
    result = 1 - cos_ramp_up(length)
    return result

#create a low pass filter of length lmax + 1 that is 1 all the way 
#up until the last delta_l entries where it decreases down to 0
#in the same functional form as a cosine
def low_pass(l_cutoff, delta_l = 50):
    low_ell_pass = jnp.ones(l_cutoff - delta_l + 1)
    high_ell_filter = cos_ramp_down(delta_l)
    screen = jnp.concatenate([low_ell_pass, high_ell_filter], axis = 0)
    return screen

def get_d_matrix(cf, cn):
    pre_factor = jnp.deg2rad(5/ARCMIN_PER_DEGREE)**2
    identity = jnp.ones(cn.shape)
    d_matrix = jnp.sqrt((cf + pre_factor * identity + 2*cn) * jnp.nan_to_num(1 / cf, posinf = 0, neginf = 0))
    return d_matrix

def batch_simulated_trials(num_trials=10, N=256, theta_pix=2,
                           uk_arcmin_t=10, lmax=17000):
    
    #frozen wrapper function whose only input is seed to be run
    #in parallel using jax.vmap
    def parallel_sim(seed):
        return load_sim(N = N, theta_pix = theta_pix,
                    uk_arcmin_t = uk_arcmin_t,
                    seed = seed, lmax = lmax)
    
    #built a list of seeds to run in parallel
    seeds = []
    for _ in range(num_trials):
        seeds.append(np.random.randint(0, 2**31))
    seeds = jnp.asarray(seeds)

    #run the load_sim() method in parallel for num_trials
    #total number of randomly generated seeds
    parallelized_load_sim = jax.vmap(parallel_sim)
    trial_results = parallelized_load_sim(seeds)
    return trial_results

def real_fourier_2_full_plane(real_fourier_field):
    #just do something stupid like this because reflections
    #make my head hurt...
    full_fourier_field = jfft.fft2(jfft.irfft2(real_fourier_field))
    return full_fourier_field

#TODO this also seems to be working...
def cross_correlation(field_1, field_2, theta_pix):
    ell_1, cl_1 = power_spectra(field_1, theta_pix)
    _, cl_2 = power_spectra(field_2, theta_pix)
    _, cl_cross = power_spectra(field_1, theta_pix, field_2)
    rho = cl_cross / jnp.sqrt(cl_1 * cl_2)
    return ell_1, rho

#NOTE this seems to be working for all the fields I have tested it with...
def power_spectra(field_1, theta_pix, field_2 = None, delta_l = 50, lmax = 17000):
    l_edges = np.arange(0, lmax, delta_l)
    #convert rfft2 to full fft2
    N, _ = field_1.shape
    if field_2 is None:
        field_2 = field_1
    field_1 = real_fourier_2_full_plane(field_1)
    field_2 = real_fourier_2_full_plane(field_2)
    ls, d = gen_fourier_grid(N, theta_pix)
    ls = real_fourier_2_full_plane(ls)
    scale_factor = N**2/d**2
    mask_1 = jnp.where(ls.flatten() > jnp.min(l_edges), True, False)
    mask_2 = jnp.where(ls.flatten() < jnp.max(l_edges), True, False)
    total_mask = mask_1 * mask_2
    ls_masked = jnp.real(ls.flatten()[total_mask])
    field_1_masked = field_1.flatten()[total_mask]
    field_2_masked = field_2.flatten()[total_mask]
    cl_obs =  jnp.real(field_1_masked * jnp.conj(field_2_masked)) / scale_factor 
    weights = (jnp.nan_to_num(1 / (2 / (2*ls_masked + 1)), nan = 0))
    normalization, _ = np.histogram(ls_masked, bins = l_edges, weights = weights)
    cl, _ = np.histogram(ls_masked, bins = l_edges, weights = weights * cl_obs)
    ell, _ = np.histogram(ls_masked, bins = l_edges, weights = weights * ls_masked)
    cl_normalized = cl / normalization
    ell_normalized = ell / normalization
    #now we need to filter off the NaNs
    cl_nan_mask = jnp.where(jnp.isnan(cl_normalized), False, True)
    ell_nan_mask = jnp.where(jnp.isnan(cl_normalized), False, True)
    cl_normalized = cl_normalized[cl_nan_mask]
    ell_normalized = ell_normalized[ell_nan_mask]
    return ell_normalized, cl_normalized

def bin_power_spectra(unbinned_cls, N, theta_pix, delta_l = 50, lmax = 17000):
    l_edges = np.arange(0, lmax, delta_l)
    ls, _ = gen_fourier_grid(N, theta_pix)
    ls = real_fourier_2_full_plane(ls)
    #scale_factor = N**2/d**2
    mask_1 = jnp.where(ls.flatten() > jnp.min(l_edges), True, False)
    mask_2 = jnp.where(ls.flatten() < jnp.max(l_edges), True, False)
    total_mask = mask_1 * mask_2
    ls_masked = jnp.real(ls.flatten()[total_mask])
    #cl_obs =  unbinned_cls / scale_factor #NOTE is the scale factor here necessary physically or not?
    weights = (jnp.nan_to_num(1 / (2 / (2*ls_masked + 1)), nan = 0))
    normalization, _ = np.histogram(ls_masked, bins = l_edges, weights = weights)
    cl, _ = np.histogram(ls_masked, bins = l_edges, weights = weights * unbinned_cls)
    ell, _ = np.histogram(ls_masked, bins = l_edges, weights = weights * ls_masked)
    cl_normalized = cl / normalization
    ell_normalized = ell / normalization
    #now we need to filter off the NaNs
    cl_nan_mask = jnp.where(jnp.isnan(cl_normalized), False, True)
    ell_nan_mask = jnp.where(jnp.isnan(cl_normalized), False, True)
    cl_normalized = cl_normalized[cl_nan_mask]
    ell_normalized = ell_normalized[ell_nan_mask]
    return ell_normalized, cl_normalized

#OKAY this is pretty based, we are gonna start using this a lot more...
def run_batched_julia_sims(uk_arcmin_t, N, theta_pix, num_trials = 100, lmax = 17000):
    julia.seval("using CMBLensing")
    julia.seval(f"""
        T = Float64
        pol = :I
        Cℓ = camb(ℓmax = {lmax})
        sim_list = []
        for trial in 1:{num_trials}
            field_list = []
            (;f, f̃, ϕ, ds) = load_sim(
                    Cℓ = Cℓ,
                    θpix = {theta_pix},
                    T = T,
                    pol = pol,
                    Nside = {N},
                    μKarcminT = {uk_arcmin_t}
                )
            push!(field_list, f)
            push!(field_list, ϕ)
            push!(field_list, ds.d)
            push!(field_list, f̃)
            push!(field_list, ds.d - ds.M * ds.B * f̃)
            push!(field_list, ds.M * ds.B * f̃)
            push!(sim_list, field_list)
        end

        #loop over the data and find the average Cl of phi, f, and data
        cls_format = get_Cℓ(sim_list[1][1])
        cls_length = length(cls_format.Cℓ)
        ell = cls_format.ℓ
        cl_pp_total = zeros(cls_length)
        cl_tt_total = zeros(cls_length)
        cl_dd_total = zeros(cls_length)
        cl_ll_total = zeros(cls_length)
        cl_nn_total = zeros(cls_length)
        cl_ss_total = zeros(cls_length)
        for trial in 1:{num_trials}
            cl_tt_total .+= get_Cℓ(sim_list[trial][1]).Cℓ
            cl_pp_total .+= get_Cℓ(sim_list[trial][2]).Cℓ
            cl_dd_total .+= get_Cℓ(sim_list[trial][3]).Cℓ
            cl_ll_total .+= get_Cℓ(sim_list[trial][4]).Cℓ
            cl_nn_total .+= get_Cℓ(sim_list[trial][5]).Cℓ
            cl_ss_total .+= get_Cℓ(sim_list[trial][6]).Cℓ
        end
        cl_tt_avg = cl_tt_total ./ {num_trials}
        cl_pp_avg = cl_pp_total ./ {num_trials}
        cl_dd_avg = cl_dd_total ./ {num_trials}
        cl_ll_avg = cl_ll_total ./ {num_trials}
        cl_nn_avg = cl_nn_total ./ {num_trials}
        cl_ss_avg = cl_ss_total ./ {num_trials}
        """)
    #initialize and populate a dictionary of results
    results = {
        "ell": np.asarray(julia.ell),
        "data": np.asarray(julia.cl_dd_avg),
        "white_noise": np.asarray(julia.cl_nn_avg),
        "sum_total": np.asarray(julia.cl_ss_avg),
        "unlensed_temp": np.asarray(julia.cl_tt_avg),
        "lensed_temp": np.asarray(julia.cl_ll_avg),
        "phi": np.asarray(julia.cl_pp_avg),
    }
    return results

#TODO
#1. Refactor / clean up and organize
#2. Add more unit tests
#3. Sync up project with GIT...
#4. Start working on polarizations: L, L^-1, L^Dagger, logpdf, load_sim(), wiener_filter(), grad_phi, grad_f, ...
#5. Investigate other runtime and accuracy boosts...

def get_avg_cls(theta_pix, num_trials = 100, N = 256, uk_arcmin_t = 10, lmax = 17000):
    #initialize the output dictionary
    results = {
        "data": jnp.asarray([]),
        "lensed_temp": jnp.asarray([]),
        "unlensed_temp": jnp.asarray([]),
        "white_noise": jnp.asarray([]),
        "sum_total": jnp.asarray([]),
        "phi": jnp.asarray([]),
    }
    #run load_sim for num_trials
    trial_results = batch_simulated_trials(num_trials = num_trials, N = N, theta_pix = theta_pix, uk_arcmin_t = uk_arcmin_t, lmax = lmax)
    #rows of trial results represent one singular call of load_sim()
    for field in ["data", "lensed_temp", "unlensed_temp", "phi", "white_noise", "sum_total"]:
        for trial in range(num_trials):
            _, cls = power_spectra(jfft.rfft2(trial_results[field][trial]), theta_pix)
            if len(results[field]) == 0:
                results[field] = cls
            else:
                results[field] += cls
        results[field] = results[field] / num_trials
    return results

def f_sky(N, theta_pix):
        dx = np.deg2rad(theta_pix / 60)
        map_area = N**2 * dx**2 * (180 / np.pi)**2
        full_sky_area = 40_000
        return map_area/full_sky_area

def log_c_ell_variance(ell, f_sky, delta_l):
    result = 2/(delta_l * (2*ell + 1) * f_sky)
    return result

# def log_c_ell_variance_with_noise(ell, f_sky, delta_l, cls, nls):
#     result = 2 * (1 + nls / cls)**2 / (delta_l * (2*ell + 1) * f_sky)
#     return result

# def c_ell_variance_with_noise(ell, f_sky, delta_l, cls, nls):
#     result = 2 * (nls + cls)**2 / (delta_l * (2*ell + 1) * f_sky)
#     return result

def plot_log_cls(l_bins, cls_1, cls_2, std, title, frac_diff = False, label_1 = "Julia", label_2 = "Python"):

    if frac_diff:
        plt.figure()
        plt.plot(l_bins, 1 - cls_1/cls_2, label = "Fractional Difference", color = "Green")
        plt.legend()
        plt.xlabel("ell bins")
        plt.ylabel("fractional difference of cl_average")
        plt.title(title)
        plt.show(block=False)
        return
    
    plt.figure()
    plt.fill_between(l_bins, 
                    jnp.log(cls_1) - std, 
                    jnp.log(cls_1) + std, 
                    color="blue", 
                    label = label_1,
                    alpha=0.5)
    plt.fill_between(l_bins, 
                    jnp.log(cls_2) - std, 
                    jnp.log(cls_2) + std, 
                    color="red", 
                    label = label_2,
                    alpha=0.5)
    plt.legend()
    plt.xlabel("ell bins")
    plt.ylabel("log of cl_average")
    plt.title(title)
    plt.show(block=False)
    return

def generate_auto_spectra_validation_plots(num_trials = 100, N = 256, uk_arcmin_t = 10, lmax = 17000, theta_pix = 2):

    python_results = get_avg_cls(theta_pix = theta_pix, num_trials = num_trials, N = N, uk_arcmin_t = uk_arcmin_t, lmax = lmax)
    cl_tt_avg_python = python_results["unlensed_temp"]
    cl_ll_avg_python = python_results["lensed_temp"]
    cl_dd_avg_python = python_results["data"]
    cl_pp_avg_python = python_results["phi"]
    cl_nn_avg_python = python_results["white_noise"]
    cl_ss_avg_python = python_results["sum_total"]

    julia_results = run_batched_julia_sims(theta_pix = theta_pix, num_trials = num_trials, N = N, uk_arcmin_t = uk_arcmin_t, lmax = lmax)
    ell = julia_results["ell"]
    cl_tt_avg_julia = julia_results["unlensed_temp"]
    cl_ll_avg_julia = julia_results["lensed_temp"]
    cl_dd_avg_julia = julia_results["data"]
    cl_pp_avg_julia = julia_results["phi"]
    cl_nn_avg_julia = julia_results["white_noise"]
    cl_ss_avg_julia = julia_results["sum_total"]

    F_SKY = f_sky(N, theta_pix)
    delta_l = 50
    cl_std = np.sqrt(log_c_ell_variance(ell, F_SKY, delta_l))
    l_bins = jnp.arange(len(ell)) 

    #-----------------------------------------------------------------
    #PHI AUTO SPECTRA COMPARISON
    #-----------------------------------------------------------------
    plot_log_cls(l_bins, cl_pp_avg_python, cl_pp_avg_julia, cl_std, "Average Cl^Phi_Phi")

    #---------------------------------------------------------------------------
    #UNLENSED TEMP AUTO SPECTRA COMPARISON
    #---------------------------------------------------------------------------
    plot_log_cls(l_bins, cl_tt_avg_python, cl_tt_avg_julia, cl_std, "Average Cl^TT")

    #---------------------------------------------------------------------------
    #LENSED TEMP AUTO SPECTRA COMPARISON
    #---------------------------------------------------------------------------
    plot_log_cls(l_bins, cl_ll_avg_python, cl_ll_avg_julia, cl_std, "Average Cl^LL")

    #---------------------------------------------------------------------------
    #M * B * L * f AUTO SPECTRA COMPARISON
    #---------------------------------------------------------------------------
    plot_log_cls(l_bins, cl_ss_avg_python, cl_ss_avg_julia, cl_std, "Average Cl^SS")

    #---------------------------------------------------------------------------
    #NOISE AUTO SPECTRA COMPARISON
    #---------------------------------------------------------------------------
    plot_log_cls(l_bins, cl_nn_avg_python, cl_nn_avg_julia, cl_std, "Average Cl^NN")

    #---------------------------------------------------------------------------
    #DATA AUTO SPECTRA COMPARISON (i.e. Data = M * B * L * f + N)
    #---------------------------------------------------------------------------
    plot_log_cls(l_bins, cl_dd_avg_python, cl_dd_avg_julia, np.sqrt(2)*cl_std, "Average Cl^DD")
    return

if __name__ == "__main__":
    generate_auto_spectra_validation_plots()
    print("Done!")
