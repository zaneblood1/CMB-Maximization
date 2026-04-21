from imports import *
from functions import *
from constants import *
# from juliacall import Main as julia
# from juliacall import Pkg
from jax import config
#config.update("jax_disable_jit", True)

#NOTE for the time being I would only recommend running this with the default params...
#noise level, theta_pix, N, seed, and lmax could all be changed comfortably but more
#validation needs to be done before other params can be set
def load_sim(N, theta_pix, master_seed = None, uk_arcmin_t = 3, H0 = None, 
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
        As = As,
        nt = nt, 
        ns = ns,
        lmax = lmax_prime,
        tau = tau, 
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
    dl_te_scalar = jnp.asarray(power_spectra["unlensed_scalar"][:,3])
    dl_ee_scalar = jnp.asarray(power_spectra["unlensed_scalar"][:,1])
    dl_bb_scalar = jnp.asarray(power_spectra["unlensed_scalar"][:,2])

    dl_tt_tensor = jnp.asarray(power_spectra["tensor"][:,0])
    dl_te_tensor = jnp.asarray(power_spectra["tensor"][:,3])
    dl_ee_tensor = jnp.asarray(power_spectra["tensor"][:,1])
    dl_bb_tensor = jnp.asarray(power_spectra["tensor"][:,2])

    dl_tt_total = jnp.asarray(power_spectra["total"][:,0])
    dl_ee_total = jnp.asarray(power_spectra["total"][:,1])
    dl_bb_total = jnp.asarray(power_spectra["total"][:,2])

    cl_tt_scalar = dl2cl(dl_tt_scalar, lmax, lmax_prime)
    cl_te_scalar = dl2cl(dl_te_scalar, lmax, lmax_prime)
    cl_ee_scalar = dl2cl(dl_ee_scalar, lmax, lmax_prime)
    cl_bb_scalar = dl2cl(dl_bb_scalar, lmax, lmax_prime)

    cl_tt_tensor = dl2cl(dl_tt_tensor, lmax, lmax_prime)
    cl_te_tensor = dl2cl(dl_te_tensor, lmax, lmax_prime)
    cl_ee_tensor = dl2cl(dl_ee_tensor, lmax, lmax_prime)
    cl_bb_tensor = dl2cl(dl_bb_tensor, lmax, lmax_prime)

    cl_tt_total = dl2cl(dl_tt_total, lmax, lmax_prime)
    cl_ee_total = dl2cl(dl_ee_total, lmax, lmax_prime)
    cl_bb_total = dl2cl(dl_bb_total, lmax, lmax_prime)

    #lensing potential Dl's
    dl_pp = jnp.asarray(results.get_lens_potential_cls(lmax = lmax_prime - 1)[:,0])
    cl_pp = dl2cl(dl_pp, lmax, lmax_prime, is_phi = True)

    #compute a meshgrid of fourier modes and also
    #return the pixel width in radians
    ls, d = gen_fourier_grid(jnp.zeros((N,N//2+1)), theta_pix)

    #calculate the noise Cl's
    noise_cls_tt, noise_cls_te, noise_cls_ee, noise_cls_bb = noise_cls(lmax_prime, uk_arcmin_t)

    #TODO use the following style of code to re-write the RNG code parts in the simulation method
    #Convert the integer seed to a JAX PRNGKey
    KEY = jax.random.PRNGKey(master_seed)
    #Split the key into N independent sub-keys
    KEYS = jax.random.split(KEY, 100)

    #given the Cl's from each type of field,
    #generate these fields using Gaussian statistics
    ell = jnp.arange(2, lmax).astype(jnp.float64)
    cphi = covar_matrix_from_cls(N, d, ls, ell, cl_pp, origin_value = 0)
    phi = field_from_covar(N, cphi, KEYS[0])

    #the cf covariance matrix is the sum of the tensor and scalar matrices
    cf_tt_scalar = covar_matrix_from_cls(N, d, ls, ell, cl_tt_scalar, origin_value = 0)
    cf_tt_tensor = covar_matrix_from_cls(N, d, ls, ell, cl_tt_tensor, origin_value = 0)
    cf_te_scalar = covar_matrix_from_cls(N, d, ls, ell, cl_te_scalar, origin_value = 0)
    cf_te_tensor = covar_matrix_from_cls(N, d, ls, ell, cl_te_tensor, origin_value = 0)
    cf_ee_scalar = covar_matrix_from_cls(N, d, ls, ell, cl_ee_scalar, origin_value = 0)
    cf_ee_tensor = covar_matrix_from_cls(N, d, ls, ell, cl_ee_tensor, origin_value = 0)
    cf_bb_scalar = covar_matrix_from_cls(N, d, ls, ell, cl_bb_scalar, origin_value = 0)
    cf_bb_tensor = covar_matrix_from_cls(N, d, ls, ell, cl_bb_tensor, origin_value = 0)
    cf_tt = cf_tt_scalar + cf_tt_tensor
    cf_te = cf_te_scalar + cf_te_tensor
    cf_ee = cf_ee_scalar + cf_ee_tensor
    cf_bb = cf_bb_scalar + cf_bb_tensor
    #the lensed cf i.e. "cfl" is also needed for the quadratic estimate
    cfl_tt = covar_matrix_from_cls(N, d, ls, ell, cl_tt_total, origin_value = 0)
    cfl_ee = covar_matrix_from_cls(N, d, ls, ell, cl_ee_total, origin_value = 0)
    cfl_bb = covar_matrix_from_cls(N, d, ls, ell, cl_bb_total, origin_value = 0)
    unlensed_temp_i = field_from_covar(N, cf_tt, KEYS[1])
    unlensed_temp_e = field_from_covar(N, cf_ee, KEYS[2])
    unlensed_temp_b = field_from_covar(N, cf_bb, KEYS[3])
    #the lensed field is just found by lensing the unlensed field
    pix_width = jnp.deg2rad(theta_pix / ARCMIN_PER_DEGREE)
    lensed_temp_i, lensed_temp_e, lensed_temp_b = lense_ieb(unlensed_temp_i, unlensed_temp_e, 
                                                  unlensed_temp_b, phi, pix_width, N, theta_pix)

    #compute the mask and beam which are needed to simulate the data field
    l_cutoff = 3000
    m_tt = get_mask(l_cutoff, N, d, ls)
    m_te = jnp.zeros(m_tt.shape)
    m_ee = m_tt
    m_bb = m_tt

    b_tt = get_beam(N, d, ls, lmax_prime)
    b_te = jnp.zeros(b_tt.shape)
    b_ee = b_tt
    b_bb = b_tt

    #the data field is M * B * L * f + n where n ~ N(0, Cn) i.e. "white noise"
    ell_prime = jnp.arange(2, lmax_prime)
    cn_tt = covar_matrix_from_cls(N, d, ls, ell_prime, noise_cls_tt, origin_value = 0)
    cn_te = covar_matrix_from_cls(N, d, ls, ell_prime, noise_cls_te, origin_value = 0)
    cn_ee = covar_matrix_from_cls(N, d, ls, ell_prime, noise_cls_ee, origin_value = 0)
    cn_bb = covar_matrix_from_cls(N, d, ls, ell_prime, noise_cls_bb, origin_value = 0)
    #NOTE the sum of seeds below is necessary to get rid of possible possible correlations
    #between lensed_temp and white noise... In fact, I should probably re-write
    #the whole seed generating code to start with a given seed and then continuosly
    #update itself for the next seed method
    white_noise_tt = field_from_covar(N, cn_tt, KEYS[4])
    white_noise_ee = field_from_covar(N, cn_ee, KEYS[5])
    white_noise_bb = field_from_covar(N, cn_bb, KEYS[6])

    #find the [I, E, B] matrix product MB_[I, E, B] of Mask_[I, E, B] x Beam_[I, E, B]
    #now take the result and find the resulting matrix vector product given by
    #result_[I, E, B] = MB_[I, E, B] x lensed_field_[I, E, B]... However, these matrices
    #are diagonal so it is actually just term by term
    sum_total_i = jfft.irfft2(m_tt * b_tt * jfft.rfft2(lensed_temp_i))
    sum_total_e = jfft.irfft2(m_ee * b_ee * jfft.rfft2(lensed_temp_e))
    sum_total_b = jfft.irfft2(m_bb * b_bb * jfft.rfft2(lensed_temp_b))
    data_i = sum_total_i + white_noise_tt
    data_e = sum_total_e + white_noise_ee
    data_b = sum_total_b + white_noise_bb

    #the D matrix is used in mixing and map estimation...
    d_matrix_tt, d_matrix_te, d_matrix_ee, d_matrix_bb = \
        get_d_matrix(cf_tt, cf_te, cf_ee, cf_bb, 
                     cn_tt, cn_te, cn_ee, cn_bb)

    #NOTE this seems to be working for the most part... I.e. comparing Nphi
    #from julia and python directly gives huge percent difference because of numerical precision?
    #But... We only ever use Nphi^-1 which is a lot close (~ 0.02% difference) and in practice
    #this is added to Cphi^-1 so the quantity we really care about is hessian = Cphi^-1 + Nphi^-1
    #and the error here is on the order of ~ 1e-5
    #nphi = quadratic_estimate(cn, cf, cfl, m, b, d) / nphi_fac #NOTE uncomment for TT Nphi
    nphi = quadratic_estimate_v2(cf_ee, cf_bb, cfl_ee, cfl_bb, cn_ee, cn_bb, m_ee, m_bb, b_ee, b_bb, pix_width) / nphi_fac

    #MAP estimate doesn't use G so set it to 1 for the time being
    g = jnp.ones(d_matrix_tt.shape)

    #return all the generated fiels and their covariance matrices...
    results = {}
    results["cfl_tt"] = cfl_tt
    results["cfl_ee"] = cfl_ee
    results["cfl_bb"] = cfl_bb
    results["nphi"] = nphi
    results["unlensed_temp_i"] = unlensed_temp_i
    results["unlensed_temp_e"] = unlensed_temp_e
    results["unlensed_temp_b"] = unlensed_temp_b
    results["phi"] = phi
    results["cn_tt"] = cn_tt
    results["cn_te"] = cn_te
    results["cn_ee"] = cn_ee
    results["cn_bb"] = cn_bb
    results["cf_tt"] = cf_tt
    results["cf_te"] = cf_te
    results["cf_ee"] = cf_ee
    results["cf_bb"] = cf_bb
    results["cphi"] = cphi
    results["m_tt"] = m_tt
    results["m_te"] = m_te
    results["m_ee"] = m_ee
    results["m_bb"] = m_bb
    results["b_tt"] = b_tt
    results["b_te"] = b_te
    results["b_ee"] = b_ee
    results["b_bb"] = b_bb
    results["d_tt"] = d_matrix_tt
    results["d_te"] = d_matrix_te
    results["d_ee"] = d_matrix_ee
    results["d_bb"] = d_matrix_bb
    results["g"] = g
    results["white_noise_i"] = white_noise_tt
    results["white_noise_e"] = white_noise_ee
    results["white_noise_b"] = white_noise_bb
    results["lensed_temp_i"] = lensed_temp_i
    results["lensed_temp_e"] = lensed_temp_e
    results["lensed_temp_b"] = lensed_temp_b
    results["sum_total_i"] = sum_total_i
    results["sum_total_e"] = sum_total_e
    results["sum_total_b"] = sum_total_b
    results["data_i"] = data_i
    results["data_e"] = data_e
    results["data_b"] = data_b
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
    for (type_1, type_2) in [(0, 0), (1, 1), (0, 1), (1, 0)]:
        qe_norm_at_position = get_qe_norm_at_position(tf, ct, sigma_temp_total, pix_width, type_1, type_2)
        qe_norm_at_position = get_specific_derivative(qe_norm_at_position, pix_width, type_1, type_2)
        qe_sum += abs(qe_norm_at_position)

    return jnp.nan_to_num(1 / qe_sum, nan = 0, posinf = 0, neginf = 0)

@partial(jax.jit, static_argnums = (0,))
def get_indices(length):
    parts = [unique_permutations(num_0s, length - num_0s) for num_0s in range(length + 1)]
    return jnp.concatenate(parts, axis = 0)

@partial(jax.jit, static_argnums = (0, 1))
def unique_permutations(n0, n1):
    total_length = n0 + n1
    res = []
    for indices in combinations(range(total_length), n0):
        p = [1] * total_length
        for i in indices:
            p[i] = 0
        res.append(p)
    return jnp.array(res, dtype = jnp.int32).reshape(-1, total_length)

@jax.jit
def quadratic_estimate_v2(cf_ee, cf_bb, cfl_ee, cfl_bb, cn_ee, cn_bb, mask_ee, mask_bb, beam_ee, beam_bb, pix_width):

    qe_sum = jnp.zeros(cf_ee.shape, dtype = jnp.complex128)

    for (i, j) in get_indices(2):

        internal = quadratic_estimate_internal(cf_ee, cf_bb, cfl_ee, cfl_bb, 
                                               cn_ee, cn_bb, mask_ee, mask_bb, 
                                               beam_ee, beam_bb, pix_width, i, j)
        qe_sum += jnp.abs(get_specific_derivative(jfft.rfft2(internal), pix_width, i, j))

    return jnp.nan_to_num(1 / qe_sum, nan = 0, posinf = 0, neginf = 0)

@jax.jit
def quadratic_estimate_internal(cf_ee, cf_bb, cfl_ee, cfl_bb, cn_ee, cn_bb, mask_ee, mask_bb, beam_ee, beam_bb, pix_width, i, j):

    epsilon = levi_civita_3d()
    tf2e = (mask_ee * beam_ee)**2
    tf2b = (mask_bb * beam_bb)**2
    sigma_e_tot = tf2e * cfl_ee + cn_ee
    sigma_b_tot = tf2b * cfl_bb + cn_bb
    N, _ = cf_ee.shape
    qe_sum = np.zeros((N, N), dtype = jnp.float64)

    for (k, l, m, n, p, q) in get_indices(6):

        qe_sum += 4 * epsilon[m, p, 2] * epsilon[n, q, 2] \
                    * get_qe_norm_at_position_v2(tf2e, cf_ee, sigma_e_tot, 
                                                 tf2b, cf_bb, sigma_b_tot, 
                                                 pix_width, i, j, k, l, m, n, p, q)
    return qe_sum

def get_qe_norm_at_position_v2(tf2e, cf_ee, sigma_e_tot, tf2b, cf_bb, sigma_b_tot, pix_width, i, j, k, l, m, n, p, q):
    result = jfft.irfft2(qe_leg_v2(tf2e * cf_ee**2 / sigma_e_tot, pix_width, {"encapsulated": jnp.array([i, j]), "isolated": jnp.array([k, l, m, n])})) \
             * jfft.irfft2(qe_leg_v2(tf2b / sigma_b_tot, pix_width, {"encapsulated": jnp.array([]), "isolated": jnp.array([k, l, p, q])})) \
             - 2*jfft.irfft2(qe_leg_v2(tf2e * cf_ee / sigma_e_tot, pix_width, {"encapsulated": jnp.array([i]), "isolated": jnp.array([k, l, m, n])})) \
             * jfft.irfft2(qe_leg_v2(tf2b * cf_bb / sigma_b_tot, pix_width, {"encapsulated": jnp.array([j]), "isolated": jnp.array([k, l, p, q])})) \
             + jfft.irfft2(qe_leg_v2(tf2e / sigma_e_tot, pix_width, {"encapsulated": jnp.array([]), "isolated": jnp.array([k, l, m, n])})) \
             * jfft.irfft2(qe_leg_v2(tf2b * cf_bb**2 / sigma_b_tot, pix_width, {"encapsulated": jnp.array([i, j]), "isolated": jnp.array([k, l, p, q])}))
    return result

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

@jax.jit
def qe_leg_internal(field, pix_width, num_derivatives, num_x, num_y):

    #start by converting nan to 0 in case we quit early and 
    #just return the field
    field = jnp.nan_to_num(field, nan = 0)

    #for zero derivatives just return the field
    def quit_early(field, pix_width, num_derivatives, num_x, num_y):
        _ = pix_width #quiet the linter...
        _ = num_derivatives
        _ = num_x
        _ = num_y
        return field * (1.0 + 0.0j)
    
    def take_partials(field, pix_width, num_derivatives, num_x, num_y):
        #compute the laplacian raised to the n/2 power
        laplacian_power = jnp.sqrt(laplacian_2d(field, pix_width)**num_derivatives)
        
        #compute the specific derivative term
        N, _ = field.shape
        kx = 2 * jnp.pi * jfft.fftfreq(N, pix_width)
        ky = 2 * jnp.pi * jfft.rfftfreq(N, pix_width)
        KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
        x_deriv  = 1j * KX
        y_deriv  = 1j * KY
        qe_leg_value = (x_deriv**num_x * y_deriv**num_y) * field
        qe_leg_value = qe_leg_value / laplacian_power

        #gracefully handle any NaNs
        return jnp.nan_to_num(qe_leg_value, nan = 0)
    
    operands = field, pix_width, num_derivatives, num_x, num_y
    result = jax.lax.cond(
        jnp.equal(num_derivatives, 0),
        quit_early,
        take_partials,
        *operands
    )

    return result

@jax.jit
def qe_leg_v2(field, pix_width, indices):
    num_derivatives = len(indices["isolated"])
    squashed_indices = jnp.concatenate((indices["encapsulated"], indices["isolated"]), axis = 0)
    num_y = jnp.sum(squashed_indices)
    num_x = len(squashed_indices) - num_y
    return qe_leg_internal(field, pix_width, num_derivatives, num_x, num_y)

@jax.jit
def laplacian_2d(field, pix_width):
    Nx, _ = field.shape
    Ny = Nx
    kx = 2 * jnp.pi * jfft.fftfreq(Nx, pix_width)
    ky = 2 * jnp.pi * jfft.rfftfreq(Ny, pix_width)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    laplacian = KX**2 + KY**2
    return laplacian

@jax.jit
def get_specific_derivative(field, pix_width, type_1 = None, type_2 = None):

    #0th order derivative of a field is the field
    if type_1 is None and type_2 is None:
        return field
    
    #derivative calculations should be done in fourier space
    #for this specific QE Lef calculation
    x, y, xx, xy, yy = get_fourier_derivatives(field, pix_width)
    first_derivatives = jnp.array([x, y])
    second_derivatives = jnp.array([xx, yy, xy])

    #1st order derivatives
    if type_2 is None:
        return first_derivatives[type_1]
        
    # #similar second order derivatives
    # elif jnp.equal(type_1, type_2):
    #     return second_derivatives[type_1]
    # #mixed second order derivatives
    # else:
    #     return second_derivatives[2]

    def return_double_2nd_order(second_derivatives, type_1):
        return second_derivatives[type_1]
    
    def return_mixed_2nd_order(second_derivatives, type_1):
        _ = type_1
        return second_derivatives[2]

    operands = second_derivatives, type_1
    result = jax.lax.cond(
        jnp.equal(type_1, type_2),
        return_double_2nd_order,
        return_mixed_2nd_order,
        *operands
    )

    return result

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

def field_from_covar(N, covar_matrix, seed = None):
    #make results reproduceable if so desired
    # if seed is not None:
    #     key = jax.random.key(seed)
    if seed is None:
        seed = jax.random.key(np.random.randint(0, 2**31))
    #real and imaginary Gaussian fields both with mean 0, variance 1
    real_dist = jax.random.normal(seed, shape=(N, N//2 + 1))
    imag_dist = 1j * jax.random.normal(seed, shape=(N, N//2 + 1))
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
    cn_tt = jnp.deg2rad(uk_arcmin_t/ARCMIN_PER_DEGREE)**2 * nls / bls
    cn_tt = jnp.nan_to_num(cn_tt, nan = 0.0, posinf = 0.0, neginf = 0.0)
    cn_te = jnp.zeros(cn_tt.shape)
    cn_ee = 2 * cn_tt
    cn_bb = 2 * cn_tt
    return cn_tt, cn_te, cn_ee, cn_bb

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

@jax.jit
def levi_civita_3d():
    i, j, k = jnp.meshgrid(jnp.arange(3), jnp.arange(3), jnp.arange(3), indexing='ij')
    return (i - j) * (j - k) * (k - i) / 2

def get_d_matrix(cf_tt, cf_te, cf_ee, cf_bb, cn_tt, cn_te, cn_ee, cn_bb):

    pre_factor = jnp.deg2rad(5/ARCMIN_PER_DEGREE)**2
    identity = jnp.ones(cn_tt.shape)

    cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee = invert_block_matrix(cf_tt, cf_te, cf_te, cf_ee)
    cf_inv_bb = reciprocal_matrix(cf_bb)

    cf_inv_tt = jnp.nan_to_num(cf_inv_tt, nan = 0, posinf = 0, neginf = 0)
    cf_inv_te = jnp.nan_to_num(cf_inv_te, nan = 0, posinf = 0, neginf = 0)
    cf_inv_et = jnp.nan_to_num(cf_inv_et, nan = 0, posinf = 0, neginf = 0)
    cf_inv_ee = jnp.nan_to_num(cf_inv_ee, nan = 0, posinf = 0, neginf = 0)
    cf_inv_bb = jnp.nan_to_num(cf_inv_bb, nan = 0, posinf = 0, neginf = 0)

    sum_tt = (cf_tt + pre_factor * identity + 2*cn_tt)
    sum_te = (cf_te + 2*cn_te)
    sum_ee = (cf_ee + pre_factor * identity + 2*cn_ee)
    sum_bb = (cf_bb + pre_factor * identity + 2*cn_bb)

    d_matrix_tt, d_matrix_te, d_matrix_et, d_matrix_ee, d_matrix_bb = \
        ieb_matrix_mult(sum_tt, sum_te, sum_te, sum_ee, sum_bb, 
                        cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee, cf_inv_bb)

    #NOTE I do not know why but the distinction between d_et and d_te seems to matter
    #a lot here... The current usage of just solely d_et seems to give results that
    #match Julia much better for some reason than actually doing the correct math...
    d_matrix_tt, d_matrix_te, d_matrix_et, d_matrix_ee = \
        ieb_matrix_sqrt(d_matrix_tt, d_matrix_et, d_matrix_et, d_matrix_ee)
    d_matrix_bb = jnp.sqrt(d_matrix_bb)

    return d_matrix_tt, d_matrix_te, d_matrix_ee, d_matrix_bb



def batch_simulated_trials(num_trials=10, N=256, theta_pix=2,
                           uk_arcmin_t=10, lmax=17000):
    
    #frozen wrapper function whose only input is seed to be run
    #in parallel using jax.vmap
    def parallel_sim(seed):
        return load_sim(N = N, theta_pix = theta_pix,
                    uk_arcmin_t = uk_arcmin_t,
                    master_seed = seed, lmax = lmax)
    
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
    ls, d = gen_fourier_grid(field_1, theta_pix)
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

#TODO refactor and clean up these very similar methods... stay DRY, Do not Repeat Yourself
def bin_power_spectra(unbinned_cls, N, theta_pix, delta_l = 50, lmax = 17000):
    l_edges = np.arange(0, lmax, delta_l)
    ls, _ = gen_fourier_grid(jnp.zeros((N, N)), theta_pix)
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
#TODO actually use juliacall once we figure out why it is corrupting everything
def run_batched_julia_sims(uk_arcmin_t, N, theta_pix, num_trials = 100, lmax = 17000):
    # julia.seval("using CMBLensing")
    # julia.seval(f"""
    #     T = Float64
    #     pol = :I
    #     Cℓ = camb(ℓmax = {lmax})
    #     sim_list = []
    #     for trial in 1:{num_trials}
    #         field_list = []
    #         (;f, f̃, ϕ, ds) = load_sim(
    #                 Cℓ = Cℓ,
    #                 θpix = {theta_pix},
    #                 T = T,
    #                 pol = pol,
    #                 Nside = {N},
    #                 μKarcminT = {uk_arcmin_t}
    #             )
    #         push!(field_list, f)
    #         push!(field_list, ϕ)
    #         push!(field_list, ds.d)
    #         push!(field_list, f̃)
    #         push!(field_list, ds.d - ds.M * ds.B * f̃)
    #         push!(field_list, ds.M * ds.B * f̃)
    #         push!(sim_list, field_list)
    #     end

    #     #loop over the data and find the average Cl of phi, f, and data
    #     cls_format = get_Cℓ(sim_list[1][1])
    #     cls_length = length(cls_format.Cℓ)
    #     ell = cls_format.ℓ
    #     cl_pp_total = zeros(cls_length)
    #     cl_tt_total = zeros(cls_length)
    #     cl_dd_total = zeros(cls_length)
    #     cl_ll_total = zeros(cls_length)
    #     cl_nn_total = zeros(cls_length)
    #     cl_ss_total = zeros(cls_length)
    #     for trial in 1:{num_trials}
    #         cl_tt_total .+= get_Cℓ(sim_list[trial][1]).Cℓ
    #         cl_pp_total .+= get_Cℓ(sim_list[trial][2]).Cℓ
    #         cl_dd_total .+= get_Cℓ(sim_list[trial][3]).Cℓ
    #         cl_ll_total .+= get_Cℓ(sim_list[trial][4]).Cℓ
    #         cl_nn_total .+= get_Cℓ(sim_list[trial][5]).Cℓ
    #         cl_ss_total .+= get_Cℓ(sim_list[trial][6]).Cℓ
    #     end
    #     cl_tt_avg = cl_tt_total ./ {num_trials}
    #     cl_pp_avg = cl_pp_total ./ {num_trials}
    #     cl_dd_avg = cl_dd_total ./ {num_trials}
    #     cl_ll_avg = cl_ll_total ./ {num_trials}
    #     cl_nn_avg = cl_nn_total ./ {num_trials}
    #     cl_ss_avg = cl_ss_total ./ {num_trials}
    #     """)

    #initialize and populate a dictionary of results
    results = {
        "ell": jnp.asarray(precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/ell.npz")),
        "data": jnp.asarray(precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_dd_avg.npz")),
        "white_noise": jnp.asarray(precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_nn_avg.npz")),
        "sum_total": jnp.asarray(precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_ss_avg.npz")),
        "unlensed_temp": jnp.asarray(precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_tt_avg.npz")),
        "lensed_temp": jnp.asarray(precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_ll_avg.npz")),
        "phi": jnp.asarray(precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_pp_avg.npz")),
    }
    return results

#TODO
#1. Refactor / clean up and organize
#2. Add more unit tests
#3. Sync up project with GIT...
#4. Start working on polarizations: L, L^-1, L^Dagger, logpdf, load_sim(), wiener_filter(), grad_phi, grad_f, ...
#5. Investigate other runtime and accuracy boosts...

def get_avg_cls(theta_pix, num_trials = 100, N = 256, uk_arcmin_t = 10, lmax = 17000, delta_l = 50):
    #initialize the output dictionary
    results = {
        "data_i": jnp.asarray([]),
        "data_e": jnp.asarray([]),
        "data_b": jnp.asarray([]),
        "lensed_temp_i": jnp.asarray([]),
        "lensed_temp_e": jnp.asarray([]),
        "lensed_temp_b": jnp.asarray([]),
        "unlensed_temp_i": jnp.asarray([]),
        "unlensed_temp_e": jnp.asarray([]),
        "unlensed_temp_b": jnp.asarray([]),
        "white_noise_i": jnp.asarray([]),
        "white_noise_e": jnp.asarray([]),
        "white_noise_b": jnp.asarray([]),
        "sum_total_i": jnp.asarray([]),
        "sum_total_e": jnp.asarray([]),
        "sum_total_b": jnp.asarray([]),
        "phi": jnp.asarray([]),
    }
    #run load_sim for num_trials
    trial_results = batch_simulated_trials(num_trials = num_trials, N = N, theta_pix = theta_pix, uk_arcmin_t = uk_arcmin_t, lmax = lmax)
    #rows of trial results represent one singular call of load_sim()
    for suffix in ["i", "e", "b"]:
        for field in [f"data_{suffix}", f"lensed_temp_{suffix}", f"unlensed_temp_{suffix}", 
                      f"white_noise_{suffix}", f"sum_total_{suffix}"]:
            for trial in range(num_trials):
                _, cls = power_spectra(jfft.rfft2(trial_results[field][trial]), theta_pix, delta_l = delta_l)
                if len(results[field]) == 0:
                    results[field] = cls
                else:
                    results[field] += cls
            results[field] = results[field] / num_trials

    for trial in range(num_trials):
        _, cls = power_spectra(jfft.rfft2(trial_results["phi"][trial]), theta_pix, delta_l = delta_l)
        if len(results["phi"]) == 0:
            results["phi"] = cls
        else:
            results["phi"] += cls
    results["phi"] = results["phi"] / num_trials
    return results

def f_sky(N, theta_pix):
        dx = np.deg2rad(theta_pix / 60)
        map_area = N**2 * dx**2 * (180 / np.pi)**2
        full_sky_area = 40_000
        return map_area/full_sky_area

def log_c_ell_variance(ell, f_sky, delta_l):
    result = 2/(delta_l * (2*ell + 1) * f_sky)
    return result

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

    python_results = get_avg_cls(theta_pix = theta_pix, num_trials = num_trials, N = N, 
                                 uk_arcmin_t = uk_arcmin_t, lmax = lmax, delta_l=50)
    cl_tt_i_avg_python = python_results["unlensed_temp_i"]
    cl_tt_e_avg_python = python_results["unlensed_temp_e"]
    cl_tt_b_avg_python = python_results["unlensed_temp_b"]
    cl_ll_i_avg_python = python_results["lensed_temp_i"]
    cl_ll_e_avg_python = python_results["lensed_temp_e"]
    cl_ll_b_avg_python = python_results["lensed_temp_b"]
    cl_dd_i_avg_python = python_results["data_i"]
    cl_dd_e_avg_python = python_results["data_e"]
    cl_dd_b_avg_python = python_results["data_b"]
    cl_pp_avg_python = python_results["phi"]
    cl_nn_i_avg_python = python_results["white_noise_i"]
    cl_nn_e_avg_python = python_results["white_noise_e"]
    cl_nn_b_avg_python = python_results["white_noise_b"]
    cl_ss_i_avg_python = python_results["sum_total_i"]
    cl_ss_e_avg_python = python_results["sum_total_e"]
    cl_ss_b_avg_python = python_results["sum_total_b"]

    # julia_results = run_batched_julia_sims(theta_pix = theta_pix, num_trials = num_trials, N = N, uk_arcmin_t = uk_arcmin_t, lmax = lmax)
    ell = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/ell.npz")
    cl_tt_i_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_tt_i_avg.npz")
    cl_tt_e_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_tt_e_avg.npz")
    cl_tt_b_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_tt_b_avg.npz")
    cl_ll_i_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_ll_i_avg.npz")
    cl_ll_e_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_ll_e_avg.npz")
    cl_ll_b_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_ll_b_avg.npz")
    cl_dd_i_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_dd_i_avg.npz")
    cl_dd_e_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_dd_e_avg.npz")
    cl_dd_b_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_dd_b_avg.npz")
    cl_pp_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_pp_avg.npz")
    cl_nn_i_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_nn_i_avg.npz")
    cl_nn_e_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_nn_e_avg.npz")
    cl_nn_b_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_nn_b_avg.npz")
    cl_ss_i_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_ss_i_avg.npz")
    cl_ss_e_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_ss_e_avg.npz")
    cl_ss_b_avg_julia = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/cl_ss_b_avg.npz")

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
    plot_log_cls(l_bins, cl_tt_i_avg_python, cl_tt_i_avg_julia, cl_std, "Average Cl^TT, I")
    plot_log_cls(l_bins, cl_tt_e_avg_python, cl_tt_e_avg_julia, cl_std, "Average Cl^TT, E")
    plot_log_cls(l_bins, cl_tt_b_avg_python, cl_tt_b_avg_julia, cl_std, "Average Cl^TT, B")

    #---------------------------------------------------------------------------
    #LENSED TEMP AUTO SPECTRA COMPARISON
    #---------------------------------------------------------------------------
    plot_log_cls(l_bins, cl_ll_i_avg_python, cl_ll_i_avg_julia, cl_std, "Average Cl^LL, I")
    plot_log_cls(l_bins, cl_ll_e_avg_python, cl_ll_e_avg_julia, cl_std, "Average Cl^LL, E")
    plot_log_cls(l_bins, cl_ll_b_avg_python, cl_ll_b_avg_julia, cl_std, "Average Cl^LL, B")

    #---------------------------------------------------------------------------
    #M * B * L * f AUTO SPECTRA COMPARISON
    #---------------------------------------------------------------------------
    plot_log_cls(l_bins, cl_ss_i_avg_python, cl_ss_i_avg_julia, cl_std, "Average Cl^SS, I")
    plot_log_cls(l_bins, cl_ss_e_avg_python, cl_ss_e_avg_julia, cl_std, "Average Cl^SS, E")
    plot_log_cls(l_bins, cl_ss_b_avg_python, cl_ss_b_avg_julia, cl_std, "Average Cl^SS, B")

    #---------------------------------------------------------------------------
    #NOISE AUTO SPECTRA COMPARISON
    #---------------------------------------------------------------------------
    plot_log_cls(l_bins, cl_nn_i_avg_python, cl_nn_i_avg_julia, cl_std, "Average Cl^NN, I")
    plot_log_cls(l_bins, cl_nn_e_avg_python, cl_nn_e_avg_julia, cl_std, "Average Cl^NN, E")
    plot_log_cls(l_bins, cl_nn_b_avg_python, cl_nn_b_avg_julia, cl_std, "Average Cl^NN, B")

    #---------------------------------------------------------------------------
    #DATA AUTO SPECTRA COMPARISON (i.e. Data = M * B * L * f + N)
    #---------------------------------------------------------------------------
    plot_log_cls(l_bins, cl_dd_i_avg_python, cl_dd_i_avg_julia, np.sqrt(2)*cl_std, "Average Cl^DD, I")
    plot_log_cls(l_bins, cl_dd_e_avg_python, cl_dd_e_avg_julia, np.sqrt(2)*cl_std, "Average Cl^DD, E")
    plot_log_cls(l_bins, cl_dd_b_avg_python, cl_dd_b_avg_julia, np.sqrt(2)*cl_std, "Average Cl^DD, B")
    return

# def generate_camb_auto_spectra_validation_plots(num_trials = 100, N = 256, uk_arcmin_t = 10, lmax = 17000, theta_pix = 2, delta_l = 50):

#     python_results = get_avg_cls(theta_pix = theta_pix, num_trials = num_trials, N = N, 
#                                  uk_arcmin_t = uk_arcmin_t, lmax = lmax, delta_l = delta_l)
#     cl_tt_i_avg_python = python_results["unlensed_temp_i"]
#     cl_tt_e_avg_python = python_results["unlensed_temp_e"]
#     cl_tt_b_avg_python = python_results["unlensed_temp_b"]
#     cl_ll_i_avg_python = python_results["lensed_temp_i"]
#     cl_ll_e_avg_python = python_results["lensed_temp_e"]
#     cl_ll_b_avg_python = python_results["lensed_temp_b"]
#     cl_dd_i_avg_python = python_results["data_i"]
#     cl_dd_e_avg_python = python_results["data_e"]
#     cl_dd_b_avg_python = python_results["data_b"]
#     cl_pp_avg_python = python_results["phi"]
#     cl_nn_i_avg_python = python_results["white_noise_i"]
#     cl_nn_e_avg_python = python_results["white_noise_e"]
#     cl_nn_b_avg_python = python_results["white_noise_b"]
#     cl_ss_i_avg_python = python_results["sum_total_i"]
#     cl_ss_e_avg_python = python_results["sum_total_e"]
#     cl_ss_b_avg_python = python_results["sum_total_b"]

#     # H0 = None 
#     # ombh2 = 0.0224567 
#     # omch2 = 0.118489 
#     # cosmomc_theta = 0.0104098
#     # r = 0.2
#     # mnu = 0.06 
#     # tau = 0.055, 
#     # As = np.exp(3.043) * 1e-10 
#     # nt = -0.2/8 #i.e. -r/8
#     # ns = 0.968602
#     # lmax = 17000
#     # k_pivot = 0.002
#     # Alens = 1

#     # #the lower-valued lmax_prime is used to get Dl's and Cl's from
#     # #camb and then we linearly extrapolate in log-log space higer ell
#     # #Dl's and Cl's using the higher original lmax value
#     # lmax_prime = min(lmax, 5000)
#     # #first generate the camb parameters object
#     # pars = camb.set_params(
#     #     H0 = H0, 
#     #     ombh2 = ombh2, 
#     #     omch2 = omch2, 
#     #     cosmomc_theta = cosmomc_theta,
#     #     r = r,
#     #     mnu = mnu, 
#     #     As = As,
#     #     nt = nt, 
#     #     ns = ns,
#     #     lmax = lmax_prime,
#     #     tau = tau, 
#     #     pivot_scalar = k_pivot,
#     #     pivot_tensor = k_pivot,
#     #     Alens = Alens)
    
#     pars = camb.set_params(
#         H0 = None, 
#         ombh2 = 0.0224567, 
#         omch2 = 0.118489, 
#         cosmomc_theta = 0.0104098,
#         r = 0.2,
#         mnu = 0.06, 
#         As = np.exp(3.043) * 1e-10,
#         nt = -0.025, 
#         ns = 0.968602,
#         lmax = 5000,
#         tau = 0.055, 
#         pivot_scalar = 0.002,
#         pivot_tensor = 0.002,
#         Alens = 1)
    
#     lmax_prime = 5000
#     pars.max_l_tensor = 2*lmax_prime
#     pars.max_eta_k_tensor = 4*lmax_prime
#     pars.WantScalars = True
#     pars.WantTensors = True
#     pars.DoLensing = True
#     pars.set_nonlinear_lensing(True)

#     #calculate results for these parameters
#     results = camb.get_results(pars)

#     #get the Dl's from camb
#     power_spectra = results.get_cmb_power_spectra(pars, lmax = lmax_prime - 1, CMB_unit = "muK")

#     #temperature Dl's
#     dl_tt_scalar = jnp.asarray(power_spectra["unlensed_scalar"][:,0])
#     dl_te_scalar = jnp.asarray(power_spectra["unlensed_scalar"][:,3])
#     dl_ee_scalar = jnp.asarray(power_spectra["unlensed_scalar"][:,1])
#     dl_bb_scalar = jnp.asarray(power_spectra["unlensed_scalar"][:,2])

#     dl_tt_tensor = jnp.asarray(power_spectra["tensor"][:,0])
#     dl_te_tensor = jnp.asarray(power_spectra["tensor"][:,3])
#     dl_ee_tensor = jnp.asarray(power_spectra["tensor"][:,1])
#     dl_bb_tensor = jnp.asarray(power_spectra["tensor"][:,2])

#     dl_tt_total = jnp.asarray(power_spectra["total"][:,0])
#     dl_ee_total = jnp.asarray(power_spectra["total"][:,1])
#     dl_bb_total = jnp.asarray(power_spectra["total"][:,2])

#     cl_tt_scalar = dl2cl(dl_tt_scalar, lmax, lmax_prime)
#     cl_te_scalar = dl2cl(dl_te_scalar, lmax, lmax_prime)
#     cl_ee_scalar = dl2cl(dl_ee_scalar, lmax, lmax_prime)
#     cl_bb_scalar = dl2cl(dl_bb_scalar, lmax, lmax_prime)

#     cl_tt_tensor = dl2cl(dl_tt_tensor, lmax, lmax_prime)
#     cl_te_tensor = dl2cl(dl_te_tensor, lmax, lmax_prime)
#     cl_ee_tensor = dl2cl(dl_ee_tensor, lmax, lmax_prime)
#     cl_bb_tensor = dl2cl(dl_bb_tensor, lmax, lmax_prime)

#     cl_tt_total = dl2cl(dl_tt_total, lmax, lmax_prime)
#     cl_ee_total = dl2cl(dl_ee_total, lmax, lmax_prime)
#     cl_bb_total = dl2cl(dl_bb_total, lmax, lmax_prime)

#     #lensing potential Dl's
#     dl_pp = jnp.asarray(results.get_lens_potential_cls(lmax = lmax_prime - 1)[:,0])
#     cl_pp = dl2cl(dl_pp, lmax, lmax_prime, is_phi = True)
#     cl_pp_camb_binned = bin_power_spectra(cl_pp, N, theta_pix, delta_l = 50, lmax = 17000)

#     plt.figure()
#     plt.plot(cl_pp_avg_python/cl_pp)
#     plt.show()
#     return

if __name__ == "__main__":
    #NOTE ~ 1 minute for 100 trials withOUT nphi
    #results = batch_simulated_trials(num_trials = 100)
    #results = load_sim(N = 256, theta_pix = 2,
    #                   uk_arcmin_t = 10,
    #                   seed = 67, lmax = 17_000)
    generate_auto_spectra_validation_plots(num_trials = 100, N = 256, uk_arcmin_t = 10, lmax = 17000, theta_pix = 2)
    #generate_camb_auto_spectra_validation_plots(num_trials = 10, N = 256, uk_arcmin_t = 10, lmax = 17000, theta_pix = 2, delta_l = 50)
    #field_1 = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/field_E.npz")
    #field_2 = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/field_B.npz")
    #ground_cls = precision_load("/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/f_Cl_EB.npz")
    #l_bins, predict_cls = power_spectra(field_1, theta_pix = 2, field_2 = field_2)
    print("Done!")
    #TODO Nphi, Cfl_[TT, TE, EE, BB] and auto-spectra validation.....
