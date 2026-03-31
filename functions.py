#imports
from imports import *
from matplotlib.animation import FuncAnimation, FFMpegWriter

#reusable plotting code to standardize plots
def plot_heat_map(heatmap, title, x_label, y_label, color_bar_label, save_fig = False, file_path = "", clim = None):
    plt.figure(figsize=(6, 5))
    if clim == None:
        plt.imshow(heatmap, cmap='coolwarm', origin='lower')
    else:
        plt.imshow(heatmap, cmap='coolwarm', origin='lower', clim = clim)
    plt.colorbar(label=color_bar_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if save_fig == True:
        plt.savefig(file_path)
        plt.close()
        return
    plt.show()
    return

#This function is mostly used in unit testing benchmarks
def percent_diff_2d(ground, predict):
    if np.linalg.norm(ground) != 0:
        percent_diff = np.linalg.norm(predict - ground) / np.linalg.norm(ground)
    else: #avoid divide by zero errors
        percent_diff = np.linalg.norm(predict - ground)
    return percent_diff

#reusable function for computing standard data science metrics
def compute_data_science_metrics(ground_truth, prediction):

    #Mean Squared Error
    mse = jnp.mean((ground_truth - prediction)**2)
    print("Mean Squared Error = "+str(mse))

    #normalized mean squared error
    nmse = mse / jnp.var(ground_truth)
    print("Normalized Mean Squared Error = "+str(nmse))

    #normalized root mean squared error
    rmse = jnp.sqrt(jnp.mean((prediction - ground_truth)**2))
    nrmse = rmse / jnp.std(ground_truth)
    print("Normalized Root Mean Squared Error = "+str(nrmse))

    #R^2 value
    numerator = jnp.sum((prediction - ground_truth)**2)
    denominator = jnp.sum((ground_truth - jnp.mean(ground_truth))**2)
    r2 = 1 - (numerator / denominator)
    print("R^2 value = "+str(r2))

    #return a tuple of the values of interest
    return mse, nmse, nrmse, r2

def precision_load(file_path):
    return jnp.array(np.load(file_path))

@jax.jit
def matrix_adjoint(matrix):
    #return the complex conjugate transpose
    return jnp.conj(matrix).T

# @jax.jit
# def reshape_diagonal(diagonal_matrix): #TODO get rid of magic numbers here
#     diagonal_length = len(diagonal_matrix)
#     num_rows = jnp.sqrt(diagonal_length).astype(jnp.int64)
#     num_cols = num_rows//2+1
#     return diagonal_matrix.reshape((num_rows, num_cols))

@jax.jit
def reshape_diagonal(diagonal_matrix):
    #apparently ndim is known ahead of time by jax so we do not
    #need to use a jax.lax.cond statement here
    if diagonal_matrix.ndim == 2:
        return diagonal_matrix
    diagonal_length = diagonal_matrix.shape[0]
    #for some reason, using numpy operations here and casting to
    #int does not throw an error whereas using jnp.astype does...
    num_rows = int(np.sqrt(2*diagonal_length))
    num_cols = num_rows // 2 + 1
    return diagonal_matrix.reshape((num_rows, num_cols))


@jax.jit
def logdet(matrix, fourier_weights):
    #take the natural log of the absolute value of each element in the covariance matrix
    #and then perform element-wise multiplication by the fourier weights
    log_abs_matrix = jnp.where(matrix != 0, jnp.log(jnp.abs(matrix)), 0)*fourier_weights[jnp.newaxis, :]
    #find the log determinants of thes resulting matrix
    log_det_value = jnp.sum(log_abs_matrix)
    log_det_sign = jnp.prod(jnp.sign(jnp.where(matrix != 0, matrix, 1)))
    return log_det_value, log_det_sign

@jax.jit
def reciprocal_matrix(matrix):
    #where each element is not zero return the reciprocal value otherwise if zero stay at zero
    return jnp.where(matrix != 0, 1.0/matrix, 0.0)

@jax.jit
def log_pdf_contribution(field_array, field_covar_inv, fourier_weights, num_pixels):
        #This statement is equivalent to sum(field^dagger * covar^-1 * field) with proper normalization
        return jnp.real(jnp.sum(jnp.conj(field_array) * field_covar_inv * field_array * fourier_weights[jnp.newaxis, :] * (1/num_pixels)))

#-----------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------- Custom AutoDiff Gradients of the phi contribution to the logpdf function using JAX ------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------

#Define a wrapper function specifically for gradients of the phi^Dagger * c_phi^-1 * phi term
@jax.custom_vjp
@jax.jit
def log_pdf_phi_contribution(phi_array, phi_covar_inv, fourier_weights, num_pixels):
    return log_pdf_contribution(phi_array, phi_covar_inv, fourier_weights, num_pixels)

#Forward pass of the custom VJP
@jax.jit
def log_pdf_phi_contribution_forward_pass(phi_array, phi_covar_inv, fourier_weights, num_pixels):
    
    #Compute primal output
    contribution = log_pdf_contribution(phi_array, phi_covar_inv, fourier_weights, num_pixels)

    #return the value of the unlensed field and also store any data needed by backward pass.
    return contribution, (phi_array, phi_covar_inv, fourier_weights, num_pixels)

#Backward pass of the custom vjp
@jax.jit
def log_pdf_phi_contribution_backwards_pass(res, g):
    
    #unpack the necessary data
    (phi_array, phi_covar_inv, fourier_weights, num_pixels) = res 

    #we need to rescale the gradient w.r.t. phi (Marius does this as well...)
    #(AutoDiff computes only a fraction of this result)
    #this is due to the fact that in practice we needed to multiply by the fourier weights and divide by
    #the number of pixels when finding the logpdf value, but in theory the gradient is just -C_phi^-1*phi for the phi_contribution
    grad_phi = 2*phi_covar_inv*phi_array

    #Use autodiff for the rest of the inputs.
    def log_pdf_phi_contribution_only_others(*other_inputs):
        return log_pdf_contribution(phi_array, *other_inputs)

    #Now we get their VJPs in one call:
    _, vjp_fun = jax.vjp(log_pdf_phi_contribution_only_others, phi_covar_inv, fourier_weights, num_pixels)
    other_gradients = vjp_fun(g) 

    #Return a gradient for EVERY input (either via a custom analytical gradient or via AutoDiff)
    return (g*grad_phi, *other_gradients)

# -----------------------------------------------------------------------------------------------------
# Register the two-pass rule for the first order contribution to the gradient of logpdf w.r.t. phi
# -----------------------------------------------------------------------------------------------------
log_pdf_phi_contribution.defvjp(log_pdf_phi_contribution_forward_pass, log_pdf_phi_contribution_backwards_pass)

@jax.jit
def get_spatial_derivatives(f, pix_width):
    Nx, Ny = f.shape
    kx = 2 * jnp.pi * jfft.fftfreq(Nx, pix_width)
    ky = 2 * jnp.pi * jfft.rfftfreq(Ny, pix_width)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    F = jfft.rfft2(f)
    Fx  = jfft.irfft2(1j * KX * F, s=f.shape)
    Fy  = jfft.irfft2(1j * KY * F, s=f.shape)
    Fxx = jfft.irfft2(-(KX**2) * F, s=f.shape)
    Fyy = jfft.irfft2(-(KY**2) * F, s=f.shape)
    Fxy = jfft.irfft2(-(KX * KY) * F, s=f.shape)

    #return the real valued fields
    return Fx, Fy, Fxx, Fxy, Fyy

@jax.jit
def get_fourier_derivatives(f, pix_width):
    Nx, _ = f.shape
    Ny = Nx
    kx = 2 * jnp.pi * jfft.fftfreq(Nx, pix_width)
    ky = 2 * jnp.pi * jfft.rfftfreq(Ny, pix_width)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    Fx  = 1j * KX * f
    Fy  = 1j * KY * f
    Fxx = -(KX**2) * f
    Fyy = -(KY**2) * f
    Fxy = -(KX * KY) * f

    #return the complex valued fourier fields
    return Fx, Fy, Fxx, Fxy, Fyy

@jax.jit
def get_m_matrix_components_at_time(time, d2phi_dx2, d2phi_dxdy, d2phi_dy2):
    #the derivatives of phi do not change with time so we can reuse their values
    m_xx = 1 + time*d2phi_dx2
    m_xy = time*d2phi_dxdy
    m_yy = 1 + time*d2phi_dy2
    m_yx = m_xy #NOTE this matrix is symmetric
    #return the components of the magnification matrix
    return m_xx, m_xy, m_yx, m_yy

@jax.jit
def get_inverse_matrix_components(m_xx, m_xy, m_yy, eps=1e-12):
    det = m_xx * m_yy - m_xy * m_xy
    #avoid divide by zero errors by supplying a small epsilon
    inv_det = 1.0 / (det+eps) 
    m_inv_xx =  m_yy * inv_det
    m_inv_xy = -m_xy * inv_det
    m_inv_yy =  m_xx * inv_det
    return m_inv_xx, m_inv_xy, m_inv_yy

@jax.jit
def get_p_vector_components(dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time):
    #begin by getting the inverse magnification matrix components
    m_xx, m_xy, _, m_yy = get_m_matrix_components_at_time(time, d2phi_dx2, d2phi_dxdy, d2phi_dy2) 
    #m_xy = m_yx so only one off-diagonal component is needed
    m_inv_xx, m_inv_xy, m_inv_yy = get_inverse_matrix_components(m_xx, m_xy, m_yy)
    #given the gradients and inverse magnification matrix components compute and return the p-vector components
    p_x = dphi_dx*m_inv_xx + dphi_dy*m_inv_xy
    p_y = dphi_dx*m_inv_xy + dphi_dy*m_inv_yy
    return p_x, p_y

#compute the v-vector components of v_vec = M^-1 * grad(f)
@jax.jit
def get_v_vector_components(time, f, d2phi_dx2, d2phi_dxdy, d2phi_dy2, pix_width):

    #the derivatives of f change throughout time so we need to compute them at each call
    df_dx, df_dy, _, _, _ = get_spatial_derivatives(f, pix_width)

    #get the inverse magnification matrix components
    m_xx, m_xy, _, m_yy = get_m_matrix_components_at_time(time, d2phi_dx2, d2phi_dxdy, d2phi_dy2) 
    #m_xy = m_yx via symmetry so only one off-diagonal component is needed
    m_inv_xx, m_inv_xy, m_inv_yy = get_inverse_matrix_components(m_xx, m_xy, m_yy)

    #compute the v vector components given the input data
    v_x = df_dx*m_inv_xx + df_dy*m_inv_xy
    v_y = df_dy*m_inv_yy + df_dx*m_inv_xy

    #return the calculated values
    return v_x, v_y

@jax.jit
def get_adjoint_lensing_term(f, grad_phi_x, grad_phi_y, d2_phi_dx2, d2_phi_dxdy, d2_phi_dy2, t, pix_width):
    p_vec_x, p_vec_y = get_p_vector_components(grad_phi_x, grad_phi_y, d2_phi_dx2, d2_phi_dxdy, d2_phi_dy2, t)
    divergence_x, _, _, _, _ = get_spatial_derivatives(p_vec_x*f, pix_width) # equivalent to <--> d/dx (f * (grad_phi * M^-1)_x)
    _, divergence_y, _, _, _ = get_spatial_derivatives(p_vec_y*f, pix_width) # equivalent to <--> d/dy (f * (grad_phi * M^-1)_y)
    adjoint_term = (divergence_x + divergence_y)
    return adjoint_term

@jax.jit
def get_standard_lensing_term(f, grad_phi_x, grad_phi_y, d2_phi_dx2, d2_phi_dxdy, d2_phi_dy2, t, pix_width):
    #compute the vector components of v = M^{-1} * grad_f
    vx, vy = get_v_vector_components(t, f, d2_phi_dx2, d2_phi_dxdy, d2_phi_dy2, pix_width)
    #return df/dt = grad_phi^T * M^{-1} * grad_f
    standard_lensing_term = grad_phi_x*vx + grad_phi_y*vy
    return standard_lensing_term

@jax.jit
def get_delta_phi_roc(f, delta_f, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width):
    #We need to begin by taking the point-wise product of 
    #the delta_f field and the x and y components of the gradient of f
    df_dx, df_dy, _, _, _ = get_spatial_derivatives(f, pix_width)
    fdf_product_x = delta_f * df_dx
    fdf_product_y = delta_f * df_dy
    #now we take the components of this "fdf product" vector and multiply
    #by the inverse magnification matrix on the left...
    #Let's first get the inverse magnification matrix components
    m_xx, m_xy, _, m_yy = get_m_matrix_components_at_time(time, d2phi_dx2, d2phi_dxdy, d2phi_dy2) 
    m_inv_xx, m_inv_xy, m_inv_yy = get_inverse_matrix_components(m_xx, m_xy, m_yy)
    #now we can compute the components we want
    m_inv_fdf_x = m_inv_xx*fdf_product_x + m_inv_xy*fdf_product_y
    m_inv_fdf_y = m_inv_xy*fdf_product_x + m_inv_yy*fdf_product_y
    #next we take the divergence of this peculiar vector
    delta_phi_div_term_x, _, _, _, _ = get_spatial_derivatives(m_inv_fdf_x, pix_width)
    _, delta_phi_div_term_y, _, _, _ = get_spatial_derivatives(m_inv_fdf_y, pix_width)
    delta_phi_div_term = delta_phi_div_term_x + delta_phi_div_term_y
    #we are almost there... Now we need to compute the tensor product between the m_inv_fdf vector and the standard p vector
    #then apply a "laplacian" style operator to these components
    p_x, p_y = get_p_vector_components(dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time)
    w_xx = p_x * m_inv_fdf_x
    w_xy = p_x * m_inv_fdf_y
    w_yx = p_y * m_inv_fdf_x
    w_yy = p_y * m_inv_fdf_y
    _, _, laplacian_xx, _, _  = get_spatial_derivatives(time*w_xx, pix_width)
    _, _, _, laplacian_xy, _ = get_spatial_derivatives(time*w_xy, pix_width)
    _, _, _, laplacian_yx, _ = get_spatial_derivatives(time*w_yx, pix_width)
    _, _, _, _, laplacian_yy = get_spatial_derivatives(time*w_yy, pix_width)
    laplacian_sum = laplacian_xx + laplacian_xy + laplacian_yx + laplacian_yy
    #the final form of the time rate of change of delta_phi is the laplacian sum minus the divergence term
    d_delta_phi_dt = laplacian_sum + delta_phi_div_term
    return d_delta_phi_dt

@jax.jit
def lensing_gradients_integration_step(time, y, args):
    #unpack the args array
    dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, pix_width = args

    #reshape 1D coupled raveled fields into 2D fields
    f, delta_f, delta_phi = y
    shape = dphi_dx.shape
    f = f.reshape(shape)
    #delta_f is a small parameter we will integrate in tandem with delta_phi
    delta_f = delta_f.reshape(shape)
    delta_phi = delta_phi.reshape(shape)

    #calculate the three rates of change for f, delta_phi and delta_f:
    #1. The df/dt term is just the normal lensing term applied to the full field "f"
    df_dt = get_standard_lensing_term(f, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width)

    #2. The d_delta_f/dt term appears to be the adjoint lensing term applied to delta_f
    d_delta_f_dt = get_adjoint_lensing_term(delta_f, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width)

    #3. The d_delta_phi_dt term is a little more involved... See the function definition for the inner working
    d_delta_phi_dt = get_delta_phi_roc(f, delta_f, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width)

    #return the three coupled ODE dynamics raveled up into 1D arrays
    return df_dt.ravel(), d_delta_f_dt.ravel(), d_delta_phi_dt.ravel()

@jax.jit
def single_lense_flow_step(t, y, args):
    #unpack the args array
    grad_phi_x, grad_phi_y, d2_phi_dx2, d2_phi_dxdy, d2_phi_dy2, pix_width, adjoint = args
    #reshape y into 2D field
    shape = grad_phi_x.shape
    f = y.reshape(shape)
    #the rate of change per pixel (i.e. the form of df/dt) has a different form 
    #depending on whether we are taking the adjoint or not
    operands = f, grad_phi_x, grad_phi_y, d2_phi_dx2, d2_phi_dxdy, d2_phi_dy2, t, pix_width
    rate = jax.lax.cond(
        adjoint,
        get_adjoint_lensing_term, #the adjoint rate of change is given by -1*divergence(p_vec*f)
        get_standard_lensing_term, #the non-adjoint rate of change is given by (grad_phi * v) <--> (grad_phi * M^-1 * grad_f)
        *operands
    )
    #return the flattened vector
    return rate.ravel() 

@jax.jit
def lense_flow(f, phi, pix_width, n, direction, adjoint, num_pixels, rescale_and_conjugate):

    #quiet the linter since this input is only used in AutoDiff code
    _ = num_pixels
    _ = rescale_and_conjugate

    #precompute phi partials
    grad_phi_x, grad_phi_y, d2_phi_dx2, d2_phi_dxdy, d2_phi_dy2 = get_spatial_derivatives(phi, pix_width)

    #default is forward lensing operations
    def forward(_):
        t0, t1 = 0.0, 1.0
        dt0 = 1.0/n
        return t0, t1, dt0

    #set to negative 1 for inverse lensing operations
    def inverse(_):
        t0, t1 = 1.0, 0.0
        dt0 = -1.0/n
        return t0, t1, dt0

    #jax trace requires if statements to be written like this
    t0, t1, dt0 = jax.lax.cond(
        jnp.equal(direction, INVERSE_LENSE),
        inverse,
        forward,
        operand = None
    )

    #ravel up 2D array into 1D array since this is required for the diffrax ode solver
    y0 = f.ravel()
    #store extra arguments in a single array
    args = (grad_phi_x, grad_phi_y, d2_phi_dx2, d2_phi_dxdy, d2_phi_dy2, pix_width, adjoint)
    #define a single step
    single_step_dynamics = ODETerm(single_lense_flow_step)
    ode_solver_method = Tsit5() #diffrax equivalent of RK45 need to use a non-stiff solver to avoid singluar matrix inversions
    stepsize_controller = PIDController(rtol = PID_CONTROLLER_RTOL, atol = PID_CONTROLLER_ATOL, dtmax=1/n) #use adaptive step sizes for ideally higher precision

    #call the ode solver
    sol = diffeqsolve(
        single_step_dynamics,
        ode_solver_method,
        t0=t0, #initial time
        t1=t1, #final time
        dt0=dt0, #initial guess for step
        y0=y0, #initial conditons
        args=args, #extra arguments
        stepsize_controller = stepsize_controller,
        adjoint = RecursiveCheckpointAdjoint(),
        max_steps = 100_000 #default is 4096
    )

    #get the last entry in the solution array
    y_final = jnp.asarray(sol.ys)[-1]
    return y_final.reshape(grad_phi_x.shape) #reshape the flattened form into the 2D format

@jax.jit
def get_m_or_b_dagger(m_or_b_fourier):

    #go back to real space, compute adjoint, transform back to fourier space
    m_or_b_matrix_real = jfft.irfft2(m_or_b_fourier)
    m_or_b_dagger_real = matrix_adjoint(m_or_b_matrix_real) #adjoint is complex conjugate transpose
    m_or_b_dagger_fourier = jfft.rfft2(m_or_b_dagger_real)

    return m_or_b_dagger_fourier

@jax.jit
def gradf_logpdf(f, phi, data, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, pix_width, num_pixels):

    #The analytical gradient of logpdf w.r.t f is given by:
    #grad_f = (L^dagger*B^dagger*M^dagger) * (C_n^-1) * (d - M*B*L*f) - (C_f^-1)*f
    #See e.g. marius' julia code in dataset.jl

    #get (C_f^-1)
    cf_matrix = reshape_diagonal(cf_diagonal)
    f_covar_inv = reciprocal_matrix(cf_matrix)
    #get (C_n^-1)
    cn_matrix = reshape_diagonal(cn_diagonal)
    noise_covar_inv = reciprocal_matrix(cn_matrix)

    #perform lense flow on f to compute L*f
    f_lensed = jfft.rfft2(lense_flow(jfft.irfft2(f), jfft.irfft2(phi), \
                                            pix_width, DEFAULT_NUM_LENSE_STEPS, FORWARD_LENSE, False, num_pixels, False))
    
    #IS THE BELOW CALCULATION OF M AND B DAGGER ACCURATE OR NECESSARY?
    #properly format the B and M matrices and their adjoints
    m_matrix_fourier = reshape_diagonal(m_diagonal)
    #IS THE BELOW CALCULATION OF M AND B DAGGER ACCURATE OR NECESSARY?
    #properly format the B and M matrices and their adjoints
    b_matrix_fourier = reshape_diagonal(b_diagonal)
    m_dagger_fourier = get_m_or_b_dagger(m_matrix_fourier) 
    b_dagger_fourier = get_m_or_b_dagger(b_matrix_fourier)
    
    #perform the adjoint of lenseflow on (B^dagger*M^dagger) * (C_n^-1) * (d - M*B*L*f)
    lhs = jfft.rfft2(lense_flow(jfft.irfft2(b_dagger_fourier*m_dagger_fourier*noise_covar_inv*\
        (data-m_matrix_fourier*b_matrix_fourier*f_lensed)), jfft.irfft2(phi), \
        pix_width, DEFAULT_NUM_LENSE_STEPS, INVERSE_LENSE, True, num_pixels, False))
        #backwards in time and use the adjoint version

    #add this result to C_f^-1 and return the result
    rhs = -1*f_covar_inv*f
    grad_f = lhs + rhs
    return grad_f

#-----------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------ Custom AutoDiff Gradients for LenseFlow using JAX --------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------

#Define a wrapper function that returns the same values as lense flow but has custom gradients defined for each of its parameters
#AND for BOTH the FORWARD and BACKWARDS directions!
@jax.custom_vjp
@jax.jit
def lense_flow_wrapper(f, phi, pix_width, n, direction, adjoint, num_pixels, rescale_and_conjugate):
    return lense_flow(f, phi, pix_width, n, direction, adjoint, num_pixels, rescale_and_conjugate)

#Forward pass of the custom VJP
@jax.jit
def lense_flow_forward_pass(f, phi, pix_width, n, direction, adjoint, num_pixels, rescale_and_conjugate):
    
    #Compute primal output
    f_transformed = lense_flow(f, phi, pix_width, n, direction, adjoint, num_pixels, rescale_and_conjugate)

    #return the value of the transformed field and also store any data needed by backward pass.
    return f_transformed, (f, phi, pix_width, n, direction, adjoint, f_transformed, num_pixels, rescale_and_conjugate)

#Backward pass of the custom vjp
@jax.jit
def lense_flow_backwards_pass(res, g):
    
    #unpack the necessary data
    (f, phi, pix_width, n, direction, adjoint, f_transformed, num_pixels, rescale_and_conjugate) = res 

    #get the value of the gradients of the lensing operator w.r.t. f and phi
    #NOTE WE NEED TO USE THE TRANSFORMED FIELD AS THE INPUT HERE AND INTEGRATE IN THE OPPOSITE DIRECTION
    _, delta_f, delta_phi = get_lensing_operator_gradients(phi, f_transformed, g, pix_width, -direction, n)
    
    def undo_logpdf_operations(delta_f, delta_phi):
        num_rows, _ = delta_phi.shape
        STANDARD_FOURIER_WEIGHTS = 2*jnp.ones(num_rows//2+1, dtype =jnp.complex128) #TODO remove magic number
        STANDARD_FOURIER_WEIGHTS = STANDARD_FOURIER_WEIGHTS.at[0].set(1)
        STANDARD_FOURIER_WEIGHTS = STANDARD_FOURIER_WEIGHTS.at[-1].set(1)

        delta_f_fourier = jnp.conj(jfft.rfft2(delta_f) / STANDARD_FOURIER_WEIGHTS[jnp.newaxis, :])
        delta_phi_fourier = jnp.conj(jfft.rfft2(delta_phi) / STANDARD_FOURIER_WEIGHTS[jnp.newaxis, :])

        delta_f = jfft.irfft2((num_pixels)*delta_f_fourier)
        delta_phi = jfft.irfft2((num_pixels)*delta_phi_fourier)
        return delta_f, delta_phi
    
    def maintain_current_deltas(delta_f, delta_phi):
        return delta_f, delta_phi

    operands = delta_f, delta_phi
    delta_f, delta_phi = jax.lax.cond(
        rescale_and_conjugate,
        undo_logpdf_operations,
        maintain_current_deltas,
        *operands
    )

    #Use autodiff for the rest of the inputs.
    def lense_flow_only_others(*other_inputs):
        return lense_flow(f, phi, *other_inputs)

    #Now we get their VJPs in one call:
    _, vjp_fun = jax.vjp(lense_flow_only_others, pix_width, n, direction, adjoint, num_pixels, rescale_and_conjugate)
    other_gradients = vjp_fun(g) 

    #Return a gradient for EVERY input (either via a custom analytical gradient or via AutoDiff)
    return (delta_f, delta_phi, *other_gradients)

# -----------------------------------------------------------
# Register the two-pass rule for the inverse lensing operator
# -----------------------------------------------------------
lense_flow_wrapper.defvjp(lense_flow_forward_pass, lense_flow_backwards_pass)

#---------------------------------------------------------------------------------------------
#LogPDF using the unlensed parametrizations f and phi as inputs
#---------------------------------------------------------------------------------------------
@jax.jit
def logpdf_unlensed_param(f_array, phi_array, data_array,
              b_diagonal, m_diagonal, 
              cf_diagonal, cphi_diagonal, cn_diagonal,
              f_lambda_array, phi_lambda_array, data_lambda_array, 
              cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width):

    #reshape the covariance matrix diagonals (need to specify Fortran style reshaping as Julia uses to be consistent)
    cf_matrix = reshape_diagonal(cf_diagonal)
    cphi_matrix = reshape_diagonal(cphi_diagonal)
    cn_matrix = reshape_diagonal(cn_diagonal)

    #find the log determinants of these matrices
    log_det_f_value, log_det_f_sign = logdet(cf_matrix, cf_lambda_array)
    log_det_phi_value, log_det_phi_sign = logdet(cphi_matrix, cphi_lambda_array)
    log_det_noise_value, log_det_noise_sign = logdet(cn_matrix, cn_lambda_array)
    
    #even though we reshaped the diagonal matrices to be rectangular, 
    #we compute the inverse as if they were diagonal matrices because they originally were
    f_covar_inv = reciprocal_matrix(cf_matrix)
    phi_covar_inv = reciprocal_matrix(cphi_matrix)
    noise_covar_inv = reciprocal_matrix(cn_matrix)
    
    #need to explicitly calculate f_lensed ourselves if we want to have the grad_f of logpdf be accurate
    #when computed by an AD (auto diff)
    #lense the field f, apply the M and B transformations and then plug into noise contribution term of logPDF
    f_lensed_array = lense_flow_wrapper(jfft.irfft2(f_array), jfft.irfft2(phi_array), \
                               pix_width, DEFAULT_NUM_LENSE_STEPS, FORWARD_LENSE, False, num_pixels, True)
    f_lensed_array = m_diagonal * b_diagonal * jfft.rfft2(f_lensed_array)

    #compute the contribution from each field term (the [:, jnp.newaxis] syntax is necessary to do
    #the kind of element wise multiplication we want to do for this type of calculation)
    f_contribution = log_pdf_contribution(f_array, f_covar_inv, f_lambda_array, num_pixels)
    phi_contribution = log_pdf_phi_contribution(phi_array, phi_covar_inv, phi_lambda_array, num_pixels)
    noise_contribution = log_pdf_contribution((data_array - f_lensed_array), noise_covar_inv, data_lambda_array, num_pixels)

    #the resulting logpdf value is the sum of the f, phi, and noise contributions 
    #along with their logdeterminants divided by negative 2
    result = -1*(f_contribution + phi_contribution + noise_contribution \
                 + log_det_f_value * log_det_f_sign + log_det_phi_value * log_det_phi_sign \
                 + log_det_noise_value * log_det_noise_sign)/2

    return result

#LogPDF function using the lensed parametrizations f_lensed and phi as inputs
@jax.jit
def logpdf_lensed_param(f_lensed_array, phi_array, data_array,
              b_diagonal, m_diagonal, 
              cf_diagonal, cphi_diagonal, cn_diagonal, 
              f_lambda_array, phi_lambda_array, data_lambda_array, 
              cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width):

    #reshape the covariance matrix diagonals (need to specify Fortran style reshaping as Julia uses to be consistent)
    cf_matrix = reshape_diagonal(cf_diagonal)
    cphi_matrix = reshape_diagonal(cphi_diagonal)
    cn_matrix = reshape_diagonal(cn_diagonal)

    #find the log determinants of these matrices
    log_det_f_value, log_det_f_sign = logdet(cf_matrix, cf_lambda_array)
    log_det_phi_value, log_det_phi_sign = logdet(cphi_matrix, cphi_lambda_array)
    log_det_noise_value, log_det_noise_sign = logdet(cn_matrix, cn_lambda_array)

    #even though we reshaped the diagonal matrices to be rectangular, 
    #we compute the inverse as if they were diagonal matrices because they originally were
    f_covar_inv = reciprocal_matrix(cf_matrix)
    phi_covar_inv = reciprocal_matrix(cphi_matrix)
    noise_covar_inv = reciprocal_matrix(cn_matrix)

    #need to explicitly calculate f_lensed ourselves if we want to have the grad_f of logpdf be accurate
    #when computed by an AD (auto diff)
    #DE-lense the field f_lensed to get f
    f_array = lense_flow_wrapper(jfft.irfft2(f_lensed_array), jfft.irfft2(phi_array), \
                               pix_width, DEFAULT_NUM_LENSE_STEPS, INVERSE_LENSE, False, num_pixels, True)
    #Transform back into Fourier space...
    f_array = jfft.rfft2(f_array)
    
    #compute the contribution from each field term (the [:, jnp.newaxis] syntax is necessary to do
    #the kind of element wise multiplication we want to do for this type of calculation)
    f_contribution = log_pdf_contribution(f_array, f_covar_inv, f_lambda_array, num_pixels)
    phi_contribution = log_pdf_phi_contribution(phi_array, phi_covar_inv, phi_lambda_array, num_pixels)

    #apply the B and M masks / filters to the lensed field
    f_lensed_array = m_diagonal * b_diagonal * f_lensed_array
    noise_contribution = log_pdf_contribution((data_array - f_lensed_array), noise_covar_inv, data_lambda_array, num_pixels)

    #the resulting logpdf value is the sum of the f, phi, and noise contributions 
    #along with their logdeterminants divided by negative 2
    result = -1*(f_contribution + phi_contribution + noise_contribution \
                 + log_det_f_value * log_det_f_sign + log_det_phi_value * log_det_phi_sign \
                 + log_det_noise_value * log_det_noise_sign)/2

    return result

#---------------------------------------------------------------------------------------------------------------------
#------------------------------------- Custom AutoDiff Gradients for logpdf ------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

#create a wrapper around the logpdf function using the unlensed parametrization
#in order to create custom AutoDiff gradients w.r.t. f and phi
@jax.custom_vjp
@jax.jit
def logpdf(f_array, phi_array, data_array,
                    b_diagonal, m_diagonal,
                    cf_diagonal, cphi_diagonal, cn_diagonal,
                    f_lambda_array, phi_lambda_array, data_lambda_array, 
                    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width):
    return logpdf_unlensed_param(f_array, phi_array, data_array, 
                    b_diagonal, m_diagonal,
                    cf_diagonal, cphi_diagonal, cn_diagonal,
                    f_lambda_array, phi_lambda_array, data_lambda_array, 
                    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)

#Forward pass of the custom VJP
@jax.jit
def logpdf_forward(f_array, phi_array, data_array,
                    b_diagonal, m_diagonal,
                    cf_diagonal, cphi_diagonal, cn_diagonal,
                    f_lambda_array, phi_lambda_array, data_lambda_array, 
                    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width):
    
    #Compute primal output
    value = logpdf_unlensed_param(f_array, phi_array, data_array,
                    b_diagonal, m_diagonal,
                    cf_diagonal, cphi_diagonal, cn_diagonal,
                    f_lambda_array, phi_lambda_array, data_lambda_array, 
                    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)

    #Store any data needed by backward pass.
    return value, (f_array, phi_array, data_array,
                    b_diagonal, m_diagonal,
                    cf_diagonal, cphi_diagonal, cn_diagonal,
                    f_lambda_array, phi_lambda_array, data_lambda_array, 
                    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)

#Backward pass (custom derivative of logpdf w.r.t. f and use autodiff w.r.t. phi and other parameters...)
@jax.jit
def logpdf_backwards(res, g):
    
    #unpack the necessary data
    (f_array, phi_array, data_array,
    b_diagonal, m_diagonal,
    cf_diagonal, cphi_diagonal, cn_diagonal,
    f_lambda_array, phi_lambda_array, data_lambda_array, 
    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width) = res 

    #get the value of the analytical gradient of logpdf w.r.t. f
    #we must use the unlensed field as the input to gradf_logpdf, NOT the lensed field
    gradf = gradf_logpdf(f_array, phi_array, data_array, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, pix_width, num_pixels)
    
    #Autodiff for gradient w.r.t. phi (should hook into dL^-1/dphi via AutoDiff chain rule...)
    grad_phi = (jax.grad(logpdf_unlensed_param, argnums = 1)(f_array, phi_array, data_array,
                                            b_diagonal, m_diagonal,
                                            cf_diagonal, cphi_diagonal, cn_diagonal,
                                            f_lambda_array, phi_lambda_array, data_lambda_array, 
                                            cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width))
    
    # Use autodiff for the rest of the inputs.
    # We must generate VJPs for the other inputs to logpdf which we do not care about differentiating w.r.t.
    # The easiest way is: define a helper function that only treats each of the "other" parameters as variables.
    def logpdf_only_others(*other_inputs):
        return logpdf_unlensed_param(f_array, phi_array, *other_inputs)
    
    # def logpdf_only_others(*other_inputs):
    #     return logpdf_unlensed_param(f_array, *other_inputs)

    #Now we get their VJPs in one call:
    _, vjp_fun = jax.vjp(logpdf_only_others, data_array,
                                            b_diagonal, m_diagonal, 
                                            cf_diagonal, cphi_diagonal, cn_diagonal,
                                            f_lambda_array, phi_lambda_array, data_lambda_array, 
                                            cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)
    others = vjp_fun(g) 

    #Return a gradient for EVERY input of logpdf_wrapped (either via a custom analytical gradient or via AutoDiff)
    return (g*gradf, g*grad_phi, *others)

    #  #Now we get their VJPs in one call:
    # _, vjp_fun = jax.vjp(logpdf_only_others, phi_array, data_array,
    #                                         b_diagonal, m_diagonal, 
    #                                         cf_diagonal, cphi_diagonal, cn_diagonal,
    #                                         f_lambda_array, phi_lambda_array, data_lambda_array, 
    #                                         cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)
    # others = vjp_fun(g) 

    # #Return a gradient for EVERY input of logpdf_wrapped (either via a custom analytical gradient or via AutoDiff)
    # return (g*gradf, *others)

# -----------------------------------------------------------
# Register the two-pass rule for the logpdf function
# -----------------------------------------------------------
logpdf.defvjp(logpdf_forward, logpdf_backwards)

#create a wrapper around the logpdf function using the LENSED parametrization
#in order to create custom AutoDiff gradients w.r.t. f and phi
@jax.custom_vjp
@jax.jit
def logpdf_lensed_wrapper(f_lensed_array, phi_array, data_array,
                    b_diagonal, m_diagonal,
                    cf_diagonal, cphi_diagonal, cn_diagonal,
                    f_lambda_array, phi_lambda_array, data_lambda_array, 
                    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width):
    return logpdf_lensed_param(f_lensed_array, phi_array, data_array, 
                    b_diagonal, m_diagonal,
                    cf_diagonal, cphi_diagonal, cn_diagonal,
                    f_lambda_array, phi_lambda_array, data_lambda_array, 
                    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)

#Forward pass of the custom VJP
@jax.jit
def logpdf_lensed_forward(f_lensed_array, phi_array, data_array,
                    b_diagonal, m_diagonal,
                    cf_diagonal, cphi_diagonal, cn_diagonal,
                    f_lambda_array, phi_lambda_array, data_lambda_array, 
                    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width):
    
    #Compute primal output
    value = logpdf_lensed_param(f_lensed_array, phi_array, data_array,
                    b_diagonal, m_diagonal,
                    cf_diagonal, cphi_diagonal, cn_diagonal,
                    f_lambda_array, phi_lambda_array, data_lambda_array, 
                    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)

    #Store any data needed by backward pass.
    return value, (f_lensed_array, phi_array, data_array,
                    b_diagonal, m_diagonal,
                    cf_diagonal, cphi_diagonal, cn_diagonal,
                    f_lambda_array, phi_lambda_array, data_lambda_array, 
                    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)

#Backward pass (custom derivative of logpdf w.r.t. f and use autodiff w.r.t. phi and other parameters...)
@jax.jit
def logpdf_lensed_backwards(res, g):
    
    #unpack the necessary data
    (f_lensed_array, phi_array, data_array,
    b_diagonal, m_diagonal,
    cf_diagonal, cphi_diagonal, cn_diagonal,
    f_lambda_array, phi_lambda_array, data_lambda_array, 
    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width) = res 

    #get the value of the analytical gradient of logpdf w.r.t. f
    #we must use the unlensed field as the input to gradf_logpdf, NOT the lensed field
    f_array = jfft.rfft2(lense_flow(jfft.irfft2(f_lensed_array), jfft.irfft2(phi_array), \
                         pix_width, DEFAULT_NUM_LENSE_STEPS, INVERSE_LENSE, False, num_pixels, False))
    gradf = gradf_logpdf(f_array, phi_array, data_array, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, pix_width, num_pixels)
    #NOTE should the above gradient w.r.t. f be the same as in the unlensed param? Could be a point of error if this
    #ends up not working out in the future...

    #Autodiff for gradient w.r.t. phi (should hook into dL^-1/dphi via AutoDiff chain rule...)
    grad_phi = (jax.grad(logpdf_lensed_param, argnums = 1)(f_lensed_array, phi_array, data_array,
                                            b_diagonal, m_diagonal,
                                            cf_diagonal, cphi_diagonal, cn_diagonal,
                                            f_lambda_array, phi_lambda_array, data_lambda_array, 
                                            cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width))
    
    # Use autodiff for the rest of the inputs.
    # We must generate VJPs for the other inputs to logpdf which we do not care about differentiating w.r.t.
    # The easiest way is: define a helper function that only treats each of the "other" parameters as variables.
    def logpdf_only_others(*other_inputs):
        return logpdf_lensed_param(f_lensed_array, phi_array, *other_inputs)

    #Now we get their VJPs in one call:
    _, vjp_fun = jax.vjp(logpdf_only_others, data_array,
                                            b_diagonal, m_diagonal, 
                                            cf_diagonal, cphi_diagonal, cn_diagonal,
                                            f_lambda_array, phi_lambda_array, data_lambda_array, 
                                            cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)
    others = vjp_fun(g) 

    #Return a gradient for EVERY input of logpdf_wrapped (either via a custom analytical gradient or via AutoDiff)
    return (g*gradf, g*grad_phi, *others)

# -----------------------------------------------------------
# Register the two-pass rule for the logpdf function
# -----------------------------------------------------------
logpdf_lensed_wrapper.defvjp(logpdf_lensed_forward, logpdf_lensed_backwards)

@jax.jit
def get_lensing_operator_gradients(phi, f, delta_f_init, pix_width, direction=1, n=10):

    #precompute phi partials - need to convert back to real space to do this
    dphi_dx, dphi_dy, d2_phi_dx2, d2_phi_dxdy, d2_phi_dy2 = get_spatial_derivatives(phi, pix_width)

    #default is forward integration operations
    def forward(_):
        t0, t1 = 0.0, 1.0
        dt0 = 1.0/n
        return t0, t1, dt0

    #set to negative 1 for inverse integration operations
    def inverse(_):
        t0, t1 = 1.0, 0.0
        dt0 = -1.0/n
        return t0, t1, dt0

    #jax trace requires if statements to be written like this
    t0, t1, dt0 = jax.lax.cond(
        jnp.equal(direction, INVERSE_LENSE),
        inverse,
        forward,
        operand = None
    )

    #initialize delta_phi to be 0 as stated in Marius' paper
    shape = dphi_dx.shape
    delta_phi = jnp.zeros(shape)
    #ravel up the three 2D arrays into 1D arrays since this is required for the diffrax ode solver
    y0 = (f.ravel(), delta_f_init.ravel(), delta_phi.ravel())
    #store extra arguments in a single array
    args = (dphi_dx, dphi_dy, d2_phi_dx2, d2_phi_dxdy, d2_phi_dy2, pix_width)

    #define a single step
    single_step_dynamics = ODETerm(lensing_gradients_integration_step)
    ode_solver_method = Tsit5() #diffrax equivalent of RK45 need to use a non-stiff solver to avoid singluar matrix inversions
    stepsize_controller = PIDController(rtol = PID_CONTROLLER_RTOL, atol = PID_CONTROLLER_ATOL) #use adaptive step sizes for ideally higher precision

    #call the ode solver
    sol = diffeqsolve(
        single_step_dynamics,
        ode_solver_method,
        t0=t0, #initial time
        t1=t1, #final time
        dt0=dt0, #initial guess for step
        y0=y0, #initial conditons
        args=args,
        stepsize_controller = stepsize_controller
    )

    #get the last entry in the solution array for the gradient term:
    #Index 0 holds the lensed field term, Index 1 the delta_f term, and Index 3 the delta_phi term
    f_real = jnp.asarray(sol.ys[0][-1])
    f_real = f_real.reshape(shape) #reshape flattened vector back into 2D shape

    delta_f_real = jnp.asarray(sol.ys[1][-1])
    delta_f_real = delta_f_real.reshape(shape)

    delta_phi_real = jnp.asarray(sol.ys[2][-1])
    delta_phi_real = delta_phi_real.reshape(shape)

    #return the tuple of fields 
    return f_real, delta_f_real, delta_phi_real

@jax.jit
def wiener_filter_matrix_operator(f, phi, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, pix_width, num_pixels):
    #The A matrix operator is the gradient of logpdf w.r.t. f evaluated at d = 0 and other inputs at their current values
    data = jnp.zeros(f.shape)
    return (gradf_logpdf(f, phi, data, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, pix_width, num_pixels))  

@jax.jit
def wiener_filter_b_vector(phi, data, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, pix_width, num_pixels):
    #The b vector is the gradient of logpdf w.r.t. f evaluated at f = 0 and other inputs at their current values
    f = jnp.zeros(data.shape)
    return -1*gradf_logpdf(f, phi, data, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, pix_width, num_pixels)

@jax.jit
def hessian_logpdf_preconditioner(cf_diagonal, b_diagonal, m_diagonal, cn_diagonal):

    #find the inverse covariance matrices needed
    cf_covar_inv = reciprocal_matrix(reshape_diagonal(cf_diagonal))
    cn_covar_inv = reciprocal_matrix(reshape_diagonal(cn_diagonal))

    #find the M and B dagger matrices
    b_matrix = reshape_diagonal(b_diagonal)
    m_matrix = reshape_diagonal(m_diagonal)

    b_dagger = get_m_or_b_dagger(b_matrix)
    m_dagger = get_m_or_b_dagger(m_matrix)

    #the preconditioner is equal to Cf^-1 + B^Dagger * M^Dagger * Cn^-1 * M * B
    preconditioner = cf_covar_inv + b_dagger * m_dagger * cn_covar_inv * m_matrix * b_matrix
    return preconditioner

#this is the matrix we multiply grad_phi and the step_size by 
#to find the approximate step in the phi direction during gradient descent
@jax.jit
def phi_gradient_hessian(cphi_diagonal, nphi_diagonal):
    
    cphi_inv = reciprocal_matrix(reshape_diagonal(cphi_diagonal))
    nphi_inv = reciprocal_matrix(reshape_diagonal(nphi_diagonal))
    
    #hessian = Cphi^-1 + Nphi^-1
    hessian = cphi_inv + nphi_inv
    return hessian

#wiener filter implementation
@jax.jit
def wiener_filter(f, phi, data, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, fourier_weights, pix_width, num_pixels):

    #Compute the b array which is the gradient w.r.t. f of the logpdf 
    #function evaluated at f = 0, d = d, phi = phi, etc...
    b = wiener_filter_b_vector(phi, data, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, pix_width, num_pixels)

    #use a preconditioner to speed up the calculation
    preconditioner = hessian_logpdf_preconditioner(cf_diagonal, b_diagonal, m_diagonal, cn_diagonal)

    #take the conjugate gradient of A @ f = b and solve for f assuming
    f_wiener_filtered = conjugate_gradient(phi, m_diagonal, b_diagonal,
            cf_diagonal, cn_diagonal, pix_width, 
            b, f, preconditioner, fourier_weights, num_pixels, maxiter = 500, tol = 1e-1)

    #return the value of f found via conjugate gradient. This is the
    #wiener filtered version of f... this will be the real space version
    return f_wiener_filtered

#given f and phi and the mixing matrices in fourier space, return the mixed f and mixed phi in fourier space
@jax.jit
def mix(f, phi, d_diagonal, g_diagonal, pix_width, num_pixels):

    #reshape the diagonals into the propoer format
    d_matrix = reshape_diagonal(d_diagonal)
    g_matrix = reshape_diagonal(g_diagonal)

    #f_mixed = L(phi) * D * f
    f_mixed = jfft.rfft2(lense_flow(jfft.irfft2(d_matrix * f), jfft.irfft2(phi), \
                         pix_width, DEFAULT_NUM_LENSE_STEPS, FORWARD_LENSE, False, num_pixels, False))
    
    #phi_mixed = G * phi
    phi_mixed = g_matrix * phi

    #return the mixed tuple in fourier space
    return f_mixed, phi_mixed

@jax.jit
def unmix(f_mixed, phi_mixed, d_diagonal, g_diagonal, pix_width, num_pixels):

    #apply the inverse G operator to phi mixed to get back the original phi
    g_matrix = reshape_diagonal(g_diagonal)
    g_inv = reciprocal_matrix(g_matrix)
    phi = g_inv * phi_mixed

    #delense the field f_mixed then multiply by the inverse D matrix
    #to get back the original field f
    d_matrix = reshape_diagonal(d_diagonal)
    d_inv = reciprocal_matrix(d_matrix)
    f = d_inv * jfft.rfft2(lense_flow_wrapper(jfft.irfft2(f_mixed), jfft.irfft2(phi), \
                              pix_width, DEFAULT_NUM_LENSE_STEPS, INVERSE_LENSE, False, num_pixels, False))
    
    #return the unmixed f and phi
    return f, phi

#try to define a custom conjugate gradient which works in fourier space rather than real space
@partial(jax.jit, static_argnames = ["maxiter", "tol"])
def conjugate_gradient(phi, m_diagonal, b_diagonal,
            cf_diagonal, cn_diagonal, pix_width, 
            b, x, M, fourier_weights, num_pixels, maxiter = 500, tol = 1e-1):

    #Compute the A matrix which is the gradient w.r.t. f of the logpdf
    #function evaluated at f = f, d = 0, phi = phi, etc..
    def A(field):
        return wiener_filter_matrix_operator(
            field, phi, m_diagonal, b_diagonal,
            cf_diagonal, cn_diagonal, pix_width, 
            num_pixels
        )

    r = b - A(x) #compute the 1st residual base on the initial guess x0
    M_inv = jnp.where(M != 0, 1/M, 0)
    z =  M_inv * r #compute z = (M^-1 @ r) = (M \ r) since M is diagonal 
    p = z #copy the value of z
    #compute the dot product in fourier space between r and z
    res = jnp.real(jnp.sum(jnp.conj(r) * z * fourier_weights[jnp.newaxis, :] * (1/num_pixels)))
    #define the initial state
    initial_state = (x, res, res, p, r, 0)

    def loop_condition(state):
        _, _, res_curr, _, _, step_idx = state
        return jnp.logical_and(res_curr >= tol, step_idx < maxiter)
    
    def main_loop(state):
        x, res, res_curr, p, r, step_idx = state
        #compute Ap = A(p)
        Ap = A(p)
        #compute alpha = res / dot(p, Ap)
        alpha = res / jnp.real(jnp.sum(jnp.conj(p) * Ap * fourier_weights[jnp.newaxis, :] * (1/num_pixels)))
        #compute x = x + alpha * p (alpha is a number, p is an array, x is an array)
        x = x + alpha * p
        #compute r = r - alpha * Ap
        r = r - alpha * Ap
        #compute z = (M \ r) = (M^-1 @ r)
        z = M_inv * r
        #current value of the residual
        res_curr = jnp.real(jnp.sum(jnp.conj(r) * z * fourier_weights[jnp.newaxis, :] * (1/num_pixels)))
        #update the p value
        p = z + (res_curr / res) * p
        #otherwise update previous res value and keep looping
        res = res_curr
        #update the step index
        step_idx += 1
        return (x, res, res_curr, p, r, step_idx)
    
    #perform a jitted while-loop
    final_state = jax.lax.while_loop(loop_condition, main_loop, initial_state)

    #after max iterations have been looped through return the final value
    return final_state[0]

#define function which computes the predicted phi, f, and f_lensed
@partial(jax.jit, static_argnames = ["num_steps",  "num_trials"])
def run_gradient_descent_v1(data_set, num_steps, num_trials = 10):

    #unpack the necessary data from the data set object
    data = data_set["data"]
    m_diagonal = data_set["m_diagonal"]
    b_diagonal = data_set["b_diagonal"]
    d_diagonal = data_set["d_diagonal"]
    g_diagonal = data_set["g_diagonal"]
    cf_diagonal = data_set["cf_diagonal"]
    cn_diagonal = data_set["cn_diagonal"]
    cphi_diagonal = data_set["cphi_diagonal"]
    nphi_diagonal = data_set["nphi_diagonal"]
    f_lambda_array = data_set["f_lambda_array"] #TODO replace all these specific fourier weights with just the standard fourier weights...
    phi_lambda_array = data_set["phi_lambda_array"]
    data_lambda_array = data_set["data_lambda_array"]
    cf_lambda_array = data_set["cf_lambda_array"]
    cphi_lambda_array = data_set["cphi_lambda_array"]
    cn_lambda_array = data_set["cn_lambda_array"]
    pix_width = data_set["pix_width"]
    num_pixels = data_set["num_pixels"]
    
    #compute number of rows and columns in fourier space given map size
    num_rows, num_cols = data.shape

    #initialize the history object to store step ny step info
    history = {
        "alpha_values": jnp.zeros(num_steps, dtype = jnp.float64),
        "logpdf_values": jnp.zeros(num_steps, dtype = jnp.float64)
    }

    #set starting guesses for f, phi, and f_lensed to all zeros
    f_predict = phi_predict = jnp.zeros((num_rows, num_cols), dtype=jnp.complex128)

    #compute the hessian which we will use to calculate the step in the phi direction 
    hessian = reciprocal_matrix(phi_gradient_hessian(cphi_diagonal, nphi_diagonal))

    def loop_condition(state):
        f_predict, phi_predict, data, m_diagonal, b_diagonal, cf_diagonal, \
        d_diagonal, g_diagonal, cphi_diagonal, cn_diagonal, \
        phi_lambda_array, data_lambda_array, cf_lambda_array, cphi_lambda_array, \
        cn_lambda_array, f_lambda_array, pix_width, num_pixels, prev_grad_norm, alpha, step_idx = state
        return step_idx < num_steps
    
    def main_loop(state):

        #unpack the current state
        f_predict, phi_predict, data, m_diagonal, b_diagonal, cf_diagonal, \
        d_diagonal, g_diagonal, cphi_diagonal, cn_diagonal, \
        phi_lambda_array, data_lambda_array, cf_lambda_array, cphi_lambda_array, \
        cn_lambda_array, f_lambda_array, pix_width, num_pixels, prev_grad_norm, alpha, step_idx = state

        #compute the wiener filter of f
        f_predict = wiener_filter(f_predict, phi_predict, data, m_diagonal, \
                            b_diagonal, cf_diagonal, cn_diagonal, f_lambda_array, pix_width, num_pixels) 
        
        #must mix f and phi before taking the gradient
        f_mixed, phi_mixed = mix(f_predict, phi_predict, d_diagonal, g_diagonal, pix_width, num_pixels)
        mixed_grad_phi = mixed_phi_gradient(f_mixed, phi_mixed, data,
                                b_diagonal, m_diagonal, d_diagonal, g_diagonal,
                                cf_diagonal, cphi_diagonal, cn_diagonal,
                                f_lambda_array, phi_lambda_array, data_lambda_array, 
                                cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)
        
        #grad_norm = jnp.linalg.norm(mixed_grad_phi)
        #norm_perc_change = 1 - grad_norm/prev_grad_norm
        #jax.debug.print("step number = {}, delta norm = {}", step_idx, norm_perc_change, ordered = True)

        def min_logpdf(eta):
            return logpdf_at_phi_new(eta, grad_phi = mixed_grad_phi, f_predict = f_mixed, phi_predict = phi_mixed, \
                    data = data, b_diagonal = b_diagonal, m_diagonal = m_diagonal, cf_diagonal = cf_diagonal, \
                    cphi_diagonal = cphi_diagonal, cn_diagonal = cn_diagonal, d_diagonal = d_diagonal, \
                    g_diagonal = g_diagonal, f_lambda_array = f_lambda_array, \
                    phi_lambda_array = phi_lambda_array, data_lambda_array = data_lambda_array, \
                    cf_lambda_array = cf_lambda_array, cphi_lambda_array = cphi_lambda_array, \
                    cn_lambda_array = cn_lambda_array, num_pixels = num_pixels, pix_width = pix_width, hessian = hessian)

        #vmap style line search over 10 values on the range 0.1 to 2 times the previous alpha
        line_search = jnp.linspace(0.1, 2*alpha, num_trials)
        v_min_logpdf = jax.vmap(min_logpdf)
        results = v_min_logpdf(line_search)
        best_idx = jnp.argmin(results)
        alpha = line_search[best_idx]
        
        #update phi based on the best alpha choice we found from the vmap search
        phi_predict = phi_predict + alpha * hessian * mixed_grad_phi
        #jax.debug.print("step number = {}, best alpha = {}", step_idx, alpha, ordered = True)

        #update the step index
        step_idx += 1
        #set previous grad norm to current norm
        #prev_grad_norm = grad_norm

        return (f_predict, phi_predict, data, m_diagonal, b_diagonal, cf_diagonal, \
                d_diagonal, g_diagonal, cphi_diagonal, cn_diagonal, \
                phi_lambda_array, data_lambda_array, cf_lambda_array, cphi_lambda_array, \
                cn_lambda_array, f_lambda_array, pix_width, num_pixels, prev_grad_norm, alpha, step_idx)
    
    #perform a jitted while-loop
    initial_state = f_predict, phi_predict, data, m_diagonal, b_diagonal, cf_diagonal, \
                    d_diagonal, g_diagonal, cphi_diagonal, cn_diagonal, \
                    phi_lambda_array, data_lambda_array, cf_lambda_array, cphi_lambda_array, \
                    cn_lambda_array, f_lambda_array, pix_width, num_pixels, 1, 1, 0 
    final_state = jax.lax.while_loop(loop_condition, main_loop, initial_state)

    #return the three predicted values and the history object
    f_predict = final_state[0]
    phi_predict = final_state[1]
    return f_predict, phi_predict, history

#define function which computes the predicted phi, f, and f_lensed
@partial(jax.jit, static_argnames = ["num_steps"])
def run_gradient_descent_v2(data_set, num_steps):

    #unpack the necessary data from the data set object
    data = data_set["data"]
    m_diagonal = data_set["m_diagonal"]
    b_diagonal = data_set["b_diagonal"]
    d_diagonal = data_set["d_diagonal"]
    g_diagonal = data_set["g_diagonal"]
    cf_diagonal = data_set["cf_diagonal"]
    cn_diagonal = data_set["cn_diagonal"]
    cphi_diagonal = data_set["cphi_diagonal"]
    nphi_diagonal = data_set["nphi_diagonal"]
    f_lambda_array = data_set["f_lambda_array"] #TODO replace all these specific fourier weights with just the standard fourier weights...
    phi_lambda_array = data_set["phi_lambda_array"]
    data_lambda_array = data_set["data_lambda_array"]
    cf_lambda_array = data_set["cf_lambda_array"]
    cphi_lambda_array = data_set["cphi_lambda_array"]
    cn_lambda_array = data_set["cn_lambda_array"]
    pix_width = data_set["pix_width"]
    num_pixels = data_set["num_pixels"]
    
    #compute number of rows and columns in fourier space given map size
    num_rows, num_cols = data.shape

    #initialize the history object to store step ny step info
    history = {
        "logpdf_values": jnp.zeros(num_steps, dtype = jnp.float64)
    }

    #set starting guesses for f, phi, and f_lensed to all zeros
    f_predict = phi_predict = jnp.zeros((num_rows, num_cols), dtype=jnp.complex128)

    #compute the hessian which we will use to calculate the step in the phi direction 
    hessian = reciprocal_matrix(phi_gradient_hessian(cphi_diagonal, nphi_diagonal))

    def loop_condition(state):
        f_predict, phi_predict, data, m_diagonal, b_diagonal, cf_diagonal, \
        d_diagonal, g_diagonal, cphi_diagonal, cn_diagonal, \
        phi_lambda_array, data_lambda_array, cf_lambda_array, cphi_lambda_array, \
        cn_lambda_array, f_lambda_array, pix_width, num_pixels, prev_grad_norm, step_idx = state
        return step_idx < num_steps
    
    def main_loop(state):

        #unpack the current state
        f_predict, phi_predict, data, m_diagonal, b_diagonal, cf_diagonal, \
        d_diagonal, g_diagonal, cphi_diagonal, cn_diagonal, \
        phi_lambda_array, data_lambda_array, cf_lambda_array, cphi_lambda_array, \
        cn_lambda_array, f_lambda_array, pix_width, num_pixels, prev_grad_norm, step_idx = state

        #compute the wiener filter of f
        f_predict = wiener_filter(f_predict, phi_predict, data, m_diagonal, \
                            b_diagonal, cf_diagonal, cn_diagonal, f_lambda_array, pix_width, num_pixels) 
        
        #must mix f and phi before taking the gradient
        f_mixed, phi_mixed = mix(f_predict, phi_predict, d_diagonal, g_diagonal, pix_width, num_pixels)
        mixed_grad_phi = mixed_phi_gradient(f_mixed, phi_mixed, data,
                                b_diagonal, m_diagonal, d_diagonal, g_diagonal,
                                cf_diagonal, cphi_diagonal, cn_diagonal,
                                f_lambda_array, phi_lambda_array, data_lambda_array, 
                                cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)
        
        grad_norm = jnp.linalg.norm(mixed_grad_phi)
        norm_perc_change = 1 - grad_norm/prev_grad_norm
        #jax.debug.print("step number = {}, delta norm = {}", step_idx, norm_perc_change, ordered = True)

        #NOTE in this version of the descent algo, we just use a constant alpha
        #step size of 1 to avoid the random and inefficient line search, which seemingly adds
        #minimal accuracy improvements??        
        phi_predict = phi_predict + hessian * mixed_grad_phi

        #update the step index
        step_idx += 1
        #set previous grad norm to current norm
        prev_grad_norm = grad_norm

        return (f_predict, phi_predict, data, m_diagonal, b_diagonal, cf_diagonal, \
                d_diagonal, g_diagonal, cphi_diagonal, cn_diagonal, \
                phi_lambda_array, data_lambda_array, cf_lambda_array, cphi_lambda_array, \
                cn_lambda_array, f_lambda_array, pix_width, num_pixels, prev_grad_norm, step_idx)
    
    #perform a jitted while-loop
    initial_state = f_predict, phi_predict, data, m_diagonal, b_diagonal, cf_diagonal, \
                    d_diagonal, g_diagonal, cphi_diagonal, cn_diagonal, \
                    phi_lambda_array, data_lambda_array, cf_lambda_array, cphi_lambda_array, \
                    cn_lambda_array, f_lambda_array, pix_width, num_pixels, 1, 0 
    final_state = jax.lax.while_loop(loop_condition, main_loop, initial_state)

    #return the three predicted values and the history object
    f_predict = final_state[0]
    phi_predict = final_state[1]
    return f_predict, phi_predict, history

#define function which computes the predicted phi, f, and f_lensed
#NOTE this function is NOT JIT-ted even though all the functions inside of it
#are JIT-ted (minus the line search...)
def run_gradient_descent_v3(data_set, num_steps, save_animation = False):

    #unpack the necessary data from the data set object
    # data = data_set["data"]
    # m_diagonal = data_set["m_diagonal"]
    # b_diagonal = data_set["b_diagonal"]
    # d_diagonal = data_set["d_diagonal"]
    # g_diagonal = data_set["g_diagonal"]
    # cf_diagonal = data_set["cf_diagonal"]
    # cn_diagonal = data_set["cn_diagonal"]
    # cphi_diagonal = data_set["cphi_diagonal"]
    # nphi_diagonal = data_set["nphi_diagonal"]
    # f_lambda_array = data_set["f_lambda_array"] #TODO replace all these specific fourier weights with just the standard fourier weights...
    # phi_lambda_array = data_set["phi_lambda_array"]
    # data_lambda_array = data_set["data_lambda_array"]
    # cf_lambda_array = data_set["cf_lambda_array"]
    # cphi_lambda_array = data_set["cphi_lambda_array"]
    # cn_lambda_array = data_set["cn_lambda_array"]
    # pix_width = data_set["pix_width"]
    # num_pixels = data_set["num_pixels"]

    data = data_set["data"]
    m_diagonal = data_set["m"]
    b_diagonal = data_set["b"]
    d_diagonal = data_set["d_matrix"]
    g_diagonal = data_set["g"]
    cf_diagonal = data_set["cf"]
    cn_diagonal = data_set["cn"]
    cphi_diagonal = data_set["cphi"]
    nphi_diagonal = data_set["nphi"]
    f_lambda_array = data_set["fourier_weights"] #TODO replace all these specific fourier weights with just the standard fourier weights...
    phi_lambda_array = data_set["fourier_weights"]
    data_lambda_array = data_set["fourier_weights"]
    cf_lambda_array = data_set["fourier_weights"]
    cphi_lambda_array = data_set["fourier_weights"]
    cn_lambda_array = data_set["fourier_weights"]
    pix_width = data_set["pix_width"]
    num_pixels = data_set["num_pixels"]
    
    #compute number of rows and columns in fourier space given map size
    num_rows, num_cols = data.shape

    #set starting guesses for f, phi, and f_lensed to all zeros
    f_predict = phi_predict = jnp.zeros((num_rows, num_cols), dtype=jnp.complex128)

    #initialize the history object to store step ny step info
    history = {
        "alpha_values": jnp.zeros(num_steps, dtype = jnp.float64),
        "logpdf_values": jnp.zeros(num_steps, dtype = jnp.float64),
        "phi_frames": [jfft.irfft2(phi_predict)],
        "f_frames": [jfft.irfft2(f_predict)],
    }

    #compute the hessian which we will use to calculate the step in the phi direction 
    hessian = reciprocal_matrix(phi_gradient_hessian(cphi_diagonal, nphi_diagonal))

    #loop for num_steps iterations
    for step_idx in range(num_steps):

        #compute the wiener filter of f
        f_predict = wiener_filter(f_predict, phi_predict, data, m_diagonal, \
                            b_diagonal, cf_diagonal, cn_diagonal, f_lambda_array, pix_width, num_pixels) 

        #must mix f and phi before taking the gradient
        f_mixed, phi_mixed = mix(f_predict, phi_predict, d_diagonal, g_diagonal, pix_width, num_pixels)
        mixed_grad_phi = mixed_phi_gradient(f_mixed, phi_mixed, data,
                                b_diagonal, m_diagonal, d_diagonal, g_diagonal,
                                cf_diagonal, cphi_diagonal, cn_diagonal,
                                f_lambda_array, phi_lambda_array, data_lambda_array, 
                                cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)
        #jax.debug.print("step number = {}, grad phi norm = {}", step_idx, jnp.linalg.norm(mixed_grad_phi), ordered = True)

        #create a new partial function whose only input is alpha for use in the brent optimizer (we want to
        #minimize the value of this function w.r.t. the single parameter alpha)
        min_logpdf = partial(logpdf_at_phi_new, grad_phi = mixed_grad_phi, f_predict = f_mixed, phi_predict = phi_mixed, \
                    data = data, b_diagonal = b_diagonal, m_diagonal = m_diagonal, cf_diagonal = cf_diagonal, \
                    cphi_diagonal = cphi_diagonal, cn_diagonal = cn_diagonal, d_diagonal = d_diagonal, \
                    g_diagonal = g_diagonal, f_lambda_array = f_lambda_array, \
                    phi_lambda_array = phi_lambda_array, data_lambda_array = data_lambda_array, \
                    cf_lambda_array = cf_lambda_array, cphi_lambda_array = cphi_lambda_array, \
                    cn_lambda_array = cn_lambda_array, num_pixels = num_pixels, pix_width = pix_width, hessian = hessian) 
        
        #call the optimizer to find which alpha minimizes the best
        optimizer = minimize(min_logpdf, 1, method = 'BFGS') 
        alpha = optimizer.x
        #jax.debug.print("step number = {}, alpha = {}", step_idx, jnp.linalg.norm(alpha), ordered = True)
        #now update phi with the best alpha choice
        phi_predict = phi_predict + alpha * hessian * mixed_grad_phi

        if save_animation:
            history["phi_frames"].append(jfft.irfft2(phi_predict))
            history["f_frames"].append(jfft.irfft2(f_predict))

    # if save_animation:
    #     create_mp4_from_time_series(history["phi_frames"], "Phi Movie")
    #     create_mp4_from_time_series(history["f_frames"], "F Movie")

    #return the three predicted values and the history object
    return f_predict, phi_predict, history

# def create_mp4_from_time_series(time_series, title):
#     fig, ax = plt.subplots()
#     im = ax.imshow(time_series[0], animated=True, cmap = 'coolwarm', origin = 'lower')

#     def update(frame):
#         #im.set_data(time_series[frame])
#         im = ax.imshow(time_series[frame], animated=True, cmap = 'coolwarm', origin = 'lower')
#         return [im]

#     ani = FuncAnimation(fig, update, frames = len(time_series), blit = True)
#     writer = FFMpegWriter(fps = 20, bitrate = 1800)
#     ani.save(f"{title}.mp4", writer = writer)
#     return


#helper function for the brent minimization procedure
@jax.jit
def logpdf_at_phi_new(alpha, grad_phi, f_predict, phi_predict, data,
                      b_diagonal, m_diagonal, cf_diagonal, cphi_diagonal, cn_diagonal, d_diagonal, g_diagonal,
                      f_lambda_array, phi_lambda_array, data_lambda_array, cf_lambda_array, 
                      cphi_lambda_array, cn_lambda_array, num_pixels, pix_width, hessian):
    
    #the test phi will be the current phi prediction minus a step
    #in the opposite direction of the gradient of logpdf w.r.t. phi
    test_phi = phi_predict + alpha * hessian * grad_phi

    #try unmixing f by the test phi! before we compute the logpdf
    f_unmixed, phi_unmixed = unmix(f_predict, test_phi, d_diagonal, g_diagonal, pix_width, num_pixels)

    #compute the value of logpdf at this test phi value
    logpdf_value = logpdf(f_unmixed, phi_unmixed, 
                    data, b_diagonal, m_diagonal, cf_diagonal, cphi_diagonal, cn_diagonal,
                    f_lambda_array, phi_lambda_array, data_lambda_array, cf_lambda_array,
                    cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)
    
    #return the negative one times the logpdf
    return -1*logpdf_value

#try to standardize the loading of data
#This should essentially be every thing found in ds; in the julia code
def load_data_set(file_path_to_folder):

    #initialize a set / map
    data_set ={}

    #load the .npz type of files into their respective keys
    data_set["data"] = precision_load(file_path_to_folder + "data_array.npz")
    data_set["m_diagonal"] = precision_load(file_path_to_folder + "m_diagonal.npz")
    data_set["b_diagonal"] = precision_load(file_path_to_folder + "b_diagonal.npz")
    data_set["g_diagonal"] = precision_load(file_path_to_folder + "g_diagonal.npz")
    data_set["d_diagonal"] = precision_load(file_path_to_folder + "d_diagonal.npz")
    data_set["cf_diagonal"] = precision_load(file_path_to_folder + "cf_diagonal.npz")
    data_set["cf_lensed_diagonal"] = precision_load(file_path_to_folder + "cf_lensed_diagonal.npz")
    data_set["cn_diagonal"] = precision_load(file_path_to_folder + "cn_diagonal.npz")
    data_set["cphi_diagonal"] = precision_load(file_path_to_folder + "cphi_diagonal.npz")
    data_set["nphi_diagonal"] = precision_load(file_path_to_folder + "nphi_diagonal.npz")
    data_set["f_lambda_array"] = precision_load(file_path_to_folder + "f_lambda_array.npz")
    data_set["phi_lambda_array"] = precision_load(file_path_to_folder + "phi_lambda_array.npz")
    data_set["data_lambda_array"] = precision_load(file_path_to_folder + "data_lambda_array.npz")
    data_set["cf_lambda_array"] = precision_load(file_path_to_folder + "cf_lambda_array.npz")
    data_set["cphi_lambda_array"] = precision_load(file_path_to_folder + "cphi_lambda_array.npz")
    data_set["cn_lambda_array"] = precision_load(file_path_to_folder + "cn_lambda_array.npz")

    #read the scalar values from their text files and load into their respective keys
    with open(file_path_to_folder + "pix_width.txt", "r") as f:
        pix_width = float(f.read().strip()) 
    with open(file_path_to_folder + "num_pixels.txt", "r") as f:
        num_pixels = float(f.read().strip()) 
    data_set["pix_width"] = pix_width
    data_set["num_pixels"] = num_pixels

    #return the updated object
    return data_set

#create a custom logpdf function in the mixed parametrization and rely on jax autodiff
#to carry out the chain rule completely itself...
@jax.jit
def logpdf_mixed(f_mixed, phi_mixed, data_array,
                    b_diagonal, m_diagonal, d_diagonal, g_diagonal,
                    cf_diagonal, cphi_diagonal, cn_diagonal,
                    f_lambda_array, phi_lambda_array, data_lambda_array, 
                    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width):
    #it takes in the mixed f and mixed phi and then unmixes them
    #and then evaluates the logpdf in the unlensed parametrization
    f_array, phi_array = unmix(f_mixed, phi_mixed, d_diagonal, g_diagonal, pix_width, num_pixels)
    return logpdf(f_array, phi_array, data_array, 
                    b_diagonal, m_diagonal,
                    cf_diagonal, cphi_diagonal, cn_diagonal,
                    f_lambda_array, phi_lambda_array, data_lambda_array, 
                    cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)

#TODO implement...
#should be similar to the above function, but this should
#only be comparison data / unit testing data NOT found in ds; in the Julia code
def load_ground_truth_data():
    return

#returns the quantity D^-1 * (dL^-1/dphi_mixed * f) * grad_f
@jax.jit
def mixing_jacobian_phi_component(f_mixed, phi_mixed, grad_f, d_diagonal, pix_width, num_pixels):

    #define a subfunction of just phi to apply the jax.vjp operator on
    #NOTE this must be an R^(N x N) --> R^(N x N) function in order for jax.vjp to behave properly!
    def inverse_lense(phi_mixed):
        f = jfft.rfft2(lense_flow(jfft.irfft2(f_mixed), \
            phi_mixed, pix_width, DEFAULT_NUM_LENSE_STEPS, INVERSE_LENSE, False, num_pixels, False))
        #NOTE I do not think we need to define a custom adjoint for d_inv * f because
        #we are not differentiating w.r.t. D or f here just phi whereas his D \ v adjoint
        #seems to only apply when differentiating w.r.t. D or v... Plus the same error is
        #present even when just comparing the pullbacks of L \ f in either language
        d_inv = reciprocal_matrix(reshape_diagonal(d_diagonal))
        result = d_inv * f
        return jfft.irfft2(result)
    
    #call the vjp with phi in real space and subsequently apply to grad_f in real space
    _, pullback = jax.vjp(inverse_lense, jfft.irfft2(phi_mixed))
    #Try zeroing out the imaginary or real parts of grad_f before applying pullback?
    (differential,) = pullback(jfft.irfft2(grad_f))
    return jfft.rfft2(differential)

#analytical form of the mixed phi gradient using the chain rule
@jax.jit
def mixed_phi_gradient(f_mixed, phi_mixed, data_array,
            b_diagonal, m_diagonal, d_diagonal, g_diagonal,
            cf_diagonal, cphi_diagonal, cn_diagonal,
            f_lambda_array, phi_lambda_array, data_lambda_array, 
            cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width):

    #unlense the fields
    f, phi = unmix(f_mixed, phi_mixed, d_diagonal, g_diagonal, pix_width, num_pixels)

    #compute the phi gradient in the unlensed and unmixed parametrization
    grad_phi = jax.grad(logpdf, argnums = 1)(f, phi, data_array,
                b_diagonal, m_diagonal, cf_diagonal, cphi_diagonal, cn_diagonal,
                f_lambda_array, phi_lambda_array, data_lambda_array, 
                cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)

    #compute the f gradient in the unlensed and unmixed parametrization
    grad_f = jax.grad(logpdf, argnums = 0)(f, phi, data_array,
                b_diagonal, m_diagonal, cf_diagonal, cphi_diagonal, cn_diagonal,
                f_lambda_array, phi_lambda_array, data_lambda_array, 
                cf_lambda_array, cphi_lambda_array, cn_lambda_array, num_pixels, pix_width)

    # grad_f = jnp.ones(data_array.shape, dtype = jnp.complex128)
    # grad_phi = jnp.ones(data_array.shape, dtype = jnp.complex128)

    #add these together using the proper jacobian factors from the chain rule
    chain_rule_term = mixing_jacobian_phi_component(f_mixed, phi_mixed, grad_f, d_diagonal, pix_width, num_pixels) 
    grad_phi_mixed = grad_phi + chain_rule_term
    return grad_phi_mixed