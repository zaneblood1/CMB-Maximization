#imports
from imports import *
#from matplotlib.animation import FuncAnimation, FFMpegWriter

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
    plt.show(block = False)
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

#compute the log(determinant) of a 2 x 2 block matrix
#of the form [A & B \\ C & D]
#where each submatrix is assumed to be diagonal
@jax.jit
def block_matrix_logdet(a, b, c, d):
    fourier_weights = get_fourier_weights(a.shape)
    #mathematically we return ln(det(A)) + ln(det(D - C * A^-1 * B))
    term_1_val, term_1_sign = logdet(a, fourier_weights)
    term_2_val, term_2_sign = logdet(d - c * reciprocal_matrix(a) * b, fourier_weights)
    return term_1_sign * term_1_val + term_2_sign * term_2_val

#compute the log(determinant) of a 3 x 3 block
#polarazation matrix of the form:
# [ TT & TE &  0 ]
# [ TE & EE &  0 ]
# [ 0  &  0 &  BB]
@jax.jit
def polarization_matrix_logdet(tt, te, ee, bb):
    #the return format should be the log determinant of the 
    #upper 2 x 2 block plus the logdeterminant of the BB term
    fourier_weights = get_fourier_weights(bb.shape)
    bb_logdet_val, bb_logdet_sign = logdet(bb, fourier_weights)
    return block_matrix_logdet(tt, te, te, ee) + bb_logdet_val * bb_logdet_sign

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
@jax.jit
def logpdf_contribution(field_array, field_covar_inv, fourier_weights, num_pixels):
        #This statement is equivalent to sum(field^dagger * covar^-1 * field) with proper normalization
        return jnp.real(jnp.sum(jnp.conj(field_array) * field_covar_inv * field_array * fourier_weights * (1/num_pixels)))

@jax.jit
def polarized_logpdf_contribution(t_field, e_field, b_field, tt_covar, te_covar, ee_covar, bb_covar, num_pixels):
    #compute the generalized inverse matrix of a polarization matrix
    te_inverse_matrix_11, te_inverse_matrix_12, te_inverse_matrix_21, te_inverse_matrix_22 = \
        invert_block_matrix(tt_covar, te_covar, te_covar, ee_covar)
    #get the fourier weights
    fourier_weights = get_fourier_weights(t_field.shape)
    #compute the temperature only inner product t^Dagger * te_inverse_matrix_11 * t
    temp_product =  jnp.real(jnp.sum(jnp.conj(t_field) * te_inverse_matrix_11 * t_field * fourier_weights * (1/num_pixels)))
    #compute the E-mode only inner product e^Dagger * te_inverse_matrix_22 * e
    e_product =  jnp.real(jnp.sum(jnp.conj(e_field) * te_inverse_matrix_22 * e_field * fourier_weights * (1/num_pixels)))
    #compute the B-mode only inner product b^Dagger * bb_covar^-1 * b
    b_product =  jnp.real(jnp.sum(jnp.conj(b_field) * reciprocal_matrix(bb_covar) * b_field * fourier_weights * (1/num_pixels)))
    #compute the TE cross term
    te_product = jnp.real(jnp.sum(jnp.conj(t_field) * te_inverse_matrix_12 * e_field * fourier_weights * (1/num_pixels)))
    #compute the ET cross term
    et_product = jnp.real(jnp.sum(jnp.conj(e_field) * te_inverse_matrix_21 * t_field * fourier_weights * (1/num_pixels)))
    #finally sum of of the inner products together and return the result
    final_contribution = temp_product + e_product + b_product + te_product + et_product
    return final_contribution

#compute the inverse of a 2 x 2 block matrix of the form
#[A & B \\ C & D] where we assume that A, B, C, D are all diagonal
#(i.e. implying that their inverses are their reciprocals...)
#these formulas are known as Schur's terms / decomposition
@jax.jit
def invert_block_matrix(a, b, c, d):
    #(A - B * D^-1 * C)^-1
    inv_11 = reciprocal_matrix(a - b * reciprocal_matrix(d) * c)
    #D^-1 + D^-1 * C * (A - B * D^-1 * C)^-1 * B * D^-1
    inv_22 = reciprocal_matrix(d) + \
        reciprocal_matrix(d) * c * reciprocal_matrix(a - b * reciprocal_matrix(d) * c) * b * reciprocal_matrix(d)
    #-(A - B * D^-1 * C)^-1 * B * D^-1
    inv_12 = -reciprocal_matrix(a - b * reciprocal_matrix(d) * c) * b * reciprocal_matrix(d)
    #-D^-1 * C * (A - B * D^-1 * C)^-1
    inv_21 = - reciprocal_matrix(d) * c * reciprocal_matrix(a - b * reciprocal_matrix(d) * c)
    return inv_11, inv_12, inv_21, inv_22 

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

@jax.jit #TODO implement...
def get_delta_phi_iqu_roc(t, q, u, delta_t, delta_q, delta_u, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width):
    #We need to begin by taking the point-wise product of 
    #the delta_f field and the x and y components of the gradient of f
    dt_dx, dt_dy, _, _, _ = get_spatial_derivatives(t, pix_width)
    dq_dx, dq_dy, _, _, _ = get_spatial_derivatives(q, pix_width)
    du_dx, du_dy, _, _, _ = get_spatial_derivatives(u, pix_width)

    tdt_product_x = delta_t * dt_dx
    tdt_product_y = delta_t * dt_dy
    qdq_product_x = delta_q * dq_dx
    qdq_product_y = delta_q * dq_dy
    udu_product_x = delta_u * du_dx
    udu_product_y = delta_u * du_dy

    #I think this is how we compute f = I^2 + Q^2 + U^2...?
    fdf_product_x = tdt_product_x + qdq_product_x + udu_product_x
    fdf_product_y = tdt_product_y + qdq_product_y + udu_product_y

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

@jax.jit #TODO implement...
def get_delta_phi_qu_roc(q, u, delta_q, delta_u, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width):
    #We need to begin by taking the point-wise product of 
    #the delta_f field and the x and y components of the gradient of f
    dq_dx, dq_dy, _, _, _ = get_spatial_derivatives(q, pix_width)
    du_dx, du_dy, _, _, _ = get_spatial_derivatives(u, pix_width)

    qdq_product_x = delta_q * dq_dx
    qdq_product_y = delta_q * dq_dy
    udu_product_x = delta_u * du_dx
    udu_product_y = delta_u * du_dy

    #I think this is how we compute f = I^2 + Q^2 + U^2...?
    fdf_product_x = qdq_product_x + udu_product_x
    fdf_product_y = qdq_product_y + udu_product_y

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
def ieb_lensing_gradients_integration_step(time, y, args):
    #unpack the args array
    dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, pix_width = args

    #reshape 1D coupled raveled fields into 2D fields
    t, q, u, delta_t, delta_q, delta_u, delta_phi = y
    shape = dphi_dx.shape
    t = t.reshape(shape)
    q = q.reshape(shape)
    u = u.reshape(shape)
    #delta_f is a small parameter we will integrate in tandem with delta_phi
    delta_t = delta_t.reshape(shape)
    delta_q = delta_q.reshape(shape)
    delta_u = delta_u.reshape(shape)
    delta_phi = delta_phi.reshape(shape)

    #calculate the three rates of change for f, delta_phi and delta_f:
    #1. The df/dt term is just the normal lensing term applied to the full field "f"
    dtemp_dt = get_standard_lensing_term(t, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width)
    dq_dt = get_standard_lensing_term(q, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width)
    du_dt = get_standard_lensing_term(u, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width)

    #2. The d_delta_f/dt term appears to be the adjoint lensing term applied to delta_f
    d_delta_temp_dt = get_adjoint_lensing_term(delta_t, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width)
    d_delta_q_dt = get_adjoint_lensing_term(delta_q, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width)
    d_delta_u_dt = get_adjoint_lensing_term(delta_u, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width)

    #3. The d_delta_phi_dt term is a little more involved... See the function definition for the inner working
    d_delta_phi_dt = get_delta_phi_iqu_roc(t, q, u, delta_t, delta_q, delta_u, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width)
    
    #return the three coupled ODE dynamics raveled up into 1D arrays
    return dtemp_dt.ravel(), dq_dt.ravel(), du_dt.ravel(), d_delta_temp_dt.ravel(), d_delta_q_dt.ravel(), d_delta_u_dt.ravel(), d_delta_phi_dt.ravel()

@jax.jit
def eb_lensing_gradients_integration_step(time, y, args):
    #unpack the args array
    dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, pix_width = args

    #reshape 1D coupled raveled fields into 2D fields
    q, u, delta_q, delta_u, delta_phi = y
    shape = dphi_dx.shape
    q = q.reshape(shape)
    u = u.reshape(shape)
    #delta_f is a small parameter we will integrate in tandem with delta_phi
    delta_q = delta_q.reshape(shape)
    delta_u = delta_u.reshape(shape)
    delta_phi = delta_phi.reshape(shape)

    #calculate the three rates of change for f, delta_phi and delta_f:
    #1. The df/dt term is just the normal lensing term applied to the full field "f"
    dq_dt = get_standard_lensing_term(q, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width)
    du_dt = get_standard_lensing_term(u, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width)

    #2. The d_delta_f/dt term appears to be the adjoint lensing term applied to delta_f
    d_delta_q_dt = get_adjoint_lensing_term(delta_q, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width)
    d_delta_u_dt = get_adjoint_lensing_term(delta_u, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, d2phi_dy2, time, pix_width)

    #3. The d_delta_phi_dt term is a little more involved... See the function definition for the inner working
    d_delta_phi_dt = get_delta_phi_qu_roc(q, u, delta_q, delta_u, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dxdy, 
                                          d2phi_dy2, time, pix_width)

    #return the three coupled ODE dynamics raveled up into 1D arrays
    return dq_dt.ravel(), du_dt.ravel(), d_delta_q_dt.ravel(), d_delta_u_dt.ravel(), d_delta_phi_dt.ravel()


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

#@partial(jax.jit, static_argnames = ["N", "theta_pix"])
@jax.jit
def qu2eb(q_field, u_field, N, theta_pix):
    N, _ = q_field.shape
    #generate mesh grid of fourier modes of size (N, N // 2 + 1)
    lx, ly, _ = gen_mesh_grid(q_field, theta_pix)
    #NOTE the transformation below mimics the error that Marius' makes
    #so we can compare apples to apples :)
    lx = lx.at[:N//2+1, -1].set(-1*lx[:N//2+1, -1])
    #compute the arctangent at each point in the meshgrid
    l_angle = jnp.arctan2(lx, ly)
    #E and B are then given in terms of U and Q by a rotation in fourier space
    e_field = -q_field * jnp.cos(2*l_angle) - u_field * jnp.sin(2*l_angle)
    b_field = q_field * jnp.sin(2*l_angle) - u_field * jnp.cos(2*l_angle)
    # weights = 2*jnp.ones(129,)
    # weights = weights.at[0].set(1)
    # weights = weights.at[-1].set(1)
    return e_field, b_field

#@partial(jax.jit, static_argnames = ["N", "theta_pix"])
@jax.jit
def eb2qu(e_field, b_field, N, theta_pix):
    #generate mesh grid of fourier modes of size (N, N // 2 + 1)
    N, _ = e_field.shape
    lx, ly, _ = gen_mesh_grid(e_field, theta_pix)
    #NOTE the transformation below mimics the error that Marius' makes
    #so we can compare apples to apples :)
    lx = lx.at[:N//2+1, -1].set(-1*lx[:N//2+1, -1])
    #compute the arctangent at each point in the meshgrid
    l_angle = jnp.arctan2(lx, ly)
    #Q and U are then given in terms of E and B by a rotation in fourier space
    q_field = -e_field * jnp.cos(2*l_angle) + b_field * jnp.sin(2*l_angle)
    u_field = -e_field * jnp.sin(2*l_angle) - b_field * jnp.cos(2*l_angle)
    # weights = 2*jnp.ones(129,)
    # weights = weights.at[0].set(1)
    # weights = weights.at[-1].set(1)
    return q_field, u_field

#@partial(jax.jit, static_argnames = ["N", "theta_pix"])
@jax.jit
def gen_fourier_grid(field, theta_pix):
    lx, ly, d = gen_mesh_grid(field, theta_pix)
    #find the magnitude of total l at each point in this meshgrid l = sqrt{lx^2+ly^2}
    ls =  jnp.sqrt(lx**2 + ly**2)
    return ls, d

#@partial(jax.jit, static_argnames = ["N", "theta_pix"])
@jax.jit
def gen_mesh_grid(field, theta_pix):
    N, _ = field.shape
    #1 deg = 60' so convert arcmins per pixel to deg per pixel and then deg per pixel to rad per pixel
    d = jnp.deg2rad(theta_pix / ARCMIN_PER_DEGREE)
    #create an array of real frequencies in the x-direction lx[] using fft.rfreq
    lx = 2 * jnp.pi * jfft.rfftfreq(N, d) #d is the spacing in radians per pixel
    #create an array of fully complex frequencies in the y-direction ly[] using fft.freq
    ly = 2 * jnp.pi * jfft.fftfreq(N, d)
    #create a mesh-grid of these frequencies
    lx, ly = jnp.meshgrid(lx, ly) #i.e. repeat lx for length of ly and vice versa
    return lx, ly, d

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
    #TODO CHANGE THIS BACK TO WRAPPER!!!!!!
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

    return jnp.real(result)

@partial(jax.jit, static_argnames = ["field_shape"])
def get_fourier_weights(field_shape):
    rows, _ = field_shape
    length = rows // 2 + 1
    weights = 2 * jnp.ones((length,), dtype = jnp.complex128) 
    weights = weights.at[0].set(1)
    weights = weights.at[-1].set(1)
    return weights

#@partial(jax.jit, static_argnames = ["N", "theta_pix"])
@jax.jit
def logpdf_ieb_unlensed_param(f_intensity, f_e_mode, f_b_mode, f_covar_tt, f_covar_te, f_covar_ee, f_covar_bb,
            data_intensity, data_e_mode, data_b_mode, noise_covar_tt, noise_covar_te, noise_covar_ee, noise_covar_bb, 
            phi, phi_covar, mask_tt, mask_ee, mask_bb, beam_tt, beam_ee, beam_bb, 
            N, theta_pix, pix_width):
    
    #compute the three log determinant contributions
    logdet_phi_val, logdet_phi_sign = logdet(phi_covar, get_fourier_weights(phi_covar.shape))
    logdet_cphi = logdet_phi_val * logdet_phi_sign
    logdet_cf = polarization_matrix_logdet(f_covar_tt, f_covar_te, f_covar_ee, f_covar_bb)
    logdet_cn = polarization_matrix_logdet(noise_covar_tt, noise_covar_te, noise_covar_ee, noise_covar_bb)

    #compute the phi^2 term which does not involve fancy matrix inversion techniques
    fourier_weights = 2 * jnp.ones(129)
    fourier_weights = fourier_weights.at[0].set(1)
    fourier_weights = fourier_weights.at[-1].set(1)
    #NOTE I made a stupid "new" function without the space between log and pdf that takes in properly
    #space fourier weights already... Need to switch to using this everywhere and standardize the weight code
    phi_squared = log_pdf_phi_contribution(phi, reciprocal_matrix(phi_covar), fourier_weights, N**2)

    #compute the data^2 and f^2 terms which involve taking the inverse of a 2 x 2 block matrix using Schur's terms
    f_squared = polarized_logpdf_contribution(f_intensity, f_e_mode, f_b_mode, f_covar_tt, \
                                                   f_covar_te, f_covar_ee, f_covar_bb, N**2)

    #lense the I, E, and B fields (by first converting to the I, Q, U basis) to compute the data term
    f_intensity_lensed, f_e_lensed, f_b_lensed = lense_ieb_wrapper(jfft.irfft2(f_intensity), jfft.irfft2(f_e_mode), \
                         jfft.irfft2(f_b_mode), jfft.irfft2(phi), pix_width, N, theta_pix, 
                         n = 10, direction = +1, adjoint = False, rescale_and_conjugate = True)
    f_intensity_lensed = jfft.rfft2(f_intensity_lensed)
    f_e_lensed = jfft.rfft2(f_e_lensed)
    f_b_lensed = jfft.rfft2(f_b_lensed)
    # f_q_mode, f_u_mode = eb2qu(f_e_mode, f_b_mode, N, theta_pix)
    # f_q_lensed = jfft.rfft2(lense_flow(jfft.irfft2(f_q_mode), jfft.irfft2(phi), pix_width, 10, +1, False, num_pixels, False))
    # f_u_lensed = jfft.rfft2(lense_flow(jfft.irfft2(f_u_mode), jfft.irfft2(phi), pix_width, 10, +1, False, num_pixels, False))
    # f_e_lensed, f_b_lensed = qu2eb(f_q_lensed, f_u_lensed, N, theta_pix)

    #compute the deltas / differences between (data - M * B * L * f) for I, E, & B
    delta_intensity = data_intensity - mask_tt * beam_tt * f_intensity_lensed
    delta_e_mode = data_e_mode - mask_ee * beam_ee * f_e_lensed
    delta_b_mode = data_b_mode - mask_bb * beam_bb * f_b_lensed
    data_squared = polarized_logpdf_contribution(delta_intensity, delta_e_mode, delta_b_mode, noise_covar_tt, \
                                                   noise_covar_te, noise_covar_ee, noise_covar_bb, N**2)

    #add all the terms together and divide by -2
    return jnp.real(-(data_squared + f_squared + phi_squared + logdet_cf + logdet_cn + logdet_cphi)/2)

@jax.jit
def logpdf_eb_unlensed_param(f_e_mode, f_b_mode, f_covar_ee, f_covar_bb,
            data_e_mode, data_b_mode, noise_covar_ee, noise_covar_bb, 
            phi, phi_covar, mask_ee, mask_bb, beam_ee, beam_bb, 
            N, theta_pix, pix_width):
    
    #compute the three log determinant contributions
    weights = get_fourier_weights(phi_covar.shape)
    logdet_phi_val, logdet_phi_sign = logdet(phi_covar, weights)
    logdet_cphi = logdet_phi_val * logdet_phi_sign

    logdet_cf_ee_val, logdet_cf_ee_sign = logdet(f_covar_ee, weights)
    logdet_cf_ee = logdet_cf_ee_val * logdet_cf_ee_sign
    logdet_cf_bb_val, logdet_cf_bb_sign = logdet(f_covar_bb, weights)
    logdet_cf_bb = logdet_cf_bb_val * logdet_cf_bb_sign
    logdet_cf = logdet_cf_ee + logdet_cf_bb

    logdet_cn_ee_val, logdet_cn_ee_sign = logdet(noise_covar_ee, weights)
    logdet_cn_ee = logdet_cn_ee_val * logdet_cn_ee_sign
    logdet_cn_bb_val, logdet_cn_bb_sign = logdet(noise_covar_bb, weights)
    logdet_cn_bb = logdet_cn_bb_val * logdet_cn_bb_sign
    logdet_cn = logdet_cn_ee + logdet_cn_bb

    #compute the phi^2 term which does not involve fancy matrix inversion techniques
    fourier_weights = 2 * jnp.ones(129)
    fourier_weights = fourier_weights.at[0].set(1)
    fourier_weights = fourier_weights.at[-1].set(1)
    #NOTE I made a stupid "new" function without the space between log and pdf that takes in properly
    #space fourier weights already... Need to switch to using this everywhere and standardize the weight code
    phi_squared = log_pdf_phi_contribution(phi, reciprocal_matrix(phi_covar), fourier_weights, N**2)

    #compute the data^2 and f^2 terms which involve taking the inverse of a 2 x 2 block matrix using Schur's terms
    f_squared_ee = logpdf_contribution(f_e_mode, reciprocal_matrix(f_covar_ee), fourier_weights, N**2)
    f_squared_bb = logpdf_contribution(f_b_mode, reciprocal_matrix(f_covar_bb), fourier_weights, N**2)

    #lense the I, E, and B fields (by first converting to the I, Q, U basis) to compute the data term
    # f_intensity_lensed, f_e_lensed, f_b_lensed = lense_ieb_wrapper(jfft.irfft2(f_intensity), jfft.irfft2(f_e_mode), \
    #                      jfft.irfft2(f_b_mode), jfft.irfft2(phi), pix_width, N, theta_pix, 
    #                      n = 10, direction = +1, adjoint = False, rescale_and_conjugate = True)
    # f_intensity_lensed = jfft.rfft2(f_intensity_lensed)
    # f_e_lensed = jfft.rfft2(f_e_lensed)
    # f_b_lensed = jfft.rfft2(f_b_lensed)
    f_e_lensed, f_b_lensed = lense_eb_wrapper(jfft.irfft2(f_e_mode), \
                         jfft.irfft2(f_b_mode), jfft.irfft2(phi), pix_width, N, theta_pix, 
                         n = 10, direction = +1, adjoint = False, rescale_and_conjugate = True)
    f_e_lensed = jfft.rfft2(f_e_lensed)
    f_b_lensed = jfft.rfft2(f_b_lensed)

    #compute the deltas / differences between (data - M * B * L * f) for I, E, & B
    delta_e_mode = data_e_mode - mask_ee * beam_ee * f_e_lensed
    delta_b_mode = data_b_mode - mask_bb * beam_bb * f_b_lensed
    data_squared_ee = logpdf_contribution(delta_e_mode, reciprocal_matrix(noise_covar_ee), fourier_weights, N**2)
    data_squared_bb = logpdf_contribution(delta_b_mode, reciprocal_matrix(noise_covar_bb), fourier_weights, N**2)

    #add all the terms together and divide by -2
    return -jnp.real(data_squared_ee + data_squared_bb + f_squared_ee + 
                     f_squared_bb + phi_squared + logdet_cf + logdet_cn + logdet_cphi)/2
    #return (logdet_cphi + logdet_cf + logdet_cn)
#@partial(jax.jit, static_argnames = ["N", "theta_pix"])
@jax.jit
def logpdf_ieb_lensed_param(f_intensity_lensed, f_e_lensed, f_b_lensed, f_covar_tt, f_covar_te, f_covar_ee, f_covar_bb,
            data_intensity, data_e_mode, data_b_mode, noise_covar_tt, noise_covar_te, noise_covar_ee, noise_covar_bb, 
            phi, phi_covar, mask_tt, mask_ee, mask_bb, beam_tt, beam_ee, beam_bb, 
            N, theta_pix, pix_width):
    
    num_pixels = N**2
    
    #compute the three log determinant contributions
    logdet_phi_val, logdet_phi_sign = logdet(phi_covar, get_fourier_weights(phi_covar.shape))
    logdet_cphi = logdet_phi_val * logdet_phi_sign
    logdet_cf = polarization_matrix_logdet(f_covar_tt, f_covar_te, f_covar_ee, f_covar_bb)
    logdet_cn = polarization_matrix_logdet(noise_covar_tt, noise_covar_te, noise_covar_ee, noise_covar_bb)

    #compute the phi^2 term which does not involve fancy matrix inversion techniques
    fourier_weights = get_fourier_weights(phi.shape)
    #NOTE I made a stupid "new" function without the space between log and pdf that takes in properly
    #space fourier weights already... Need to switch to using this everywhere and standardize the weight code
    phi_squared = logpdf_contribution(phi, reciprocal_matrix(phi_covar), fourier_weights, num_pixels)

    #compute the data^2 and f^2 terms which involve taking the inverse of a 2 x 2 block matrix using Schur's terms
    f_intensity, f_e_mode, f_b_mode = lense_ieb_wrapper(jfft.irfft2(f_intensity_lensed), jfft.irfft2(f_e_lensed), jfft.irfft2(f_b_lensed), 
                                                jfft.irfft2(phi), pix_width, N, theta_pix, n = 10, 
                                                direction = -1, adjoint = False, rescale_and_conjugate = True)
    f_intensity = jfft.rfft2(f_intensity)
    f_e_mode = jfft.rfft2(f_e_mode)
    f_b_mode = jfft.rfft2(f_b_mode)
    #TODO there is a huge discrepancy with the f^2 term in the unlensed v.s. lensed params since the only differences
    #between the original B mode and the unlensed B mode is right where the cf_bb^-1 matrix amplifies this discrepancy
    #by 10^9 orders of magnitude...
    f_squared = polarized_logpdf_contribution(f_intensity, f_e_mode, f_b_mode, f_covar_tt, \
                                                   f_covar_te, f_covar_ee, f_covar_bb, num_pixels)

    #compute the deltas / differences between (data - M * B * L * f) for I, E, & B
    delta_intensity = data_intensity - mask_tt * beam_tt * f_intensity_lensed
    delta_e_mode = data_e_mode - mask_ee * beam_ee * f_e_lensed
    delta_b_mode = data_b_mode - mask_bb * beam_bb * f_b_lensed
    data_squared = polarized_logpdf_contribution(delta_intensity, delta_e_mode, delta_b_mode, noise_covar_tt, \
                                                   noise_covar_te, noise_covar_ee, noise_covar_bb, num_pixels)

    #add all the terms together and divide by -2
    return -(data_squared + f_squared + phi_squared + logdet_cf + logdet_cn + logdet_cphi)/2

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

#@partial(jax.jit, static_argnames = ["N", "theta_pix"])
@jax.jit
def get_ieb_lensing_operator_gradients(phi, t_field, e_field, b_field, cotangent, pix_width, N, theta_pix, direction=1, n=10):

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
    delta_t_init = cotangent[0]
    delta_e_init = cotangent[1]
    delta_b_init = cotangent[2]

    delta_q_init, delta_u_init = eb2qu(jfft.rfft2(delta_e_init), jfft.rfft2(delta_b_init), N, theta_pix)
    delta_q_init = jfft.irfft2(delta_q_init)
    delta_u_init = jfft.irfft2(delta_u_init)

    q_field, u_field = eb2qu(jfft.rfft2(e_field), jfft.rfft2(b_field), N, theta_pix)
    q_field = jfft.irfft2(q_field)
    u_field = jfft.irfft2(u_field)

    y0 = (t_field.ravel(), q_field.ravel(), u_field.ravel(), \
          delta_t_init.ravel(), delta_q_init.ravel(), delta_u_init.ravel(), \
          delta_phi.ravel())
    
    #store extra arguments in a single array
    args = (dphi_dx, dphi_dy, d2_phi_dx2, d2_phi_dxdy, d2_phi_dy2, pix_width)

    #define a single step
    single_step_dynamics = ODETerm(ieb_lensing_gradients_integration_step)
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
    t_real = jnp.asarray(sol.ys[0][-1])
    t_real = t_real.reshape(shape) #reshape flattened vector back into 2D shape
    q_real = jnp.asarray(sol.ys[1][-1])
    q_real = q_real.reshape(shape) #reshape flattened vector back into 2D shape
    u_real = jnp.asarray(sol.ys[2][-1])
    u_real = u_real.reshape(shape) #reshape flattened vector back into 2D shape

    delta_t_real = jnp.asarray(sol.ys[3][-1])
    delta_t_real = delta_t_real.reshape(shape)
    delta_q_real = jnp.asarray(sol.ys[4][-1])
    delta_q_real = delta_q_real.reshape(shape)
    delta_u_real = jnp.asarray(sol.ys[5][-1])
    delta_u_real = delta_u_real.reshape(shape)

    delta_phi_real = jnp.asarray(sol.ys[6][-1])
    delta_phi_real = delta_phi_real.reshape(shape)

    #convert Q,U back to E, B
    e_real, b_real = qu2eb(jfft.rfft2(q_real), jfft.rfft2(u_real), N, theta_pix)
    e_real = jfft.irfft2(e_real)
    b_real = jfft.irfft2(b_real)
    delta_e_real, delta_b_real = qu2eb(jfft.rfft2(delta_q_real), jfft.rfft2(delta_u_real), N, theta_pix)
    delta_e_real = jfft.irfft2(delta_e_real)
    delta_b_real = jfft.irfft2(delta_b_real)

    #return the tuple of fields 
    return t_real, e_real, b_real, delta_t_real, delta_e_real, delta_b_real, delta_phi_real

@jax.jit
def get_eb_lensing_operator_gradients(phi, e_field, b_field, cotangent, pix_width, N, theta_pix, direction=1, n=10):

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
    delta_e_init = cotangent[0]
    delta_b_init = cotangent[1]

    delta_q_init, delta_u_init = eb2qu(jfft.rfft2(delta_e_init), jfft.rfft2(delta_b_init), N, theta_pix)
    delta_q_init = jfft.irfft2(delta_q_init)
    delta_u_init = jfft.irfft2(delta_u_init)

    q_field, u_field = eb2qu(jfft.rfft2(e_field), jfft.rfft2(b_field), N, theta_pix)
    q_field = jfft.irfft2(q_field)
    u_field = jfft.irfft2(u_field)

    y0 = (q_field.ravel(), u_field.ravel(), \
          delta_q_init.ravel(), delta_u_init.ravel(), \
          delta_phi.ravel())
    
    #store extra arguments in a single array
    args = (dphi_dx, dphi_dy, d2_phi_dx2, d2_phi_dxdy, d2_phi_dy2, pix_width)

    #define a single step
    single_step_dynamics = ODETerm(eb_lensing_gradients_integration_step)
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
    q_real = jnp.asarray(sol.ys[0][-1])
    q_real = q_real.reshape(shape) #reshape flattened vector back into 2D shape
    u_real = jnp.asarray(sol.ys[1][-1])
    u_real = u_real.reshape(shape) #reshape flattened vector back into 2D shape

    delta_q_real = jnp.asarray(sol.ys[2][-1])
    delta_q_real = delta_q_real.reshape(shape)
    delta_u_real = jnp.asarray(sol.ys[3][-1])
    delta_u_real = delta_u_real.reshape(shape)

    delta_phi_real = jnp.asarray(sol.ys[4][-1])
    delta_phi_real = delta_phi_real.reshape(shape)

    #convert Q,U back to E, B
    e_real, b_real = qu2eb(jfft.rfft2(q_real), jfft.rfft2(u_real), N, theta_pix)
    e_real = jfft.irfft2(e_real)
    b_real = jfft.irfft2(b_real)
    delta_e_real, delta_b_real = qu2eb(jfft.rfft2(delta_q_real), jfft.rfft2(delta_u_real), N, theta_pix)
    delta_e_real = jfft.irfft2(delta_e_real)
    delta_b_real = jfft.irfft2(delta_b_real)

    #return the tuple of fields 
    return e_real, b_real, delta_e_real, delta_b_real, delta_phi_real

@jax.jit
def wiener_filter_matrix_operator(f, phi, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, pix_width, num_pixels):
    #The A matrix operator is the gradient of logpdf w.r.t. f evaluated at d = 0 and other inputs at their current values
    data = jnp.zeros(f.shape)
    return (gradf_logpdf(f, phi, data, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, pix_width, num_pixels))  

#@partial(jax.jit, static_argnames = ["N", "theta_pix"])
@jax.jit
def wiener_filter_matrix_operator_ieb(t_field, e_field, b_field, phi, 
                                      mask_tt, mask_te, mask_et, mask_ee, mask_bb, 
                                      beam_tt, beam_te, beam_et, beam_ee, beam_bb,
                                      cf_tt, cf_te, cf_et, cf_ee, cf_bb, 
                                      cn_tt, cn_te, cn_et, cn_ee, cn_bb, 
                                      pix_width, N, theta_pix):
    #The A matrix operator is the gradient of logpdf w.r.t. f evaluated at d = 0 and other inputs at their current values
    t_data = jnp.zeros(t_field.shape)
    e_data = jnp.zeros(t_field.shape)
    b_data = jnp.zeros(t_field.shape)
    return gradf_logpdf_ieb(t_field, e_field, b_field, 
                            t_data, e_data, b_data, phi,
                            cn_tt, cn_ee, cn_bb, cn_te, cn_et,
                            cf_tt, cf_ee, cf_bb, cf_te, cf_et,
                            mask_tt, mask_ee, mask_bb, mask_te, mask_et,
                            beam_tt, beam_ee, beam_bb, beam_te, beam_et,
                            N, theta_pix, pix_width)

@jax.jit
def wiener_filter_matrix_operator_eb(e_field, b_field, phi, 
                                      mask_ee, mask_bb, 
                                      beam_ee, beam_bb,
                                      cf_ee, cf_bb, 
                                      cn_ee, cn_bb, 
                                      pix_width, N, theta_pix):
    #The A matrix operator is the gradient of logpdf w.r.t. f evaluated at d = 0 and other inputs at their current values
    e_data = jnp.zeros(e_field.shape)
    b_data = jnp.zeros(e_field.shape)
    return gradf_logpdf_eb(e_field, b_field, 
                            e_data, b_data, phi,
                            cn_ee, cn_bb,
                            cf_ee, cf_bb,
                            mask_ee, mask_bb,
                            beam_ee, beam_bb,
                            N, theta_pix, pix_width)

@jax.jit
def wiener_filter_b_vector(phi, data, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, pix_width, num_pixels):
    #The b vector is the gradient of logpdf w.r.t. f evaluated at f = 0 and other inputs at their current values
    f = jnp.zeros(data.shape)
    return -1*gradf_logpdf(f, phi, data, m_diagonal, b_diagonal, cf_diagonal, cn_diagonal, pix_width, num_pixels)

#@partial(jax.jit, static_argnames = ["N", "theta_pix"])
@jax.jit
def wiener_filter_b_vector_ieb(phi, t_data, e_data, b_data, mask_tt, mask_ee, mask_bb, mask_te, mask_et, 
                           beam_tt, beam_ee, beam_bb, beam_te, beam_et, cf_tt, cf_ee, cf_bb, cf_te, cf_et, 
                           cn_tt, cn_ee, cn_bb, cn_te, cn_et, pix_width, N, theta_pix):
    #The b vector is the gradient of logpdf w.r.t. f evaluated at f = 0 and other inputs at their current values
    t_field = jnp.zeros(t_data.shape)
    e_field = jnp.zeros(t_data.shape)
    b_field = jnp.zeros(t_data.shape)
    gradf_t, gradf_e, gradf_b = gradf_logpdf_ieb(t_field, e_field, b_field, 
                                                t_data, e_data, b_data, phi,
                                                cn_tt, cn_ee, cn_bb, cn_te, cn_et,
                                                cf_tt, cf_ee, cf_bb, cf_te, cf_et,
                                                mask_tt, mask_ee, mask_bb, mask_te, mask_et,
                                                beam_tt, beam_ee, beam_bb, beam_te, beam_et,
                                                N, theta_pix, pix_width)
    return -1*gradf_t, -1*gradf_e, -1*gradf_b

@jax.jit
def wiener_filter_b_vector_eb(phi, e_data, b_data, mask_ee, mask_bb,  
                                beam_ee, beam_bb, cf_ee, cf_bb,  
                                cn_ee, cn_bb, pix_width, N, theta_pix):
    #The b vector is the gradient of logpdf w.r.t. f evaluated at f = 0 and other inputs at their current values
    e_field = jnp.zeros(e_data.shape)
    b_field = jnp.zeros(e_data.shape)
    gradf_e, gradf_b = gradf_logpdf_eb(e_field, b_field, 
                                        e_data, b_data, phi,
                                        cn_ee, cn_bb,
                                        cf_ee, cf_bb,
                                        mask_ee, mask_bb,
                                        beam_ee, beam_bb,
                                        N, theta_pix, pix_width)
    return -1*gradf_e, -1*gradf_b

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

@jax.jit
def hessian_logpdf_preconditioner_ieb(cf_tt, cf_te, cf_et, cf_ee, cf_bb, 
                                      beam_tt, beam_te, beam_et, beam_ee, beam_bb,
                                      mask_tt, mask_te, mask_et, mask_ee, mask_bb, 
                                      cn_tt, cn_te, cn_et, cn_ee, cn_bb):

    #find the inverse covariance matrices needed
    cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee = invert_block_matrix(cf_tt, cf_te, cf_et, cf_ee)
    cf_inv_bb = reciprocal_matrix(cf_bb)
    cn_inv_tt, cn_inv_te, cn_inv_et, cn_inv_ee = invert_block_matrix(cn_tt, cn_te, cn_et, cn_ee)
    cn_inv_bb = reciprocal_matrix(cn_bb)

    #the preconditioner is then equal to Cf^-1 + B^Dagger * M^Dagger * Cn^-1 * M * B
    #TODO seemingly each day an object oriented approach with overrides for common operations
    #such as mult, div, scale, add, sub is making more and more sense... Otherwise we have to do
    #stupid repetitive stuff like this chaining here to perform a simple matrix multiplication of 5 block matrices
    mb_tt, mb_te, mb_et, mb_ee, mb_bb = ieb_matrix_mult(mask_tt, mask_te, mask_et, mask_ee, mask_bb, 
                                                        beam_tt, beam_te, beam_et, beam_ee, beam_bb)
    cn_inv_mb_tt, cn_inv_mb_te, cn_inv_mb_et, cn_inv_mb_ee, cn_inv_mb_bb = \
                                        ieb_matrix_mult(cn_inv_tt, cn_inv_te, cn_inv_et, cn_inv_ee, 
                                                        cn_inv_bb, mb_tt, mb_te, mb_et, mb_ee, mb_bb)
    mcn_inv_mb_tt, mcn_inv_mb_te, mcn_inv_mb_et, mcn_inv_mb_ee, mcn_inv_mb_bb = \
                                        ieb_matrix_mult(mask_tt, mask_te, mask_et, mask_ee, mask_bb, 
                                        cn_inv_mb_tt, cn_inv_mb_te, cn_inv_mb_et, cn_inv_mb_ee, cn_inv_mb_bb)
    bmcn_inv_mb_tt, bmcn_inv_mb_te, bmcn_inv_mb_et, bmcn_inv_mb_ee, bmcn_inv_mb_bb = \
                        ieb_matrix_mult(beam_tt, beam_te, beam_et, beam_ee, beam_bb, 
                                        mcn_inv_mb_tt, mcn_inv_mb_te, mcn_inv_mb_et, mcn_inv_mb_ee, mcn_inv_mb_bb)
    preconditioner_tt = cf_inv_tt + bmcn_inv_mb_tt
    preconditioner_te = cf_inv_te + bmcn_inv_mb_te
    preconditioner_et = cf_inv_et + bmcn_inv_mb_et
    preconditioner_ee = cf_inv_ee + bmcn_inv_mb_ee
    preconditioner_bb = cf_inv_bb + bmcn_inv_mb_bb
    return preconditioner_tt, preconditioner_te, preconditioner_et, preconditioner_ee, preconditioner_bb

@jax.jit
def hessian_logpdf_preconditioner_eb(cf_ee, cf_bb, 
                                      beam_ee, beam_bb,
                                      mask_ee, mask_bb, 
                                      cn_ee, cn_bb):

    #find the inverse covariance matrices needed
    cn_inv_ee = reciprocal_matrix(cn_ee)
    cn_inv_bb = reciprocal_matrix(cn_bb)
    cf_inv_ee = reciprocal_matrix(cf_ee)
    cf_inv_bb = reciprocal_matrix(cf_bb)

    #the preconditioner is then equal to Cf^-1 + B^Dagger * M^Dagger * Cn^-1 * M * B
    #TODO seemingly each day an object oriented approach with overrides for common operations
    #such as mult, div, scale, add, sub is making more and more sense... Otherwise we have to do
    #stupid repetitive stuff like this chaining here to perform a simple matrix multiplication of 5 block matrices
    preconditioner_ee = cf_inv_ee + beam_ee * mask_ee * cn_inv_ee * mask_ee * beam_ee
    preconditioner_bb = cf_inv_bb + beam_bb * mask_bb * cn_inv_bb * mask_bb * beam_bb
    return preconditioner_ee, preconditioner_bb

#TODO implement wiener filter in TEB / TQU polarizations. In this basis, the A matrix operator maps an f_[T, E, B] vector to another
#Af_[T, E, B] vector. The b vector is a length 3 b_[T, E, B] vector...The Hessian logpdf preconditioner is a TEB block covariance matrix
#which maps [T, E, B] vectors to --> [T, E, B] vectors...

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

#wiener filter implementation
#@partial(jax.jit, static_argnames = ["N", "theta_pix"])
@jax.jit
def wiener_filter_ieb(f_t, f_e, f_b, phi, t_data, e_data, b_data, 
                      mask_tt, mask_ee, mask_bb, mask_te, mask_et, 
                      beam_tt, beam_ee, beam_bb, beam_te, beam_et, 
                      cf_tt, cf_ee, cf_bb, cf_te, cf_et, 
                      cn_tt, cn_ee, cn_bb, cn_te, cn_et, 
                      pix_width, N, theta_pix):

    #Compute the b array which is the gradient w.r.t. f of the logpdf 
    #function evaluated at f = 0, d = d, phi = phi, etc...
    b_t, b_e, b_b = wiener_filter_b_vector_ieb(phi, t_data, e_data, b_data, mask_tt, mask_ee, mask_bb, mask_te, mask_et, 
                           beam_tt, beam_ee, beam_bb, beam_te, beam_et, cf_tt, cf_ee, cf_bb, cf_te, cf_et, 
                           cn_tt, cn_ee, cn_bb, cn_te, cn_et, pix_width, N, theta_pix)

    #use a preconditioner to speed up the calculation
    preconditioner_tt, preconditioner_te, preconditioner_et, preconditioner_ee, preconditioner_bb = \
                                      hessian_logpdf_preconditioner_ieb(
                                        cf_tt, cf_te, cf_et, cf_ee, cf_bb, 
                                        beam_tt, beam_te, beam_et, beam_ee, beam_bb,
                                        mask_tt, mask_te, mask_et, mask_ee, mask_bb, 
                                        cn_tt, cn_te, cn_et, cn_ee, cn_bb)

    #take the conjugate gradient of A @ f = b and solve for f assuming
    f_t_wiener_filtered, f_e_wiener_filtered, f_b_wiener_filtered = \
        conjugate_gradient_ieb(phi, mask_tt, mask_te, mask_et, mask_ee, mask_bb,
                                beam_tt, beam_te, beam_et, beam_ee, beam_bb,
                                cf_tt, cf_te, cf_et, cf_ee, cf_bb, 
                                cn_tt, cn_te, cn_et, cn_ee, cn_bb, 
                                b_t, b_e, b_b, f_t, f_e, f_b, 
                                preconditioner_tt, preconditioner_te, preconditioner_et,
                                preconditioner_ee, preconditioner_bb, 
                                N, pix_width, theta_pix, maxiter = 500, tol = 1e-1)

    #return the value of f found via conjugate gradient. This is the
    #wiener filtered version of f... this will be the real space version
    return f_t_wiener_filtered, f_e_wiener_filtered, f_b_wiener_filtered

@jax.jit
def wiener_filter_eb(f_e, f_b, phi, e_data, b_data, 
                      mask_ee, mask_bb, 
                      beam_ee, beam_bb, 
                      cf_ee, cf_bb, 
                      cn_ee, cn_bb, 
                      pix_width, N, theta_pix):

    #Compute the b array which is the gradient w.r.t. f of the logpdf 
    #function evaluated at f = 0, d = d, phi = phi, etc...
    b_e, b_b = wiener_filter_b_vector_eb(phi, e_data, b_data, mask_ee, mask_bb,  
                                         beam_ee, beam_bb, cf_ee, cf_bb,  
                                         cn_ee, cn_bb, pix_width, N, theta_pix)
    #NOTE!!! b_e and b_b are not matching 
    #use a preconditioner to speed up the calculation
    preconditioner_ee, preconditioner_bb = hessian_logpdf_preconditioner_eb(cf_ee, cf_bb, 
                                                                        beam_ee, beam_bb,
                                                                        mask_ee, mask_bb, 
                                                                        cn_ee, cn_bb)

    #take the conjugate gradient of A @ f = b and solve for f assuming
    f_e_wiener_filtered, f_b_wiener_filtered = conjugate_gradient_eb(phi, mask_ee, mask_bb,
                                                beam_ee, beam_bb,
                                                cf_ee, cf_bb, 
                                                cn_ee, cn_bb, 
                                                b_e, b_b, f_e, f_b, 
                                                preconditioner_ee, preconditioner_bb, 
                                                N, pix_width, theta_pix, maxiter = 500, tol = 1e-1)

    #return the value of f found via conjugate gradient. This is the
    #wiener filtered version of f... this will be the real space version
    return f_e_wiener_filtered, f_b_wiener_filtered

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

#@partial(jax.jit, static_argnames = ["N", "theta_pix"])
@jax.jit
def mix_ieb(f_t, f_e, f_b, phi, d_tt, d_te, d_et, d_ee, d_bb, g_diagonal, pix_width, N, theta_pix):

    #reshape the diagonals into the propoer format
    g_matrix = reshape_diagonal(g_diagonal)

    df_t, df_e, df_b = ieb_mat_vec_product(d_tt, d_te, d_et, d_ee, d_bb, f_t, f_e, f_b)

    #f_mixed = L(phi) * D * f
    f_t_mixed, f_e_mixed, f_b_mixed = lense_ieb(jfft.irfft2(df_t), jfft.irfft2(df_e), jfft.irfft2(df_b), \
                      jfft.irfft2(phi), pix_width, N, theta_pix, DEFAULT_NUM_LENSE_STEPS, \
                      FORWARD_LENSE, adjoint = False, rescale_and_conjugate = False)
    f_t_mixed = jfft.rfft2(f_t_mixed)
    f_e_mixed = jfft.rfft2(f_e_mixed)
    f_b_mixed = jfft.rfft2(f_b_mixed)
    
    #phi_mixed = G * phi
    phi_mixed = g_matrix * phi

    #return the mixed tuple in fourier space
    return f_t_mixed, f_e_mixed, f_b_mixed, phi_mixed

@jax.jit
def mix_eb(f_e, f_b, phi, d_ee, d_bb, g_diagonal, pix_width, N, theta_pix):

    #reshape the diagonals into the propoer format
    g_matrix = reshape_diagonal(g_diagonal)
    df_e = d_ee * f_e
    df_b = d_bb * f_b

    #f_mixed = L(phi) * D * f
    f_e_mixed, f_b_mixed = lense_eb(jfft.irfft2(df_e), jfft.irfft2(df_b), \
                      jfft.irfft2(phi), pix_width, N, theta_pix, DEFAULT_NUM_LENSE_STEPS, \
                      FORWARD_LENSE, adjoint = False, rescale_and_conjugate = False)
    f_e_mixed = jfft.rfft2(f_e_mixed)
    f_b_mixed = jfft.rfft2(f_b_mixed)
    
    #phi_mixed = G * phi
    phi_mixed = g_matrix * phi

    #return the mixed tuple in fourier space
    return f_e_mixed, f_b_mixed, phi_mixed

#@partial(jax.jit, static_argnames = ["N", "theta_pix"])
@jax.jit
def unmix_ieb(f_t_mixed, f_e_mixed, f_b_mixed, phi_mixed, d_tt, d_te, d_et, d_ee, d_bb, g_diagonal, pix_width, N, theta_pix):

    #apply the inverse G operator to phi mixed to get back the original phi
    g_matrix = reshape_diagonal(g_diagonal)
    g_inv = reciprocal_matrix(g_matrix)
    phi = g_inv * phi_mixed

    #delense the field f_mixed then multiply by the inverse D matrix
    #to get back the original field f
    d_inv_tt, d_inv_te, d_inv_et, d_inv_ee = invert_block_matrix(d_tt, d_te, d_et, d_ee)
    d_inv_bb = reciprocal_matrix(d_bb)
    f_t, f_e, f_b = lense_ieb(jfft.irfft2(f_t_mixed), jfft.irfft2(f_e_mixed), jfft.irfft2(f_b_mixed), \
                      jfft.irfft2(phi), pix_width, N, theta_pix, DEFAULT_NUM_LENSE_STEPS, \
                      INVERSE_LENSE, adjoint = False, rescale_and_conjugate = False)
    f_t, f_e, f_b = ieb_mat_vec_product(d_inv_tt, d_inv_te, d_inv_et, d_inv_ee, d_inv_bb, \
                                        jfft.rfft2(f_t), jfft.rfft2(f_e), jfft.rfft2(f_b))
    #return the unmixed f and phi
    return f_t, f_e, f_b, phi

@jax.jit
def unmix_eb(f_e_mixed, f_b_mixed, phi_mixed, d_ee, d_bb, g_diagonal, pix_width, N, theta_pix):

    #apply the inverse G operator to phi mixed to get back the original phi
    g_matrix = reshape_diagonal(g_diagonal)
    g_inv = reciprocal_matrix(g_matrix)
    phi = g_inv * phi_mixed

    #delense the field f_mixed then multiply by the inverse D matrix
    #to get back the original field f
    d_inv_ee = reciprocal_matrix(d_ee)
    d_inv_bb = reciprocal_matrix(d_bb)
    f_e, f_b = lense_eb(jfft.irfft2(f_e_mixed), jfft.irfft2(f_b_mixed), \
                      jfft.irfft2(phi), pix_width, N, theta_pix, DEFAULT_NUM_LENSE_STEPS, \
                      INVERSE_LENSE, adjoint = False, rescale_and_conjugate = False)
    f_e = d_inv_ee * jfft.rfft2(f_e)
    f_b = d_inv_bb * jfft.rfft2(f_b)
    #return the unmixed f and phi
    return f_e, f_b, phi

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

#try to define a custom conjugate gradient which works in fourier space rather than real space
@partial(jax.jit, static_argnames = ["maxiter", "tol"])
def conjugate_gradient_ieb(phi, mask_tt, mask_te, mask_et, mask_ee, mask_bb,
                            beam_tt, beam_te, beam_et, beam_ee, beam_bb,
                            cf_tt, cf_te, cf_et, cf_ee, cf_bb, 
                            cn_tt, cn_te, cn_et, cn_ee, cn_bb, 
                            b_vec_t, b_vec_e, b_vec_b, x_vec_t, x_vec_e, x_vec_b, 
                            preconditioner_tt, preconditioner_te, preconditioner_et,
                            preconditioner_ee, preconditioner_bb, 
                            N, pix_width, theta_pix, maxiter = 500, tol = 1e-1):

    #Compute the A matrix which is the gradient w.r.t. f of the logpdf
    #function evaluated at f = f, d = 0, phi = phi, etc..
    def linear_operator(t_field, e_field, b_field):
        return wiener_filter_matrix_operator_ieb(
            t_field, e_field, b_field, phi, 
            mask_tt, mask_te, mask_et, mask_ee, mask_bb, 
            beam_tt, beam_te, beam_et, beam_ee, beam_bb,
            cf_tt, cf_te, cf_et, cf_ee, cf_bb, 
            cn_tt, cn_te, cn_et, cn_ee, cn_bb, 
            pix_width, N, theta_pix)

    Af_t, Af_e, Af_b = linear_operator(x_vec_t, x_vec_e, x_vec_b) #compute the 1st residual base on the initial guess x0
    r_t, r_e, r_b = ieb_vec_add_sub(b_vec_t, b_vec_e, b_vec_b, Af_t, Af_e, Af_b, add_sub = -1)
    preconditioner_inv_tt, preconditioner_inv_te, preconditioner_inv_et, preconditioner_inv_ee = \
        invert_block_matrix(preconditioner_tt, preconditioner_te, preconditioner_et, preconditioner_ee)
    preconditioner_inv_bb = reciprocal_matrix(preconditioner_bb)
    #compute z = (M^-1 @ r) = (M \ r) since M is diagonal 
    z_t, z_e, z_b = ieb_mat_vec_product(preconditioner_inv_tt, preconditioner_inv_te, preconditioner_inv_et, \
                            preconditioner_inv_ee, preconditioner_inv_bb, r_t, r_e, r_b)
    #copy the values of z
    p_t = z_t
    p_e = z_e
    p_b = z_b 
    #compute the dot products in fourier space between r and z
    res = ieb_dot_product(r_t, r_e, r_b, z_t, z_e, z_b, N**2)
    #define the initial state
    initial_state = (x_vec_t, x_vec_e, x_vec_b, res, res, p_t, p_e, p_b, r_t, r_e, r_b, 0)
    #get standard fourier weights
    fourier_weights = get_fourier_weights(x_vec_t.shape)

    def loop_condition(state):
        _, _, _, _, res_curr, _, _, _, _, _, _, step_idx = state
        return jnp.logical_and(res_curr >= tol, step_idx < maxiter)
    
    def main_loop(state):
        x_vec_t, x_vec_e, x_vec_b, res, res_curr, p_t, p_e, p_b, r_t, r_e, r_b, step_idx = state
        #compute Ap = A(p)
        Ap_t, Ap_e, Ap_b = linear_operator(p_t, p_e, p_b)
        #compute alpha = res / dot(p, Ap)
        p_dot_Ap = ieb_dot_product(p_t, p_e, p_b, Ap_t, Ap_e, Ap_b, N**2)
        alpha = res / p_dot_Ap
        #compute x = x + alpha * p (alpha is a number, p is an array, x is an array)
        x_vec_t, x_vec_e, x_vec_b = ieb_vec_add_sub(x_vec_t, x_vec_e, x_vec_b, \
                                        alpha * p_t, alpha * p_e, alpha * p_b)
        #compute r = r - alpha * Ap
        r_t, r_e, r_b = ieb_vec_add_sub(r_t, r_e, r_b, \
                        alpha * Ap_t, alpha * Ap_e, alpha * Ap_b, add_sub = -1)
        #compute z = (M \ r) = (M^-1 @ r)
        z_t, z_e, z_b = ieb_mat_vec_product(preconditioner_inv_tt, preconditioner_inv_te, preconditioner_inv_et, \
                            preconditioner_inv_ee, preconditioner_inv_bb, r_t, r_e, r_b)
        #current value of the residual
        res_curr = ieb_dot_product(r_t, r_e, r_b, z_t, z_e, z_b, N**2)
        #update the p value
        res_ratio = (res_curr / res)
        p_t, p_e, p_b = ieb_vec_add_sub(z_t, z_e, z_b, \
                        res_ratio * p_t, res_ratio * p_e, res_ratio * p_b)
        #otherwise update previous res value and keep looping
        res = res_curr
        #update the step index
        step_idx += 1
        return (x_vec_t, x_vec_e, x_vec_b, res, res_curr, p_t, p_e, p_b, r_t, r_e, r_b, step_idx)
    
    #perform a jitted while-loop
    final_state = jax.lax.while_loop(loop_condition, main_loop, initial_state)

    #after max iterations have been looped through return the final value
    return final_state[0], final_state[1], final_state[2]

@partial(jax.jit, static_argnames = ["maxiter", "tol"])
def conjugate_gradient_eb(phi, mask_ee, mask_bb,
                            beam_ee, beam_bb,
                            cf_ee, cf_bb, 
                            cn_ee, cn_bb, 
                            b_vec_e, b_vec_b, x_vec_e, x_vec_b, 
                            preconditioner_ee, preconditioner_bb, 
                            N, pix_width, theta_pix, maxiter = 500, tol = 1e-1):

    #Compute the A matrix which is the gradient w.r.t. f of the logpdf
    #function evaluated at f = f, d = 0, phi = phi, etc..
    def linear_operator(e_field, b_field):
        return wiener_filter_matrix_operator_eb(e_field, b_field, phi, 
                                                mask_ee, mask_bb, 
                                                beam_ee, beam_bb,
                                                cf_ee, cf_bb, 
                                                cn_ee, cn_bb, 
                                                pix_width, N, theta_pix)

    Af_e, Af_b = linear_operator(x_vec_e, x_vec_b) #compute the 1st residual base on the initial guess x0
    r_e = b_vec_e - Af_e
    r_b = b_vec_b - Af_b
    preconditioner_inv_ee = reciprocal_matrix(preconditioner_ee)
    preconditioner_inv_bb = reciprocal_matrix(preconditioner_bb)
    #compute z = (M^-1 @ r) = (M \ r) since M is diagonal 
    z_e = preconditioner_inv_ee * r_e
    z_b = preconditioner_inv_bb * r_b
    #copy the values of z
    p_e = z_e
    p_b = z_b 
    #get standard fourier weights
    fourier_weights = get_fourier_weights(phi.shape)
    #compute the dot products in fourier space between r and z
    res_e = jnp.real(jnp.sum(jnp.conj(r_e) * z_e * fourier_weights * (1/N**2)))
    res_b = jnp.real(jnp.sum(jnp.conj(r_b) * z_b * fourier_weights * (1/N**2)))
    res = res_e + res_b
    #define the initial state
    initial_state = (x_vec_e, x_vec_b, res, res, p_e, p_b, r_e, r_b, 0)

    def loop_condition(state):
        _, _, _, res_curr, _, _, _, _, step_idx = state
        return jnp.logical_and(res_curr >= tol, step_idx < maxiter)
    
    def main_loop(state):
        x_vec_e, x_vec_b, res, res_curr, p_e, p_b, r_e, r_b, step_idx = state
        #compute Ap = A(p)
        Ap_e, Ap_b = linear_operator(p_e, p_b)
        #compute alpha = res / dot(p, Ap)
        pe_dot_Ape = jnp.real(jnp.sum(jnp.conj(p_e) * Ap_e * fourier_weights * (1/N**2)))
        pb_dot_Apb = jnp.real(jnp.sum(jnp.conj(p_b) * Ap_b * fourier_weights * (1/N**2)))
        p_dot_Ap = pe_dot_Ape + pb_dot_Apb
        alpha = res / p_dot_Ap
        #compute x = x + alpha * p (alpha is a number, p is an array, x is an array)
        x_vec_e = x_vec_e + alpha * p_e
        x_vec_b = x_vec_b + alpha * p_b
        #compute r = r - alpha * Ap
        r_e = r_e - alpha * Ap_e
        r_b = r_b - alpha * Ap_b
        #compute z = (M \ r) = (M^-1 @ r)
        z_e = preconditioner_inv_ee * r_e
        z_b = preconditioner_inv_bb * r_b
        #current value of the residual
        rz_e = jnp.real(jnp.sum(jnp.conj(r_e) * z_e * fourier_weights * (1/N**2)))
        rz_b = jnp.real(jnp.sum(jnp.conj(r_b) * z_b * fourier_weights * (1/N**2)))
        res_curr = rz_e + rz_b
        #update the p value
        res_ratio = (res_curr / res)
        p_e = z_e + res_ratio * p_e
        p_b = z_b + res_ratio * p_b
        #otherwise update previous res value and keep looping
        res = res_curr
        #update the step index
        step_idx += 1
        return (x_vec_e, x_vec_b, res, res_curr, p_e, p_b, r_e, r_b, step_idx)
    
    #perform a jitted while-loop
    final_state = jax.lax.while_loop(loop_condition, main_loop, initial_state)

    #after max iterations have been looped through return the final value
    return final_state[0], final_state[1]



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

    # data = data_set["data"]
    # m_diagonal = data_set["m"]
    # b_diagonal = data_set["b"]
    # d_diagonal = data_set["d_matrix"]
    # g_diagonal = data_set["g"]
    # cf_diagonal = data_set["cf"]
    # cn_diagonal = data_set["cn"]
    # cphi_diagonal = data_set["cphi"]
    # nphi_diagonal = data_set["nphi"]
    # f_lambda_array = data_set["fourier_weights"] #TODO replace all these specific fourier weights with just the standard fourier weights...
    # phi_lambda_array = data_set["fourier_weights"]
    # data_lambda_array = data_set["fourier_weights"]
    # cf_lambda_array = data_set["fourier_weights"]
    # cphi_lambda_array = data_set["fourier_weights"]
    # cn_lambda_array = data_set["fourier_weights"]
    # pix_width = data_set["pix_width"]
    # num_pixels = data_set["num_pixels"]
    
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

#TODO implement full IEB MLE Descent Algorithm!!
def run_gradient_descent_ieb(data_set, num_steps):

    #unpack the necessary data from the data set object
    data_t = data_set["data_t"]
    data_e = data_set["data_e"]
    data_b = data_set["data_b"]
    mask_tt = data_set["mask_tt"]
    mask_te = data_set["mask_te"]
    mask_et = data_set["mask_et"]
    mask_ee = data_set["mask_ee"]
    mask_bb = data_set["mask_bb"]
    beam_tt = data_set["beam_tt"]
    beam_te = data_set["beam_te"]
    beam_et = data_set["beam_et"]
    beam_ee = data_set["beam_ee"]
    beam_bb = data_set["beam_bb"]
    d_tt = data_set["d_tt"]
    d_te = data_set["d_te"]
    d_et = data_set["d_et"]
    d_ee = data_set["d_ee"]
    d_bb = data_set["d_bb"]
    g_diagonal = data_set["g_diagonal"]
    cf_tt = data_set["cf_tt"]
    cf_te = data_set["cf_te"]
    cf_et = data_set["cf_et"]
    cf_ee = data_set["cf_ee"]
    cf_bb = data_set["cf_bb"]
    cn_tt = data_set["cn_tt"]
    cn_te = data_set["cn_te"]
    cn_et = data_set["cn_et"]
    cn_ee = data_set["cn_ee"]
    cn_bb = data_set["cn_bb"]
    cphi_diagonal = data_set["cphi_diagonal"]
    nphi_diagonal = data_set["nphi_diagonal"]
    pix_width = data_set["pix_width"]
    
    #compute number of rows and columns in fourier space given map size
    num_rows, num_cols = data_t.shape
    N = 256
    theta_pix = 2

    #set starting guesses for f, phi, and f_lensed to all zeros
    t_predict = jnp.zeros((num_rows, num_cols), dtype=jnp.complex128)
    e_predict = jnp.zeros((num_rows, num_cols), dtype=jnp.complex128)
    b_predict = jnp.zeros((num_rows, num_cols), dtype=jnp.complex128)
    phi_predict = jnp.zeros((num_rows, num_cols), dtype=jnp.complex128)

    #compute the hessian which we will use to calculate the step in the phi direction 
    hessian = reciprocal_matrix(phi_gradient_hessian(cphi_diagonal, nphi_diagonal))

    #loop for num_steps iterations
    for step_idx in range(num_steps):

        #compute the wiener filter of f
        t_predict, e_predict, b_predict = wiener_filter_ieb(t_predict, e_predict, b_predict, phi_predict, 
                                                            data_t, data_e, data_b, 
                                                            mask_tt, mask_ee, mask_bb, mask_te, mask_et, 
                                                            beam_tt, beam_ee, beam_bb, beam_te, beam_et, 
                                                            cf_tt, cf_ee, cf_bb, cf_te, cf_et, 
                                                            cn_tt, cn_ee, cn_bb, cn_te, cn_et, 
                                                            pix_width, N, theta_pix)

        #must mix f and phi before taking the gradient
        t_mixed, e_mixed, b_mixed, phi_mixed = mix_ieb(t_predict, e_predict, b_predict, phi_predict, 
                                                       d_tt, d_te, d_et, d_ee, d_bb, 
                                                       g_diagonal, pix_width, N, theta_pix)
        mixed_grad_phi = mixed_phi_gradient_ieb(t_mixed, e_mixed, b_mixed, phi_mixed, data_t, data_e, data_b,
                                                beam_tt, beam_ee, beam_bb, beam_te, beam_te, mask_tt, mask_ee, mask_bb, mask_te, mask_te,
                                                d_tt, d_te, d_te, d_ee, d_bb, g_diagonal,
                                                cf_tt, cf_ee, cf_bb, cf_te, cf_te, cphi_diagonal, 
                                                cn_tt, cn_ee, cn_bb, cn_te, cn_te, pix_width, N, theta_pix)
        #jax.debug.print("step number = {}, grad phi norm = {}", step_idx, jnp.linalg.norm(mixed_grad_phi), ordered = True)

        #create a new partial function whose only input is alpha for use in the brent optimizer (we want to
        #minimize the value of this function w.r.t. the single parameter alpha)
        min_logpdf = partial(logpdf_at_phi_new_ieb, grad_phi = mixed_grad_phi, t_mixed = t_mixed, e_mixed = e_mixed, b_mixed = b_mixed, 
                      phi_predict = phi_mixed, data_t = data_t, data_e = data_e, data_b = data_b,
                      beam_tt = beam_tt, beam_ee = beam_ee, beam_bb = beam_bb, mask_tt = mask_tt, mask_ee = mask_ee, 
                      mask_bb = mask_bb, cf_tt = cf_tt, cf_te = cf_te, cf_ee = cf_ee, cf_bb = cf_bb, 
                      cphi_diagonal = cphi_diagonal, cn_tt = cn_tt, cn_te = cn_te, cn_ee = cn_ee, cn_bb = cn_bb,
                      d_tt = d_tt, d_te = d_te, d_et = d_et, d_ee = d_ee, d_bb = d_bb, g_diagonal = g_diagonal,
                      pix_width = pix_width, N = N, theta_pix = theta_pix, hessian = hessian)
        
        #call the optimizer to find which alpha minimizes the best
        optimizer = minimize(min_logpdf, 1, method = 'BFGS') 
        alpha = optimizer.x
        #jax.debug.print("step number = {}, alpha = {}", step_idx, jnp.linalg.norm(alpha), ordered = True)
        #now update phi with the best alpha choice
        phi_predict = phi_predict + alpha * hessian * mixed_grad_phi

    #return the three predicted values and the history object
    return t_predict, e_predict, b_predict, phi_predict

def run_gradient_descent_eb(data_set, num_steps):

    #unpack the necessary data from the data set object
    data_e = data_set["data_e"]
    data_b = data_set["data_b"]
    mask_ee = data_set["mask_ee"]
    mask_bb = data_set["mask_bb"]
    beam_ee = data_set["beam_ee"]
    beam_bb = data_set["beam_bb"]
    d_ee = data_set["d_ee"]
    d_bb = data_set["d_bb"]
    g_diagonal = data_set["g_diagonal"]
    cf_ee = data_set["cf_ee"]
    cf_bb = data_set["cf_bb"]
    cn_ee = data_set["cn_ee"]
    cn_bb = data_set["cn_bb"]
    cphi_diagonal = data_set["cphi_diagonal"]
    nphi_diagonal = data_set["nphi_diagonal"]
    pix_width = data_set["pix_width"]
    
    #compute number of rows and columns in fourier space given map size
    num_rows, num_cols = data_e.shape
    N = 256
    theta_pix = 2

    #set starting guesses for f, phi, and f_lensed to all zeros
    e_predict = jnp.zeros((num_rows, num_cols), dtype=jnp.complex128)
    b_predict = jnp.zeros((num_rows, num_cols), dtype=jnp.complex128)
    phi_predict = jnp.zeros((num_rows, num_cols), dtype=jnp.complex128)

    #compute the hessian which we will use to calculate the step in the phi direction 
    hessian = reciprocal_matrix(phi_gradient_hessian(cphi_diagonal, nphi_diagonal))
    alphas = []
    logpdf_values = []

    #loop for num_steps iterations
    alpha_max = 1
    for step_idx in range(num_steps):

        #compute the wiener filter of f
        e_predict, b_predict = wiener_filter_eb(e_predict, b_predict, phi_predict, data_e, data_b, 
                                                mask_ee, mask_bb, beam_ee, beam_bb, 
                                                cf_ee, cf_bb, cn_ee, cn_bb, 
                                                pix_width, N, theta_pix)

        #must mix f and phi before taking the gradient
        e_mixed, b_mixed, phi_mixed = mix_eb(e_predict, b_predict, phi_predict, d_ee, d_bb, g_diagonal, pix_width, N, theta_pix)
        #e_mixed = precision_load(f"/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/fJ_mixed_E_{step_idx}.npz")
        #b_mixed = precision_load(f"/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/fJ_mixed_B_{step_idx}.npz")
        mixed_grad_phi = mixed_phi_gradient_eb(e_mixed, b_mixed, phi_mixed, data_e, data_b,
                                                beam_ee, beam_bb, mask_ee, mask_bb,
                                                d_ee, d_bb, g_diagonal,
                                                cf_ee, cf_bb, cphi_diagonal, cn_ee, cn_bb,
                                                pix_width, N, theta_pix)
        #jax.debug.print("step number = {}, grad phi norm = {}", step_idx, jnp.linalg.norm(mixed_grad_phi), ordered = True)

        #create a new partial function whose only input is alpha for use in the brent optimizer (we want to
        #minimize the value of this function w.r.t. the single parameter alpha)
        min_logpdf = partial(logpdf_at_phi_new_eb, grad_phi = mixed_grad_phi, e_mixed = e_mixed, b_mixed = b_mixed, 
                      phi_predict = phi_mixed, data_e = data_e, data_b = data_b,
                      beam_ee = beam_ee, beam_bb = beam_bb, mask_ee = mask_ee, mask_bb = mask_bb, cf_ee = cf_ee, cf_bb = cf_bb, 
                      cphi_diagonal = cphi_diagonal, cn_ee = cn_ee, cn_bb = cn_bb,
                      d_ee = d_ee, d_bb = d_bb, g_diagonal = g_diagonal,
                      pix_width = pix_width, N = N, theta_pix = theta_pix, hessian = hessian)
        
        #call the optimizer to find which alpha minimizes the best
        # optimizer = minimize(min_logpdf, 2*alpha_max, method = 'BFGS') 
        optimizer = minimize(min_logpdf, 1, method = 'BFGS') 
        alpha_max = optimizer.x
        alphas.append(alpha_max)
        logpdf_values.append(optimizer.fun)
        #jax.debug.print("step number = {}, alpha = {}", step_idx, jnp.linalg.norm(alpha), ordered = True)
        #now update phi with the best alpha choice
        phi_predict = phi_predict + alpha_max * hessian * mixed_grad_phi

    #return the three predicted values and the history object
    return e_predict, b_predict, phi_predict, alphas, logpdf_values


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

@jax.jit
def logpdf_at_phi_new_ieb(alpha, grad_phi, t_mixed, e_mixed, b_mixed, phi_predict, data_t, data_e, data_b,
                      beam_tt, beam_ee, beam_bb, mask_tt, mask_ee, mask_bb, cf_tt, cf_te, cf_ee, cf_bb, 
                      cphi_diagonal, cn_tt, cn_te, cn_ee, cn_bb,
                      d_tt, d_te, d_et, d_ee, d_bb, g_diagonal,
                      pix_width, N, theta_pix, hessian):
    
    #the test phi will be the current phi prediction minus a step
    #in the opposite direction of the gradient of logpdf w.r.t. phi
    test_phi = phi_predict + alpha * hessian * grad_phi

    #try unmixing f by the test phi! before we compute the logpdf
    t_unmixed, e_unmixed, b_unmixed, phi_unmixed = unmix_ieb(t_mixed, e_mixed, b_mixed, test_phi, 
                                                             d_tt, d_te, d_et, d_ee, d_bb, g_diagonal, 
                                                             pix_width, N, theta_pix)

    #compute the value of logpdf at this test phi value
    phi_covar = reshape_diagonal(cphi_diagonal)
    logpdf_value = logpdf_ieb_unlensed_param(t_unmixed, e_unmixed, b_unmixed, cf_tt, cf_te, cf_ee, cf_bb,
                                            data_t, data_e, data_b, cn_tt, cn_te, cn_ee, cn_bb, phi_unmixed, 
                                            phi_covar, mask_tt, mask_ee, mask_bb, beam_tt, beam_ee, beam_bb, 
                                            N, theta_pix, pix_width)
    
    #return the negative one times the logpdf
    return -1*logpdf_value

@jax.jit
def logpdf_at_phi_new_eb(alpha, grad_phi, e_mixed, b_mixed, phi_predict, data_e, data_b,
                      beam_ee, beam_bb, mask_ee, mask_bb, cf_ee, cf_bb, 
                      cphi_diagonal, cn_ee, cn_bb,
                      d_ee, d_bb, g_diagonal,
                      pix_width, N, theta_pix, hessian):
    
    #the test phi will be the current phi prediction minus a step
    #in the opposite direction of the gradient of logpdf w.r.t. phi
    test_phi = phi_predict + alpha * hessian * grad_phi

    #try unmixing f by the test phi! before we compute the logpdf
    e_unmixed, b_unmixed, phi_unmixed = unmix_eb(e_mixed, b_mixed, test_phi, 
                                                d_ee, d_bb, g_diagonal, 
                                                pix_width, N, theta_pix)

    #compute the value of logpdf at this test phi value
    phi_covar = reshape_diagonal(cphi_diagonal)
    logpdf_value = logpdf_eb_unlensed_param(e_unmixed, b_unmixed, cf_ee, cf_bb,
                                            data_e, data_b, cn_ee, cn_bb, phi_unmixed, 
                                            phi_covar, mask_ee, mask_bb, beam_ee, beam_bb, 
                                            N, theta_pix, pix_width)
    
    #return the negative one times the logpdf
    return -1*logpdf_value

#NOTE in these types of matrices do we always expect the cross-diagonal
#elements to be equivalent??? For the time being I will assume not I guess...
@jax.jit
def ieb_matrix_mult(tt1, te1, et1, ee1, bb1, tt2, te2, et2, ee2, bb2):
    # [TT_1 TE_1   0  ]    [TT_2 TE_2   0  ]
    # [ET_1 EE_1   0  ] x  [ET_2 EE_2   0  ]
    # [0    0    BB_1 ]    [0    0    BB_2 ]
    result_tt = tt1 * tt2 + te1 * et2
    result_et = et1 * tt2 + ee1 * et2
    result_te = tt1 * te2 + te1 * ee2
    result_ee = et1 * te2 + ee1 * ee2
    result_bb = bb1 * bb2 
    return result_tt, result_te, result_et, result_ee, result_bb

#NOTE The square root of a block diagonal matrix is NOT equivalent to
#the square root of the individual elements. This was found out the painful way...
@jax.jit
def ieb_matrix_sqrt(a, b, c, d):
    # [ a b ]
    # [ c d ]
    s = jnp.sqrt(a * d - b * c)
    t = jnp.sqrt(a + (d + 2*s))
    t = jnp.where(t != 0, 1/t, 0)
    result_11 = t * (a + s)
    result_12 = t * b
    result_21 = t * c
    result_22 = t * (d + s)
    return result_11, result_12, result_21, result_22

@jax.jit
def ieb_mat_vec_product(mat_tt, mat_te, mat_et, mat_ee, mat_bb, vec_t, vec_e, vec_b):
    # [TT TE   0  ]    [T]
    # [ET EE   0  ] x  [E]
    # [0  0    BB ]    [B]
    result_t = mat_tt * vec_t + mat_te * vec_e
    result_e = mat_et * vec_t + mat_ee * vec_e
    result_b = mat_bb * vec_b
    return result_t, result_e, result_b

@jax.jit
def ieb_dot_product(a_t, a_e, a_b, b_t, b_e, b_b, num_pixels):
    # [T    E    B]    [T]
    #               x  [E]
    #                  [B]
    fourier_weights = get_fourier_weights(a_t.shape)
    tt_product = jnp.real(jnp.sum(jnp.conj(a_t) * b_t * fourier_weights * (1/num_pixels)))
    ee_product = jnp.real(jnp.sum(jnp.conj(a_e) * b_e * fourier_weights * (1/num_pixels)))
    bb_product = jnp.real(jnp.sum(jnp.conj(a_b) * b_b * fourier_weights * (1/num_pixels)))
    return tt_product + ee_product + bb_product

@jax.jit
def ieb_vec_add_sub(t1, e1, b1, t2, e2, b2, add_sub = +1):
    def subtract(_):
        return t1 - t2, e1 - e2, b1 - b2
    def add(_):
        return t1 + t2, e1 + e2, b1 + b2
    t_res, e_res, b_res = jax.lax.cond(
        jnp.equal(add_sub, -1),
        subtract,
        add,
        operand = None
    )
    return t_res, e_res, b_res

#@partial(jax.jit, static_argnames = ["theta_pix", "N"])
@jax.jit
def lense_ieb(t_field, e_field, b_field, phi, pix_width, N, theta_pix, n = 10, 
              direction = +1, adjoint = False, rescale_and_conjugate = False):
    #N = jnp.sqrt(num_pixels).astype(int)
    num_pixels = N**2
    q_field, u_field = eb2qu(jfft.rfft2(e_field), jfft.rfft2(b_field), N, theta_pix)
    t_lensed = lense_flow(t_field, phi, pix_width, n, direction, adjoint, num_pixels, rescale_and_conjugate)
    q_lensed = lense_flow(jfft.irfft2(q_field), phi, pix_width, n, direction, adjoint, num_pixels, rescale_and_conjugate)
    u_lensed = lense_flow(jfft.irfft2(u_field), phi, pix_width, n, direction, adjoint, num_pixels, rescale_and_conjugate)
    e_lensed, b_lensed = qu2eb(jfft.rfft2(q_lensed), jfft.rfft2(u_lensed), N, theta_pix)
    return t_lensed, jfft.irfft2(e_lensed), jfft.irfft2(b_lensed)

@jax.jit
def lense_eb(e_field, b_field, phi, pix_width, N, theta_pix, n = 10, 
              direction = +1, adjoint = False, rescale_and_conjugate = False):
    num_pixels = N**2
    q_field, u_field = eb2qu(jfft.rfft2(e_field), jfft.rfft2(b_field), N, theta_pix)
    q_lensed = lense_flow(jfft.irfft2(q_field), phi, pix_width, n, direction, adjoint, num_pixels, rescale_and_conjugate)
    u_lensed = lense_flow(jfft.irfft2(u_field), phi, pix_width, n, direction, adjoint, num_pixels, rescale_and_conjugate)
    e_lensed, b_lensed = qu2eb(jfft.rfft2(q_lensed), jfft.rfft2(u_lensed), N, theta_pix)
    return jfft.irfft2(e_lensed), jfft.irfft2(b_lensed)

@jax.custom_vjp
@jax.jit
def lense_eb_wrapper(e_field, b_field, phi, pix_width, N, theta_pix, n, direction, adjoint, rescale_and_conjugate):
    return lense_eb(e_field, b_field, phi, pix_width, N, theta_pix, n, direction, adjoint, rescale_and_conjugate)

#Forward pass of the custom VJP
@jax.jit
def lense_eb_forward_pass(e_field, b_field, phi, pix_width, N, theta_pix, n, direction, adjoint, rescale_and_conjugate):
    
    #Compute primal output
    e_transformed, b_transformed = lense_eb(e_field, b_field, phi, pix_width, \
                                            N, theta_pix, n, direction, adjoint, rescale_and_conjugate)

    #return the value of the transformed field and also store any data needed by backward pass.
    return (e_transformed, b_transformed), (e_transformed, b_transformed, \
                                            e_field, b_field, phi, pix_width, \
                                            N, theta_pix, n, direction, adjoint, \
                                            rescale_and_conjugate)

#Backward pass of the custom vjp
@jax.jit
def lense_eb_backwards_pass(res, g):
    
    #unpack the necessary data
    (e_transformed, b_transformed, e_field, b_field, phi, pix_width, \
                    N, theta_pix, n, direction, adjoint, rescale_and_conjugate) = res 

    #get the value of the gradients of the lensing operator w.r.t. f and phi
    #NOTE WE NEED TO USE THE TRANSFORMED FIELD AS THE INPUT HERE AND INTEGRATE IN THE OPPOSITE DIRECTION
    _, _, delta_fe, delta_fb, delta_phi = get_eb_lensing_operator_gradients(phi, \
                                                        e_transformed, b_transformed, g, pix_width, 
                                                        N, theta_pix, -direction, n)
    
    def undo_logpdf_operations(delta_fe, delta_fb, delta_phi):
        num_rows, _ = delta_phi.shape
        STANDARD_FOURIER_WEIGHTS = 2*jnp.ones(num_rows//2+1, dtype =jnp.complex128) #TODO remove magic number
        STANDARD_FOURIER_WEIGHTS = STANDARD_FOURIER_WEIGHTS.at[0].set(1)
        STANDARD_FOURIER_WEIGHTS = STANDARD_FOURIER_WEIGHTS.at[-1].set(1)

        delta_fe_fourier = jnp.conj(jfft.rfft2(delta_fe) / STANDARD_FOURIER_WEIGHTS[jnp.newaxis, :])
        delta_fb_fourier = jnp.conj(jfft.rfft2(delta_fb) / STANDARD_FOURIER_WEIGHTS[jnp.newaxis, :])
        delta_phi_fourier = jnp.conj(jfft.rfft2(delta_phi) / STANDARD_FOURIER_WEIGHTS[jnp.newaxis, :])

        delta_fe = jfft.irfft2((N**2)*delta_fe_fourier)
        delta_fb = jfft.irfft2((N**2)*delta_fb_fourier)
        delta_phi = jfft.irfft2((N**2)*delta_phi_fourier)
        return delta_fe, delta_fb, delta_phi
    
    def maintain_current_deltas(delta_fe, delta_fb, delta_phi):
        return delta_fe, delta_fb, delta_phi

    operands = delta_fe, delta_fb, delta_phi
    delta_fe, delta_fb, delta_phi = jax.lax.cond(
        rescale_and_conjugate,
        undo_logpdf_operations,
        maintain_current_deltas,
        *operands
    )

    #Use autodiff for the rest of the inputs.
    def lense_eb_only_others(*other_inputs):
        return lense_eb(e_field, b_field, phi, *other_inputs)

    #Now we get their VJPs in one call:
    _, vjp_fun = jax.vjp(lense_eb_only_others, pix_width, N, theta_pix, n, direction, adjoint, rescale_and_conjugate)
    other_gradients = vjp_fun(g) 

    #Return a gradient for EVERY input (either via a custom analytical gradient or via AutoDiff)
    return (delta_fe, delta_fb, delta_phi, *other_gradients)

# -----------------------------------------------------------
# Register the two-pass rule for the inverse lensing operator
# -----------------------------------------------------------
lense_eb_wrapper.defvjp(lense_eb_forward_pass, lense_eb_backwards_pass)

#Define a wrapper function that returns the same values as lense flow but has custom gradients defined for each of its parameters
#AND for BOTH the FORWARD and BACKWARDS directions!
#@partial(jax.jit, static_argnames = ["theta_pix", "N"])
@jax.custom_vjp
#@partial(jax.jit, static_argnames = ["theta_pix", "N"])
@jax.jit
def lense_ieb_wrapper(t_field, e_field, b_field, phi, pix_width, N, theta_pix, n, direction, adjoint, rescale_and_conjugate):
    return lense_ieb(t_field, e_field, b_field, phi, pix_width, N, theta_pix, n, direction, adjoint, rescale_and_conjugate)

#Forward pass of the custom VJP
#@partial(jax.jit, static_argnames = ["theta_pix", "N"])
@jax.jit
def lense_ieb_forward_pass(t_field, e_field, b_field, phi, pix_width, N, theta_pix, n, direction, adjoint, rescale_and_conjugate):
    
    #Compute primal output
    t_transformed, e_transformed, b_transformed = lense_ieb(t_field, e_field, b_field, phi, pix_width, \
                                                N, theta_pix, n, direction, adjoint, rescale_and_conjugate)

    #return the value of the transformed field and also store any data needed by backward pass.
    return (t_transformed, e_transformed, b_transformed), (t_transformed, e_transformed, b_transformed, \
                                                           t_field, e_field, b_field, phi, pix_width, \
                                                           N, theta_pix, n, direction, adjoint, \
                                                           rescale_and_conjugate)

#Backward pass of the custom vjp
@jax.jit
def lense_ieb_backwards_pass(res, g):
    
    #unpack the necessary data
    (t_transformed, e_transformed, b_transformed, t_field, e_field, b_field, phi, pix_width, \
                    N, theta_pix, n, direction, adjoint, rescale_and_conjugate) = res 

    #get the value of the gradients of the lensing operator w.r.t. f and phi
    #NOTE WE NEED TO USE THE TRANSFORMED FIELD AS THE INPUT HERE AND INTEGRATE IN THE OPPOSITE DIRECTION
    _, _, _,  delta_ft, delta_fe, delta_fb, delta_phi = get_ieb_lensing_operator_gradients(phi, t_transformed, \
                                                        e_transformed, b_transformed, g, pix_width, N, theta_pix, -direction, n)
    
    def undo_logpdf_operations(delta_ft, delta_fe, delta_fb, delta_phi):
        num_rows, _ = delta_phi.shape
        STANDARD_FOURIER_WEIGHTS = 2*jnp.ones(num_rows//2+1, dtype =jnp.complex128) #TODO remove magic number
        STANDARD_FOURIER_WEIGHTS = STANDARD_FOURIER_WEIGHTS.at[0].set(1)
        STANDARD_FOURIER_WEIGHTS = STANDARD_FOURIER_WEIGHTS.at[-1].set(1)

        delta_ft_fourier = jnp.conj(jfft.rfft2(delta_ft) / STANDARD_FOURIER_WEIGHTS[jnp.newaxis, :])
        delta_fe_fourier = jnp.conj(jfft.rfft2(delta_fe) / STANDARD_FOURIER_WEIGHTS[jnp.newaxis, :])
        delta_fb_fourier = jnp.conj(jfft.rfft2(delta_fb) / STANDARD_FOURIER_WEIGHTS[jnp.newaxis, :])
        delta_phi_fourier = jnp.conj(jfft.rfft2(delta_phi) / STANDARD_FOURIER_WEIGHTS[jnp.newaxis, :])

        delta_ft = jfft.irfft2((N**2)*delta_ft_fourier)
        delta_fe = jfft.irfft2((N**2)*delta_fe_fourier)
        delta_fb = jfft.irfft2((N**2)*delta_fb_fourier)
        delta_phi = jfft.irfft2((N**2)*delta_phi_fourier)
        return delta_ft, delta_fe, delta_fb, delta_phi
    
    def maintain_current_deltas(delta_ft, delta_fe, delta_fb, delta_phi):
        return delta_ft, delta_fe, delta_fb, delta_phi

    operands = delta_ft, delta_fe, delta_fb, delta_phi
    delta_ft, delta_fe, delta_fb, delta_phi = jax.lax.cond(
        rescale_and_conjugate,
        undo_logpdf_operations,
        maintain_current_deltas,
        *operands
    )

    #Use autodiff for the rest of the inputs.
    def lense_ieb_only_others(*other_inputs):
        return lense_ieb(t_field, e_field, b_field, phi, *other_inputs)

    #Now we get their VJPs in one call:
    _, vjp_fun = jax.vjp(lense_ieb_only_others, pix_width, N, theta_pix, n, direction, adjoint, rescale_and_conjugate)
    other_gradients = vjp_fun(g) 

    #Return a gradient for EVERY input (either via a custom analytical gradient or via AutoDiff)
    return (delta_ft, delta_fe, delta_fb, delta_phi, *other_gradients)

# -----------------------------------------------------------
# Register the two-pass rule for the inverse lensing operator
# -----------------------------------------------------------
lense_ieb_wrapper.defvjp(lense_ieb_forward_pass, lense_ieb_backwards_pass)

#@partial(jax.jit, static_argnames = ["theta_pix", "N"])
@jax.jit
def gradf_logpdf_ieb(t_field, e_field, b_field, 
                     t_data, e_data, b_data, phi,
                     cn_tt, cn_ee, cn_bb, cn_te, cn_et,
                     cf_tt, cf_ee, cf_bb, cf_te, cf_et,
                     mask_tt, mask_ee, mask_bb, mask_te, mask_et,
                     beam_tt, beam_ee, beam_bb, beam_te, beam_et,
                     N, theta_pix, pix_width):
    
    #NOTE up to here is okay... error on the order of 1e-5 to 1e-4 at max...
    #first compute [I, E, B]_lensed
    t_lensed, e_lensed, b_lensed = lense_ieb(jfft.irfft2(t_field), jfft.irfft2(e_field), jfft.irfft2(b_field), jfft.irfft2(phi), \
                                             pix_width, N, theta_pix)
    t_lensed = jfft.rfft2(t_lensed)
    e_lensed = jfft.rfft2(e_lensed)
    b_lensed = jfft.rfft2(b_lensed)

    #compute M * B * [I, E, B]_lensed #NOTE these matrices match EXACTLY (diff = 0.0)
    mb_tt, mb_te, mb_et, mb_ee, mb_bb = ieb_matrix_mult(mask_tt, mask_te, mask_et, mask_ee, mask_bb, \
                                                        beam_tt, beam_te, beam_et, beam_ee, beam_bb)
    #NOTE up to here is good, had stoopid typo in matrix-vector product definition
    t_signal, e_signal, b_signal = ieb_mat_vec_product(mb_tt, mb_te, mb_et, mb_ee, mb_bb, \
                                                       t_lensed, e_lensed, b_lensed)

    #compute the vector difference diff_vec_[I, E, B] = {data_[I, E, B] - (M * B * f_lensed_[I, E, B])}
    diff_t, diff_e, diff_b = ieb_vec_add_sub(t_data, e_data, b_data, \
                                             t_signal, e_signal, b_signal, \
                                             add_sub = -1)

    #Now take the diff_vec and multiply by the inverse of Cn
    cn_inv_tt, cn_inv_te, cn_inv_et, cn_inv_ee = invert_block_matrix(cn_tt, cn_te, cn_et, cn_ee)
    cn_inv_bb = reciprocal_matrix(cn_bb)
    t_result, e_result, b_result = ieb_mat_vec_product(cn_inv_tt, cn_inv_te, cn_inv_et, cn_inv_ee, cn_inv_bb, \
                                                       diff_t, diff_e, diff_b)
    
    #Now, take the result of the above and apply L^Dagger * B^Dagger * M^Dagger NOTE that the default M and B matrices
    #are Hermitian in the sense that their complex conjugate transpose is equal to the original matrix...
    #I am not sure if this is true in general, but I will assume that to be the case here...
    t_result, e_result, b_result = ieb_mat_vec_product(mb_tt, mb_te, mb_et, mb_ee, mb_bb, \
                                                       t_result, e_result, b_result)
    t_result, e_result, b_result = lense_ieb(jfft.irfft2(t_result), jfft.irfft2(e_result), jfft.irfft2(b_result), jfft.irfft2(phi), pix_width, \
                                             N, theta_pix, direction = -1, adjoint = True)
    t_result = jfft.rfft2(t_result)
    e_result = jfft.rfft2(e_result)
    b_result = jfft.rfft2(b_result)
    
    #The last step is to subtract off Cf^-1 * f
    cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee = invert_block_matrix(cf_tt, cf_te, cf_et, cf_ee)
    cf_inv_bb = reciprocal_matrix(cf_bb)
    cff_prod_tt, cff_prod_ee, cff_prod_bb = ieb_mat_vec_product(cf_inv_tt, cf_inv_te, cf_inv_et, cf_inv_ee, cf_inv_bb, \
                                                                t_field, e_field, b_field)
    t_result, e_result, b_result = ieb_vec_add_sub(t_result, e_result, b_result, \
                                                   cff_prod_tt, cff_prod_ee, cff_prod_bb, \
                                                   add_sub = -1)
    return t_result, e_result, b_result

@jax.jit
def gradf_logpdf_eb(e_field, b_field, 
                     e_data, b_data, phi,
                     cn_ee, cn_bb,
                     cf_ee, cf_bb,
                     mask_ee, mask_bb,
                     beam_ee, beam_bb,
                     N, theta_pix, pix_width):
    
    #NOTE up to here is okay... error on the order of 1e-5 to 1e-4 at max...
    #first compute [I, E, B]_lensed
    e_lensed, b_lensed = lense_eb(jfft.irfft2(e_field), jfft.irfft2(b_field), jfft.irfft2(phi), pix_width, N, theta_pix)
    e_lensed = jfft.rfft2(e_lensed)
    b_lensed = jfft.rfft2(b_lensed)

    #compute M * B * [I, E, B]_lensed #NOTE these matrices match EXACTLY (diff = 0.0)
    # e_signal = mask_ee * mask_bb * e_lensed
    # b_signal = beam_ee * beam_bb * b_lensed
    e_signal = mask_ee * beam_ee * e_lensed
    b_signal = mask_bb * beam_bb * b_lensed

    #compute the vector difference diff_vec_[I, E, B] = {data_[I, E, B] - (M * B * f_lensed_[I, E, B])}
    diff_e = e_data - e_signal
    diff_b = b_data - b_signal

    #Now take the diff_vec and multiply by the inverse of Cn
    cn_inv_ee = reciprocal_matrix(cn_ee)
    cn_inv_bb = reciprocal_matrix(cn_bb)
    # e_result = mask_ee * mask_bb * cn_inv_ee * diff_e
    # b_result = beam_ee * beam_bb * cn_inv_bb * diff_b
    e_result = mask_ee * beam_ee * cn_inv_ee * diff_e
    b_result = mask_bb * beam_bb * cn_inv_bb * diff_b

    e_result, b_result = lense_eb(jfft.irfft2(e_result), jfft.irfft2(b_result), jfft.irfft2(phi), \
                                  pix_width, N, theta_pix, direction = -1, adjoint = True)
    e_result = jfft.rfft2(e_result)
    b_result = jfft.rfft2(b_result)
    
    #The last step is to subtract off Cf^-1 * f
    cf_inv_ee = reciprocal_matrix(cf_ee)
    cf_inv_bb = reciprocal_matrix(cf_bb)
    e_result = e_result - cf_inv_ee * e_field
    b_result = b_result - cf_inv_bb * b_field
    return e_result, b_result

#try to standardize the loading of data
#This should essentially be every thing found in ds; in the julia code
def load_data_set(file_path_to_folder):

    #initialize a set / map
    data_set ={}

    #load the .npz type of files into their respective keys
    data_set["data"] = precision_load(file_path_to_folder + "data_t_field_I.npz")
    data_set["m_diagonal"] = reshape_diagonal(precision_load(file_path_to_folder + "mask_covar_tt.npz"))
    data_set["b_diagonal"] = reshape_diagonal(precision_load(file_path_to_folder + "beam_covar_tt.npz"))
    data_set["g_diagonal"] = precision_load(file_path_to_folder + "g_diagonal.npz")
    data_set["d_diagonal"] = precision_load(file_path_to_folder + "d_tt.npz")
    data_set["cf_diagonal"] = precision_load(file_path_to_folder + "cf_covar_tt.npz")
    data_set["cf_lensed_diagonal"] = precision_load(file_path_to_folder + "cfl_covar_tt.npz")
    data_set["cn_diagonal"] = precision_load(file_path_to_folder + "cn_covar_tt.npz")
    data_set["cphi_diagonal"] = precision_load(file_path_to_folder + "cphi_diagonal.npz")
    data_set["nphi_diagonal"] = precision_load(file_path_to_folder + "nphi_diagonal_I.npz")
    lambda_array = 2 * jnp.ones(129)
    lambda_array = lambda_array.at[0].set(1)
    lambda_array = lambda_array.at[-1].set(1)
    data_set["f_lambda_array"] = lambda_array
    data_set["phi_lambda_array"] = lambda_array
    data_set["data_lambda_array"] = lambda_array
    data_set["cf_lambda_array"] = lambda_array
    data_set["cphi_lambda_array"] = lambda_array
    data_set["cn_lambda_array"] = lambda_array

    #read the scalar values from their text files and load into their respective keys
    # with open(file_path_to_folder + "pix_width.txt", "r") as f:
    #     pix_width = float(f.read().strip()) 
    # with open(file_path_to_folder + "num_pixels.txt", "r") as f:
    #     num_pixels = float(f.read().strip()) 
    data_set["pix_width"] = jnp.deg2rad(2 / ARCMIN_PER_DEGREE)
    data_set["num_pixels"] = 256 * 256

    #return the updated object
    return data_set

def load_ieb_data_set(file_path_to_folder):
    data_set = {}
    data_set["data_t"] = precision_load(str(file_path_to_folder) + "data_t_field_IP.npz")
    data_set["data_e"] = precision_load(str(file_path_to_folder) + "data_e_field_IP.npz")
    data_set["data_b"] = precision_load(str(file_path_to_folder) + "data_b_field_IP.npz")
    data_set["mask_tt"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "mask_covar_tt.npz"))
    data_set["mask_te"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "mask_covar_te.npz"))
    data_set["mask_et"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "mask_covar_te.npz"))
    data_set["mask_ee"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "mask_covar_ee.npz"))
    data_set["mask_bb"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "mask_covar_bb.npz"))
    data_set["beam_tt"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "beam_covar_tt.npz"))
    data_set["beam_te"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "beam_covar_te.npz"))
    data_set["beam_et"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "beam_covar_te.npz"))
    data_set["beam_ee"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "beam_covar_ee.npz"))
    data_set["beam_bb"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "beam_covar_bb.npz"))
    data_set["d_tt"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "d_tt.npz"))
    data_set["d_te"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "d_te.npz"))
    data_set["d_et"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "d_te.npz"))
    data_set["d_ee"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "d_ee.npz"))
    data_set["d_bb"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "d_bb.npz"))
    data_set["g_diagonal"] = precision_load(str(file_path_to_folder) + "g_diagonal.npz")
    data_set["cf_tt"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "cf_covar_tt.npz"))
    data_set["cf_te"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "cf_covar_te.npz"))
    data_set["cf_et"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "cf_covar_te.npz"))
    data_set["cf_ee"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "cf_covar_ee.npz"))
    data_set["cf_bb"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "cf_covar_bb.npz"))
    data_set["cn_tt"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "cn_covar_tt.npz"))
    data_set["cn_te"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "cn_covar_te.npz"))
    data_set["cn_et"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "cn_covar_te.npz"))
    data_set["cn_ee"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "cn_covar_ee.npz"))
    data_set["cn_bb"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "cn_covar_bb.npz"))
    data_set["cphi_diagonal"] = precision_load(str(file_path_to_folder) + "cphi_diagonal.npz")
    data_set["nphi_diagonal"] = precision_load(str(file_path_to_folder) + "nphi_diagonal_IP.npz")
    #read the scalar values from their text files and load into their respective keys
    # with open(file_path_to_folder + "pix_width.txt", "r") as f:
    #     pix_width = float(f.read().strip()) 
    # with open(file_path_to_folder + "num_pixels.txt", "r") as f:
    #     num_pixels = float(f.read().strip()) 
    # with open(file_path_to_folder + "theta_pix.txt", "r") as f:
    #     theta_pix = float(f.read().strip()) 
    #TODO actually implement this not hard-coded...
    data_set["pix_width"] = jnp.deg2rad(2 / ARCMIN_PER_DEGREE) #pix_width
    data_set["num_pixels"] = 256 * 256 #num_pixels
    data_set["theta_pix"] = 2 #theta_pix
    return data_set

def load_eb_data_set(file_path_to_folder):
    data_set = {}
     
    data_set["data_e"] = precision_load(str(file_path_to_folder) + "data_e_field_P.npz")
    data_set["data_b"] = precision_load(str(file_path_to_folder) + "data_b_field_P.npz")
     
    data_set["mask_ee"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "mask_covar_ee.npz"))
    data_set["mask_bb"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "mask_covar_bb.npz"))
     
    data_set["beam_ee"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "beam_covar_ee.npz"))
    data_set["beam_bb"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "beam_covar_bb.npz"))
     
    data_set["d_ee"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "d_ee.npz"))
    data_set["d_bb"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "d_bb.npz"))
    data_set["g_diagonal"] = precision_load(str(file_path_to_folder) + "g_diagonal.npz")
     
    data_set["cf_ee"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "cf_covar_ee.npz"))
    data_set["cf_bb"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "cf_covar_bb.npz"))
     
    data_set["cn_ee"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "cn_covar_ee.npz"))
    data_set["cn_bb"] = reshape_diagonal(precision_load(str(file_path_to_folder) + "cn_covar_bb.npz"))

    data_set["cphi_diagonal"] = precision_load(str(file_path_to_folder) + "cphi_diagonal.npz")
    data_set["nphi_diagonal"] = precision_load(str(file_path_to_folder) + "nphi_diagonal_IP.npz")
    #read the scalar values from their text files and load into their respective keys
    # with open(file_path_to_folder + "pix_width.txt", "r") as f:
    #     pix_width = float(f.read().strip()) 
    # with open(file_path_to_folder + "num_pixels.txt", "r") as f:
    #     num_pixels = float(f.read().strip()) 
    # with open(file_path_to_folder + "theta_pix.txt", "r") as f:
    #     theta_pix = float(f.read().strip()) 
    #TODO actually implement this not hard-coded...
    data_set["pix_width"] = jnp.deg2rad(2 / ARCMIN_PER_DEGREE) #pix_width
    data_set["num_pixels"] = 256 * 256 #num_pixels
    data_set["theta_pix"] = 2 #theta_pix
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

@jax.jit
def mixing_jacobian_phi_component_ieb(f_t_mixed, f_e_mixed, f_b_mixed, phi_mixed, 
                                      grad_f_t, grad_f_e, grad_f_b,
                                      d_tt, d_te, d_et, d_ee, d_bb,
                                      pix_width, N, theta_pix):

    #define a subfunction of just phi to apply the jax.vjp operator on
    #NOTE this must be an R^(N x N) --> R^(N x N) function in order for jax.vjp to behave properly!
    def inverse_lense(phi_mixed):
        f_t, f_e, f_b = lense_ieb(jfft.irfft2(f_t_mixed), jfft.irfft2(f_e_mixed), jfft.irfft2(f_b_mixed), \
                        (phi_mixed), pix_width, N, theta_pix, DEFAULT_NUM_LENSE_STEPS, \
                        INVERSE_LENSE, adjoint = False, rescale_and_conjugate = False)
        d_inv_tt, d_inv_te, d_inv_et, d_inv_ee = invert_block_matrix(d_tt, d_te, d_et, d_ee)
        d_inv_bb = reciprocal_matrix(d_bb)
        f_t, f_e, f_b = ieb_mat_vec_product(d_inv_tt, d_inv_te, d_inv_et, d_inv_ee, d_inv_bb, \
                                            jfft.rfft2(f_t), jfft.rfft2(f_e), jfft.rfft2(f_b))
        return jfft.irfft2(f_t), jfft.irfft2(f_e), jfft.irfft2(f_b)
    
    #call the vjp with phi in real space and subsequently apply to grad_f in real space
    _, pullback = jax.vjp(inverse_lense, jfft.irfft2(phi_mixed))
    zeros = jnp.zeros(jfft.irfft2(grad_f_t).shape)
    g1 = (jfft.irfft2(grad_f_t), zeros, zeros)
    (differential_t,) = pullback(g1)
    g2 = (zeros, jfft.irfft2(grad_f_e), zeros)
    (differential_e,) = pullback(g2)
    g3 = (zeros, zeros, jfft.irfft2(grad_f_b))
    (differential_b,) = pullback(g3)
    return jfft.rfft2(differential_t), jfft.rfft2(differential_e), jfft.rfft2(differential_b)

@jax.jit
def mixing_jacobian_phi_component_eb(f_e_mixed, f_b_mixed, phi_mixed, 
                                      grad_f_e, grad_f_b,
                                      d_ee, d_bb,
                                      pix_width, N, theta_pix):

    #define a subfunction of just phi to apply the jax.vjp operator on
    #NOTE this must be an R^(N x N) --> R^(N x N) function in order for jax.vjp to behave properly!
    def inverse_lense(phi_mixed):
        f_e, f_b = lense_eb(jfft.irfft2(f_e_mixed), jfft.irfft2(f_b_mixed), \
                        (phi_mixed), pix_width, N, theta_pix, DEFAULT_NUM_LENSE_STEPS, \
                        INVERSE_LENSE, adjoint = False, rescale_and_conjugate = False)
        d_inv_ee = reciprocal_matrix(d_ee)
        d_inv_bb = reciprocal_matrix(d_bb)
        f_e = d_inv_ee * jfft.rfft2(f_e)
        f_b = d_inv_bb * jfft.rfft2(f_b)
        return jfft.irfft2(f_e), jfft.irfft2(f_b)
    
    #call the vjp with phi in real space and subsequently apply to grad_f in real space
    _, pullback = jax.vjp(inverse_lense, jfft.irfft2(phi_mixed))
    zeros = jnp.zeros(jfft.irfft2(grad_f_e).shape)
    g0 = (jfft.irfft2(grad_f_e), zeros)
    (differential_e,) = pullback(g0)
    g1 = (zeros, jfft.irfft2(grad_f_b))
    (differential_b,) = pullback(g1)
    return jfft.rfft2(differential_e), jfft.rfft2(differential_b)

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

#@partial(jax.jit, static_argnames = ["N", "theta_pix"])
@jax.jit
def mixed_phi_gradient_ieb(f_t_mixed, f_e_mixed, f_b_mixed, phi_mixed, data_t, data_e, data_b,
                            beam_tt, beam_ee, beam_bb, beam_te, beam_et, mask_tt, mask_ee, mask_bb, mask_te, mask_et,
                            d_tt, d_te, d_et, d_ee, d_bb, g_diagonal,
                            cf_tt, cf_ee, cf_bb, cf_te, cf_et, cphi_diagonal, cn_tt, cn_ee, cn_bb, cn_te, cn_et,
                            pix_width, N, theta_pix):

    #unlense the fields
    f_t, f_e, f_b, phi = unmix_ieb(f_t_mixed, f_e_mixed, f_b_mixed, phi_mixed, \
                                   d_tt, d_te, d_et, d_ee, d_bb, g_diagonal, pix_width, \
                                   N, theta_pix)

    #compute the phi gradient in the unlensed and unmixed parametrization
    phi_covar = reshape_diagonal(cphi_diagonal)
    grad_phi = jax.grad(logpdf_ieb_unlensed_param, argnums = 14)(
                        f_t, f_e, f_b, cf_tt, cf_te, cf_ee, cf_bb,
                        data_t, data_e, data_b, cn_tt, cn_te, cn_ee, cn_bb, 
                        phi, phi_covar, mask_tt, mask_ee, 
                        mask_bb, beam_tt, beam_ee, beam_bb, 
                        N, theta_pix, pix_width)

    #compute the f gradient in the unlensed and unmixed parametrization
    grad_f_t, grad_f_e, grad_f_b = gradf_logpdf_ieb(f_t, f_e, f_b, 
                                    data_t, data_e, data_b, phi,
                                    cn_tt, cn_ee, cn_bb, cn_te, cn_et,
                                    cf_tt, cf_ee, cf_bb, cf_te, cf_et,
                                    mask_tt, mask_ee, mask_bb, mask_te, mask_et,
                                    beam_tt, beam_ee, beam_bb, beam_te, beam_et,
                                    N, theta_pix, pix_width)

    #add these together using the proper jacobian factors from the chain rule
    chain_rule_term_t, chain_rule_term_e, chain_rule_term_b = \
        mixing_jacobian_phi_component_ieb(f_t_mixed, f_e_mixed, f_b_mixed, phi_mixed, 
                                          grad_f_t, grad_f_e, grad_f_b,
                                          d_tt, d_te, d_et, d_ee, d_bb,
                                          pix_width, N, theta_pix)
    grad_phi_mixed = grad_phi + chain_rule_term_t + chain_rule_term_e + chain_rule_term_b
    return grad_phi_mixed

@jax.jit
def mixed_phi_gradient_eb(f_e_mixed, f_b_mixed, phi_mixed, data_e, data_b,
                            beam_ee, beam_bb, mask_ee, mask_bb,
                            d_ee, d_bb, g_diagonal,
                            cf_ee, cf_bb, cphi_diagonal, cn_ee, cn_bb,
                            pix_width, N, theta_pix):

    #unlense the fields
    f_e, f_b, phi = unmix_eb(f_e_mixed, f_b_mixed, phi_mixed, d_ee, 
                             d_bb, g_diagonal, pix_width, N, theta_pix)

    #compute the phi gradient in the unlensed and unmixed parametrization
    phi_covar = reshape_diagonal(cphi_diagonal)
    grad_phi = jax.grad(logpdf_eb_unlensed_param, argnums = 8)(
                        f_e, f_b, cf_ee, cf_bb,
                        data_e, data_b, cn_ee, cn_bb, 
                        phi, phi_covar, mask_ee, mask_bb, beam_ee, beam_bb, 
                        N, theta_pix, pix_width)

    #compute the f gradient in the unlensed and unmixed parametrization
    grad_f_e, grad_f_b = gradf_logpdf_eb(
                                    f_e, f_b, 
                                    data_e, data_b, phi,
                                    cn_ee, cn_bb,
                                    cf_ee, cf_bb,
                                    mask_ee, mask_bb,
                                    beam_ee, beam_bb,
                                    N, theta_pix, pix_width)

    #add these together using the proper jacobian factors from the chain rule
    chain_rule_term_e, chain_rule_term_b = \
        mixing_jacobian_phi_component_eb(f_e_mixed, f_b_mixed, phi_mixed, 
                                          grad_f_e, grad_f_b,
                                          d_ee, d_bb,
                                          pix_width, N, theta_pix)
    grad_phi_mixed = grad_phi + chain_rule_term_e + chain_rule_term_b
    return grad_phi_mixed