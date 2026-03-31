import jax.numpy as jnp
PID_CONTROLLER_RTOL = 1e-5 #relative tolerance for Diffrax integrator
PID_CONTROLLER_ATOL = 1e-5 #absolute tolerance for Diffrax integrator
DEFAULT_NUM_LENSE_STEPS = 10 #default number of lensing steps for LenseFlow
INVERSE_LENSE = -1 #flag for inverse lensing is represented by negative 1
FORWARD_LENSE = 1 #flag for forward lensing
ARCMIN_PER_DEGREE = 60 #arcminute to degree conversion factor
BEAM_TRANSFER_SCALAR = 8*jnp.log(2)