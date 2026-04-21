# using CMBLensing
# using NPZ
# using Zygote

using Pkg; Pkg.activate("/home/zane-blood/CMBLensing.jl/Project.toml")
using NPZ
using CMBLensing

#run a simulation using pre-existing julia code
println("\nrunning simulation...\n")
(;f, f̃, ϕ, ds) = load_sim(
    θpix      = 2,
    Nside     = 256,
    T         = Float64,
    pol       = :I, #change this to temperature rather than polarization for the time being
    μKarcminT = 1,
    L         = LenseFlow(10),
    seed = 1
    # seed      = 0, #Keep these commented out for the time being....
    # pixel_mask_kwargs = (edge_padding_deg=1, apodization_deg=0, num_ptsrcs=0),
    # bandpass_mask     = LowPass(5000)
)

#write the necessary Julia data to storage in a folder in the project directory
cwd = pwd() * "/cmb_lensing/"
file_path = cwd * "/julia_ground_truth_data/"

#store all the necesary arrays and fields
println("writing simulation data to storage...\n")
npzwrite(file_path * "cf_diagonal.npz", (ds.Cf.op.diag))
npzwrite(file_path * "cf_lensed_diagonal.npz", (ds.Cf̃.diag))
npzwrite(file_path * "cphi_diagonal.npz", (ds.Cϕ.op.diag))
npzwrite(file_path * "cn_diagonal.npz", (ds.Cn̂.diag))
npzwrite(file_path * "m_diagonal.npz", transpose(ds.M̂.diag.arr))
npzwrite(file_path * "b_diagonal.npz", transpose(ds.B̂.diag.arr))
npzwrite(file_path * "f_array.npz", transpose(f.arr))
npzwrite(file_path * "f_map_array.npz", transpose(Map(f).arr))
npzwrite(file_path * "phi_array.npz", transpose(ϕ.arr))
npzwrite(file_path * "phi_map_array.npz", transpose(Map(ϕ).arr))
npzwrite(file_path * "f_lensed_array.npz",transpose( ds.M̂.diag.arr .* ds.B̂.diag.arr .* f̃[:Il]))
npzwrite(file_path * "f_lensed_no_b_no_m_array.npz", transpose(f̃[:Il]))
npzwrite(file_path * "f_lensed_map_array.npz", transpose(Map(f̃).arr))
npzwrite(file_path * "data_array.npz", transpose(ds.d.arr))
npzwrite(file_path * "f_lambda_array.npz", (f.λ_rfft))
npzwrite(file_path * "phi_lambda_array.npz", (ϕ.λ_rfft))
npzwrite(file_path * "data_lambda_array.npz", (ds.d.λ_rfft))
npzwrite(file_path * "cf_lambda_array.npz", (ds.Cf.op.diag.λ_rfft))
npzwrite(file_path * "cphi_lambda_array.npz", (ds.Cϕ.op.diag.λ_rfft))
npzwrite(file_path * "cn_lambda_array.npz", (ds.Cn.diag.λ_rfft))
npzwrite(file_path * "m_lambda_array.npz", (ds.M̂.diag.λ_rfft))
npzwrite(file_path * "b_lambda_array.npz", (ds.B̂.diag.λ_rfft))
npzwrite(file_path * "cf_lensed_lambda_array.npz", (ds.Cf̃.diag.λ_rfft))
npzwrite(file_path * "f_lensed_lambda_array.npz", (f̃.λ_rfft))

#matrices needed for the mixed parametrization
npzwrite(file_path * "d_diagonal.npz", (ds.D.op.diag))
npzwrite(file_path * "g_diagonal.npz", (ds.G.op.diag))

#the mixed parametrization ground truth
(;f°, ϕ°) = mix(ds; f, ϕ)
npzwrite(file_path * "f_mixed_array.npz", transpose(f°.arr))
npzwrite(file_path * "phi_mixed_array.npz", transpose(ϕ°.arr))

#Store the pre-conditioner as well...
Hess_preconditioner = CMBLensing.Hessian_logpdf_preconditioner(:f, ds).diag
npzwrite(file_path * "preconditioner.npz", (Hess_preconditioner))

L = ds.L
Lϕ = L(ϕ)
npzwrite(file_path * "f_adjoint_lensed_array.npz", transpose((Lϕ'*f).arr))

#store the value of the logpdf function, pixel width, and total number of pixels
open(file_path * "logpdf_value.txt", "w") do io
    println(io, logpdf(ds; f, ϕ))
end
open(file_path * "pix_width.txt", "w") do io
    pix_width = ds.d.metadata.Δx
    println(io, pix_width)
end
open(file_path * "num_pixels.txt", "w") do io
    Nx = ds.d.metadata.Nx
    Ny = ds.d.metadata.Ny
    num_pixels = Nx * Ny
    println(io, num_pixels)
end

#compute the wiener filter 
println("calculating wiener filters...\n")
zero_f = 0 .* f 
phi_init = ϕ #store the current value of phi

f_wf_at_f_phi, = argmaxf_logpdf(ds, (;ϕ); fstart = f, conjgrad_kwargs = (tol = 1e-1, nsteps = 500));
f_wf_at_f0_phi, = argmaxf_logpdf(ds, (;ϕ); fstart = zero_f, conjgrad_kwargs = (tol = 1e-1, nsteps = 500));

ϕ = 0 .* ϕ #momentarily set phi to zero

f_wf_at_f_phi0, = argmaxf_logpdf(ds, (;ϕ); fstart = f, conjgrad_kwargs = (tol = 1e-1, nsteps = 500)); 
f_wf_at_f0_phi0, = argmaxf_logpdf(ds, (;ϕ); fstart = zero_f, conjgrad_kwargs = (tol = 1e-1, nsteps = 500)); 

#set phi back to what it originally was...
ϕ = phi_init

npzwrite(file_path * "wiener_filter_at_f0_phi0.npz", transpose(f_wf_at_f0_phi0.arr))
npzwrite(file_path * "wiener_filter_at_f0_phi.npz", transpose(f_wf_at_f0_phi.arr))
npzwrite(file_path * "wiener_filter_at_f_phi0.npz", transpose(f_wf_at_f_phi0.arr))
npzwrite(file_path * "wiener_filter_at_f_phi.npz", transpose(f_wf_at_f_phi.arr))

#Computing logpdf gradient ground truths
println("computing logpdf gradients...\n")
g_logpdf_wrt_ϕ = gradient(ϕ -> logpdf(ds; f, ϕ), ϕ)[1];
g_logpdf_wrt_f = gradient(f -> logpdf(ds; f, ϕ), f)[1];

#specifically computing grad_phi_logpdf at f = true f & phi = gaussian...
#start by creating the gaussian lensing potential
pix_width = ds.d.metadata.Δx
num_rows = num_cols = 256
decay_rate = 500
magnitude = 0.125*1e-4
gaussian_data = zeros(Float64, 256, 256)
for row_idx = 1:num_rows
    for col_idx = 1:num_cols
        gaussian_data[row_idx, col_idx] = magnitude*exp(-decay_rate*(pix_width^2*((row_idx-num_rows/2)^2+(col_idx-num_cols/2)^2)))
    end
end
gaussian_phi = FlatMap(gaussian_data, θpix=2)
npzwrite(file_path * "gaussian_phi.npz", transpose(gaussian_phi.arr))

#then take gradient in unlensed / unmixed parametrization
phi_init = ϕ #use temp variables to not screw up the kernel
ϕ = gaussian_phi
grad_phi_gauss_unmixed = gradient(ϕ -> logpdf(ds; f, ϕ), ϕ)[1];
npzwrite(file_path * "grad_phi_gauss_unmixed.npz", transpose(grad_phi_gauss_unmixed.arr))
ϕ = phi_init

#Compute b vector, A matrix ground truths as well
b_vector = -1*gradient(f -> logpdf(ds; f, ϕ), zero(f))[1]; #Negative 1 is crucial here to avoid exploding solutions
Hess = FuncOp(f -> (CMBLensing.gradientf_logpdf(ds; f, ϕ, d=zero(ds.d))))
A_matrix = Hess * f

npzwrite(file_path * "grad_phi.npz", transpose(g_logpdf_wrt_ϕ.arr))
npzwrite(file_path * "grad_f.npz", transpose(g_logpdf_wrt_f.arr))
npzwrite(file_path * "beta_vector_ground.npz", transpose(b_vector.arr))
npzwrite(file_path * "alpha_matrix_ground.npz", transpose(A_matrix.arr))

#computing the negδvelocityᴴ integration results
println("computing lensing operator gradients...\n")
L = ds.L
Lϕ = L(ϕ)

#gradients of forward lensing operator
Lϕ_fwd = precompute!!(Lϕ, f)
f̃ = Lϕ_fwd * f
Δ = CMBLensing.pinv(Diagonal(ds.Cf.op.diag)) * f #use this as an arbitrary δf initial
Lϕ_back = precompute!!(Lϕ_fwd, Δ)
(_, δf, δϕ) = @⌛ Lϕ.odesolve(CMBLensing.negδvelocityᴴ(Lϕ_back, FieldTuple(f, Δ))..., Lϕ.t₁ => Lϕ.t₀)
npzwrite(file_path * "delta_f_forward_array.npz", transpose(δf.arr))
npzwrite(file_path * "delta_phi_forward_array.npz", transpose(δϕ.arr))

#gradients of inverse lensing operator
Lϕ_fwd = precompute!!(Lϕ, f)
f = Lϕ_fwd \ f̃
Δ = CMBLensing.pinv(Diagonal(ds.Cf.op.diag)) * f̃ #use this as an arbitrary δf̃ initial
Lϕ_back = precompute!!(Lϕ_fwd, Δ)
(_, δf, δϕ) = @⌛ Lϕ.odesolve(CMBLensing.negδvelocityᴴ(Lϕ_back, FieldTuple(f, Δ))..., Lϕ.t₀ => Lϕ.t₁)
npzwrite(file_path * "delta_f_inverse_array.npz", transpose(δf.arr))
npzwrite(file_path * "delta_phi_inverse_array.npz", transpose(δϕ.arr))
println("done generating data...\n")
