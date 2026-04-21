#The following lines might be necessary to point to the right active environment
using Pkg
Pkg.activate("/home/zane-blood/CMBLensing.jl/Project.toml")
Pkg.instantiate(); Pkg.resolve(); Pkg.precompile()

#Once the correct active environment is set, call CMBLensing
using CMBLensing, PythonPlot
using NPZ, LinearAlgebra, ProgressMeter, NamedTupleTools


#creating simulation data
(;f, f̃, ϕ, ds) = load_sim(
    θpix      = 2,
    Nside     = 256,
    T         = Float64,
    pol       = :IP, #change this to temperature rather than polarization for the time being
    seed = 1
    # μKarcminT = 1,
    # L         = LenseFlow(10)
    # seed      = 0, #Keep these commented out for the time being....
    # pixel_mask_kwargs = (edge_padding_deg=1, apodization_deg=0, num_ptsrcs=0),
    # bandpass_mask     = LowPass(5000)
)

#write the necessary Julia data to storage in a folder in the project directory
cwd = pwd()
file_path = cwd * "/julia_maximization_debug/"

#store all the necesary arrays and fields
# println("writing simulation data to storage...\n")
# npzwrite(file_path * "cf_diagonal.npz", ds.Cf.op.diag)
# npzwrite(file_path * "cf_lensed_diagonal.npz", ds.Cf̃.diag)
# npzwrite(file_path * "cphi_diagonal.npz", ds.Cϕ.op.diag)
# npzwrite(file_path * "nphi_diagonal.npz", ds.Nϕ.diag)
# npzwrite(file_path * "cn_diagonal.npz", ds.Cn̂.diag)
# npzwrite(file_path * "m_diagonal.npz", ds.M̂.diag.arr)
# npzwrite(file_path * "b_diagonal.npz", ds.B̂.diag.arr)
# npzwrite(file_path * "f_array.npz", f.arr)
# npzwrite(file_path * "f_map_array.npz", Map(f).arr)
# npzwrite(file_path * "phi_array.npz", ϕ.arr)
# npzwrite(file_path * "phi_map_array.npz", Map(ϕ).arr)
# npzwrite(file_path * "f_lensed_array.npz", ds.M̂.diag.arr .* ds.B̂.diag.arr .* f̃[:Il])
# npzwrite(file_path * "f_lensed_no_b_no_m_array.npz", f̃[:Il])
# npzwrite(file_path * "f_lensed_map_array.npz", Map(f̃).arr)
# npzwrite(file_path * "data_array.npz", ds.d.arr)
# npzwrite(file_path * "f_lambda_array.npz", f.λ_rfft)
# npzwrite(file_path * "phi_lambda_array.npz", ϕ.λ_rfft)
# npzwrite(file_path * "data_lambda_array.npz", ds.d.λ_rfft)
# npzwrite(file_path * "cf_lambda_array.npz", ds.Cf.op.diag.λ_rfft)
# npzwrite(file_path * "d_lambda_array.npz", ds.D.op.diag.λ_rfft)
# npzwrite(file_path * "cphi_lambda_array.npz", ds.Cϕ.op.diag.λ_rfft)
# npzwrite(file_path * "cn_lambda_array.npz", ds.Cn.diag.λ_rfft)
# npzwrite(file_path * "m_lambda_array.npz", ds.M̂.diag.λ_rfft)
# npzwrite(file_path * "b_lambda_array.npz", ds.B̂.diag.λ_rfft)
# npzwrite(file_path * "cf_lensed_lambda_array.npz", ds.Cf̃.diag.λ_rfft)
# npzwrite(file_path * "f_lensed_lambda_array.npz", f̃.λ_rfft)

# #matrices needed for the mixed parametrization
# npzwrite(file_path * "d_diagonal.npz", ds.D.op.diag)
# npzwrite(file_path * "g_diagonal.npz", ds.G.op.diag)

# # #the mixed parametrization ground truth
# # (;f°, ϕ°) = mix(ds; f, ϕ)
# # npzwrite(file_path * "f_mixed_array.npz", f°.arr)
# # npzwrite(file_path * "phi_mixed_array.npz", ϕ°.arr)

# #Store the pre-conditioner as well...
# Hess_preconditioner = CMBLensing.Hessian_logpdf_preconditioner(:f, ds).diag
# npzwrite(file_path * "preconditioner.npz", Hess_preconditioner)

# L = ds.L
# Lϕ = L(ϕ)
# npzwrite(file_path * "f_adjoint_lensed_array.npz", (Lϕ'*f))

# #store the value of the logpdf function, pixel width, and total number of pixels
# # open(file_path * "logpdf_value.txt", "w") do io
# #     println(io, logpdf(ds; f, ϕ))
# # end
# open(file_path * "pix_width.txt", "w") do io
#     pix_width = ds.d.metadata.Δx
#     println(io, pix_width)
# end
# open(file_path * "num_pixels.txt", "w") do io
#     Nx = ds.d.metadata.Nx
#     Ny = ds.d.metadata.Ny
#     num_pixels = Nx * Ny
#     println(io, num_pixels)
# end

## wiener filter
function argmaxf_logpdf(
    ds :: CMBLensing.DataSet,
    Ω :: CMBLensing.NamedTuple, 
    d = ds.d;
    fstart = nothing, 
    preconditioner = :diag, 
    conjgrad_kwargs = (tol=1e-1,nsteps=500),
    offset = false,
)
    
    Hess_preconditioner = CMBLensing.Hessian_logpdf_preconditioner(:f, ds)
    zero_f = zero(diag(ds.Cf))

    # the following will give the argmax for any model with Gaussian P(f,d|z...)
    b  = -CMBLensing.gradientf_logpdf(ds; f=zero_f, d=d,       Ω...)
    # if isfile("/home/zane-blood/Desktop/cmb_lensing/b_array.npz") == false
    #     npzwrite("/home/zane-blood/Desktop/cmb_lensing/b_array.npz", b.arr)
    # end
    a₀ =  CMBLensing.gradientf_logpdf(ds; f=zero_f, d=zero(d), Ω...)
    offset && (b += a₀)
    Hess = FuncOp(f -> (CMBLensing.gradientf_logpdf(ds; f, d=zero(d), Ω...) - a₀))
    conjugate_gradient(Hess_preconditioner, Hess, b, (isnothing(fstart) ? zero_f : fstart); conjgrad_kwargs...)

end

@⌛ function conjugate_gradient(
    M,
    A,
    b,
    x = zero(b);
    nsteps       = length(b),
    tol          = sqrt(eps()),
    progress     = false,
    callback     = nothing,
    history_keys = nothing,
    history_mod  = 1
)
    get_history() = isnothing(history_keys) ? nothing : select((;i,x,p,r,res,t),history_keys)

    # debug_file_path = "/home/zane-blood/Desktop/cmb_lensing/cg_debug.txt"
    # #debugging print statements to a text_file.. only debug on the first call of the wf 
    # open(debug_file_path, "w") do io
    #     write(io, "begin debug...\n")
    # end

    t₀ = time()
    T = real(eltype(x)) # allow `dot` to return a higher precision but keep vector its original eltype
    i = 1
    r = b - A*x
    # if isfile("/home/zane-blood/Desktop/cmb_lensing/b_array.npz") == false
    #     npzwrite("/home/zane-blood/Desktop/cmb_lensing/r_array.npz", r.arr)
    # end
    
    # if x == zero(b)
    #     open(debug_file_path, "a") do io
    #         write(io, "size of x = " * string(size(x.arr)) * "\n")
    #         # write(io, "x printed out looks like: \n")
    #         # println(io, x)
    #         # write(io, "x.arr printed out looks like: \n")
    #         # println(io, x.arr)
    #         write(io, "size of b = " * string(size(b.arr)) * "\n")
    #         # write(io, "b printed out looks like: \n")
    #         # println(io, b)
    #         # write(io, "b.arr printed out looks like: \n")
    #         # println(io, b.arr)
    #         write(io, "size of r = " * string(size(r.arr)) * "\n")
    #         # write(io, "r printed out looks like: \n")
    #         # println(io, r)
    #         # write(io, "r.arr printed out looks like: \n")
    #         # println(io, r.arr)
    #         write(io, "size of M = " * string(size(M.diag)) * "\n")
    #         # write(io, "M printed out looks like: \n")
    #         # println(io, M)
    #         # write(io, "M.diag printed out looks like: \n")
    #         # println(io, M.diag)
    #     end
    #     if isfile("/home/zane-blood/Desktop/cmb_lensing/M_array.npz") == false
    #         npzwrite("/home/zane-blood/Desktop/cmb_lensing/M_array.npz", M.diag)
    #     end
    # end

    z = M \ r

    # if isfile("/home/zane-blood/Desktop/cmb_lensing/z_array.npz") == false
    #     npzwrite("/home/zane-blood/Desktop/cmb_lensing/z_array.npz", z.arr)
    # end

    p = z
    bestres = res = res₀ = T(dot(r,z))
    @assert !isnan(res)
    bestx = x
    t    = time() - t₀
    history = [get_history()]

    # if isfile("/home/zane-blood/Desktop/cmb_lensing/res.txt") == false
    #     open("/home/zane-blood/Desktop/cmb_lensing/res.txt", "w") do io
    #         write(io, "res = " * string(res) * "\n")
    #     end
    # end

    # open("/home/zane-blood/Desktop/cmb_lensing/x_array.txt", "w") do io
    #         write(io, string(x.arr) * "\n")
    # end
    prog = Progress(100, (progress!=false ? progress : Inf), "Conjugate Gradient: ")
    for outer i = 2:nsteps
        Ap   = @⌛ A * p
        α    = res / dot(p,Ap)
        x    = x + T(α) * p
        r    = r - T(α) * Ap
        z    = M \ r
        res′ = T(dot(r,z))
        p    = z + (res′ / res) * p
        res  = res′
        t    = time() - t₀
        # open("/home/zane-blood/Desktop/cmb_lensing/res.txt", "a") do io
        #     write(io, "res = " * string(res) * "\n")
        # end
        # if i % 10 == 0
        #     npzwrite("/home/zane-blood/Desktop/cmb_lensing/x_array_" * string(i) * ".npz", x.arr)
        # end
        
        if all(res<bestres)
            bestres,bestx = res,x
            # open(debug_file_path, "a") do io
            #     write(io, "Hit If-1 @ index =" * string(i) * "\n")
            # end
        end
        if !isnothing(callback)
            # open(debug_file_path, "a") do io
            #     write(io, "Hit If-2 @ index =" * string(i) * "\n")
            # end
            callback(i,x,res)
        end
        if (i%history_mod) == 0
            push!(history, get_history())
            # open(debug_file_path, "a") do io
            #     write(io, "Hit If-3 @ index =" * string(i) * "\n")
            # end
        end
        if all(res<tol)
            # open(debug_file_path, "a") do io
            #     write(io, "Hit If-4 Break @ index =" * string(i) * "\n")
            # end
            break
        end
        
        # update progress bar to whichever we've made the most progress on,
        # logarithmically reaching the toleranace limit or doing the maximum
        # number of steps
        if progress
            progress_nsteps = round(Int,100*(i-1)/(nsteps-1))
            progress_tol = res isa AbstractFloat ? round(Int,100^min(1, (log10(res/res₀)) / log10(tol/res₀))) : 0
            ProgressMeter.update!(prog, max(progress_nsteps,progress_tol))
        end
    end
    ProgressMeter.finish!(prog)
    (bestx, history)
end

function sample_f(rng::CMBLensing.AbstractRNG, ds::CMBLensing.DataSet, Ω, d=ds.d; kwargs...)
    # the following will give a sample for any model with Gaussian P(f,d|z...)
    sim = simulate(rng, ds; Ω...)
    Δf, history = argmaxf_logpdf(ds, Ω, d - sim.d; kwargs..., offset=true)
    sim.f + Δf, history
end
sample_f(ds::CMBLensing.DataSet, args...; kwargs...) = sample_f(Random.default_rng(), ds, args...; kwargs...)

# allows specific DataSets to override this as a performance
# optimization, since Zygote is ~50% slower than the old hand-written
# code even after the above hack. shouldn't need this once we have
# Diffractor. the following is the fallback which just uses Zygote:
gradientf_logpdf(ds::CMBLensing.DataSet; f, Ω...) = gradient(f -> logpdf(ds; f, Ω...), f)[1]

MAP_joint(ds::CMBLensing.DataSet, args...; kwargs...) = MAP_joint((;), ds, args...; kwargs...)
function MAP_joint(
    θ, 
    ds :: CMBLensing.DataSet,
    Ωstart = FieldTuple(ϕ=Map(zero(diag(ds.Cϕ))));
    nsteps = 20,
    minsteps = 0,
    fstart = nothing,
    αtol = 1e-4,
    gradtol = 0,
    αmax = nothing,
    prior_deprojection_factor = 0,
    nburnin_update_hessian = Inf,
    progress::Bool = true,
    conjgrad_kwargs = (tol=1e-1, nsteps=500),
    quasi_sample = false,
    history_keys = (:logpdf,),
    aggressive_gc = false,
)

    if isfinite(nburnin_update_hessian)
        keys((;Ωstart...,)) == (:ϕ,) || error("nburnin_update_hessian only implemented for (f,ϕ)-only maximization.")
    end

    sample_or_argmax_f = 
        quasi_sample == false ? argmaxf_logpdf :
        quasi_sample == true ? sample_f : 
        quasi_sample isa AbstractRNG ? (args...; kwargs...) -> sample_f(copy(quasi_sample), args...; kwargs...) : 
        error("`quasi_sample` should be true, false, or an AbstractRNG")

    dsθ = copy(ds(θ))
    dsθ.G = I # MAP estimate is invariant to G so avoid wasted computation

    
    history = []
    pbar = Progress(nsteps, (progress ? 0 : Inf), "MAP_joint: ")
    ProgressMeter.update!(pbar)
    
    prevΩ = prevΩ° = prev_∇Ω°_logpdf = HΩ° = showvalues = nothing
    Ω = Ωstart
    f = prevf = fstart
    α = 1
    αmax_initial = αmax
    t_f_total = t_ϕ_total = 0

    for step = 1:nsteps

        ## f step
        t_f = @elapsed begin
            (f, argmaxf_logpdf_history) = @⌛ argmaxf_logpdf( #try changing this to just argmaxf_logpdf?
                dsθ, 
                (;Ω..., θ);
                fstart = prevf, 
                conjgrad_kwargs = (history_keys=(:i,:res), progress=false, conjgrad_kwargs...)
            )
            aggressive_gc && cuda_gc()

            # #DEBUG
            # cwd = pwd()
            # file_path = cwd * "/julia_maximization_debug/"
            # npzwrite(file_path * "f_ground_step_" * string(step) * ".npz", f.arr)
            # #DEBUG

        end

        # #DEBUG try looking at value of wiener filter here...
        # "/home/zane-blood/Desktop/cmb_lensing/res.txt"
        # npzwrite("/home/zane-blood/Desktop/cmb_lensing/f_wf_ground.npz", f.arr)

        # gradient
        t_ϕ = @elapsed begin
            ## ϕ step
            @unpack f° = (Ω° = mix(dsθ; f, Ω..., θ))
            Ω° = FieldTuple(delete(Ω°, (:f°, :θ)))

            #DEBUG
            #cwd = pwd()
            #file_path = cwd * "/julia_maximization_debug/"
            # npzwrite(file_path * "f_map_joint_mixed.npz", f°.arr)
            #npzwrite(file_path * "phi_map_joint_mixed_" * string(step) * ".npz", Ω°.ϕ°.arr)
            #DEBUG

            #maybe try creating a similar line without using a "NamedTuple but just
            #phi_mixed in both julia and python and see if the two match...
            ∇Ω°_logpdf, = @⌛ gradient(Ω°->logpdf(Mixed(dsθ); f°, Ω°..., θ), Ω°)

            ϕ° = Ω°.ϕ°
            grad_phi_test1 = @⌛ gradient(ϕ°->logpdf(Mixed(dsθ); f°, ϕ°), ϕ°)[1]

            ϕ = Ω.ϕ
            grad_phi_test2 = @⌛ gradient(ϕ->logpdf(dsθ; f, ϕ), ϕ)[1]

            #DEBUG
            cwd = pwd()
            file_path = "/home/zane-blood/Desktop/cmb_lensing/cmb_lensing/julia_maximization_debug/"
            npzwrite(file_path * "grad_phi_mixed_ground.npz", ∇Ω°_logpdf.ϕ°.arr)
            #DEBUG

            # Hessian
            if step > nburnin_update_hessian
                HΩ°⁻¹_unsmooth = Diagonal(abs.(Fourier(Ω°.ϕ° - prevΩ°.ϕ°) ./ Fourier(∇Ω°_logpdf.ϕ° - prev_∇Ω°_logpdf.ϕ°)))
                HΩ°⁻¹_smooth = Cℓ_to_Cov(:I, f.proj, smooth(ℓ⁴*cov_to_Cℓ(HΩ°⁻¹_unsmooth), xscale=:log, yscale=:log, smoothing=0.05)/ℓ⁴)
                HΩ° = Diagonal(FieldTuple(ϕ°=diag(pinv(HΩ°⁻¹_smooth))))
            elseif HΩ° == nothing
                HΩ° = CMBLensing.Hessian_logpdf_preconditioner(keys((;Ω°...,)), dsθ)
            end
            # line search
            ΔΩ° = pinv(HΩ°) * ∇Ω°_logpdf

            #DEBUG
            # cwd = pwd()
            # file_path = cwd * "/julia_maximization_debug/"
            # open(file_path * "H_omega_field_names.txt", "w") do io
            #     write(io, string(fieldnames(typeof(HΩ°))) * "\n")
            # end

            #npzwrite(file_path * "delta_omega_" * string(step) * ".npz", ΔΩ°.ϕ°.arr)
        
            # npzwrite(file_path * "H_Omega.npz", HΩ°.diag.ϕ°.arr)
            #DEBUG

            T = real(eltype(f))
            if prior_deprojection_factor != 0
                ΔΩ°_perp = pinv(HΩ°) * gradient(ΔΩ° -> logprior(dsθ; unmix(dsθ; f°, ΔΩ°...)...), ΔΩ°)[1]
                ΔΩ° .-= T(prior_deprojection_factor * dot(ΔΩ°,ΔΩ°_perp) * pinv(dot(ΔΩ°_perp,ΔΩ°_perp))) .* ΔΩ°_perp
            end
            αmax = @something(αmax_initial, 2α)
            soln = CMBLensing.@ondemand(Optim.optimize)(T(0), T(αmax), CMBLensing.@ondemand(Optim.Brent)(); abs_tol=T(αtol)) do α
                Ω°′ = Ω° + T(α) * ΔΩ°
                total_logpdf = @⌛(sum(unbatch(-(logpdf(Mixed(dsθ); f°, Ω°′..., θ)))))
                isnan(total_logpdf) ? T(α/αmax) * prevfloat(T(Inf)) : total_logpdf # workaround for https://github.com/JuliaNLSolvers/Optim.jl/issues/828
            end
            α = T(soln.minimizer)
            Ω° += α * ΔΩ°
            # open(file_path * "alpha_values.txt", "a") do io
            #     write(io, string(α) * "\n")
            # end
            # cwd = pwd()
            # file_path = cwd * "/julia_maximization_debug/"
            # npzwrite(file_path * "phi_ground_step_" * string(step) * ".npz", Ω°.ϕ°.arr)
            # open(file_path * "H_omega_field_names.txt", "w") do io
            #     write(io, string(fieldnames(typeof(HΩ°))) * "\n")
            # end
            # npzwrite(file_path * "delta_omega.npz", ΔΩ°.ϕ°.arr)
        end
        
        ## finalize
        _logpdf = @⌛ logpdf(Mixed(dsθ); f°, Ω°..., θ)
        Ω = delete(unmix(dsθ; f°, Ω°..., θ), (:f, :θ))
        ΔΩ°_norm = norm(ΔΩ°)
        total_logpdf = sum(unbatch(_logpdf))
        showvalues = [
            ("step",       step), 
            ("logpdf",     join(map(x->CMBLensing.@sprintf("%.2f",x), [unbatch(_logpdf)...]), ", ")),
            ("α",          α),
            ("ΔΩ°_norm",   CMBLensing.@sprintf("%.2g", ΔΩ°_norm)),
            ("CG",         "$(length(argmaxf_logpdf_history)) iterations ($(CMBLensing.@sprintf("%.2f",t_f)) sec)"), 
            ("Linesearch", "$(soln.iterations) bisections ($(CMBLensing.@sprintf("%.2f",t_ϕ)) sec)")
        ]
        next!(pbar; showvalues)
        push!(history, select((;f°,f,Ω°...,Ω...,∇Ω°_logpdf,total_logpdf,α,αmax,ΔΩ°,ΔΩ°_norm,logpdf=_logpdf,HΩ°,argmaxf_logpdf_history), history_keys))
        
        # early stop based on tolerance
        if (step > minsteps) && (norm(ΔΩ°) < gradtol)
            break
        end
        prevf, prevΩ, prevΩ°, prev_∇Ω°_logpdf = f, Ω, Ω°, ∇Ω°_logpdf

    end

    ProgressMeter.finish!(pbar)
    ProgressMeter.updateProgress!(pbar; showvalues)
    (;f, Ω..., history)

end

fJ, ϕJ, history = MAP_joint(ds, nsteps=1, progress=true);
# f_wf_at_f0_phi, = argmaxf_logpdf(ds, (;ϕ); fstart = zero(f), conjgrad_kwargs = (tol = 1e-1, nsteps = 500));
# cwd = pwd()
# file_path = cwd * "/julia_maximization_debug/"
# npzwrite(file_path * "phi_ground_final.npz", ϕJ.arr)
# npzwrite(file_path * "f_ground_final.npz", fJ.arr)

# open(file_path * "logpdf_values.txt", "a") do io
#     for idx in 1:length(history)
#         write(io, string(history[idx][1]) * "\n")
#     end
# end