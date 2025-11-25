
using Plots, Plots.Measures, Printf, LoopVectorization, BenchmarkTools, CUDA
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)

#handle packages
const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=false)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=false)
end
using Plots, Plots.Measures, Printf



#macro d_xa(A) esc(:( $A[ix+1, iy]-$A[ix,iy]  )) end
#macro d_ya(A) esc(:( $A[ix, iy+1]-$A[ix,iy]  )) end



@parallel function update_Pf!(Pf, qDx, qDy, _β_dτ_dx, _β_dτ_dy)
    @all(Pf) = @all(Pf) - @d_xa(qDx) * _β_dτ_dx  +  @d_ya(qDy) * _β_dτ_dy
    return nothing

end

@parallel function compute_flux!(qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)

    @inn_x(qDx) = @inn_x(qDx) - (@inn_x(qDx) + k_ηf_dx * @d_xa(Pf)) * _1_θ_dτ
    @inn_y(qDy) = @inn_y(qDy) - (@inn_y(qDy) + k_ηf_dy * @d_ya(Pf)) * _1_θ_dτ

    return nothing
end


function Pf_diffusion_2D(do_check = true)
    # physics
    lx, ly = 20.0, 20.0
    k_ηf   = 1.0
    # numerics
    nx, ny  = 16*32, 16*32 
    ϵtol    = 1e-8
    ncheck  = ceil(Int, 0.25max(nx, ny))
    cfl     = 1.0 / sqrt(2.1)
    re      = 2π
    maxiter = 500

    # derived numerics
    dx, dy  = lx / nx, ly / ny
    xc, yc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)#??
    θ_dτ    = max(lx, ly) / re / cfl / min(dx, dy)
    β_dτ    = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))

    #array initialization
    Pf      = Data.Array(@. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2))
    qDx     = @zeros(nx + 1, ny    )
    qDy     = @zeros(nx    , ny + 1)
    r_Pf    = @zeros(nx    , ny    )
        
    #precompute divisions
    k_ηf_dx, k_ηf_dy = k_ηf/dx, k_ηf/dy
    _1_θ_dτ = 1.0./(1.0 + θ_dτ)
    _β_dτ_dx = 1.0/(β_dτ*dx)
    _β_dτ_dy = 1.0/(β_dτ*dy)

    # iteration loop
    t_tic = 0.0
    iter = 1; err_Pf = 2ϵtol
    while err_Pf >= ϵtol && iter <= maxiter && do_check
        #time measure after "runup"
        if iter == 11
            t_tic = Base.time()
        end

        @parallel compute_flux!(qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
        @parallel update_Pf!(Pf, qDx, qDy, _β_dτ_dx, _β_dτ_dy)
       
        if do_check && iter % ncheck == 0
            r_Pf .= diff(qDx, dims=1) ./ dx .+ diff(qDy, dims=2) ./ dy
            err_Pf = maximum(abs.(r_Pf))
            @printf("  iter/nx=%.1f, err_Pf=%1.3e\n", iter / nx, err_Pf)
            display(heatmap(xc, yc, Array(Pf)'; xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), aspect_ratio=1, c=:turbo, clim=(0, 1)))
        end
        iter += 1; niter += 1
        
    end
    t_it = 0.0
    niter = iter - 11

    if !do_check
        if useThreads
            #warm up phase
            compute_t!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _β_dτ_dx, _β_dτ_dy)
            #directly compute t_it with benchmarktools
            t_it = @elapsed begin
                compute_t!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _β_dτ_dx, _β_dτ_dy) 
            end
        else
            #warm up phase
            compute!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _β_dτ_dx, _β_dτ_dy, threads, blocks)
            #directly compute t_it with benchmarktools
            t_it = @elapsed begin
                compute!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _β_dτ_dx, _β_dτ_dy, threads, blocks) 
            end
        end
    else
        t_it = (Base.time() - t_tic)/niter
    end  
        
    Aeff = 2*3 * nx*ny * 8 / 1e9 #2 = read and write, 3 = amount of arrays, nx"ny gridsize, *8/1e9 convert to Gb
    Teff = Aeff / t_it
    @printf("Time = %1.3f sec, Aeff = %1.3f GB, Teff = %1.3f GB/s \n", t_it * niter, Aeff, Teff)
    return Teff, Array(Pf)
end




#run
Teff, _ = Pf_diffusion_2D()