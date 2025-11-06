
using Plots, Plots.Measures, Printf, LoopVectorization, BenchmarkTools, CUDA
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)


macro d_xa(A) esc(:( $A[ix+1, iy]-$A[ix,iy]  )) end
macro d_ya(A) esc(:( $A[ix, iy+1]-$A[ix,iy]  )) end


    #CPU Functions =#
function compute_flux_t!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
    @tturbo for iy=1:ny
        for ix= 1:nx-1
            @inbounds qDx[ix+1, iy] -= (qDx[ix+1, iy] + k_ηf_dx *@d_xa(Pf)) * _1_θ_dτ
        end
    end
        
    @tturbo for iy=1:ny-1
        for ix= 1:nx
            @inbounds qDy[ix, iy+1] -= (qDy[ix, iy+1] + k_ηf_dy *@d_ya(Pf)) * _1_θ_dτ
        end
    end
    return nothing
end

function update_Pf_t!(nx, ny, Pf, qDx, qDy, _β_dτ_dx, _β_dτ_dy)
    @tturbo for iy=1:ny
        for ix=1:nx
            @inbounds Pf[ix, iy] -= (@d_xa(qDx)) * _β_dτ_dx  +  (@d_ya(qDy)) * _β_dτ_dy
        end
    end

    return nothing
end


    #= GPU Kernels =#
function update_Pf!(nx, ny, Pf, qDx, qDy, _β_dτ_dx, _β_dτ_dy)

    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if ix >= 1 && iy >=1 && ix <= nx && iy <= ny
        
        @inbounds begin
            Pf[ix, iy] -= (qDx[ix+1, iy] - qDx[ix, iy]) * _β_dτ_dx  +  (qDy[ix, iy+1] - qDy[ix, iy]) * _β_dτ_dy
        end
    end

    return nothing
end

function compute_flux!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)

    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if iy >=1 && iy <= ny && ix >=1 && ix <= nx-1

        @inbounds qDx[ix+1, iy] -= (qDx[ix+1, iy] + k_ηf_dx *(Pf[ix+1, iy] - Pf[ix, iy])) * _1_θ_dτ 
    end

    if iy >=1 && iy <= ny-1 && ix >=1 && ix <= nx  
        @inbounds qDy[ix, iy+1] -= (qDy[ix, iy+1] + k_ηf_dy *(Pf[ix, iy+1] - Pf[ix, iy])) * _1_θ_dτ
    end


    return nothing
end


   

function compute!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _β_dτ_dx, _β_dτ_dy, threads, blocks)

    @cuda threads=threads blocks=blocks compute_flux!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
    synchronize()
    @cuda threads=threads blocks=blocks update_Pf!(nx, ny, Pf, qDx, qDy, _β_dτ_dx, _β_dτ_dy)
    synchronize()
    return nothing
end

function compute_t!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _β_dτ_dx, _β_dτ_dy)

    compute_flux_t!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
    update_Pf_t!(nx, ny, Pf, qDx, qDy, _β_dτ_dx, _β_dτ_dy)
    return nothing
end



function Pf_diffusion_2D(nx, ny , maxiter;do_check = false, useThreads = false)
    # physics
    lx, ly = 20.0, 20.0
    k_ηf   = 1.0
    # numerics
    ϵtol    = 1e-8
    ncheck  = ceil(Int, 0.25max(nx, ny))
    cfl     = 1.0 / sqrt(2.1)
    re      = 2π
    threads = (32, 4)
    blocks = blocks  = (nx ÷ threads[1], ny ÷ threads[2])
    #fill up nx and ny to be multiples of threads
    nx = Int(threads[1]*blocks[1])
    ny = Int(threads[2]*blocks[2])

    # derived numerics
    dx, dy  = lx / nx, ly / ny
    xc, yc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)#??
    θ_dτ    = max(lx, ly) / re / cfl / min(dx, dy)
    β_dτ    = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))

    if useThreads==false
        # GPU array initialisation
        Pf = CuArray(@. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2))
        
        qDx     = CUDA.zeros(Float64, nx + 1, ny)
        qDy     = CUDA.zeros(Float64, nx, ny + 1)
        r_Pf    = CUDA.zeros(nx, ny)
        
    else#CPU array initialization
        Pf      = @. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2)
        qDx     = zeros(Float64, nx + 1, ny)
        qDy     = zeros(Float64, nx, ny + 1)
        r_Pf    = zeros(nx, ny)
    end
        
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

        if useThreads
            compute_t!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _β_dτ_dx, _β_dτ_dy)
        else
            compute!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _β_dτ_dx, _β_dτ_dy, threads, blocks)
        end
       
        if do_check && iter % ncheck == 0
            r_Pf .= diff(qDx, dims=1) ./ dx .+ diff(qDy, dims=2) ./ dy
            err_Pf = maximum(abs.(r_Pf))
            @printf("  iter/nx=%.1f, err_Pf=%1.3e\n", iter / nx, err_Pf)
            display(heatmap(xc, yc, Array(Pf)'; xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), aspect_ratio=1, c=:turbo, clim=(0, 1)))
        end
        iter += 1
        
    end
    t_it = 0.0
    niter = iter - 11

    if !do_check
            
        #directly compute t_it with benchmarktools
        t_it = @belapsed compute!($nx, $ny, $qDx, $qDy, $Pf, $k_ηf_dx, $k_ηf_dy, $_1_θ_dτ, $_β_dτ_dx, $_β_dτ_dy) 
    else
        t_it = (Base.time() - t_tic)/niter
    end
   
        
    Aeff = 2*3 * nx*ny * 8 / 1e9 #2 = read and write, 3 = amount of arrays, nx"ny gridsize, *8/1e9 convert to Gb
    Teff = Aeff / t_it
    @printf("Time = %1.3f sec, Aeff = %1.3f GB, Teff = %1.3f GB/s \n", t_it * niter, Aeff, Teff)
    return Array(Pf)
end

nx = ny = 127
maxiter = 50
Pf_gpu = Pf_diffusion_2D(nx, ny, maxiter, do_check = true, useThreads = false)
Pf_cpu = Pf_diffusion_2D(nx, ny, maxiter, do_check = true, useThreads = true)

print("max error: $(maximum(Pf_gpu - Pf_cpu))")
flush(stdout)