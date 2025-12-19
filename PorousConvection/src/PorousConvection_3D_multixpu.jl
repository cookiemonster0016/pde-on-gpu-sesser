using ImplicitGlobalGrid
import MPI
using Plots, Printf
include("utils.jl")

using ImplicitGlobalGrid
import MPI
using Plots, Printf
using ParallelStencil
using ParallelStencil.FiniteDifferences3D

const USE_GPU = true
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=false)
end

@views avx(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])
@views avz(A) = 0.5 .* (A[:, :, 1:end-1] .+ A[:, :, 2:end])

max_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))

    # r_D  .= diff(qDx, dims=1) *_dx + diff(qDy, dims=2) * _dy
    # r_T   .= (-dTdt) .- (diff(diff(T[: , 2:end-1], dims=1), dims=1) ./ (dx*dx) .+ diff(diff(T[2:end-1, :], dims=2), dims=2) ./ (dy*dy))

    """
    Computes the pressure and temperature residual r_D and r_T
    using the material derivative of the temperature: dTdt, 
    the pressure fluxes: qDx, qDy, qDz,
    the current temperature: T
    and precomputed divisions:
    _dx = 1/dx,
    _dxdx = 1 / dx²
    
    """
@parallel function get_residual!(r_D, r_T, dTdt, qDx, qDy, qDz, T, _dx, _dy, _dz, _dxdx, _dydy, _dzdz)
    @all(r_D) = @d_xa(qDx)*_dx + @d_ya(qDy)*_dy + @d_za(qDz)*_dz
    @all(r_T) = -@all(dTdt) - ( @d2_xi(T)*_dxdx + @d2_yi(T)*_dydy + @d2_zi(T)*_dzdz )
    return nothing
end

# parallel boundary conditions -> x, y update, z = fixed
@parallel_indices (iy, iz) function bc_x!(A)
    A[1  , iy, iz] = A[2    , iy, iz]
    A[end, iy, iz] = A[end-1, iy, iz]
    return
end

@parallel_indices (ix, iz) function bc_y!(A)
    A[ix, 1, iz] = A[ix, 2, iz]
    A[ix, end, iz] = A[ix, end-1, iz]
    return
end



"""
Updates the Temperature in 3 dimensions : T

using the temperature fluxes: qTx, qTy, qTz
and precomputed divisions:
_dx = 1/ dx ...,
_β_dτ_Tp_dt = 1 / (β_dτ_T + dt)
"""

#T[2:end-1, 2:end-1] .-= (dTdt .+ (diff(qTx, dims=1) ./ dx .+ diff(qTy, dims=2) ./ dy)) ./ (1.0 / dt + β_dτ_T)
@parallel function update_temperature!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _β_dτ_Tp_dt)
    #todo is this d_xi or d_xa?
    @inn(T) = @inn(T)- (@all(dTdt) + @d_xa(qTx) * _dx + @d_ya(qTy) * _dy + @d_za(qTz) * _dz) * _β_dτ_Tp_dt

    return nothing
end




            # dTdt[ix, iy] = (T[ix+1, iy+1] - T_old[ix+1, iy+1]) * _dt + 
            #                         (max.(qDx[ix+1, iy+1], 0.0) * (T[ix+1, iy+1] - T[ix, iy+1]) * _dx +        #forward x dir
            #                         min.((qDx)[ix+2, iy+1], 0.0) * (T[ix+2, iy+1] - T[ix+1, iy+1]) * _dx +              #backward x dir
            #                         max.((qDy)[ix+1, iy+1], 0.0) * (T[ix+1, iy+1] - T[ix+1, iy]) * _dy +          #forward y dir
            #                         min.((qDy)[ix+1, iy+2], 0.0) * (T[ix+1, iy+2] - T[ix+1, iy+1]) * _dy) * _ϕ       #backward y dir

"""
Computes the material derivative of the temperature: dTdt
using The current and the old temperature: T, Told,
The pressure fluxes: qDx, qDy, qDz
and the precomputed divisions:
_dx = 1/dx, ...
_dt = 1 / dt, 
_ϕ = 1 / ϕ
"""            

@parallel_indices (ix, iy, iz) function compute_materialDerivative!(dTdt, T, T_old, qDx, qDy, qDz, _dx, _dy, _dz, _dt, _ϕ)
    
    dTdt[ix, iy, iz] = (T[ix+1, iy+1, iz+1] - T_old[ix+1, iy+1, iz+1]) * _dt + 
        (
            max.(qDx[ix+1, iy+1, iz+1], 0.0) * (T[ix+1, iy+1, iz+1] - T[ix, iy+1, iz+1]) * _dx +        #forward x dir
            min.(qDx[ix+2, iy+1, iz+1], 0.0) * (T[ix+2, iy+1, iz+1] - T[ix+1, iy+1, iz+1]) * _dx +    #backward x dir

            max.(qDy[ix+1, iy+1, iz+1], 0.0) * (T[ix+1, iy+1, iz+1] - T[ix+1, iy, iz+1]) * _dy +          #forward y dir
            min.(qDy[ix+1, iy+2, iz+1], 0.0) * (T[ix+1, iy+2, iz +1] - T[ix+1, iy+1, iz+1]) * _dy +     #backward y dir

            max(qDz[ix+1, iy+1, iz+1], 0.0) * (T[ix+1, iy+1, iz+1] - T[ix+1, iy+1, iz  ]) * _dz +   #forward z dir
            min(qDz[ix+1, iy+1, iz+2], 0.0) * (T[ix+1, iy+1, iz+2] - T[ix+1, iy+1, iz+1]) * _dz     #backward z dir 

        ) * _ϕ

    return nothing
end


"""
computes the temperature flux in 3 dimensions : qTx, qTy, qTz
using the temperature: T
and precomputed divisions:
_dx = 1/ dx ...,
_θ_dτ_Dp1 = 1 / θ_dτ_D + 1
 λ_ρCp_dx =  λ_ρCp / dx ...,

"""

#    @all(qTx) = @all(qTx) - ((@all(qTx) + λ_ρCp_dx * @d_xi(T)) *_θ_dτ_Tp1)
#   @all(qTy) = @all(qTy) - ((@all(qTy) + λ_ρCp_dy * @d_yi(T)) *_θ_dτ_Tp1)        
@parallel function compute_temperatureFlux!(qTx, qTy, qTz, T, λ_ρCp_dx, λ_ρCp_dy, λ_ρCp_dz, _θ_dτ_Tp1)

    @all(qTx) = @all(qTx) - ((@all(qTx) + λ_ρCp_dx * @d_xi(T)) *_θ_dτ_Tp1)
    @all(qTy) = @all(qTy) - ((@all(qTy) + λ_ρCp_dy * @d_yi(T)) *_θ_dτ_Tp1)
    @all(qTz) = @all(qTz) - ((@all(qTz) + λ_ρCp_dz * @d_zi(T)) *_θ_dτ_Tp1)

    return nothing
end
           

"""
Updates the pressure in 3 dimensions : P
using forces in x, y and z direction, αρgx, αρgy, αρgz,
the pressure fluxes: qDx, qDy, qDz
and precomputed divisions:
_dx = 1/ dx ...,
_θ_dτ_D = 1 / θ_dτ_D
"""
# P  .-= (diff(qDx, dims =1) ./ dx + diff(qDy, dims = 2)./dy) ./ β_dτ_D
@parallel function update_P!(P, qDx, qDy, qDz, _β_dτ_D, _dx, _dy, _dz)
    @all(P) = @all(P) - (@d_xa(qDx)*_dx + @d_ya(qDy)*_dy + @d_za(qDz)*_dz) * _β_dτ_D
    return nothing
end


"""
computes the pressure flux in 3 dimensions : qDx, qDy, qDz
using forces in x, y and z direction, αρgx, αρgy, αρgz
and precomputed divisions:
_dx = 1/ dx ...,
_θ_dτ_Dp1 = 1 / θ_dτ_D + 1
"""
    # @inn_x(qDx) = @inn_x(qDx) - ((@inn_x(qDx) + k_ηf * (@d_xa(P)*_dx - αρgx .* @av_xa(T)))* _θ_dτ_Dp1)
    # @inn_y(qDy) = @inn_y(qDy) - ((@inn_y(qDy) + k_ηf * (@d_ya(P)*_dy - αρgy .* @av_ya(T))) * _θ_dτ_Dp1)
@parallel function compute_flux!(qDx, qDy, qDz, P, T, k_ηf, _dx, _dy, _dz, _θ_dτ_Dp1, αρgx, αρgy, αρgz)

    @inn_x(qDx) = @inn_x(qDx) - ((@inn_x(qDx) + k_ηf * (@d_xa(P)*_dx - αρgx .* @av_xa(T))) * _θ_dτ_Dp1)
    @inn_y(qDy) = @inn_y(qDy) - ((@inn_y(qDy) + k_ηf * (@d_ya(P)*_dy - αρgy .* @av_ya(T))) * _θ_dτ_Dp1)
    @inn_z(qDz) = @inn_z(qDz) - ((@inn_z(qDz) + k_ηf * (@d_za(P)*_dz - αρgz .* @av_za(T))) * _θ_dτ_Dp1)
    return nothing
end

"""
Does a simulation of porous convection.
It can run on multiple CPUs or GPUs
"""
@views function porous_convection_3D(do_viz = true)
   
   #global grid init
    nz          = 127
    nx,ny       = 2 * (nz + 1) - 1, nz

    me, dims    = init_global_grid(nx, ny, nz, select_device = false)  # init global grid and more
    b_width     = (8, 8, 4)                     # for comm / comp overlap
    nt=6000
    maxiter=10*max(nx_g(), ny_g(), nz_g())
    ncheck  = ceil(2max(nx_g(), ny_g(), nz_g())) 

    if(me==0) @printf("starting initialization") end

    # physics
    lx      = 40.0
    ly      = 20.0
    lz      = 20.0
    k_ηf       = 1.0
    αρgx, αρgy, αρgz = 0.0, 0.0, 1.0
    αρg        = sqrt(αρgx^2 + αρgy^2 + αρgz^2)
    ΔT         = 200.0
    ϕ          = 0.1
    Ra         = 1000.0
    λ_ρCp      = 1 / Ra * (αρg * k_ηf * ΔT * lz / ϕ) # This changes, because now z is up
 
    # numerics
    dx      = lx / nx_g()
    dy      = ly / ny_g()
    dz      = lz / nz_g()
    ϵtol    = 1e-6
    nvis    = 100
    dtd     = min(dx, dy, dz)^2 / λ_ρCp / 4.1
    r_D     = @zeros( nx, ny, nz)

    # derived numerics
    xc = LinRange(-lx/2 + dx/2, lx/2 - dx/2, nx)
    yc = LinRange(-ly/2 + dy/2, ly/2 - dy/2, ny)

    zc = LinRange(-lz + dz/2, -dz/2, nz)

    cfl     = 1.0/sqrt(3.1)

    # pressure PT
    re_D    = 4π
    θ_dτ_D  = max(lx, ly, lz) / re_D / (cfl * min(dy, dx, dz))
    β_dτ_D  = k_ηf * re_D / (cfl * min(dx , dy, dz) * max(lx, ly, lz))

    # array initialisation

    dTdt        = @zeros(nx - 2, ny - 2, nz-2)#time derivative of T at cell centers
    r_T         = @zeros(nx - 2, ny - 2, nz-2)#redsidual
    qTx         = @zeros(nx - 1, ny - 2, nz-2)#temperature flux in x dir at faces
    qTy         = @zeros(nx - 2, ny - 1, nz-2)#temperature flux in y dir at faces
    qTz         = @zeros(nx - 2, ny - 2, nz-1)#temperature flux in z dir at faces

    # pressure
    P       = @zeros(nx, ny, nz)#at cell centers
    qDx     = @zeros(nx + 1, ny, nz)#flux on faces --> now also exteriour faces
    qDy     = @zeros(nx, ny + 1, nz)
    qDz     = @zeros(nx, ny, nz + 1)

    #initial condition for Temperature at cell centers
    T  = @zeros(nx, ny, nz)
    T .= Data.Array([ΔT * exp(-(x_g(ix, dx, T) + dx / 2 - lx / 2)^2
                            -(y_g(iy, dy, T) + dy / 2 - ly / 2)^2
                            -(z_g(iz, dz, T) + dz / 2 - lz / 2)^2) for ix = 1:size(T, 1), iy = 1:size(T, 2), iz = 1:size(T, 3)])
    T[:, :, 1  ] .=  ΔT / 2
    T[:, :, end] .= -ΔT / 2
    update_halo!(T)
    T_old = copy(T)

    iframe = 0

    k = ceil(Int, ny/2)

    #procomputations
    k_ηf_dx, k_ηf_dy, k_ηf_dz = k_ηf/dx, k_ηf/dy, k_ηf/dz
    _θ_dτ_Dp1 = 1.0./(1.0 + θ_dτ_D)
    _β_dτ_D = 1.0/(β_dτ_D)
    
    λ_ρCp_dx = λ_ρCp/dx
    λ_ρCp_dy = λ_ρCp/dy
    λ_ρCp_dz = λ_ρCp/dz

    _dx = 1.0/dx
    _dy = 1.0/dy
    _dz = 1.0/dz

    _ϕ = 1.0/ϕ

    _dxdx = 1.0/(dx*dx)
    _dydy = 1.0/(dy*dy)
    _dzdz = 1.0/(dz*dz)

    #prepare for visualization
    if do_viz
        ENV["GKSwstype"]="nul"
        if (me==0) if isdir("viz3Dmpi_out")==false mkdir("viz3Dmpi_out") end; loadpath="viz3Dmpi_out/"; anim=Animation(loadpath,String[]); println("Animation directory: $(anim.dir)") end
        nx_v, ny_v, nz_v = (nx - 2) * dims[1], (ny - 2) * dims[2], (nz - 2) * dims[3]
        (nx_v * ny_v * nz_v * sizeof(Data.Number) > 0.8 * Sys.free_memory()) && error("Not enough memory for visualization.")
        T_v   = zeros(nx_v, ny_v, nz_v) # global array for visu
        T_inn = zeros(nx - 2, ny - 2, nz - 2) # no halo local array for visu
        xi_g, zi_g = LinRange(-lx / 2 + dx + dx / 2, lx / 2 - dx - dx / 2, nx_v), LinRange(-lz + dz + dz / 2, -dz - dz / 2, nz_v) # inner points only
        iframe = 0
    end

    if(me==0) @printf("starting timeloop") end

    # time loop
    for it in 1:nt

        T_old .= T

        # set time step size
        dt = if it == 1
            0.1 * min(dx, dy, dz) / (αρg * ΔT * k_ηf)
        else
            min(5.0 * min(dx, dy, dz) / (αρg * ΔT * k_ηf), ϕ * min(dx / max_g(abs.(qDx)), dy / max_g(abs.(qDy)), dz / max_g(abs.(qDz))) / 3.1)
        end

        _dt = 1.0/dt

        # iteration loop
        iter = 1; err_D = 2ϵtol; err_T = 2ϵtol
        while err_D >= ϵtol && err_D >=ϵtol && iter <= maxiter

            #adjust pseud-transient parameters
            re_T    = π + sqrt(π^2 + lz^2 / λ_ρCp / dt)
            θ_dτ_T  = max(lx, ly, lz) / re_T / cfl / min(dx, dy, dz)
            β_dτ_T  = (re_T * λ_ρCp) / (cfl * min(dx, dy, dz) * max(lx, ly, lz))
            _β_dτ_Tp_dt = 1.0 / (1.0 / dt + β_dτ_T)
            _θ_dτ_Tp1 = 1.0 / (θ_dτ_T + 1.0)


            # fluid pressure update
            @hide_communication b_width begin
                @parallel compute_flux!(qDx, qDy, qDz, P, T, k_ηf, _dx, _dy, _dz, _θ_dτ_Dp1, αρgx, αρgy, αρgz)
                update_halo!(qDx, qDy, qDz)
            end

            #qDx on exteriour faces stays 0
            @parallel update_P!(P, qDx, qDy, qDz, _β_dτ_D, _dx, _dy, _dz)     

            #this is only calculating the fluxes
            #-λ Cp * grad T
            @parallel compute_temperatureFlux!(qTx, qTy, qTz, T, λ_ρCp_dx, λ_ρCp_dy, λ_ρCp_dz, _θ_dτ_Tp1)

            #temperature advection update
            #material derivative --> time derivative + advection term with upwinding
            @parallel (1:size(dTdt,1), 1:size(dTdt,2), 1:size(dTdt,3)) compute_materialDerivative!(dTdt, T, T_old, qDx, qDy, qDz, _dx, _dy, _dz, _dt, _ϕ)

            #complete temperature update Tnew = T + (-dTdt + grad² T )/ factor --> grad²yT = grad(temperature flux)
            #why can i not do dTdt +  λ grad²T instead of dTdt + grad (λq) because q should be gradT in the steady??? 

            @hide_communication b_width begin

                @parallel update_temperature!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _β_dτ_Tp_dt)
                
                #boundary conditions for T 
                @parallel (1:size(T, 2), 1:size(T, 3)) bc_x!(T)
                @parallel (1:size(T, 1), 1:size(T, 3)) bc_y!(T)
                update_halo!(T)

            end

            if iter % ncheck == 0

                # res = - ϕ (-dTdt) + λ (T_xx + T_yy) =#
                @parallel get_residual!(r_D, r_T, dTdt, qDx, qDy, qDz, T, _dx, _dy, _dz, _dxdx, _dydy, _dzdz )

                err_T = max_g(abs.(r_T))
                err_D = max_g(abs.(r_D))
                #@printf(" iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", iter / nx, err_D, err_T)
            end
            iter += 1
        end
    
        if (me==0) @printf("iterations until convergence: %d / %d\n", iter, maxiter) end

        dta = ϕ * min(dx / max_g(abs.(qDx)), dy / max_g(abs.(qDy)), dz / max_g(abs.(qDz))) / 3.1
        dtd = min(dx, dy, dz)^2 / λ_ρCp / 3.1
        dt  = min(dta, dtd)

        # Visualization
        if do_viz && (it % nvis == 0)
            T_inn .= Array(T)[2:end-1, 2:end-1, 2:end-1]; gather!(T_inn, T_v)
            if me == 0
                p1 = heatmap(xi_g, zi_g, T_v[:, ceil(Int, ny_g() / 2), :]'; xlims=(xi_g[1], xi_g[end]), ylims=(zi_g[1], zi_g[end]), aspect_ratio=1, c=:turbo)
                # display(p1)
                png(p1, @sprintf("viz3Dmpi_out/%04d.png", iframe += 1))
                save_array(@sprintf("viz3Dmpi_out/out_T_%04d", iframe), convert.(Float32, T_v))
            end
        end

    end
    
    #save_array("out_T", convert.(Float32, Array(T)))
    finalize_global_grid()
    return
end

print("calling convection diffusion")
porous_convection_3D(true)