using CairoMakie, Printf
#default(size=(1200, 800), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

#handle packages
const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=false)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=false)
end

#d_xa at cell centers
#d_xi at faces

@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end,:])#staggered grid averaging x dir
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])#staggered grid averaging y dir


##from the tips

# #parallel boundary conditions
# @parallel_indices (iy) function bc_x!(A)
#     A[1  , iy] = A[2    , iy]
#     A[end, iy] = A[end-1, iy]
#     return
# end

# @parallel (1:size(T,2)) bc_x!(T)


    # dTdt .= (T[2:end-1, 2:end-1] .- T_old[2:end-1, 2:end-1]) ./ _dt .+ #time derivative 
    #                             ((max.(qDx[2:end-2, 2:end-1], 0.0) .* diff(T[1:end-1, 2:end-1], dims = 1) ./ _dx .+        #forward x dir
    #                             min.(qDx[3:end-1, 2:end-1], 0.0) .* diff(T[2:end, 2:end-1], dims=1) ./ _dx .+              #backward x dir
    #                             max.(qDy[2:end-1, 2:end-2], 0.0) .* diff(T[2:end-1, 1:end-1], dims = 2) ./ _dy .+          #forward y dir
    #                             min.(qDy[2:end-1, 3:end-1], 0.0) .* diff(T[2:end-1, 2:end ], dims = 2) ./ _dy)) ./ _ϕ       #backward y dir

@parallel function compute_materialDerivative!(dTdt, T, T_old, qDx, qDy, _dx, _dy, _dt, _ϕ)

    return nothing
end


#qTx .-= (qTx .+ λ_ρCp .* diff(T[:, 2:end-1], dims=1) ./ dx) ./ (θ_dτ_T + 1.0)#!negative flux 
#qTy .-= (qTy .+ λ_ρCp .* diff(T[2:end-1, :], dims=2) ./ dy) ./ (θ_dτ_T + 1.0)         
@parallel function compute_temperatureFlux!(qTx, qTy, T,λ_ρCp_dx, λ_ρCp_dy,  θ_dτ_T )

    @all(qTx) = @all(qTx) - ((@all(qTx) + λ_ρCp_dx * @d_xa(T)) / (θ_dτ_T + 1.0))
    @all(qTy) = @all(qTy) - ((@all(qTy) + λ_ρCp_dy * @d_ya(T)) / (θ_dτ_T + 1.0))

    return nothing
end
           

# P  .-= (diff(qDx, dims =1) ./ dx + diff(qDy, dims = 2)./dy) ./ β_dτ_D
@parallel function update_Pf!(Pf, qDx, qDy, _β_dτ_dx, _β_dτ_dy)

    @all(Pf) = @all(Pf) - @d_xa(qDx) * _β_dτ_dx  +  @d_ya(qDy) * _β_dτ_dy

    return nothing

end


# qDx[2:end-1, :] .-= (qDx[2:end-1, :] .+ k_ηf .* (diff(P, dims=1) ./ dx .- αρgx .* avx(T))) ./ (θ_dτ_D + 1.0)
# qDy[:, 2:end-1] .-= (qDy[:, 2:end-1] .+ k_ηf .* (diff(P, dims=2) ./ dy .- αρgy .* avy(T))) ./ (θ_dτ_D + 1.0)
@parallel function compute_flux!(qDx, qDy, Pf, T, k_ηf_dx, k_ηf_dy, _θ_dτ_Dp1, αρgx, αρgy)

    @inn_x(qDx) = @inn_x(qDx) - ((@inn_x(qDx) + k_ηf_dx * (@d_xa(Pf) - αρgx .* @av_xa(T)))* _θ_dτ_Dp1)
    @inn_y(qDy) = @inn_y(qDy) - ((@inn_y(qDy) + k_ηf_dy * (@d_ya(Pf) - αρgy .* @av_ya(T))) * _θ_dτ_Dp1)

    return nothing
end


@views function porous_convection_2D()
    # physics
    lx      = 40.0
    ly      = 20.0
    k_ηf       = 1.0
    αρgx, αρgy = 0.0, 1.0
    αρg        = sqrt(αρgx^2 + αρgy^2)
    ΔT         = 200.0
    ϕ          = 0.1
    Ra         = 1000.0
    λ_ρCp      = 1 / Ra * (αρg * k_ηf * ΔT * ly / ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
 
    # numerics 
    ny      = 63
    nx      = 2 * (ny + 1) -1
    dx      = lx / nx
    dy      = ly / ny
    ϵtol    = 1e-6
    maxiter = 10 * max(nx, ny)
    ncheck  = ceil(max(nx, ny))
    nt      = 500
    nvis    = 20
    dtd     = min(dx, dy)^2 / λ_ρCp / 4.1
    r_D     = zeros(Float64, nx, ny)

    # derived numerics
    xc = LinRange(-lx/2 + dx/2, lx/2 - dx/2, nx)
    yc = LinRange(-ly + dy/2, -dy/2, ny)
    cfl     = 1.0/sqrt(2.1)

    # pressure PT
    re_D    = 4π
    θ_dτ_D  = max(lx, ly) / re_D / (cfl * min(dy, dx))
    β_dτ_D  = k_ηf * re_D / (cfl * min(dx , dy) * max(lx, ly))

    # array initialisation

    dTdt        = zeros(nx - 2, ny - 2)#time derivative of T at cell centers
    r_T         = zeros(nx - 2, ny - 2)#redsidual
    qTy         = zeros(nx - 2, ny - 1)#temperature flux in y dir at faces
    qTx         = zeros(nx - 1, ny - 2)#temperature flux in x dir at faces
   
    # pressure
    P       = zeros(nx, ny)#at cell centers
    qDx     = zeros(Float64, nx + 1, ny)#flux on faces --> now also exteriour faces
    qDy     = zeros(Float64, nx, ny + 1)

    #initial condition for Temperature at cell centers
    T         = @. ΔT * exp(-xc^2 - (yc' + ly / 2)^2)
    T[:, 1] .= ΔT / 2; T[:, end] .= -ΔT / 2
    T[[1, end], :] .= T[[2, end-1], :]

    T_old = copy(T)

    #makei plot preperation
    st = 5 #amount of arrows, smaller value -> more arrows
    fig = Makie.Figure(size=(600, 800))
    ax = Makie.Axis(fig[1, 1], xlabel="x", ylabel="y", aspect=DataAspect(), title="Poreus Convection")
    hm = Makie.heatmap!(ax, xc, yc, T; colormap=:lightrainbow, colorrange=(-50, 50))
    cb = Makie.Colorbar(fig[1, 2], hm, label="Temperature")

    qDx_c = zeros(Float64, nx, ny)
    qDy_c = zeros(Float64, nx, ny)
    qDx_c .= avx(qDx)
    qDy_c .= avy(qDy)
    xar = xc[1:st:end]
    yar = yc[1:st:end]

    ar = Makie.arrows2d!(ax, xar, yar, qDx_c[1:st:end, 1:st:end], qDy_c[1:st:end, 1:st:end], normalize= true)

    #procomputations
    k_ηf_dx, k_ηf_dy = k_ηf/dx, k_ηf/dy
    _θ_dτ_Dp1 = 1.0./(1.0 + θ_dτ_D)
    _β_dτ_D_dx = 1.0/(β_dτ_D*dx)
    _β_dτ_D_dy = 1.0/(β_dτ_D*dy)
    λ_ρCp_dx = λ_ρCp/dx
    λ_ρCp_dy = λ_ρCp/dy
    _dx = 1.0/dx
    _dy = 1.0/dy
    _dt = 1.0/dt
    _ϕ = 1.0/ϕ

# time loop
    record(fig, "homework-5/porous_convection__implicit_2D_Ra=100.gif", 1:nt; framerate=20) do it
        T_old .= T

        # set time step size
        dt = if it == 1
            0.1 * min(dx, dy) / (αρg * ΔT * k_ηf)
        else
            min(5.0 * min(dx, dy) / (αρg * ΔT * k_ηf), ϕ * min(dx / maximum(abs.(qDx)), dy / maximum(abs.(qDy))) / 2.1)
        end

        # iteration loop
        iter = 1; err_D = 2ϵtol; err_T = 2ϵtol
        while err_D >= ϵtol && err_D >=ϵtol && iter <= maxiter

            #adjust pseud-transient parameters
            re_T    = π + sqrt(π^2 + ly^2 / λ_ρCp / dt)
            θ_dτ_T  = max(lx, ly) / re_T / cfl / min(dx, dy)
            β_dτ_T  = (re_T * λ_ρCp) / (cfl * min(dx, dy) * max(lx, ly))

            # fluid pressure update
            @parallel compute_flux!(qDx, qDy, P, T, k_ηf_dx, k_ηf_dy, _θ_dτ_Dp1, αρgx, αρgy)

            #qDx on exteriour faces stays 0
            @parallel update_Pf!(P, qDx, qDy, _β_dτ_D_dx, _β_dτ_D_dy)

            #this is only calculating the fluxes
            #-λ Cp * grad T
            @parallel compute_temperatureFlux!(qTx, qTy, T,λ_ρCp_dx, λ_ρCp_dy,  θ_dτ_T)

            #temperature advection update
            #material derivative --> time derivative + advection term with upwinding
            @parallel compute_materialDerivative!(dTdt, T, T_old, qDx, qDy, _dx, _dy, _dt, _ϕ)

            
            #complete temperature update Tnew = T + (-dTdt + grad² T )/ factor --> grad²yT = grad(temperature flux)
            #why can i not do dTdt +  λ grad²T instead of dTdt + grad (λq) because q should be gradT in the steady??? 
            T[2:end-1, 2:end-1] .-= (dTdt .+ (diff(qTx, dims=1) ./ dx .+ diff(qTy, dims=2) ./ dy)) ./ (1.0 / dt + β_dτ_T)
            
            #boundary conditions for T 
            T[:, 1] .=  ΔT/2
            T[:, end] .= -ΔT/2
            T[1, 2:end-1] .= T[2, 2:end-1]
            T[end, 2:end-1] .= T[end-1, 2:end-1]

            if iter % ncheck == 0
                r_D  .= diff(qDx, dims=1) ./ dx + diff(qDy, dims=2) ./ dy
                #=res = - ϕ (-dTdt) + λ (T_xx + T_yy) =#
                r_T   .= (-dTdt) .- (diff(diff(T[: , 2:end-1], dims=1), dims=1) ./ (dx*dx) .+ diff(diff(T[2:end-1, :], dims=2), dims=2) ./ (dy*dy))

                err_T = maximum(abs.(r_T))
                err_D = maximum(abs.(r_D))
                #@printf(" iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", iter / nx, err_D, err_T)
            end
            iter += 1
        end
        @printf("iterations until convergence: =%.1f\n", iter)


        dta = ϕ * min(dx / maximum(abs.(qDx)), dy / maximum(abs.(qDy))) / 2.1
        dtd = min(dx, dy)^2 / λ_ρCp / 2.1
        dt  = min(dta, dtd)

        # Visualization
        
            if it % nvis == 0
                qDx_c .= avx(qDx)
                qDy_c .= avy(qDy)
                ar[3] = qDx_c[1:st:end, 1:st:end]
                ar[4] = qDy_c[1:st:end, 1:st:end]
                hm[3] = T
                display(fig)
            end
    end
end


porous_convection_2D()