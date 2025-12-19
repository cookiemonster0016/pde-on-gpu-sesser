using CairoMakie, Printf
#default(size=(1200, 800), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end,:])#staggered grid averaging x dir
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])#staggered grid averaging y dir

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
    nx      = 127
    ny      = 50
    dx      = lx / nx
    dy      = ly / ny
    ϵtol    = 1e-8
    maxiter = 50* max(nx, ny)
    ncheck  = ceil(Int, 0.25* max(nx, ny))
    nt      = 100
    nvis    = 5
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
            qDx[2:end-1, :] .-= (qDx[2:end-1, :] .+ k_ηf .* (diff(P, dims=1) ./ dx .- αρgx .* avx(T))) ./ (θ_dτ_D + 1.0)
            qDy[:, 2:end-1] .-= (qDy[:, 2:end-1] .+ k_ηf .* (diff(P, dims=2) ./ dy .- αρgy .* avy(T))) ./ (θ_dτ_D + 1.0)
            #qDx on exteriour faces stays 0
            P  .-= (diff(qDx, dims =1) ./ dx + diff(qDy, dims = 2)./dy) ./ β_dτ_D

            #this is only calculating the fluxes
            #-λ Cp * grad T
            qTx .-= (qTx .+ λ_ρCp .* diff(T[:, 2:end-1], dims=1) ./ dx) ./ (θ_dτ_T + 1.0)#!negative flux 
            qTy .-= (qTy .+ λ_ρCp .* diff(T[2:end-1, :], dims=2) ./ dy) ./ (θ_dτ_T + 1.0)

            #temperature advection update

            #material derivative --> time derivative + advection term with upwinding        
            dTdt .= (T[2:end-1, 2:end-1] .- T_old[2:end-1, 2:end-1]) ./ dt .+ #time derivative 
                                ((max.(qDx[2:end-2, 2:end-1], 0.0) .* diff(T[1:end-1, 2:end-1], dims = 1) ./ dx .+        #forward x dir
                                min.(qDx[3:end-1, 2:end-1], 0.0) .* diff(T[2:end, 2:end-1], dims=1) ./ dx .+              #backward x dir
                                max.(qDy[2:end-1, 2:end-2], 0.0) .* diff(T[2:end-1, 1:end-1], dims = 2) ./ dy .+          #forward y dir
                                min.(qDy[2:end-1, 3:end-1], 0.0) .* diff(T[2:end-1, 2:end ], dims = 2) ./ dy)) ./ ϕ       #backward y dir


            
            #complete temperature update Tnew = T + (-dTdt + grad² T )/ factor --> grad²T = grad(temperature flux)
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