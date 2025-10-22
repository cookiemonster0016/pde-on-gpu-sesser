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
    nt      = 500
    nvis    = 5
    dtd        = min(dx, dy)^2 / λ_ρCp / 4.1
    r_Pf    = zeros(Float64, nx, ny)
    # derived numerics

    xc = LinRange(-lx/2 + dx/2, lx/2 - dx/2, nx)
    yc = LinRange(-ly + dy/2, -dy/2, ny)
    cfl     = 1.0/sqrt(2.1)

    # pressure PT
    re_D    = 2π
    θ_dτ_D  = max(lx, ly) / re_D / (cfl * min(dy, dx))
    β_dτ_D  = k_ηf * re_D / (cfl * min(dx , dy) * max(lx, ly))
    # array initialisation
   
    # pressure
    P       = zeros(nx, ny)#at cell centers
    qDx     = zeros(Float64, nx + 1, ny)#flux on faces --> now also exteriour faces
    qDy     = zeros(Float64, nx, ny + 1)

    #initial condition for Temperature at cell centers
    T         = @. ΔT * exp(-xc^2 - (yc' + ly / 2)^2)
    T[:, 1] .= ΔT / 2; T[:, end] .= -ΔT / 2
    T[[1, end], :] .= T[[2, end-1], :]

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
    record(fig, "homework-5/porous_convection_2D.mp4", 1:nt; framerate=20) do it
        #@printf("it = %d\n", it)
        # iteration loop
        iter = 1; err_Pf = 2ϵtol
        while err_Pf >= ϵtol && iter <= maxiter

            # pressure
            qDx[2:end-1, :] .-= (qDx[2:end-1, :] .+ k_ηf .* (diff(P, dims=1) ./ dx .- αρgx .* avx(T))) ./ (θ_dτ_D + 1.0)
            qDy[:, 2:end-1] .-= (qDy[:, 2:end-1] .+ k_ηf .* (diff(P, dims=2) ./ dy .- αρgy .* avy(T))) ./ (θ_dτ_D + 1.0)
            #qDx on exteriour faces stays 0
            P  .-= (diff(qDx, dims =1) ./ dx + diff(qDy, dims = 2)./dy) ./ β_dτ_D

            if iter % ncheck == 0
                r_Pf  .= diff(qDx, dims=1) ./ dx + diff(qDy, dims=2) ./ dy
                err_Pf = maximum(abs.(r_Pf))
                #@printf("  iter = %.1f × N, err_Pf = %1.3e\n", iter / nx, err_Pf)
            end
            iter += 1
        end

        @printf("iterations until convergence: =%.1f\n", iter)


        dta = ϕ * min(dx / maximum(abs.(qDx)), dy / maximum(abs.(qDy))) / 2.1
        dtd = min(dx, dy)^2 / λ_ρCp / 2.1
        dt  = min(dta, dtd)

        # temperature

        #temperature diffusion
        T[2:end-1, :] .+= dt * λ_ρCp .* diff(diff(T, dims=1) ./ dx, dims=1) ./ dx
        T[:, 2:end-1] .+= dt * λ_ρCp  .* diff( diff(T, dims=2) ./ dy, dims=2) ./ dy

        #Temperature advection ->upwind takes the correct direction because of min/max
        #max = positive velocity = forward scheme, min = negative velocity = backward scheme
        T[2:end-1, :] .-= dt/ϕ .* (max.(qDx[2:end-2, :], 0.0) .* diff(T[1:end-1, :], dims = 1) ./ dx .+
                              min.(qDx[3:end-1, :], 0.0) .* diff(T[2:end, : ], dims = 1) ./ dx)

        T[:, 2:end-1] .-= dt / ϕ .* (max.(qDy[:, 2:end-2], 0.0) .* diff(T[:, 1:end-1], dims = 2) ./ dy .+
                              min.(qDy[:, 3:end-1], 0.0) .* diff(T[:, 2:end  ], dims = 2) ./ dy)

        # Visualization
        
            if it % nvis == 0
                qDx_c .= avx(qDx)
                qDy_c .= avy(qDy)
                ar[3] = qDx_c[1:st:end, 1:st:end]
                ar[4] = qDy_c[1:st:end, 1:st:end]
                hm[3] = T
                #display(fig)
            end
    end
end


porous_convection_2D()