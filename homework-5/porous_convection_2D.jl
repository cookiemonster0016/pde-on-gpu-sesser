using Plots, Plots.Measures, Printf
default(size=(1200, 800), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

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
    Ra         = 100
    λ_ρCp      = 1 / Ra * (αρg * k_ηf * ΔT * ly / ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
 
    # numerics 
    nx      = 100
    ny = 50
    dx      = lx / nx
    dy = ly / ny
    ϵtol    = 1e-8
    maxiter = 50* max(nx, ny)
    ncheck  = ceil(Int, 0.25* max(nx, ny))
    nt      = 50
    nvis    = 5
    dtd        = min(dx, dy)^2 / λ_ρCp / 4.1
    r_Pf    = zeros(Float64, nx, ny)
    # derived numerics

    xc      = LinRange(dx / 2, lx - dx / 2, nx)
    yc      = LinRange(dy / 2, ly - dy / 2, ny)
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

    # time loop
    for it in 1:nt
        @printf("it = %d\n", it)
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
                @printf("  iter = %.1f × N, err_Pf = %1.3e\n", iter / nx, err_Pf)
            end
            iter += 1
        end
        dta = ϕ * min(dx / maximum(abs.(qDx)), dy / maximum(abs.(qDy))) / 2.1
        dtd = min(dx, dy)^2 / λ_ρCp / 2.1
        dt  = min(dta, dtd)

        # temperature

        #temperature diffusion
        T[2:end-1, :] .+= dt .* diff(λ_ρCp .* diff(T, dims=1) ./ dx, dims=1) ./ dx
        T[:, 2:end-1] .+= dt .* diff(λ_ρCp .* diff(T, dims=2) ./ dy, dims=2) ./ dy

        #Temperature advection ->upwind takes the correct direction because of min/max
        T[2:end-1, :] .-= dt .* (max.(qDx[3:end-1, :], 0.0) .* diff(T[1:end-1, :], dims = 1) ./ dx .+
                              min.(qDx[2:end-2, :], 0.0) .* diff(T[2:end, : ], dims = 1) ./ dx)

        T[:, 2:end-1] .-= dt .* (max.(qDy[:, 3:end-1], 0.0) .* diff(T[:, 1:end-1], dims = 2) ./ dy .+
                              min.(qDy[:, 2:end-2], 0.0) .* diff(T[:, 2:end  ], dims = 2) ./ dy)

        #=if it % nvis == 0
            # visualisation
            p1 = plot(xc, [T_i, T]; xlims=(0, lx), ylabel="Temperature", title="iter/nx=$(round(iter/nx,sigdigits=3))")
            p2 = plot(xc, P       ; xlims=(0, lx), xlabel="lx", ylabel="Pressure")
            display(plot(p1, p2; layout=(2, 1)))
        end=#
    end
end
porous_convection_2D()