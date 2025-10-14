using Plots, Plots.Measures, Printf

default(size=(1200, 800), framestyle=:box, label=false, grid=false, 
        margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

@views function steady_diffusion_1D()
    # Physics
    lx, ly  = 10.0, 10.0
    dc      = 1.0
    vx      = 10.0
    vy      = -10.0
    da      = 1000

    # Numerics
    nx, ny  = 200, 201
    ϵtol    = 1e-8
    maxiter = 10nx
    ncheck  = ceil(Int, 0.02nx)
    nt      = 50 # number of time steps

    # Derived numerics
    dx      = lx / nx
    dy      = ly / ny
    xc      = LinRange(dx / 2, lx - dx / 2, nx)
    yc      = LinRange(dy / 2, ly - dy / 2, ny)
    dt = min(dx / abs(vx), dy / abs(vy)) / 2
    re      = π + sqrt(π^2 + da)
    ρ       = (lx / (dc * re))^2  
    dτ      = min(dx, dy) / sqrt(1 / ρ) / sqrt(2)

    # Array initialization & initial conditions
    C       = @. exp(-(xc - lx / 4)^2 - (yc' - 3ly / 4)^2)
    C_old   = copy(C)
    ErrMat = zeros(Float64, nx-1, ny-1)
    qx      = zeros(Float64, nx - 1, ny - 2 )
    qy      = zeros(Float64, nx- 2, ny - 1)


        # visualisation
   # p1 = heatmap(xc, yc, C'; xlims=(0, lx), ylims=(0, ly), clims=(0, 1), aspect_ratio=1,
               # xlabel="lx", ylabel="ly" , title="Initial Conditions")
   # display(p1)

    #iteration evolution for the final timestep
    err_evo = zeros(Float64, maxiter)
    iter_evo = zeros(Float64, maxiter)


    # Time loop
    anim = @animate for t = 1:nt

        #save the concentration for each timestep (because there are only 10 timesteps)
        C_old.= C
        
        # Iteration loop for diffusion
        iter = 1; err = 2ϵtol;

        # Visualization for each timestep
        p1 = heatmap(xc, yc, C'; xlims=(0, lx), ylims=(0, ly), clims=(0, 1), aspect_ratio=1,
                    xlabel="lx", ylabel="ly", title="iter/nx=$(round(iter/nx,sigdigits=3))")


        while err >= ϵtol && iter <= maxiter#only update the inner matrix (2->nx-1, 2->ny-1)
            # second derivative in x
            qx .-= dτ ./ (ρ * dc .+ dτ) .* (qx .+ dc .* diff(C[:, 2: end-1], dims=1) ./ dx)
            #second derivative in y
            qy .-= dτ ./ (ρ * dc .+ dτ) .* (qy .+ dc .* diff(C[2:end-1, :], dims = 2) ./ dy)
            C[2:end-1,2:end-1] -= dτ ./ (1.0 .+ dτ /dt) .* ((C[2:end-1, 2:end-1] .- C_old[2:end-1, 2:end-1]) ./ dt + diff(qy, dims=2) ./ dy + diff(qx, dims=1) ./ dx)

            #error and iteration for plotting
            ErrMat = (C[2:end-1, 2: end-1] .- C_old[2:end-1,2:end-1]) ./ dt -  dc .* (diff(diff(C[:, 2:end-1], dims=1), dims=1) ./ (dx^2 )+(diff(diff(C[2:end-1, :], dims=2), dims=2)) ./ (dy^2 ))
            
            err = maximum(abs.((ErrMat)))
            push!(err_evo, err)
            push!(iter_evo, iter)

            #if iter % ncheck == 0
             #   err = maximum(abs.((C[2:end-1, 2:end-1] .- C_old[2:end-1, 2:end-1]) ./ dt -  dc .* diff(diff(diff(diff(C, dims=1), dims=1), dims=2), dims=2) ./ (dx^2)))
            #end 
            
            iter += 1
        end


         # transport step x dir.
        if vx > 0
            C[2:end, :] .-= dt/dx * vx .* diff(C, dims = 1)
            # Boundary condition
            #never change the boundarys ( left boundary is constant 1 and right boundary constant 0) 
        else
            C[1:end-1, :] .-= dt/dx * vx .* diff(C, dims=1)
        end


         # transport step y dir.
        if vy > 0
            C[:, 2: end] .-= dt/dy * vy .* diff(C, dims = 2)
            # Boundary condition
            #never change the boundarys ( left boundary is constant 1 and right boundary constant 0) 
        else
            C[:, 1:end-1] .-= dt/dy * vy .* diff(C, dims=2)
        end

        p1 = heatmap(xc, yc, C'; xlims=(0, lx), ylims=(0, ly),
                     clims=(0, 1), aspect_ratio=1,
                     xlabel="lx", ylabel="ly",
                     title=@sprintf("Concentration field – time step %d", t))
        p2 = plot([1, iter], [1e-8, err]; xlabel="iter", ylabel="err",
                  yscale=:log10, grid=true, markershape=:circle, markersize=8,
                  title="Error evolution")
        plot(p1, p2; layout=(2,1))

    end

    gif(anim, "homework-4/concentration_evolution.gif", fps=10)

end

steady_diffusion_1D()