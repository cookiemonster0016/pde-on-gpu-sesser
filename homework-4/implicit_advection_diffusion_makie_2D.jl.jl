using Printf, CairoMakie

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
    nt      = 30 # number of time steps

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
    #Prepare arrow plot data (every 10th cell)
    n_skip  = 10
    xc_arr  = xc[2:n_skip:end-1]  # Interior points only
    yc_arr  = yc[2:n_skip:end-1]
   # nx_arr = Int(floor((nx-2)/10))
   # ny_arr = Int(floor((ny-2)/10))
    nx_arr  = length(xc_arr) 
    ny_arr  = length(yc_arr)

    # Initialize velocity field for arrows
    qx_arr = zeros(Float64, nx_arr, ny_arr)
    qy_arr =  zeros(Float64, nx_arr, ny_arr)

    #makie plot preperation
    fig = Makie.Figure(size=(600, 800))
    ax1 = Makie.Axis(fig[1, 1], xlabel="x", ylabel="y", aspect=DataAspect(), title="Concentration field")
    ax2 = Makie.Axis(fig[2, 1], title="Error Evolution", xlabel="iteration", ylabel="error", yscale=log10, limits = ((nothing, nothing), (1e-3, 10)))

    hm = Makie.heatmap!(ax1, xc, yc, C; colormap=:viridis, colorrange=(0, 1))
    cb = Makie.Colorbar(fig[1, 2], hm, label="C")
    ar = Makie.arrows!(ax1, xc_arr, yc_arr, qx_arr, qy_arr)
    plt = Makie.scatterlines!(ax2, Float64[], Float64[]; )

    #iteration evolution for the final timestep
    err_evo = zeros(Float64, maxiter)
    iter_evo = zeros(Float64, maxiter)

    #Animation 
    record(fig, "homework-4/heatmap_arrows.mp4"; framerate=20) do io
        # Time loop
        for t = 1:nt

            #save the concentration for each timestep (because there are only 10 timesteps)
            C_old.= C
            
            # Iteration loop for diffusion
            iter = 1; err = 2ϵtol;  

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

            #vizualization update
        
            hm[3] = C# update heatmap
            plt[1] = iter_evo#update error plot
            plt[2] = err_evo
            #scatterlines!(ax2, iter_evo, err_evo)
            
            # Update arrow plot to show flux velocity field 
            #interpolation
            qx_center = 0.5 * (qx[1:end-1, :] .+ qx[2:end, :])#size: (nx-2), (ny-2)
            qy_center = 0.5 * (qy[:, 1:end-1] .+ qy[:, 2:end])#size: (nx-2), (ny-2)
            #only show every n_skip'th arrow
            qx_arr .= qx_center[1:n_skip:end, 1:n_skip:end]#size: (nx-2)%10, (ny-2)%10?
            qy_arr .= qy_center[1:n_skip:end, 1:n_skip:end]

            ar[3] = qx_arr
            ar[4] = qy_arr

            recordframe!(io)# adds this frame to animation
        
        end

    end

end

steady_diffusion_1D()