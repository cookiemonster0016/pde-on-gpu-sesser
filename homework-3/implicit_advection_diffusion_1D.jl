using Plots, Plots.Measures, Printf

default(size=(1200, 800), framestyle=:box, label=false, grid=false, 
        margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

@views function steady_diffusion_1D()
    # Physics
    lx      = 20.0
    dc      = 1.0
    vx      = 1.0  # advection velocity

    # Numerics
    nx      = 100
    ϵtol    = 1e-8
    maxiter = 50nx
    ncheck  = ceil(Int, 0.25nx)
    nt      = 10  # number of time steps

    # Derived numerics
    dx      = lx / nx
    xc      = LinRange(dx / 2, lx - dx / 2, nx)
    dt      = dx/abs(vx)
    da      = lx^2/dc/dt
    re      = π + sqrt(π^2 + da) 
    ρ       = (lx / (dc * re))^2  
    dτ      = dx / sqrt(1 / ρ) 

    # Array initialization & initial conditions
    C       = @. 1.0 + exp(-(xc - lx / 4)^2) - xc / lx
    C_i     = copy(C)
    C_old   = copy(C)
    qx      = zeros(Float64, nx - 1)

    # Plot initial conditions
    p = plot(xc, C,
             xlabel = "x Position", 
             ylabel = "Concentration",
             title = "Initial Conditions",
             linewidth = 2,
             label = "Concentration",
             grid = true)
    display(p)


    #iteration evolution for the final timestep
    final_iter = Float64[]
    final_err = Float64[]
    #save the concentration for plotting
    c_plot = zeros(Float64, nx, nt)

    # Time loop
    for t = 1:nt

        # Visualization for each timestep
        #save the concentration for each timestep (because there are only 10 timesteps)
        c_plot[:, t] .= C
        C_old.= C
        # Iteration loop for diffusion
        iter = 1; err = 2ϵtol;
        while err >= ϵtol && iter <= maxiter
            qx         .-= dτ ./ (ρ * dc .+ dτ) .* (qx .+ dc .* diff(C) ./ dx)
            C[2:end-1] .-= dτ ./ (1.0 .+ dτ /dt) .* ((C[2:end-1] .- C_old[2:end-1]) ./ dt .+ diff(qx) ./ dx)
            if iter % ncheck == 0
                err = maximum(abs.((C[2:end-1] .- C_old[2:end-1]) ./ dt -  dc .* diff(diff(C)) ./ (dx^2)))
            end 
            #save the error of the last timestep to plot later
            if t == nt
                err = maximum(abs.((C[2:end-1] .- C_old[2:end-1]) ./ dt -  dc .* diff(diff(C)) ./ (dx^2)))
                push!(final_iter, iter / nx)
                push!(final_err, err)
            end 
            iter += 1
        end

        # transport step 
        if vx > 0
            C[2:end] .-= dt/dx * vx .* diff(C)
            # Boundary condition
            #never change the boundarys ( left boundary is constant 1 and right boundary constant 0) 
        else
            C[1:end-1] .-= dt/dx * vx .* diff(C)
        end        
        
    end

    c_plot[:, nt] .= C  #save final concentration


    # Time evolution visualization (just concentration, no error plot)
    anim = @animate for t = 1:nt
        
        # plot concentration in the animation
        plot(xc, c_plot[:, t];
            xlims=(0, lx), ylims=(-0.1, 2.0),
            xlabel="x Position", ylabel="Concentration", 
            title="Time step = $t", linewidth=3, color=:blue, label="C(x,t)")
    end

    gif(anim, "anim_adv_diff.gif"; fps=8)

    #plot error
    p_err = plot(final_iter, final_err;
                xlabel = "iter/nx", ylabel = "Error",
                yscale = :log10, grid = true,
                title = "Error Convergence (Final Time Step)",
                linewidth = 2, markershape = :circle,
                color = :red, label = "Error")
    display(p_err)
    savefig(p_err, "error_convergence_adv_diff.png")


    # Plot initial and final conditions together
    p_comparison = plot(xc, C_i; 
                        xlims=(0, lx), ylims=(-0.1, 2.0),
                        xlabel="x Position", 
                        ylabel="Concentration",
                        title="Initial vs Final Conditions (t=0 and t=$nt)",
                        linewidth=3, 
                        color=:blue,
                        linestyle=:dash,
                        label="t = 0 (Initial)",
                        grid=true)

    plot!(p_comparison, xc, C; 
          linewidth=3, 
          color=:red,
          label="t = $nt (Final)")

    display(p_comparison)
    savefig(p_comparison, "adv_diff.png")
end

steady_diffusion_1D()

