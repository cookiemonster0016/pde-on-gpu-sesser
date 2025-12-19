using Plots, MAT

nprocs = (2, 2)   # processor grid
its    = 1:20   # timesteps to visualize

@views function vizme2D_mpi_gif(nprocs, its)
    fontsize = 12
    opts = (
        aspect_ratio = 1,
        yaxis = font(fontsize, "Courier"),
        xaxis = font(fontsize, "Courier"),
        ticks = nothing,
        framestyle = :box,
        titlefontsize = fontsize,
        titlefont = "Courier",
        xlabel = "Lx",
        ylabel = "Ly",
    )

    anim = @animate for it in its
        C = []
        ip = 1

        for ipx = 1:nprocs[1]
            for ipy = 1:nprocs[2]
                file = matopen("output/mpi2D_out_C_$(ip-1)_$(it).mat")
                C_loc = read(file, "C")
                close(file)

                nx_i, ny_i = size(C_loc, 1) - 2, size(C_loc, 2) - 2
                ix1 = 1 + (ipx - 1) * nx_i
                iy1 = 1 + (ipy - 1) * ny_i

                if ip == 1
                    C = zeros(nprocs[1] * nx_i, nprocs[2] * ny_i)
                end

                C[ix1:ix1+nx_i-1, iy1:iy1+ny_i-1] .= C_loc[2:end-1, 2:end-1]
                ip += 1
            end
        end

        heatmap(
            C';
            c = :turbo,
            title = "diffusion 2D MPI (it = $it)",
            xlims = (1, size(C, 1)),
            ylims = (1, size(C, 2)),
            opts...
        )
    end

    gif(anim, "diffusion_2D_MPI.gif", fps=10)
end

vizme2D_mpi_gif(nprocs, its)
