using GLMakie

function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid=open(fname, "r"); read!(fid, A); close(fid)
end

function visualise(filename, nx, ny, nz)
    lx, ly, lz = 40.0, 20.0, 20.0
    T  = zeros(Float32, nx, ny, nz)
    load_array(filename, T)
    fig = Figure(size=(800, 500))
    ax  = Axis3(fig[1, 1]; aspect=(1, 1, 0.5), title="Temperature", xlabel="lx", ylabel="ly", zlabel="lz")
    surf_T = contour!(ax, 
        0.0 .. lx,
        0.0 .. ly,
        0.0 .. lz, 
        T; 
        alpha=0.05, colormap=:turbo)
    save("$(filename).png", fig)
    return fig
end