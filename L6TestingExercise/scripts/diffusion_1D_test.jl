using Plots, Plots.Measures, Printf
default(size=(1200, 400), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)


Diff(A) = A[2:end].-A[1:end-1]

function diffusion_1D()
    # physics
    lx   = 20.0
    dc   = 1.0
    # numerics
    nx   = 200
    nvis = 2
    # derived numerics
    dx   = lx / nx
    dt   = dx^2 / dc / 2
    nt   = nx^2
    xc   = LinRange(dx / 2, lx - dx / 2, nx)
    # array initialisation
    C    = @. 0.5cos(9π * xc / lx) + 0.5
    C_i  = copy(C)
    qx   = zeros(Float64, nx - 1)
    # time loop
    for it = 1:nt
        qx          .= .-dc .* Diff(C) ./ dx
        C[2:end-1] .-= dt .* Diff(qx) ./ dx
    end
    return C, qx
end

diffusion_1D()