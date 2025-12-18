include("PorousConvection_3D_xpu.jl")
using Pkg
Pkg.add("CairoMakie")
Pkg.add("ParallelStencil")
Pkg.add("Printf")
Pkg.add("Plots")
using Plots, Printf
#using CairoMakie
#default(size=(1200, 800), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

function save_array(Aname,A)
    fname = string(Aname, ".bin")
    out = open(fname, "w"); write(out, A); close(out)
end

#handle packages
const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=false)
end

porous_convection_3D(; nx=31, ny=31, nz=15, nt=3, maxiter=30,ncheck =2, do_viz=false)

print("finished the simulation\n")
