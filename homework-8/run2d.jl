# using Pkg
# Pkg.add("CairoMakie")
# Pkg.add("ParallelStencil")
# Pkg.add("Printf")
# Pkg.add("Plots")
using Plots
using CairoMakie, Printf
#default(size=(1200, 800), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

#handle packages
const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=false)
end


include("utils.jl")

include("PorousConvection_2D_xpu.jl")

porous_convection_2D( nx=31, ny=31, nt=3, maxiter=30,ncheck =2, do_viz=true)

print("finished the simulation\n")