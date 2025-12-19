
#for euler
# using Pkg
# Pkg.add("CairoMakie")
# Pkg.add("ParallelStencil")
# Pkg.add("Printf")
# Pkg.add("Plots")

using Plots, Printf
#using CairoMakie
#default(size=(1200, 800), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

#handle packages
const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=false)
end

include("utils.jl")

include("PorousConvection_3D_xpu.jl")

porous_convection_3D()

print("finished the simulation\n")
