using Printf

include("../src/visualization.jl")

nt=20
nx=1012
ny=500
nz=250


for i in 1:nt 
    visualise(@sprintf("finished_sim2000/out_T_%04d", i), nx, ny, nz)
end