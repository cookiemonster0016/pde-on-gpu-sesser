include("Pf_diffusion_2D_Teff.jl")
include("Pf_diffusion_2D_perf.jl")
include("Pf_diffusion_2D_perf_loop_fun.jl")

using Plots
using .loop, .perf, .original

maxiter = 500
nx = ny = 16 * 2 .^ (1:8)
Toriginal = zeros(Float64, 8)
Tperf = zeros(Float64, 8)
Tthreads = zeros(Float64, 8)
Tloop = zeros(Float64, 8)


for i in 1:8
    Toriginal[i] = original.Pf_diffusion_2D(nx[i], ny[i], maxiter, do_check = false, usebm = true)
    Tperf[i] = perf.Pf_diffusion_2D(nx[i], ny[i], maxiter,do_check = false, usebm = true)
    #they always use benchmark tool anyways
    Tthreads[i] = loop.Pf_diffusion_2D(nx[i], ny[i], maxiter,do_check = false, use_threads = true)
    Tloop[i] = loop.Pf_diffusion_2D(nx[i], ny[i], maxiter,do_check = false, use_threads = false)
end

plot(nx, [Toriginal, Tperf, Tloop, Tthreads], title= "T_eff_comparison", xlabel = "nx = ny", ylabel = "Teff", labels=["original" "perf" "loop" "loop with threads" ], markershape=:circle, markersize=3, xscale=:log10)
savefig("diff_comp_noinbounds.png")