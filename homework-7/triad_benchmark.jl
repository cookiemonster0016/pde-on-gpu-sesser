using Plots, Plots.Measures, Printf, LoopVectorization, BenchmarkTools, CUDA
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)



function compute!(C2, A, C, s)
    nx, ny = size(C2)

    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if ix >= 1 && iy >=1 && ix <= nx && iy <= ny
        @inbounds C2[ix, iy] = C[ix, iy] + s * A[ix, iy]
    end
    return nothing
end


function triad_benchmark( nx, ny)

    # numerics
    nt = 2e2
    #array initialisation
    C = CUDA.rand(Float64, nx, ny)
    A = CUDA.rand(Float64, nx, ny)
    C2 = CUDA.zeros(Float64, nx, ny)
    s = rand()

    # iteration loop
    t_it = 0.0

    threads = (32, 4)
    blocks = (nx/threads[1], ny/threads[2])
    #fill up nx and ny to be multiples of threads
    nx = Int(threads[1]*blocks[1])
    ny = Int(threads[2]*blocks[2])

    #warm up phase is this necessary??
    #@cuda threads=threads blocks=blocks compute!(C2, A, C, s)
    #CUDA.synchronize()

    #compute
    t_it = @elapsed begin
        @cuda blocks=blocks threads=threads @belapsed compute!($C2, $A, $C, s)
        synchronize()
    end
    
    Aeff = ( 2 + 1 + 1 ) * nx*ny * 8 / 1e9 #2 read & 1 write, 3 = amount of arrays, nx"ny gridsize, *8/1e9 convert to Gb
    T_peak = Aeff / t_it

    return T_peak
end


nx = ny = 30 #16384

T_peak = triad_benchmark(nx, ny)
print("T_peak = $(T_peak)")