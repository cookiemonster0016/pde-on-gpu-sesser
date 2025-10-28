using Plots, Plots.Measures, Printf, LoopVectorization, BenchmarkTools
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)


function compute_ap!(C2, A, C)
    C2 .= C .+ A
    return nothing
end

function compute_kp!(C2, A, C)
nx, ny = size(C2)

    @tturbo for y in 1:ny
        for x in 1:nx
            C2[x, y] = C[x, y] + A[x, y]
        end
    end
    return nothing
end


function Pf_diffusion_2D( nx, ny; bench)

    # numerics
    nt = 2e4
    #array initialisation
    C = rand(Float64, nx, ny)
    C2 = copy(C)
    A = copy(C)

    # iteration loop
    t_tic = 0.0
    t_it_ar = 0.0
    t_it_ker = 0.0
    niter = nt -11

    if bench == :loop
        #compute with array method
        for iter = 1:nt
            if iter == 11
                t_tic = Base.time()
            end

            compute_ap!(C2, A, C)  
        end

        t_toc = Base.time() - t_tic
        t_it_ar = t_toc/(niter)

        #compute with kernel method
        for iter = 1:nt
            if iter == 11
                t_tic = Base.time()
            end
            compute_ap!(C2, A, C)  
        end

        t_toc = Base.time() - t_tic
        t_it_ker = t_toc/(niter)

    elseif bench == :btool
        
            #compute
            t_it_ar = @belapsed compute_ap!($C2, $A, $C)
            t_it_ker = @belapsed compute_kp!($C2, $A, $C)
            
    end
    
    Aeff = ( 2 + 1 + 1 ) * nx*ny * 8 / 1e9 #2 read & 1 write, 3 = amount of arrays, nx"ny gridsize, *8/1e9 convert to Gb
    Teff_ar = Aeff / t_it_ar
    Teff_ker = Aeff / t_it_ker

    return Teff_ar, Teff_ker
end

nx = ny =  16 * 2 .^ (1:8)
Teff_l_ar = zeros(Float64, 8)
Teff_bt_ker = zeros(Float64, 8)
Teff_l_ker = zeros(Float64, 8)
Teff_bt_ar = zeros(Float64, 8)

for i in 1:8
    Teff_l_ar[i], Teff_l_ker[i] = Pf_diffusion_2D(nx[i], ny[i], bench = :loop)
    Teff_bt_ar[i], Teff_bt_ker[i]= Pf_diffusion_2D(nx[i], ny[i], bench = :btool)
end

plot(nx, [Teff_bt_ar, Teff_l_ar, Teff_bt_ker, Teff_l_ker], title= "T_peak", xlabel = "nx = ny", ylabel = "Teff", labels=["benchmark array" "loop array" "benchmark kernel" "loop kernel"], markershape=:circle, markersize=3)
savefig("T_peak")