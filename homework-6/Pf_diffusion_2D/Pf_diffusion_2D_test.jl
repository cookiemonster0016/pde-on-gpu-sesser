using Plots, Plots.Measures, Printf, LoopVectorization, BenchmarkTools, Test
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)
macro d_xa(A) esc(:( $A[ix+1, iy]-$A[ix,iy]  )) end
macro d_ya(A) esc(:( $A[ix, iy+1]-$A[ix,iy]  )) end

function compute_flux!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
    @tturbo for iy=1:ny
        for ix= 1:nx-1
            @inbounds qDx[ix+1, iy] -= (qDx[ix+1, iy] + k_ηf_dx *@d_xa(Pf)) * _1_θ_dτ
        end
    end
       
    @tturbo for iy=1:ny-1
        for ix= 1:nx
            @inbounds qDy[ix, iy+1] -= (qDy[ix, iy+1] + k_ηf_dy *@d_ya(Pf)) * _1_θ_dτ
        end
    end

    return nothing
end


function update_Pf!(nx, ny, Pf, qDx, qDy, _β_dτ_dx, _β_dτ_dy)
    @tturbo for iy=1:ny
        for ix=1:nx
            @inbounds Pf[ix, iy] -= (@d_xa(qDx)) * _β_dτ_dx  +  (@d_ya(qDy)) * _β_dτ_dy
        end
    end
    return nothing
end

function compute!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _β_dτ_dx, _β_dτ_dy)
    compute_flux!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
    update_Pf!(nx, ny, Pf, qDx, qDy, _β_dτ_dx, _β_dτ_dy)
    return nothing
end

@testset "test entries" begin
    lookup_table = Dict(63  => [0.00785398056115133, 0.007853980637555755, 0.007853978592411982],
                        127 => [0.00787296974549236, 0.007849556884184108, 0.007847181374079883],
                        255 => [0.00740912103848251, 0.009143711648167267, 0.007419533048751209],
                        511 => [0.00566813765849919, 0.004348785338575644, 0.005618691590498087],)


    function Pf_diffusion_2D(;do_check = false)
        # physics
        lx, ly = 20.0, 20.0
        k_ηf   = 1.0
        
        # numerics
        maxiter = 500
        cfl     = 1.0 / sqrt(2.1)
        re      = 2π

        ns = 16 * 2 .^ (2:5) .- 1
        
        for n in ns
            
            nx = ny = n
            # derived numerics
            dx, dy  = lx / nx, ly / ny
            xc, yc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
            θ_dτ    = max(lx, ly) / re / cfl / min(dx, dy)
            β_dτ    = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
            # array initialisation
            Pf      = @. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2)
            qDx     = zeros(Float64, nx + 1, ny)
            qDy     = zeros(Float64, nx, ny + 1)
            r_Pf    = zeros(nx, ny)


            #precompute divisions
            k_ηf_dx, k_ηf_dy = k_ηf/dx, k_ηf/dy
            _1_θ_dτ = 1.0./(1.0 + θ_dτ)
            _β_dτ_dx = 1.0/(β_dτ*dx)
            _β_dτ_dy = 1.0/(β_dτ*dy)


        
            #compute for 500 iterations
            for i in 1:maxiter
                compute!(nx, ny, qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _β_dτ_dx, _β_dτ_dy) 
            end

            xtest = [5, Int(cld(0.6*lx, dx)), nx-10]
            ytest = Int(cld(0.5*ly, dy))
            
            @test Pf[xtest, ytest] ≈ lookup_table[n]

        end

        return
    end

    Pf_diffusion_2D(do_check=false)
end;