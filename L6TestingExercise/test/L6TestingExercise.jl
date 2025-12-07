using Test, LoopVectorization
include("../scripts/diffusion_1D_test.jl")

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


@testset "Diff functiontests" begin
    A=[1.0, 2.0, 4.0, 7.0]
    @test Diff(A) == [1.0, 2.0, 3.0]

    B = [5, 5, 5]
    @test Diff(B) == [0, 0]
end


@testset "diffusion function test" begin
    C, qx = diffusion_1D() # reaches equilibrium
    x = LinRange(0, 1, length(C))
    ref_sol = 1. .- x #everything is 1

    indices = sort(rand(1:length(C), 20))
    @test all(isapprox.(C[indices], ref_sol[indices], atol=1e-2))
    indices = sort(rand(1:length(qx), 20))
    @test all(isapprox.(qx[indices], 0, atol=1e-1))
end
