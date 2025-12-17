using Test
using PorousConvection
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@init_parallel_stencil(Threads, Float64, 2, inbounds=false)  # CPU for CI

@testset "2D unit" begin
    # Example: constant fields => zero flux
    nx, ny = 16, 8
    P  = @zeros(nx, ny)
    T  = @zeros(nx, ny)
    qDx = @zeros(nx+1, ny)
    qDy = @zeros(nx, ny+1)

    k_ηf = 1.0
    _dx, _dy = 1.0, 1.0
    _θ = 0.5
    αρgx, αρgy = 0.0, 0.0

    @parallel PorousConvection.compute_flux!(qDx, qDy, P, T, k_ηf, _dx, _dy, _θ, αρgx, αρgy)

    @test maximum(abs.(Array(qDx))) < 1e-12
    @test maximum(abs.(Array(qDy))) < 1e-12
end

@testset "2D reference" begin
    out = PorousConvection.porous_convection_2D(; nx=63, ny=31, nt=5, maxiter=30, do_viz=false, use_gpu=false)
    @test all(isfinite, Array(out.T))
end