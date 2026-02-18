using Test
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using Plots, Printf

@init_parallel_stencil(Threads, Float64, 3, inbounds=false)

include("../utils.jl")
include("../PorousConvection_3D_xpu.jl")


@testset "3D unit" begin
    nx, ny, nz = 12, 8, 6
    P  = @zeros(nx, ny, nz)
    T  = @zeros(nx, ny, nz)
    qDx = @zeros(nx+1, ny, nz)
    qDy = @zeros(nx, ny+1, nz)
    qDz = @zeros(nx, ny, nz+1)

    k_ηf = 1.0
    _dx = _dy = _dz = 1.0
    _θ = 0.5
    αρgx = αρgy = αρgz = 0.0

    @parallel compute_flux!(qDx, qDy, qDz, P, T, k_ηf, _dx, _dy, _dz, _θ, αρgx, αρgy, αρgz)

    @test maximum(abs.(Array(qDx))) < 1e-12
    @test maximum(abs.(Array(qDy))) < 1e-12
    @test maximum(abs.(Array(qDz))) < 1e-12
end

@testset "3D reference" begin
    nx=31
    ny=31
    nz=15
    porous_convection_3D(; nx=31, ny=31, nz=15, nt=3, maxiter=30,ncheck =2, do_viz=false)
    T  = zeros(Float32, nx, ny, nz)
    Tref  = zeros(Float32, nx, ny, nz)
    #read output file  
    load_array("out_T", Tref)
    #read reference file
    load_array("reference3d", T)
    
    @test maximum(abs.(T .- Tref)) < 1e-5
    
    @test all(isfinite, Array(T))

    print("finished reference test\n")

end