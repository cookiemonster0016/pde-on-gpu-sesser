using Test
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Plots, Printf
using CairoMakie

@init_parallel_stencil(Threads, Float64, 2, inbounds=false)

include("../utils.jl")
include("../PorousConvection_2D_xpu.jl")


@testset "2D unit" begin
    nx, ny= 12, 8
    P  = @zeros(nx, ny)
    T  = @zeros(nx, ny)
    qDx = @zeros(nx+1, ny)
    qDy = @zeros(nx, ny+1)
    k_ηf = 1.0
    _dx = _dy = 1.0
    _θ = 0.5
    αρgx = αρgy = 0.0

    @parallel compute_flux!(qDx, qDy, P, T, k_ηf, _dx, _dy, _θ, αρgx, αρgy)

    @test maximum(abs.(Array(qDx))) < 1e-12
    @test maximum(abs.(Array(qDy))) < 1e-12

end

@testset "2D reference" begin
    nx=31
    ny=31
    porous_convection_2D(; nx=31, ny=31, nt=3, maxiter=30,ncheck =2, do_viz=false)
    T  = zeros(Float32, nx, ny)
    Tref  = zeros(Float32, nx, ny)
    #read output file  
    load_array("out_T", Tref)
    #read reference file
    load_array("reference2d", T)
    
    @test maximum(abs.(T .- Tref)) < 1e-5
    
    @test all(isfinite, Array(T))

    print("finished reference test\n")

end