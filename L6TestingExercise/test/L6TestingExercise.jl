using Test
include("../scripts/diffusion_1D_test.jl")


@testset "Diff functiontests" begin
    A=[1.0, 2.0, 4.0, 7.0]
    @test Diff(A) == [1.0, 2.0, 3.0]

    B = [5, 5, 5]
    @test Diff(B) == [0, 0]
end

@testset "Diffusion 1D functiontest" begin
    C, qx = diffusion_1D()
    indices = sort(rand(1:length(C), 20))
    for i in indices
        @test isapprox(C[i], C_t[i]; atol = 1e-8)
        @test isapprox(qx[i], qx_t[i]; atol = 1e-8)
    end
end
