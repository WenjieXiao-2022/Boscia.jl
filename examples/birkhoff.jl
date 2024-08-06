using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
import HiGHS

# For bug hunting:
seed = rand(UInt64)
@show seed
Random.seed!(seed)

include("birkhoff_Blmo.jl")

# min_{X} 1/2 * || X - Xhat ||_F^2
# X ∈ P_n (permutation matrix)

n = 3

function build_objective(n)
    # generate random doubly stochastic matrix
    Xstar = rand(n, n)
    while norm(sum(Xstar, dims=1) .- 1) > 1e-6 || norm(sum(Xstar, dims=2) .- 1) > 1e-6
        Xstar ./= sum(Xstar, dims=1)
        Xstar ./= sum(Xstar, dims=2)
    end

    function f(X)
        return 1/2 * LinearAlgebra.tr(X - Xstar)
    end

    function grad!(storage, X)
        storage .= X - Xstar
        return storage
    end

    return f, grad!
end


function build_birkhoff_mip(n)
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    X = reshape(MOI.add_variables(o, n^2), n, n)

    MOI.add_constraint.(o, X, MOI.ZeroOne())
    # doubly stochastic constraints
    MOI.add_constraint.(
        o,
        vec(sum(X, dims=1, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
        MOI.EqualTo(1.0),
    )
    MOI.add_constraint.(
        o,
        vec(sum(X, dims=2, init=MOI.ScalarAffineFunction{Float64}([], 0.0))),
        MOI.EqualTo(1.0),
    )
    return Boscia.MathOptBLMO(o)
end

@testset "Birkhoff" begin
    f, grad! = build_objective(n)

    x = zeros(n, n)
    @testset "Custom BLMO" begin
        sblmo = BirkhoffBLMO(true, n, collect(1:n^2))

        lower_bounds = fill(0.0, n^2)
        upper_bounds = fill(1.0, n^2)

        x, _, result = Boscia.solve(f, grad!, sblmo, lower_bounds, upper_bounds, collect(1:n^2), n^2, verbose=true)
        @test f(x) <= f(result[:raw_solution]) + 1e-6
        @test is_simple_linear_feasible(sblmo, x)
    end

    x_mip = zeros(n,n)
    @testset "MIP BLMO" begin
        lmo = build_birkhoff_mip(n)

        x_mip, _, result_mip = Boscia.solve(f, grad!, lmo, verbose=true)
        @test f(x_mip) <= f(result_mip[:raw_solution]) + 1e-6
        @test Boscia.is_linear_feasible(lmo, x_mip)
    end 

    @show x
    @show x_mip
    @show f(x), f(x_mip)
    @test isapprox(f(x_mip), f(x), atol=1e-6, rtol=1e-2)
end
