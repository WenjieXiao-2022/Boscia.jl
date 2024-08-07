using Boscia
using FrankWolfe
using Test
using Random
using FiniteDifferences
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
import HiGHS

# For bug hunting:
seed = rand(UInt64)
seed = 0x277d2308b1e8c464
@show seed
Random.seed!(seed)

include("birkhoff_Blmo.jl")

"""
Check if the gradient using finite differences matches the grad! provided.
Copied from FrankWolfe package: https://github.com/ZIB-IOL/FrankWolfe.jl/blob/master/examples/plot_utils.jl
"""
function check_gradients(grad!, f, gradient, num_tests=10, tolerance=1.0e-5)
    for i in 1:num_tests
        random_point = rand(length(gradient))
        grad!(gradient, random_point)
        if norm(grad(central_fdm(5, 1), f, random_point)[1] - gradient) > tolerance
            @warn "There is a noticeable difference between the gradient provided and
            the gradient computed using finite differences.:\n$(norm(grad(central_fdm(5, 1), f, random_point)[1] - gradient))"
            return false
        end
    end
    return true
end

# min_{X} 1/2 * || X - Xhat ||_F^2
# X ∈ P_n (permutation matrix)

n = 3

function build_objective(n, append_by_column=true)
    # generate random doubly stochastic matrix
    Xstar = rand(n, n)
    while norm(sum(Xstar, dims=1) .- 1) > 1e-6 || norm(sum(Xstar, dims=2) .- 1) > 1e-6
        Xstar ./= sum(Xstar, dims=1)
        Xstar ./= sum(Xstar, dims=2)
    end

    function f(x)
        X = append_by_column ? reshape(x, (n,n)) : transpose(reshape(x, (n,n)))
        return 1/2 * LinearAlgebra.tr(LinearAlgebra.transpose(X .- Xstar)*(X .- Xstar))
    end

    function grad!(storage, x)
        X = append_by_column ? reshape(x, (n,n)) : transpose(reshape(x, (n,n)))
        storage .= if append_by_column
            reduce(vcat, X .- Xstar)
        else
            reduce(vcat, LinearAlgebra.transpose(X .- Xstar))
        end
        #storage .= X .- Xstar
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

    @testset "Test Derivative" begin
        gradient = rand(n^2)
        @test check_gradients(grad!, f, gradient)
    end

    x = zeros(n, n)
    @testset "Custom BLMO" begin
        sblmo = BirkhoffBLMO(true, n, collect(1:n^2))

        lower_bounds = fill(0.0, n^2)
        upper_bounds = fill(1.0, n^2)

        x, _, result = Boscia.solve(f, grad!, sblmo, lower_bounds, upper_bounds, collect(1:n^2), n^2, verbose=true, print_iter=1)
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
