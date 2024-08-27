using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
import HiGHS

include("birkhoff_Blmo.jl")

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

    x = zeros(n, n)
    @testset "Custom BLMO" begin
        sblmo = BirkhoffBLMO(true, n, collect(1:n^2))

        lower_bounds = fill(0.0, n^2)
        upper_bounds = fill(1.0, n^2)

        x, _, result = Boscia.solve(f, grad!, sblmo, lower_bounds, upper_bounds, collect(1:n^2), n^2, verbose=true)
        @test f(x) <= f(result[:raw_solution]) + 1e-6
        @test Boscia.is_simple_linear_feasible(sblmo, x)
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

# test with mixed integer constraint
#=@testset "Birkhoff polytope mixed interger" begin
    n = 4
    d = randn(n, n)
    int_vars = collect(n^2/2:n^2)
    int_num = Int(n^2 / 2 + 1)
    lower_bounds = fill(0.0, int_num)
    upper_bounds = fill(1.0, int_num)
    sblmo = BirkhoffBLMO(true, n, int_vars)
    lmo = build_birkhoff_mip(n; lower_bounds=lower_bounds, upper_bounds=upper_bounds, int_vars=int_vars)
    x = ones(n, n) ./ n
    # test without fixings
    v_if = Boscia.bounded_compute_inface_extreme_point(sblmo, d, x, lower_bounds, upper_bounds, int_vars)
    v_fw = Boscia.bounded_compute_extreme_point(sblmo, d, lower_bounds, upper_bounds, int_vars)
    v_fw_MOI = Boscia.compute_extreme_point(lmo, d)
    v_fw_MOI = vec(v_fw_MOI)
    @test norm(v_fw - v_if) ≤ n * eps()
    @test norm(v_if - v_fw_MOI) ≤ n * eps()
    fixed_col = 2
    fixed_row = 3
    # fix one transition and renormalize
    x2 = copy(x)
    x2[:, fixed_col] .= 0
    x2[fixed_row, :] .= 0
    x2[fixed_row, fixed_col] = 1
    x2 = x2 ./ sum(x2, dims=1)
    v_fixed = Boscia.bounded_compute_inface_extreme_point(sblmo, d, x2, lower_bounds, upper_bounds, int_vars)
    idx = (fixed_col-1)*n+fixed_row
    @test v_fixed[idx] == 1
    # If matrix is already a vertex, away-step can give only itself
    @test norm(Boscia.bounded_compute_inface_extreme_point(sblmo, d, v_fixed, lower_bounds, upper_bounds, int_vars) - v_fixed) ≤ eps()
    # fixed a zero only
    x3 = copy(x)
    x3[4, 3] = 0
    # fixing zeros by creating a cycle 4->3->1->4->4
    x3[4, 4] += 1 / n
    x3[1, 4] -= 1 / n
    x3[1, 3] += 1 / n
    v_zero = Boscia.bounded_compute_inface_extreme_point(sblmo, d, x3, lower_bounds, upper_bounds, int_vars)
    idx = (3-1)*n+4
    @test v_zero[idx] == 0
    idx = (4-1)*n+1
    @test v_zero[idx] == 0
    # test with fixed bounds
    lower_bounds[Int(n^2/2)] = 1.0
    sblmo = BirkhoffBLMO(true, n, int_vars)
    lmo = build_birkhoff_mip(n; lower_bounds=lower_bounds, upper_bounds=upper_bounds, int_vars=int_vars)
end=#
