using Boscia
using FrankWolfe
using Test
using Random
using FiniteDifferences
using SCIP
using LinearAlgebra
using DataFrames
using CSV
import MathOptInterface
const MOI = MathOptInterface
import HiGHS

# For bug hunting:
#seed = rand(UInt64)
#@show seed
#Random.seed!(seed)

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

#n = 8

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

function birkhoff_boscia(seed, dim; mode="custom", verbose=true, time_limit=1200, write=true)
    @show seed
    Random.seed!(seed)

    f, grad! = build_objective(dim)

    if mode == "custom"
        sblmo = BirkhoffBLMO(true, dim, collect(1:dim^2))

        lower_bounds = fill(0.0, dim^2)
        upper_bounds = fill(1.0, dim^2)

        lmo = Boscia.ManagedBoundedLMO(sblmo, lower_bounds, upper_bounds, collect(1:dim^2), dim^2)
    elseif mode == "mip"
        lmo = build_birkhoff_mip(dim)
    else
        error("Mode not known")
    end

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=verbose, time_limit=time_limit)

    total_time_in_sec=result[:total_time_in_sec]
    status = result[:status]
    if occursin("Optimal", result[:status])
        status = "OPTIMAL"
    elseif occursin("Time", result[:status])
        status = "TIME_LIMIT"
    end

    if write
        lb_list = result[:list_lb]
        ub_list = result[:list_ub]
        time_list = result[:list_time]
        list_lmo_calls = result[:list_lmo_calls_acc]
        list_active_set_size_cb = result[:list_active_set_size] 
        list_discarded_set_size_cb = result[:list_discarded_set_size]
        list_local_tightening = result[:local_tightenings]
        list_global_tightening = result[:global_tightenings]
        df_full = DataFrame(seed=seed, dimension=dim, time=time_list, lowerBound= lb_list, upperBound = ub_list, termination=status, LMOcalls = list_lmo_calls, localTighteings=list_local_tightening, globalTightenings=list_global_tightening, list_active_set_size_cb=list_active_set_size_cb,list_discarded_set_size_cb=list_discarded_set_size_cb)
        file_name_full = joinpath(@__DIR__, "csv/full_run_boscia_" * mode * "_" * string(dim) * "_" *string(seed) * "_birkhoff.csv")
        CSV.write(file_name_full, df_full, append=false)

    
        @show result[:primal_objective]
        df = DataFrame(seed=seed, dimension=dim, time=total_time_in_sec, solution=result[:primal_objective], dual_gap =result[:dual_gap], rel_dual_gap=result[:rel_dual_gap], termination=status, ncalls=result[:lmo_calls])
        file_name = joinpath(@__DIR__,"csv/boscia_" * mode * "_birkhoff_" * string(dim) * "_" * string(seed) * ".csv")
        CSV.write(file_name, df, append=false, writeheader=true)
    end
end

#= @testset "Birkhoff" begin
    f, grad! = build_objective(n)

    #=@testset "Test Derivative" begin
        gradient = rand(n^2)
        @test check_gradients(grad!, f, gradient)
    end =#

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
end =#
