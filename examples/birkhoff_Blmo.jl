## Birkhoff BLMO
using Boscia
using Hungarian
using LinearAlgebra
using SparseArrays

"""
    BirkhoffBLMO

A simple LMO that computes the extreme point given the node specific bounds on the integer variables.
Can be stateless since all of the bound management is done by the ManagedBoundedLMO.   
"""
struct BirkhoffBLMO <: Boscia.SimpleBoundableLMO
    append_by_column::Bool
    dim::Int
    int_vars::Vector{Int}
    atol::Float64
    rtol::Float64
end

BirkhoffBLMO(append_by_column, dim, int_vars)= BirkhoffBLMO(append_by_column, dim, int_vars, 1e-6, 1e-3)

"""
Computes the extreme point given an direction d, the current lower and upper bounds on the integer variables, and the set of integer variables.
"""
function Boscia.bounded_compute_extreme_point(sblmo::BirkhoffBLMO, d, lb, ub, int_vars; kwargs...) 
    n = sblmo.dim

    if size(d,2) == 1
        d = sblmo.append_by_column ? reshape(d, (n,n)) : transpose(reshape(d, (n,n)))
    end

    fixed_to_one_rows = Int[]
    fixed_to_one_cols = Int[]
    delete_ub = Int[]
    for j in 1:n
        for i in 1:n
            if lb[(j-1)*n + i] >= 1 - eps()
                if sblmo.append_by_column
                    push!(fixed_to_one_rows, i)
                    push!(fixed_to_one_cols, j)
                    append!(delete_ub, union(collect((j-1)*n+1:j*n), collect(i:n:n^2)))
                else
                    push!(fixed_to_one_rows, j)
                    push!(fixed_to_one_cols, i)
                    append!(delete_ub, union(collect((i-1)*n+1:i*n), collect(j:n:n^2)))
                end
            end
        end
    end 

    sort!(delete_ub)
    unique!(delete_ub)
    nfixed = length(fixed_to_one_cols)
    nreduced = n - nfixed
    reducedub = copy(ub)
    deleteat!(reducedub, delete_ub)

    # stores the indices of the original matrix that are still in the reduced matrix
    index_map_rows = fill(1, nreduced)
    index_map_cols = fill(1, nreduced)
    idx_in_map_row = 1
    idx_in_map_col = 1
    for orig_idx in 1:n
        if orig_idx ∉ fixed_to_one_rows
            index_map_rows[idx_in_map_row] = orig_idx
            idx_in_map_row += 1
        end
        if orig_idx ∉ fixed_to_one_cols
            index_map_cols[idx_in_map_col] = orig_idx
            idx_in_map_col += 1
        end
    end
    type = typeof(d[1,1])
    d2 = ones(Union{type, Missing}, nreduced, nreduced)
    for j in 1:nreduced
        for i in 1:nreduced
            # interdict arc when fixed to zero
            if reducedub[(j-1)*nreduced + i] <= eps()
                if sblmo.append_by_column
                    d2[i,j] = missing
                else
                    d2[j,i] = missing
                end
            else
                if sblmo.append_by_column
                    d2[i,j] = d[index_map_rows[i], index_map_cols[j]]
                else
                    d2[j,i] = d[index_map_rows[j], index_map_cols[i]]
                end
            end
        end
    end
    m = SparseArrays.spzeros(n, n)
    for (i, j) in zip(fixed_to_one_rows, fixed_to_one_cols)
        m[i, j] = 1
    end
    res_mat = Hungarian.munkres(d2)
    (rows, cols, vals) = SparseArrays.findnz(res_mat)
    @inbounds for i in eachindex(cols)
        m[index_map_rows[rows[i]], index_map_cols[cols[i]]] = (vals[i] == 2)
    end

    m = if sblmo.append_by_column
        reduce(vcat, Matrix(m))
    else
        reduce(vcat, LinearAlgebra.transpose(Matrix(m)))
    end
    return m
end

"""
The sum of each row and column has to be equal to 1.
"""
function Boscia.is_simple_linear_feasible(sblmo::BirkhoffBLMO, v::AbstractVector) 
    n = sblmo.dim
    for i in 1:n
        # append by column ? column sum : row sum 
        if !isapprox(sum(v[((i-1)*n+1):(i*n)]), 1.0, atol=1e-6, rtol=1e-3) 
            @debug "Column sum not 1: $(sum(v[((i-1)*n+1):(i*n)]))"
            return false
        end
        # append by column ? row sum : column sum
        if !isapprox(sum(v[i:n:n^2]), 1.0, atol=1e-6, rtol=1e-3)
            @debug "Row sum not 1: $(sum(v[i:n:n^2]))"
            return false
        end
    end
    return true
end 

#=function Boscia.is_simple_linear_feasible(sblmo::BirkhoffBLMO, v::AbstractMatrix) 
    n = sblmo.dim
    for i in 1:n
        # check row sum
        if !isapprox(sum(v[i, 1:n]), 1.0, atol=sblmo.atol, rtol=sblmo.rtol)
            return false
        end
        # check column sum
        if !isapprox(sum(v[1:n, i]), 1.0, atol=sblmo.atol, rtol=sblmo.rtol)
            return false
        end
    end
    return true
end=#