
"""
Build node LMO from global LMO

Four action can be taken:
- KEEP   constraint is as saved in the global bounds
- CHANGE lower/upper bound is changed to the node specific one
- DELETE custom bound from the previous node that is invalid at current node and has to be deleted
- ADD    bound has to be added for this node because it does not exist in the global bounds (e.g. variable bound is a half open interval globally) 
"""
function build_LMO(
    blmo::BoundedLinearMinimizationOracle,
    global_bounds::IntegerBounds,
    node_bounds::IntegerBounds,
    int_vars::Vector{Int},
    fixed_int_vars::Vector{Int},
)
    free_model(blmo)

    consLB_list = get_lower_bound_list(blmo)
    consUB_list = get_upper_bound_list(blmo)
    cons_delete = []

    # Lower bounds
    for c_idx in consLB_list
        if is_constraint_on_int_var(blmo, c_idx, int_vars)
            v_idx = get_int_var(blmo, c_idx)
            if is_bound_in(blmo, c_idx, global_bounds.lower_bounds)
                # Change
                if is_bound_in(blmo, c_idx, node_bounds.lower_bounds)
                    set_bound!(blmo, c_idx, node_bounds.lower_bounds[v_idx], :greaterthan)
                    # Keep
                else
                    set_bound!(blmo, c_idx, global_bounds.lower_bounds[v_idx], :greaterthan)
                end
            else
                # Delete
                push!(cons_delete, (c_idx, :greaterthan))
            end
        end
    end

    # Upper bounds
    for c_idx in consUB_list
        if is_constraint_on_int_var(blmo, c_idx, int_vars)
            v_idx = get_int_var(blmo, c_idx)
            if is_bound_in(blmo, c_idx, global_bounds.upper_bounds)
                # Change
                if is_bound_in(blmo, c_idx, node_bounds.upper_bounds)
                    set_bound!(blmo, c_idx, node_bounds.upper_bounds[v_idx], :lessthan)
                    # Keep
                else
                    set_bound!(blmo, c_idx, global_bounds.upper_bounds[v_idx], :lessthan)
                end
            else
                # Delete
                push!(cons_delete, (c_idx, :lessthan))
            end
        end
    end

    # delete constraints
    delete_bounds!(blmo, cons_delete)

    # add node specific constraints 
    # These are bounds constraints where there is no corresponding global bound
    for key in keys(node_bounds.lower_bounds)
        if !haskey(global_bounds.lower_bounds, key)
            add_bound_constraint!(blmo, key, node_bounds.lower_bounds[key], :greaterthan)
        end
    end
    for key in keys(node_bounds.upper_bounds)
        if !haskey(global_bounds.upper_bounds, key)
            add_bound_constraint!(blmo, key, node_bounds.upper_bounds[key], :lessthan)
        end
    end

    add_fixed_int_vars_vals(blmo, fixed_int_vars)

    if blmo isa ManagedBoundedLMO && (blmo.simple_lmo isa BirkhoffBLMO)
        sblmo = blmo.simple_lmo
        dim = sblmo.dim
        n = dim^2
        fixed_to_one_vars = blmo.fixed_int_vars[blmo.fixed_int_vals.==1.0]
        empty!(sblmo.fixed_to_one_rows)
        empty!(sblmo.fixed_to_one_cols)
        for fixed_to_one_var in fixed_to_one_vars
            q = (fixed_to_one_var - 1) ÷ dim
            r = (fixed_to_one_var - 1) - q * dim
            if sblmo.append_by_column
                i = r + 1
                j = q + 1
            else
                i = q + 1
                j = r + 1
            end
            push!(sblmo.fixed_to_one_rows, i)
            push!(sblmo.fixed_to_one_cols, j)
        end

        fixed_to_one_rows = copy(sblmo.fixed_to_one_rows)
        fixed_to_one_cols = copy(sblmo.fixed_to_one_cols)
        nfixed = length(fixed_to_one_rows)
        nreduced = dim - nfixed

        # stores the indices of the original matrix that are still in the reduced matrix
        index_map_rows = fill(1, nreduced)
        index_map_cols = fill(1, nreduced)
        idx_in_map_row = 1
        idx_in_map_col = 1
        for orig_idx in 1:dim
            if orig_idx ∉ fixed_to_one_rows
                index_map_rows[idx_in_map_row] = orig_idx
                idx_in_map_row += 1
            end
            if orig_idx ∉ fixed_to_one_cols
                index_map_cols[idx_in_map_col] = orig_idx
                idx_in_map_col += 1
            end
        end

        empty!(sblmo.index_map_rows)
        empty!(sblmo.index_map_cols)
        append!(sblmo.index_map_rows, index_map_rows)
        append!(sblmo.index_map_cols, index_map_cols)
    end

    return build_LMO_correct(blmo, node_bounds)
end

build_LMO(
    tlmo::TimeTrackingLMO,
    gb::IntegerBounds,
    nb::IntegerBounds,
    int_vars::Vector{Int64},
    fixed_int_vars::Vector{Int64},
) = build_LMO(tlmo.blmo, gb, nb, int_vars, fixed_int_vars)
