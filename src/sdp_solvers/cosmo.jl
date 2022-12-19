export build_COSMO_model_primal, build_COSMO_model_dual, optimize!

# https://oxfordcontrol.github.io/COSMO.jl/stable/getting_started/
function build_COSMO_constraints_dual(
    C::AbstractMatrix{T}, 
    A::Vector{S}, 
    b::AbstractVector{T};
    verbose=false
) where {T, S <: AbstractMatrix}
    m = length(A)
    n = size(C, 1)
    N = n * (n+1) ÷ 2

    verbose && @info "--- Building constraint matrix --"

    #pre_allocate (May be uneeded?)
    total_nonzeros = sum(nnz(Ai) for Ai in A)
    M_i, M_j, M_v = Vector{Int}(), Vector{Int}(), Vector{Float64}()
    sizehint!(M_i, total_nonzeros)
    sizehint!(M_j, total_nonzeros)
    sizehint!(M_v, total_nonzeros)
    for k in 1:m
        for (ind, v) in zip(findnz(vec_symm_scs(A[k]))...)
            push!(M_i, ind)
            push!(M_j, k)
            push!(M_v, v)
        end
        verbose && k % 100 == 0 && @info "Done with constraint $k of $(m+1)"
    end
    verbose && @info "Done with constraint $(m+1) of $(m+1)"
    verbose && @info "--- Done with constraints --"

    M = sparse(M_i, M_j, M_v, N, m)
    # NOTE: no minus sign; docs incorrect?
    c_vec = vec_symm_scs(C)
    constraint = COSMO.Constraint(M, c_vec, COSMO.PsdConeTriangle)
    return [constraint]
end


# Dual form model
@doc raw"""
    build_COSMO_model_dual(
        C::AbstractMatrix{T}, 
        A::Vector{S}, 
        b::AbstractVector{T};
        x0=nothing, 
        y0=nothing,
        tol=1e-4, 
        verbose=false
    ) where {T, S <: Union{AbstractMatrix, AbstractSparseMatrix}}

Given an SDP in LMI form
```math
\begin{aligned}
&& \text{minimize} & b^Ty \\
&& \text{subject to} & \sum_{i=1}^m y_iA_i + C \succeq 0 \\
\end{aligned}
```
builds and returns a COSMO model.

The model is warmstarted at primal-dual point `(x0, y0)` and set to have stopping
tolerance `tol`.
"""
function build_COSMO_model_dual(
    C::AbstractMatrix{T}, 
    A::Vector{S}, 
    b::AbstractVector{T};
    x0=nothing, 
    y0=nothing,
    tol=1e-4, 
    verbose=false
) where {T, S <: Union{AbstractMatrix, AbstractSparseMatrix}}
    verbose && @info "Setting up problem"

    # Setup problem data
    n = length(A)
    constraints = build_COSMO_constraints_dual(C, A, b; verbose=verbose)

    model = COSMO.Model()


    # KKT Solvers:
    # CholmodKKTSolver
    # QdldlKKTSolver
    # PardisoDirectKKTSolver
    # PardisoIndirectKKTSolver
    # MKLPardisoKKTSolver
    # CGIndirectKKTSolver
    # MINRESIndirectKKTSolver

    # Merge options are:
    # COSMO.ParentChildMerge
    # COSMO.CliqueGraphMerge
    #   Custom edge weighting functions can be used by defining your own
    #   CustomEdgeWeight <: AbstractEdgeWeight and a corresponding edge_metric method.
    #   By default, the ComplexityWeight <: AbstractEdgeWeight is used which computes
    #   the weight based on the cardinalities of the cliques:

    # Performance Tips:
    # Parameters
        # You could try changing any of the following parameters:
        # rho: The initial algorithm step parameter has a large influence on the convergence. Try different values between 1e-5 and 10.
        # adaptive_rho = false: You can try to disable the automatic rho adaption and use different rho values.
        # adaptive_rho_interval: This specifies after how many iterations COSMO tries to adapt the rho parameter. You can also set adaptive_rho_interval = 0 which adapts the rho parameter after the time spent iterating passes 40% of the factorisation time. This is currently the default in OSQP and works well with QPs.
        # alpha = 1.0: This disables the over-relaxation that is used in the algorithm. We recommend values between 1.0 - 1.6.
        # scaling = 0: This disables the problem scaling.
        # eps_abs and eps_rel: Check the impact of modifying the stopping accuracies.

    settings = COSMO.Settings(
        verbose=verbose,
        eps_abs=tol,
        # eps_rel=tol,
        decompose=false,
        merge_strategy=COSMO.CliqueGraphMerge,
        max_iter=25_000,
        rho=1e-3,
        alpha=1.5,
        adaptive_rho_interval=40,
        # kkt_solver=COSMO.MKLPardisoKKTSolver,
        verbose_timing=verbose,
    )

    # NOTE: this is recommended when using COSMO multithreading. It is mainly
    #       helpful when COSMO can parallelize the cone projections (e.g., if you
    #       are using chordal decomposition, etc)
    # BLAS.set_num_threads(1)

    COSMO.assemble!(
        model,
        spzeros(n, n),                  # P matrix in x'*P*x + q'*x
        b,                                  # q vector in objective (above)
        constraints,
        settings=settings,
        # x0 = 0.5*ones(n),
    )
    !isnothing(x0) && COSMO.warm_start_primal!(model, x0)
    !isnothing(y0) && COSMO.warm_start_dual!(model, y0)

    return model
end


function build_COSMO_constraints_primal(
    C::AbstractMatrix, 
    A::Vector{T}, 
    b::AbstractVector;
    ineq=false, 
    verbose=false
) where {T <: Union{AbstractSparseMatrix, AbstractMatrix}}
    m = length(A)
    n = size(A[1], 1)
    N = (n * (n + 1)) ÷ 2

    function sparse_ijk(total_nonzeros)
        M_i, M_j, M_v = Vector{Int}(), Vector{Int}(), Vector{Float64}()
        sizehint!(M_i, total_nonzeros)
        sizehint!(M_j, total_nonzeros)
        sizehint!(M_v, total_nonzeros)
        return M_i, M_j, M_v
    end

    verbose && @info "--- Building constraint matrix --"

    #pre_allocate (May be uneeded?)
    total_nonzeros = sum(nnz(Ai) for Ai in A)
    M_i, M_j, M_v = sparse_ijk(total_nonzeros)
    for k in 1:m
        for (ind, v) in zip(findnz(vec_symm_scs(A[k]))...)
            push!(M_i, k)
            push!(M_j, ind)
            push!(M_v, v)
        end
        verbose && k % 100 == 0 && @info "Done with constraint $k of $(m+1)"
    end

    M = sparse(M_i, M_j, M_v, m, N)
    # NOTE: minus sign needs to be here despite conflicting with docs
    if ineq
        constraints = [COSMO.Constraint(M, -b, COSMO.Nonnegatives)]
    else
        constraints = [COSMO.Constraint(M, -b, COSMO.ZeroSet)]
    end

    # Identity block for the PSD cone triangle
    M_i, M_j, M_v = sparse_ijk(m)
    for k in 1:N
        push!(M_i, k)
        push!(M_j, k)
        push!(M_v, 1.0)
    end
    M = sparse(M_i, M_j, M_v)
    push!(constraints, COSMO.Constraint(M, spzeros(N), COSMO.PsdConeTriangle))

    verbose && @info "--- Done with constraints --"
    return constraints
end


@doc raw"""
    build_COSMO_model_primal(
        C::AbstractMatrix{T}, 
        A::Vector{S}, 
        b::AbstractVector{T};
        ineq=false,
        x0=nothing,
        y0=nothing,
        tol=1e-4,
        verbose=false
    ) where {T, S <: Union{AbstractSparseMatrix, AbstractMatrix}}

Given an SDP in standard form
```math
\begin{aligned}
&& \text{maximize} & \mathbf{tr}(CX) \\
&& \text{subject to} & \mathbf{tr}(A_iX) = b_i, \qquad i = 1, \dots, m \\
&&& X \succeq 0,
\end{aligned}
```
builds and returns a COSMO model. If `ineq` is `true`, then the equality 
constraints are transformed into inequality constraints $\mathbf{tr}(A_iX) \le b$.

The model is warmstarted at primal-dual point `(x0, y0)` and set to have stopping
tolerance `tol`.
"""
function build_COSMO_model_primal(
    C::AbstractMatrix{T}, 
    A::Vector{S}, 
    b::AbstractVector{T};
    ineq=false,
    x0=nothing,
    y0=nothing,
    tol=1e-4,
    verbose=false
) where {T, S <: Union{AbstractSparseMatrix, AbstractMatrix}}

    verbose && @info "Setting up problem"

    # Setup problem data
    n = length(A)
    constraints = build_COSMO_constraints_primal(C, A, b; ineq=ineq, verbose=verbose)
    q = vec_symm_scs(C)   # objective qᵀx

    model = COSMO.Model()

    # KKT Solvers:
    # CholmodKKTSolver
    # QdldlKKTSolver
    # PardisoDirectKKTSolver
    # PardisoIndirectKKTSolver
    # MKLPardisoKKTSolver
    # CGIndirectKKTSolver
    # MINRESIndirectKKTSolver

    # Merge options are:
    # COSMO.ParentChildMerge
    # COSMO.CliqueGraphMerge
    #   Custom edge weighting functions can be used by defining your own
    #   CustomEdgeWeight <: AbstractEdgeWeight and a corresponding edge_metric method.
    #   By default, the ComplexityWeight <: AbstractEdgeWeight is used which computes
    #   the weight based on the cardinalities of the cliques:

    # Performance Tips:
    # Parameters
        # You could try changing any of the following parameters:
        # rho: The initial algorithm step parameter has a large influence on the convergence. Try different values between 1e-5 and 10.
        # adaptive_rho = false: You can try to disable the automatic rho adaption and use different rho values.
        # adaptive_rho_interval: This specifies after how many iterations COSMO tries to adapt the rho parameter. You can also set adaptive_rho_interval = 0 which adapts the rho parameter after the time spent iterating passes 40% of the factorisation time. This is currently the default in OSQP and works well with QPs.
        # alpha = 1.0: This disables the over-relaxation that is used in the algorithm. We recommend values between 1.0 - 1.6.
        # scaling = 0: This disables the problem scaling.
        # eps_abs and eps_rel: Check the impact of modifying the stopping accuracies.

    settings = COSMO.Settings(
        verbose=verbose,
        eps_abs=tol,
        # eps_rel=tol,
        decompose=true,
        merge_strategy=COSMO.CliqueGraphMerge,
        max_iter=25_000,
        rho=1e-3,
        alpha=1.5,
        adaptive_rho_interval=40,
        # kkt_solver=COSMO.MKLPardisoKKTSolver,
        verbose_timing=verbose,
    )

    # NOTE: this is recommended when using COSMO multithreading. It is mainly
    #       helpful when COSMO can parallelize the cone projections (e.g., if you
    #       are using chordal decomposition, etc)
    # BLAS.set_num_threads(1)
    N = length(q)
    COSMO.assemble!(
        model,
        spzeros(N,N),                  # P matrix in x'*P*x + q'*x
        q,                                  # q vector in objective (above)
        constraints,
        settings=settings,
        # x0 = 0.5*ones(n),
    )
    !isnothing(x0) && COSMO.warm_start_primal!(model, x0)
    !isnothing(y0) && COSMO.warm_start_dual!(model, y0)

    return model
end

"""
    optimize!(model::COSMO.Workspace) = COSMO.optimize!(model)

Calls the COSMO solver on the problem `model`.
"""
optimize!(model::COSMO.Workspace) = COSMO.optimize!(model)