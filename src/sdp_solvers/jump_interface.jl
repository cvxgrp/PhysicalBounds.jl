@doc raw"""
    primal_problem_solve(C, Av, b; ineq = nothing, optimizer=Hypatia.Optimizer(verbose=false), warmstart_X=nothing)

Solves the SDP
```math
\begin{aligned}
& \text{minimize} && \mathbf{tr}(CX) \\
& \text{subject to} && \mathbf{tr}(A_iX) = b_i, \qquad i = 1, \dots, m \\
&&& X \succeq 0,
\end{aligned}
```
where the $A_i$'s are stored in the vector `Av`, using the specified optimizer.

Returns the model and variable $X$, after running the solver.
"""
function primal_problem_solve(C, Av, b; ineq = nothing, optimizer=Hypatia.Optimizer(verbose=false), warmstart_X=nothing)
    @assert isnothing(ineq) || length(ineq) == length(Av)
    n = size(C, 1)
    
    mod = Model(() -> optimizer)
    @variable(mod, X[1:n, 1:n], PSD)
    if !isnothing(warmstart_X)
        for i in 1:n, j in 1:n
            set_start_value(X[i,j], warmstart_X[i,j])
        end
    end

    ip(A, X) = sum(A .* X)

    function ip(A::T, X) where T <: SparseMatrixCSC
        running_total = AffExpr()
        for (i, j, v) in zip(findnz(A)...)
            add_to_expression!(running_total, v, X[i, j])
        end
        return running_total
    end

    @objective(mod, Min, ip(C, X))

    for (i, A_i) in enumerate(Av)
        if !isnothing(ineq) && ineq[i]
            @constraint(mod, ip(A_i, X) <= b[i])
        else
            @constraint(mod, ip(A_i, X) == b[i])
        end
        # (i == 1 || i % 100 == 0) && @info "Done with constraint $i"
    end
    @info "Finished building model"
    @info "Starting optimizer..."

    JuMP.optimize!(mod)

    return mod, X
end


@doc raw"""
    dual_problem_solve(C, Av, b; optimizer=Hypatia.Optimizer(verbose=false))

Solves the 'dual' SDP
```math
\begin{aligned}
& \text{minimize} && b^Ty \\
& \text{subject to} && \sum_{i=1}^m y_iA_i + C \succeq 0 \\
\end{aligned}
```
where the $A_i$'s are stored in the vector `Av`, using the specified optimizer.

Returns the model and variable $y$, after running the solver.
"""
function dual_problem_solve(C, Av, b; optimizer=Hypatia.Optimizer(verbose=false))
    @info "Setting up problem"
    mod = Model(() -> optimizer)
    m = length(b)
    n = size(C, 1)
    @variable(mod, y[1:m])

    @objective(mod, Min, sum(b .* y))

    S = spzeros(AffExpr, n, n)

    for (i, A_i) in enumerate(Av)
        for (j, k, v) in zip(findnz(A_i)...)
            if iszero(S[j, k])
                S[j, k] = v * y[i]
            else
                add_to_expression!(S[j, k], v, y[i])
            end
        end
    end
    
    for (j, k, v) in zip(findnz(C)...)
        if iszero(S[j, k])
            S[j, k] = C[j,k]
        else
            add_to_expression!(S[j,k], C[j,k])
        end
    end
    
    @constraint(mod, S in PSDCone())

    @info "Passing problem to optimizer"
    JuMP.optimize!(mod)

    return mod, y
end
