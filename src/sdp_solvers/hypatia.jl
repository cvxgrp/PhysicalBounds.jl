# transforms an SDP of the form
# min   tr(CX)
# st    tr(AᵢX) = bᵢ
#       X ⪰ 0
# into Hypatia's primal conic form
# min   c^Tx
# st    b - Ax = 0
#       h - Gx ∈ K
@doc raw"""
    build_Hypatia_model_primal(C::AbstractMatrix{T}, A::Vector{S}, b::AbstractVector{T}) where {T <: Real, S <: Union{AbstractSparseMatrix, AbstractMatrix}}

Given an SDP in standard form
```math
\begin{aligned}
&& \text{maximize} & \mathbf{tr}(CX) \\
&& \text{subject to} & \mathbf{tr}(A_iX) = b_i, \qquad i = 1, \dots, m \\
&&& X \succeq 0,
\end{aligned}
```
builds and returns a Hypatia model.
"""
function build_Hypatia_model_primal(
    C::AbstractMatrix{T}, 
    A::Vector{S}, 
    b::AbstractVector{T}
) where {T <: Real, S <: Union{AbstractSparseMatrix, AbstractMatrix}}
    n = size(C, 1)
    m = length(b)
    N = n * (n + 1) ÷ 2

    c = Vector(vec_symm_scs(C))
    A_hypatia = spzeros(T, m, N)
    for i in 1:m
        A_hypatia[i, :] .= vec_symm_scs(A[i])
    end
    G = -I
    h = zeros(N)
    cones = Hypatia.Cones.Cone{T}[Hypatia.Cones.PosSemidefTri{T, T}(N)]
    
    model = Hypatia.Models.Model{T}(c, A_hypatia, b, G, h, cones)
    return model
end


# transforms an SDP of the form
# min   bᵀy
# st    ∑Aᵢyᵢ + C ⪰ 0
# into Hypatia's primal conic form
# min   c^Tx
# st    b - Ax = 0
#       h - Gx ∈ K
@doc raw"""
build_Hypatia_model_dual(C::AbstractMatrix{T}, A::Vector{S}, b::AbstractVector{T}) where {T <: Real, S <: AbstractMatrix{T}}

Given an SDP in LMI form
```math
\begin{aligned}
&& \text{minimize} & b^Ty \\
&& \text{subject to} & \sum_{i=1}^m y_iA_i + C \succeq 0 \\
\end{aligned}
```
builds and returns a Hypatia model.
"""
function build_Hypatia_model_dual(
    C::AbstractMatrix{T}, 
    A::Vector{S}, 
    b::AbstractVector{T}
) where {T <: Real, S <: AbstractMatrix{T}}
    c = copy(b)
    n = size(C, 1)
    m = length(b)
    N = n * (n + 1) ÷ 2
    
    b_hypatia = T[] #zeros(T, 3)
    A_hypatia = zeros(T, 0, m) #zeros(T, m, 3)
    G = spzeros(T, N, m)
    for i in 1:m
        G[:, i] .= -vec_symm_scs(A[i])
    end
    h = Vector(vec_symm_scs(C))
    cones = Hypatia.Cones.Cone{T}[Hypatia.Cones.PosSemidefTri{T, T}(N)]

    model = Hypatia.Models.Model{T}(c, A_hypatia, b_hypatia, G, h, cones)
    return model
end

"""
    optimize(model::Hypatia.Models.Model{T}; verbose=true) where {T}

Calls the Hypatia solver on the problem `model`.
"""
function optimize(model::Hypatia.Models.Model{T}; verbose=true) where {T}
    solver = Hypatia.Solvers.Solver{T}(;verbose=verbose)
    Hypatia.Solvers.load(solver, model)
    Hypatia.Solvers.solve(solver)
    return solver
end

# Utility functions:
# status = Solvers.get_status(solver)
# primal_obj = Solvers.get_primal_obj(solver)
# dual_obj = Solvers.get_dual_obj(solver)
# x = Solvers.get_x(solver)
# y = Solvers.get_y(solver)
# z = Solvers.get_z(solver)
# s = Solvers.get_s(solver)