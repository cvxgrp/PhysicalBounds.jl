# ******************************************************************************
# ******************************************************************************
# *  IMPORTANT NOTE:                                                           *
# *  This code ultimately was not used, but it was a cool interface idea for   *
# *  specifying efficiency metric objectives.                                  *
# ******************************************************************************
# ******************************************************************************


# ----------------------------------------------------------
# Variable and QuadraticExpression Atom + related operations
# ----------------------------------------------------------
mutable struct Variable{T <: Number}
    size::Int
    id::UInt64

    function Variable(n::Int; T=Float64)
        n < 0 && throw(DomainError("Variable size must be non negative"))
        this = new{T}(n)
        this.id = objectid(this)
        return this
    end
end

struct VariableAdjoint{T}
    parent::Variable
    function VariableAdjoint(v::Variable{S}) where {S}
        new{S}(v)
    end
end

Base.length(v::Variable) = v.size
Base.length(v_adj::VariableAdjoint) = length(v_adj.parent)
Base.size(v::Variable) = (v.size,)
Base.size(v_adj::VariableAdjoint) where {T <: Number} = (1, v_adj.parent.size)
Base.eltype(::Variable{T}) where {T <: Number} = T
LinearAlgebra.adjoint(v::Variable) = VariableAdjoint(v)
LinearAlgebra.adjoint(v_adj::VariableAdjoint) = v_adj.parent

mutable struct VariableMatrixProduct{T <: Number}
    variable::Variable
    mat::AbstractMatrix

    function VariableMatrixProduct(v::Variable, A::AbstractMatrix{T}) where {T <: Number}
        !issymmetric(A) && error("Matrix must be symmetric")
        (length(v) != size(A, 1)) && throw(DimensionMismatch("Variable and matrix dimension mismatch"))
        return new{T}(v, A)
    end
end

struct VariableMatrixProductAdjoint{T}
    parent::VariableMatrixProduct
    function VariableMatrixProductAdjoint(vmp::VariableMatrixProduct{S}) where {S}
        new{S}(vmp)
    end
end

Base.length(vmp::VariableMatrixProduct) = vmp.parent.size
Base.size(vmp::VariableMatrixProduct) = (vmp.parent.size,)
Base.size(vmp_adj::VariableMatrixProductAdjoint) where {T <: Number} = (1, length(vmp_adj.parent))
Base.eltype(vmp::Variable{T}) where {T <: Number} = T
LinearAlgebra.adjoint(vmp::VariableMatrixProduct) = VariableMatrixProductAdjoint(vmp)
LinearAlgebra.adjoint(vmp_adj::VariableMatrixProductAdjoint) = vmp_adj.parent


mutable struct QuadraticExpression{T <: Number}
    variable::Variable
    mat::AbstractMatrix{T}
    vec::Vector{T}
    off::T

    function QuadraticExpression(v::Variable, A::AbstractMatrix{T}, b::Vector{T}, c::T) where {T <: Number}
        # Error checking
        !issymmetric(A) && error("Matrix must be symmetric")
        (length(v) != size(A,1)  || length(v) != length(b)) && throw(DimensionMismatch("Dimension Mismatch"))

        return new{T}(v, A, b, c)
    end

    function QuadraticExpression(v::Variable, A::T, b::T, c::T) where {T <: Number}
        length(v) != 1 && throw(DimensionMismatch())
        return new{T}(
            v,
            A * ones(T, 1,1),
            [b],
            c
        )
    end
end

# Basic properties
Base.length(quad::QuadraticExpression) = length(quad.variable)
Base.zero(::Type{QuadraticExpression}) = QuadraticExpression(Variable(0), zeros(0,0), Float64[], 0.0)
Base.:-(q::QuadraticExpression) = QuadraticExpression(q.variable, -q.mat, -q.vec, -q.off)

# Evaluation
function evaluate(quad::QuadraticExpression, x::Vecctor{T})
    length(quad) != length(x) && throw(DimensionMismatch("Argument does not match variable length"))
    return dot(x, quad.mat, x) + dot(quad.vec, x) + quad.off
end

# exp = A*x + b
mutable struct LinearVectorExpression{T <: Number}
    variable::Variable
    A::AbstractMatrix{T}
    b::Vector{T}

    function LinearVectorExpression(v::Variable, A::AbstractMatrix{T}, b::Vector{T}) where {T <: Number}
        # Error checking
        (length(v) != size(A,1)  || length(v) != length(b)) && throw(DimensionMismatch("Dimension Mismatch"))
        return new{T}(v, A, b)
    end

    function LinearVectorExpression(v::Variable, A::T, b::T) where {T <: Number}
        length(v) != 1 && throw(DimensionMismatch())
        return new{T}(
            v,
            A * ones(T, 1,1),
            [b],
        )
    end
end

# Basic properties
Base.length(lin::LinearVectorExpression) = length(lin.variable)
Base.zero(::Type{LinearVectorExpression}) = LinearVectorExpression(Variable(0), zeros(0,0), Float64[])
Base.:-(lin::LinearVectorExpression) = QuadraticExpression(lin.variable, -lin.A, -lin.b)

function evaluate(lin::LinearVectorExpression, x::Vecctor{T})
    length(lin) != length(x) && throw(DimensionMismatch("Argument does not match variable length"))
    return lin.A*x + lin.b
end

# gradient: ∇(xᵀAx + bᵀx + c) = 2Ax + b
function gradient(quad::QuadraticExpression)
    return LinearVectorExpression(quad.variable, 2*quad.mat, quad.vec)
end

# Returns a QuadraticExpression that is equivalent to ||LinearVectorExpression||²
function norm_squared(lin::LinearVectorExpression)
    return QuadraticExpression(lin.variable, lin.A'*lin.A, 2*lin.A*lin.b, sum(x->x^2, lin.b))
end


# ********************************
# *          Operations          *
# ********************************

# ------ Variable -> Quad Expression ------
# Scalar Multiplication
function Base.:*(c::T, v::Variable) where {T <: Number}
    n = length(v)
    return QuadraticExpression(v, zeros(T, n, n), c*ones(T, n), zero(T))
end
Base.:*(v::Variable, c::T) where {T <: Number} = c*v

# Vector Dot Product
function LinearAlgebra.dot(a::Vector{T}, v::Variable) where {T <: Number}
    n = length(v)
    return QuadraticExpression(v, zeros(T, n, n), a, zero(T))
end
LinearAlgebra.dot(v::Variable, a::Vector{T}) where {T <: Number} = dot(a, v)
Base.:*(v_adj::VariableAdjoint, a::Vector{T}) where {T <: Number} = dot(a, v_adj.parent)
Base.:*(a_adj::Adjoint{T, Vector{T}}, v::Variable) where {T <: Number} = dot(a_adj.parent, v)

# Quadratic Form
function LinearAlgebra.dot(x::Variable, A::AbstractMatrix{T}, y::Variable) where {T <: Number}
    n = length(x)
    x.id != y.id && error("Bilinear forms not supported yet")
    return QuadraticExpression(x, A, zeros(T, n), zero(T))
end

# Matix-Variable Multiplication
function Base.:*(A::AbstractMatrix{T}, x::Variable) where {T <: Number}
    return VariableMatrixProduct(x, A)
end

function Base.:*(x::VariableAdjoint, A::AbstractMatrix{T}) where {T <: Number}
    return VariableMatrixProduct(x.parent, A)'
end

function Base.:*(vmp_adj::VariableMatrixProductAdjoint{T}, x::Variable) where {T <: Number}
    n = length(x)
    return dot(vmp_adj.parent.variable, vmp_adj.parent.mat, x)
end

function Base.:*(x_adj::VariableAdjoint, vmp::VariableMatrixProduct) where {T <: Number}
    n = length(x_adj)
    return dot(x_adj.parent, vmp.mat, vmp.variable)
end

# TODO: Does this eliminate the need for the VariableMatrixProduct type???
function Base.:*(x_adj::VariableAdjoint, A::AbstractMatrix{T}, x::Variable) where {T <: Number}
    return dot(x_adj.parent, A, x)
end

# Construct norm
function norm_squared(q::QuadraticExpression)
    !iszero(q.mat) && throw(ArgumentError("Expression must be linear"))
    return QuadraticExpression(q.variable, q.mat'*q.mat, -2*q.vec'*q.mat, q.vec'*q.vec)
end


# ------ QuadraticExpression -> QuadraticExpression ------
function Base.:*(c::T, q::QuadraticExpression) where {T <: Number}
    return QuadraticExpression(q.variable, c * q.mat, c * q.vec, c * q.off)
end
Base.:*(q::QuadraticExpression, c::T) where {T <: Number} = c*q

function Base.:+(c::T, q::QuadraticExpression) where {T <: Number}
    return QuadraticExpression(q.variable, q.mat, q.vec, q.off + c)
end
Base.:+(q::QuadraticExpression, c::T) where {T <: Number} = c+q

Base.:-(q::QuadraticExpression, c::T) where {T <: Number} = q + -c
Base.:-(c::T, q::QuadraticExpression) where {T <: Number} = c + -q

function Base.:+(q1::QuadraticExpression, q2::QuadraticExpression)
    length(q1) != length(q2) && error("Quadratic expressions must have the same size")
    q1.variable.id != q2.variable.id && error("Adding expressions of different variables not supported yet")
    return QuadraticExpression(
        q1.variable,
        q1.mat + q2.mat,
        q1.vec + q2.vec,
        q1.off + q2.off
    )
end
Base.:-(q1::QuadraticExpression, q2::QuadraticExpression) = q1 + -q2



mutable struct QuadOverQuadExpression{T <: Number}
    numerator::QuadraticExpression
    denominator::QuadraticExpression

    function QuadOverQuadExpression(q1::QuadraticExpression{T}, q2::QuadraticExpression{T}) where {T <: Number}
        (q1.variable.id != q2.variable.id) && error("Variables in numerator and denominator must be the same")
        return new{T}(q1, q2)
    end
end

Base.length(qq:QuadOverQuadExpression) = length(qq.numerator)
Base.:/(q1::QuadraticExpression, q2::QuadraticExpression) = QuadOverQuadExpression(q1, q2)


mutable struct QuadOverQuadGradient{T <: Number}
    qq::QuadOverQuadExpression{T}
    ∇numerator::LinearVectorExpression{T}
    ∇denominator::LinearVectorExpression{T}
end

function gradient(qq::QuadOverQuadExpression)
    return QuadOverQuadGradient(qq, gradient(qq.numerator), gradient(qq.denominator))
end

# TODO: this could be wayyy more efficient
function evaluate(qqg::QuadOverQuadGradient{T}, x::Vector{T}) where {T <: Number}
    length(qqg) != length(x) && throw(DimensionMismatch("x must have length $(length(qqg.qq))"))
    a = evaluate(qqg.qq.numerator, x)
    b = evaluate(qqg.denominator, x)
    return evaluate(qqg.∇numerator, x) / b - a / b^2 * evaluate(qqg.∇denominator, x)
end
