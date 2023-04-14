@doc raw"""
    contruct_matrices(physics::IntegralEquation, P::M, Q::M) where {M <: AbstractMatrix}

Constructs the matrices ``\bar P, \bar Q, \bar A_i`` as in [[Angeris et al. 2021]](https://arxiv.org/abs/2204.05243)
used to build the bounding semidefinite program.
"""
function construct_matrices(physics::IntegralEquation, c_complex)
    Gd = to_real(physics.G_design)
    Gt = to_real(physics.G_target)
    bd = to_real(physics.b_design)
    bt = to_real(physics.b_target)
    
    nt = size(bt, 1)
    nd = size(bd, 1)
    
    if eltype(c_complex) <: Int
        c_vec = zeros(nt)
        c_vec[c_complex] .= 1.0
        c_vec[c_complex .+ (nt÷2)] .= 1.0
        C = Diagonal(c_vec)
    else
        c = to_real(c_complex)
        cr = @view(c[1:length(c_complex)])
        ci = @view(c[length(c_complex)+1:end])
        C = [cr*cr' + ci*ci'    cr*ci' - ci*cr'; -cr*ci' + ci*cr'   cr*cr' + ci*ci']
    end
    QQ = I

    P̄ = zeros(nd+1, nd+1)
    Q̄ = zeros(nd+1, nd+1)
    Ā = [spzeros(nd+1, nd+1) for _ in 1:nd]


    P = Gt' * C * Gt
    p = Gt' * C * bt
    r = dot(bt, C, bt)
    P̄[1:nd, 1:nd] .= P
    P̄[1:nd, end] .= p
    P̄[end, 1:nd] .=p
    P̄[end, end] = r

    Q = Gt' * QQ * Gt
    q = Gt'*QQ*bt
    s = dot(bt, QQ, bt)
    Q̄[1:nd, 1:nd] .= Q
    Q̄[1:nd, end] .= q
    Q̄[end, 1:nd] .= q
    Q̄[end, end] = s


    for i in 1:nd
        gi = Gd[i, :]
        bi = bd[i]
        @. Ā[i][i, 1:nd] = 0.5 * gi
        @. Ā[i][1:nd, i] += 0.5 * gi
        Ā[i][i ,i] += 1.0
        Ā[i][i, end] = -bi
        Ā[i][end, i] = -bi
    end

    return P̄, Q̄, Ā
end


# https://github.com/cvxgrp/scs
"""
    unvec_symm_scs(x)

Returns a dim-by-dim symmetric matrix corresponding to `x`.
`x` is a vector of length dim*(dim + 1)/2, corresponding to a symmetric matrix
```
X = [ X11     X12/√2 ... X1k/√2
      X21/√2  X22    ... X2k/√2
      ...
      Xk1/√2  Xk2/√2 ... Xkk ],
```
where
`vec(X) = (X11, X12, X22, X13, X23, ..., Xkk)`

Note that the factor √2 preserves inner products:
`x'*c = Tr(unvec_symm(c, dim) * unvec_symm(x, dim))`
"""
function unvec_symm_scs(x::Vector{T}, dim::Int) where {T <: Number}
    X = zeros(T, dim, dim)
    idx = 1
    for i in 1:dim
        for j in 1:i
            if i == j
                X[i,j] = x[idx]
            else
                X[j,i] = X[i,j] = x[idx] / sqrt(2)
            end
            idx += 1
        end
    end
    return X
end

function unvec_symm_scs(x::Vector{T}) where {T <: Number}
    dim = Int( (-1 + sqrt(1 + 8*length(x))) / 2 )
    dim * (dim + 1) ÷ 2 != length(x) && throw(DomainError("invalid vector length"))
    return unvec_symm_scs(x, dim)
end


"""
    vec_symm_scs(X)

Returns a vectorized representation of a symmetric matrix `X`.
`vec(X) = (X11, √2*X12, X22, √2*X13, X23, ..., Xkk)`

Note that the factor √2 preserves inner products:
`x'*c = Tr(unvec_symm(c, dim) * unvec_symm(x, dim))`
"""
function vec_symm_scs(X)
    x_vec = sqrt(2).*X[LinearAlgebra.triu(trues(size(X)))]
    idx = 1
    for i in 1:size(X)[1]
        x_vec[idx] =  x_vec[idx]/sqrt(2)
        idx += i + 1
    end
    return x_vec
end


"""
    vec_symm(X)

Returns a vectorized representation of a symmetric matrix `X`.
`vec(X) = (X11, X12, X22, X13, X23, ..., Xkk)`
"""
vec_symm(X) = X[LinearAlgebra.triu(trues(size(X)))]


"""
    unvec_symm(x, dim)

Returns a dim-by-dim symmetric matrix corresponding to `x`.
`x` is a vector of length dim*(dim + 1)/2, corresponding to a symmetric matrix
```
X = [ X11  X12 ... X1k
      X21  X22 ... X2k
      ...
      Xk1  Xk2 ... Xkk ],
```
where
`vec(X) = (X11, X12, X22, X13, X23, ..., Xkk)`
"""
function unvec_symm(x::Vector{T}, dim::Int) where {T <: Number}
    X = zeros(T, dim, dim)
    idx = 1
    for i in 1:dim, j in 1:i
        X[j,i] = X[i,j] = x[idx]
        idx += 1
    end
    return X
end

"""
    unvec_symm(x)

Returns a dim-by-dim symmetric matrix corresponding to vector `x`, where `dim` 
is the unique positive solution to `dim * (dim + 1) ÷ 2 = length(x)`:
```
X = [ X11  X12 ... X1k
      X21  X22 ... X2k
      ...
      Xk1  Xk2 ... Xkk ],
```
where
`vec(X) = (X11, X12, X22, X13, X23, ..., Xkk)`
"""
function unvec_symm(x::Vector{T}) where {T <: Number}
    dim = Int( (-1 + sqrt(1 + 8*length(x))) / 2 )
    dim * (dim + 1) ÷ 2 != length(x) && throw(DomainError("invalid vector length"))
    return unvec_symm(x, dim)
end