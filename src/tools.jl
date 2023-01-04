"""
    to_real(x)

Converts `x` to real.

If `x` is a `Vector``, returns
the real and imaginary parts stacked: `[real(x)ᵀ imag(x)ᵀ]ᵀ`.

If `x` is a matrix, returns
[real(x) -imag(x) ; imag(x) real(x)]
"""
function to_real end
to_real(v::Union{Vector, SparseVector}) = [
    real.(v)
    imag.(v)
]
to_real(M::AbstractMatrix) = [
    real.(M)    -imag.(M)
    imag.(M)    real.(M)
]

@doc raw"""
    to_complex(x)

Returns a complex vector `y` corresponding to `x`, where `x` is assumed to have
the form
```math 
    x = \begin{bmatrix} \mathbf{Re}(y) \\ \mathbf{Im}(y) \end{bmatrix}.
```
"""
function to_complex end
to_complex(v::Union{Vector, SparseVector}) = (n = length(v)÷2; return v[1:n] + 1im * v[n+1:end])
