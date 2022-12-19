"""
    generate_random_sdp(n; rand_seed=0)
Generates a random dual form SDP with side dimension `n`:
`min c'*x s.t. sum(F[i]*x[i]) + G ⪰ 0`
Returns `c, F, G, xstar, D`, where `xstar` and `D` are optimal primal and dual
variables respectively
"""
function generate_random_sdp_dual(n; rand_seed=0)
    Random.seed!(rand_seed)

    D = diagm(1 .+ rand(n))
    F = Vector{SparseMatrixCSC{Float64, Int}}(undef, n)
    c = Vector{Float64}(undef, n)
    for i in 1:n
        F[i] = spzeros(n, n)
        block_size = randn() < 1.5 ? 2 : 10
        F[i][i:min(i+block_size,n), i:min(i+block_size,n)] .= 1
        c[i] = tr(D*F[i])
    end
    xstar = rand(n)
    Fx = sum(F[i]*xstar[i] for i in 1:n)
    G = -Fx

    return c, F, G, xstar, D
end


"""
    maxcut_problem(n; rand_seed=0) 

Generates a random instance of the cost matrix for a MAXCUT problem.

"""
function maxcut_problem(n; rand_seed=0, p=0.7)
    Random.seed!(rand_seed)
    Adj_mat = tril(sprand(n, n, p), -1)
    Adj_mat = Adj_mat + Adj_mat'
    C = -0.25*(Diagonal(Adj_mat*ones(n)) - Adj_mat)
    return C
end