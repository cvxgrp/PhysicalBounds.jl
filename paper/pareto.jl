using Pkg
Pkg.activate(@__DIR__)
using Plots
using Random, LinearAlgebra
using PhysicalBounds, WaveOperators
using BSON: @save, @load
using Printf

include("utils.jl")

save_folder_data = joinpath(@__DIR__, "figures", "data")
save_folder_figures = joinpath(@__DIR__, "figures")

## Construct Problem
# Basic parameters. n is the number of gridpoints per unit length
n = 60
# Width and height of the domain 
width, height = 1.6, 1
# Wavenumber of the domain
k = 2π

g = Grid(height, width, 1/n, k)

# ----- Defining things inside of the grid -----

# Add the waveguide slab, centered, of width height/3
contrast = 10
waveguide = Slab(height/2, height/4)
set_contrast!(g, waveguide, contrast)

# Compute the input mode and add it as an input to the grid
mode_input_position = 0
mode_number = 1

input_line = VerticalLine(mode_input_position)
mode = add_mode!(g, input_line, mode_number)

# Add the design, centered on the domain, of size d_height, d_width
# Make the max possible contrast = `contrast`
d_height = height/3
d_width = height/3

x_pos = height/2 - d_height/2
y_pos = width/2 - d_width/2

design_region = Rectangle((x_pos, y_pos), (d_height, d_width))
designidx = getindices(g, design_region)
set_contrast!(g, design_region, contrast)

# ---- Defining the objective ----- 

# Make a target line at the right hand side of the domain
target_line = VerticalLine(width)

# Generate the Green's function for the problem
g_functions = generate_G(g, design_region, target_line)
modes = compute_modes(g, target_line)
output_mode = 2
c = modes.vectors[:, output_mode]
@. c *= sqrt(1 + g.contrast[:,end])
normalize!(c)

# Add contrast to g functions
@. g_functions.b_target *= sqrt(1 + g.contrast[:,end])
g_functions.G_target .= Diagonal(sqrt.(1 .+ g.contrast[:,end])) * g_functions.G_target


## Heuristic solve via LBFGS
Random.seed!(0)
θ₀ = rand(length(designidx))

λs = vcat(0.0, 10 .^(1.:0.5:4), Inf)

save_dict = Dict{Float64, Dict}()
λstr(λ) = λ == Inf ? "Inf" : (λ == 0 ? "0" : "10^$(round(log10(λ); digits=2))")

for λ in λs
    tt = @timed heuristic_solve(LBFGS, g_functions, 100*c; θ₀, λ, factr=1e7, maxiter=500, maxfun=1000, iprint=10)
    fstar, θstar = tt.value
    solve_time = tt.time
    gc_time = tt.gctime

    design = deepcopy(g)
    design.contrast[designidx] .*= θstar
    sol_design = WaveOperators.solve(design)

    target_x = @. (sol_design[:, end] * sqrt(1 + g.contrast[:,end]))
    num = abs(dot(target_x, c))^2
    denom = sum(x -> abs(x)^2, target_x)
    f1 = num / denom
    f2 = num

    save_dict[λ] = Dict(
        "design" => design,
        "sol" => sol_design,
        "fstar" => fstar,
        "efficiency" => f1,
        "power" => f2,
        "thetastar" => θstar, 
        "lambda" => λ,
        "solvetime" => solve_time,
        "gctime" => gc_time
    )

    design_plt = heatmap(
        design.contrast,
        interpolate=false,
        title="Optimal Design (opt for λ = $(λstr(λ)))",
        dpi=300,
        xaxis=nothing,
        yaxis=nothing,
        ylims=(1,n+1),
        aspect_ratio=:equal
    )
    display(design_plt)

    field_plt = heatmap(
        interpolate(
            Matrix(abs.(sol_design)) / maximum(abs.(sol_design)); 
            xlims=(0, width), 
            ylims=(0, height)
        ), 
        interpolate=true,
        dpi=300,
        title="Realized Field (opt for λ = $(λstr(λ)))",
        xaxis=nothing,
        yaxis=nothing,
        aspect_ratio=:equal
    )
    display(field_plt)

    @info "Finished with design λ = $λ,\n\tpurity = $f1, power = $f2"
end

@save joinpath(save_folder_data, "pareto.bson") save_dict
@load joinpath(save_folder_data, "pareto.bson") save_dict

f1s = [save_dict[λ]["efficiency"] for λ in λs]
f2s = [save_dict[λ]["power"] for λ in λs]

p = sortperm(f2s)
pareto_plt = plot(f2s[p]/maximum(f2s), f1s[p],
    ylabel="Purity",
    xlabel="Power (normalized)",
    legend=:bottomleft,
    label="Realized Designs",
    title="Tradeoff Curve",
    dpi=300,
    shape=:diamond,
    lw=2,
    markersize=5,
)
savefig(pareto_plt, joinpath(save_folder_figures, "pareto.pdf"))


using JuMP, COSMO
P, Q, A = PhysicalBounds.construct_matrices(g_functions, c)
Av = vcat(A, [Q])
b = vcat(zeros(length(A)), [1.0])
ineq = vcat(trues(length(A)), [false])
tt = @timed PhysicalBounds.primal_problem_solve(
    -P, Av, b; ineq, optimizer=COSMO.Optimizer()
)
model, X = tt.value
@show solution_summary(model)
bound = -objective_value(model)

plot!(pareto_plt,
    f2s[p]/maximum(f2s),
    bound*ones(length(f2s)),
    label="Purity bound",
    lw=2,
    ls=:dash,
)
savefig(pareto_plt, joinpath(save_folder_figures, "pareto-bound.pdf"))
