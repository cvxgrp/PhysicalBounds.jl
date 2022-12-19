#=
# Bend Design
We show how to compute a heuristic design for a bend.
=#

## Import packages
using WaveOperators, PhysicalBounds, Plots
using Random, LinearAlgebra


#=
## Constructing the design region
We first have to setup the problem data. For a more detailed walkthrough of this
section, checkout the corresponding example in [`WaveOperators.jl`]().
=#
n = 40                                  # Number of gridpoints per unit length
width, height = 2, 2                    # Width and height of the domain
k = 2π                                  # Wavenumber of the domain
g = Grid(height, width, 1/n, k)

## ----- Defining things inside of the grid -----
## Add the waveguide slab, centered, of width height/3
contrast = 5
wg_width = height/6

## Add horizontal waveguide
midpoint_h, midpoint_w = height/2, width/2
input_waveguide = Rectangle((midpoint_h - wg_width/2, 0), (wg_width, width/2))
set_contrast!(g, input_waveguide, contrast)

## Add vertical waveguide
output_waveguide = Rectangle((0, width/2-wg_width/2), (midpoint_h+wg_width/2, wg_width))
set_contrast!(g, output_waveguide, contrast)

## Add design region
side_length = height/4
design_region = Rectangle((height/2 - side_length/2, width/2 - side_length/2), (side_length, side_length))
designidx = getindices(g, design_region)
set_contrast!(g, design_region, contrast)
heatmap(
    repeat(g.contrast, inner=(8,8)),
    interpolate=true,
    title="Design Region",
    dpi=300,
    xaxis=nothing,
    yaxis=nothing
)

#=
Now, we add an input field
=#
## Input mode
input_mode_line = VerticalLine(0.0)
add_mode!(g, input_mode_line, 1)

## Output mode
output_mode_line = HorizontalLine(0.0)
output_mode = @. sqrt(1 + g.contrast[end, :]) * compute_modes(g, output_mode_line).vectors[:, 2]

g_functions = generate_G(g, design_region, output_mode_line)
@. g_functions.b_target *= sqrt(1 + g.contrast[end, :])
g_functions.G_target .= Diagonal(@. sqrt(1 + g.contrast[end, :])) * g_functions.G_target;

#=
We can visualize this initial field. We interpolate for a nicer plot.
=#
using Interpolations: LinearInterpolation
function interpolate(img; factor=8, ylims=(0,2), xlims=(0,2))
    xx = range(xlims..., size(img, 1))
    yy = range(ylims..., size(img, 2))
    itp = LinearInterpolation((xx,yy), img)
    x2 = range(xlims..., size(img, 1)*factor)
    y2 = range(ylims..., size(img, 2)*factor)
    return [itp(x, y) for x in x2, y in y2]
end

sol_initial = WaveOperators.solve(g)
field_plt = heatmap(
    interpolate(
        Matrix(abs.(sol_initial)) / maximum(abs.(sol_initial)); 
        xlims=(0, width), 
        ylims=(0, height)
    ),
    dpi=300,
    title="Initial Field",
    xaxis=nothing,
    yaxis=nothing,
)


#=
## Heuristic deign

We compute a heuristic design using LBFGSB
=#
Random.seed!(0)
θ₀ = rand((0.0,1.0), length(designidx))
fstar, θstar = heuristic_solve(LBFGS, g_functions, output_mode; θ₀, λ=Inf, factr=1e11, maxiter=500, maxfun=1000, iprint=0)

design = deepcopy(g)
design.contrast[designidx] .*= θstar
sol_design = WaveOperators.solve(design)

design_plt = heatmap(
    repeat(design.contrast, inner=(8,8)),
    title="Optimal Design",
    dpi=300,
    xaxis=nothing,
    yaxis=nothing
)

field_plt = heatmap(
    interpolate(
        Matrix(abs.(sol_design)) / maximum(abs.(sol_design)); 
        xlims=(0, width), 
        ylims=(0, height)
    ),
    dpi=300,
    title="Realized Field",
    xaxis=nothing,
    yaxis=nothing,
)

