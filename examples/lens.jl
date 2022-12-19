#=
# Lens Bounds and Design
We show how to compute bounds and a heuristic design for a lens.
=#

## Import packages
using Plots
using Random, LinearAlgebra
using PhysicalBounds, WaveOperators
using JuMP, SCS

#=
## Constructing the design region
We first have to setup the problem data. For a more detailed walkthrough of this
section, checkout the corresponding example in [`WaveOperators.jl`]().

Note that the number of grid points `n` in this example is very low so that it
can efficienctly run during CI.
=#
n = 15                                  # Number of gridpoints per unit length
width, height = 2, 1                    # Width and height of the domain
k = 2π                                  # Wavenumber of the domain
g = Grid(height, width, 1/n, k)

## ----- Defining things inside of the grid -----
## Add the waveguide slab, centered, of width height/3
contrast = 5
d_height = 2*height/3
slab = Slab(height/2, d_height)

## Compute the input add it as a current to the grid
input_position = 0.0
input_line = ShapeSlice(slab, VerticalLine(input_position))
add_current!(g, input_line, 1)

## Add the design, centered on the domain, of size d_height, d_width
## Make the max possible contrast = `contrast`
d_width = width/4
x_pos = height/2 - d_height/2
y_pos = width/2 - d_width/2

design_region = Rectangle((x_pos, y_pos), (d_height, d_width))
set_contrast!(g, design_region, contrast)


## ---- Defining the objective ----- 
## Make a target line at the right hand side of the domain
focal_plane = VerticalLine(width)
focus_slab = Slab(height/2, height/3)
focal_spot = ShapeSlice(slab, focal_plane)

focal_spot_idx_set = Set(getindices(g, focal_spot))
focal_spot_idx = findall(idx -> idx in focal_spot_idx_set, getindices(g, focal_plane))

## Generate the Green's function for the problem
g_functions = generate_G(g, design_region, focal_plane);

#=
We can visualize the design region and the initial field (without a design).
=#
designidx = getindices(g, design_region)
heatmap(g.contrast, title="Design Region")
sol_initial = WaveOperators.solve(g)
plt_input = heatmap(abs.(sol_initial), title="Initial Field")

#=
## Heuristic deign

First, we use LBFGS-B to find a heuristic design
=#
Random.seed!(0)
θ₀ = zeros(length(designidx))
fstar, θstar = heuristic_solve(LBFGS, g_functions, focal_spot_idx; 
    θ₀, factr=0, maxiter=1000, maxfun=1000, iprint=0)
design = deepcopy(g)
design.contrast[designidx] .*= θstar
heatmap(design.contrast, title="Heuristic Design")

#=
We can visualize the resulting field.
=#
sol_design = WaveOperators.solve(design)
heatmap(abs.(sol_design), title="Final Field")

#=
## Bound computed using an SDP.
We compare the efficiency of our design to the upper bound.
=#
P, Q, A = PhysicalBounds.construct_matrices(g_functions, focal_spot_idx)
Av = vcat(A, [Q])
b = vcat(zeros(length(A)), [1.0])
ineq = vcat(trues(length(A)), [false])
model, X = PhysicalBounds.primal_problem_solve(-P, Av, b; ineq, optimizer=SCS.Optimizer())
purity_bound = -objective_value(model)
eff = sum(abs.(sol_design[:, end][focal_spot_idx]).^2) / norm(sol_design[:, end])^2
println("Focusing efficiency: $eff")
println("Efficiency bound = $purity_bound")
#=
We see that our heuristic design is quite close to the upper bound.
=#
