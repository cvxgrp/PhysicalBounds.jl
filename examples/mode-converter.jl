#=
# Mode Converter Bounds and Design
We show how to compute bounds and a heuristic design for a mode converter. This
example is similar to that in the numerical experiments section of
[Bounds on Efficiency Metrics in Photonics](https://arxiv.org/abs/2204.05243).
For the exact code used in the paper, please see the `paper` folder of the
Github repo.
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
# Basic parameters. n is the number of gridpoints per unit length
n = 15                                  # Number of gridpoints per unit length
width, height = 2, 1                    # Width and height of the domain
k = 2π                                  # Wavenumber of the domain
g = Grid(height, width, 1/n, k)

## ----- Defining things inside of the grid -----
## Add the waveguide slab, centered, of width height/3
contrast = 5
waveguide = Slab(height/2, height/3)
set_contrast!(g, waveguide, contrast)

## Compute the input mode and add it as an input to the grid
mode_input_position = 0
mode_number = 1
input_line = VerticalLine(mode_input_position)
mode = add_mode!(g, input_line, mode_number)

## Add the design, centered on the domain, of size d_height, d_width
## Make the max possible contrast = `contrast`
d_height = 2*height/3
d_width = width/2
x_pos = height/2 - d_height/2
y_pos = width/2 - d_width/2

design_region = Rectangle((x_pos, y_pos), (d_height, d_width))
designidx = getindices(g, design_region)
set_contrast!(g, design_region, contrast)


## ---- Defining the objective ----- 
## Make a target line at the right hand side of the domain
target_line = VerticalLine(width)

## Generate the Green's function for the problem
g_functions = generate_G(g, design_region, target_line)
modes = compute_modes(g, target_line)
output_mode = 2
c = @. sqrt(1 + g.contrast[:,end]) * modes.vectors[:, output_mode]
c ./= norm(c);

#=
We can visualize the design region and the initial field (without a design).
=#
deisgnidx = getindices(g, design_region)
heatmap(g.contrast, title="Design Region")
sol_initial = WaveOperators.solve(g)
plt_input = heatmap(abs.(sol_initial), title="Initial Field")

#=
## Heuristic deign

First, we use LBFGS-B to find a heuristic design
=#
## Heuristic solve via LBFGS
Random.seed!(0)
θ₀ = rand(length(designidx))
fstar, θstar = heuristic_solve(LBFGS, g_functions, c; 
    θ₀=θ₀, factr=1e11, maxiter=500, maxfun=1000, iprint=0)
design = deepcopy(g)
design.contrast[designidx] .*= θstar
heatmap(design.contrast, title="Heuristic Design")

#=
We can visualize the resulting field. Note that by only optimizing for purity,
we end up with significantly reduced power at the output.
=#
sol_design = WaveOperators.solve(design)
heatmap(abs.(sol_design), title="Final Field")

#=
## Bound computed using an SDP.
We compare the mode purity of our design to the upper bound.
=#
P, Q, A = PhysicalBounds.construct_matrices(g_functions, c)
Av = vcat(A, [Q])
b = vcat(zeros(length(A)), [1.0])
ineq = vcat(trues(length(A)), [false])
model, X = PhysicalBounds.primal_problem_solve(-P, Av, b; ineq, optimizer=SCS.Optimizer())
purity_bound = -objective_value(model)
purity = abs(c'*sol_design[:, end])^2 / norm(sol_design[:, end])^2
println("Purtity = $purity")
println("Purity bound = $purity_bound")
#=
We see that our heuristic design is quite close to the upper bound.
=#
