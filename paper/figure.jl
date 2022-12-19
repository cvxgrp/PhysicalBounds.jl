using Plots
using Random, LinearAlgebra
using PhysicalBounds, WaveOperators
using BSON: @save, @load
using Printf
using JuMP

include("utils.jl")

save_folder_figures = joinpath(@__DIR__, "figures-new")
save_folder_data = joinpath(@__DIR__, "figures-new", "data")
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
mode_out = modes.vectors[:, 2]
mode_in = modes.vectors[:, 1]


## Figures
in_fig = plot(
    abs.(mode_in),
    grid=false,
    yaxis=nothing,
    xaxis=nothing,
    legend=false,
    lw=3,
    color=:black,
    # background=false,
    axis=false
)
out_fig = plot(
    abs.(mode_out),
    grid=false,
    yaxis=nothing,
    xaxis=nothing,
    legend=false,
    lw=3,
    color=:black,
    # background=false,
    axis=false
)
savefig(in_fig, joinpath(save_folder_figures, "setup_input.png"))
savefig(out_fig, joinpath(save_folder_figures, "setup_output.png"))

designidx = getindices(g, design_region)
heatmap(g.contrast, title="Design Region")


@load joinpath(save_folder_data, "mode_power.bson") save_dict
design = save_dict["design"]
sol_design = save_dict["sol"]

fig = heatmap( 
    repeat(design.contrast, inner=(8,8)) / maximum(design.contrast),
    # opacity=0.2,
    interpolate=false,
    # title="Optimal Design (opt for $filename)",
    dpi=300,
    xaxis=nothing,
    yaxis=nothing
)

heatmap!(fig,
    interpolate(
        Matrix(abs.(sol_design)) / maximum(abs.(sol_design)); 
        xlims=(0, width), 
        ylims=(0, height)
    ),
    interpolate=true,
    dpi=300,
    opacity=0.8,
    # title="Realized Field (opt for $filename)",
    xaxis=nothing,
    yaxis=nothing,
    zaxis=nothing,
    colorbar=nothing
)

display(fig)
savefig(fig, joinpath(save_folder_figures, "setup.png"))
