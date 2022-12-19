using Pkg
Pkg.activate(@__DIR__)
using Plots
using Random, LinearAlgebra
using PhysicalBounds, WaveOperators
using BSON: @save, @load
using Printf

include("utils.jl")

save_folder_data = joinpath(@__DIR__, "figures-new", "data")
save_folder_figures = joinpath(@__DIR__, "figures-new")

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
prefix = "mode"
Random.seed!(0)
# θ₀ = rand((0.0,1.0), length(designidx))
θ₀ = rand(length(designidx))

for λ in [0.0, Inf]
    filename = iszero(λ) ? "purity" : "power"
    tt = redirect_to_log_files(joinpath(save_folder_data, prefix * "_" * filename * ".log"), joinpath(@__DIR__, prefix * ".err")) do
        @timed heuristic_solve(LBFGS, g_functions, 100*c; θ₀, λ, factr=1e7, maxiter=500, maxfun=1000, iprint=10)
    end
    fstar, θstar = tt.value
    solve_time = tt.time
    gc_time = tt.gctime

    design = deepcopy(g)
    design.contrast[designidx] .*= θstar
    sol_design = WaveOperators.solve(design)

    target_x = @. (sol_design[:, end] * sqrt(1 + g.contrast[:,end]))
    # target_x = sol_design[:, end]
    num = abs(dot(target_x, c))^2
    denom = sum(x -> abs(x)^2, target_x)
    f1 = num / denom
    f2 = num

    save_dict = Dict(
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
        title="Optimal Design (opt for $filename)",
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
        title="Realized Field (opt for $filename)",
        xaxis=nothing,
        yaxis=nothing,
        aspect_ratio=:equal
    )
    display(field_plt)

    @save joinpath(save_folder_data, "$(prefix)_$filename.bson") save_dict
    @info "Finished with design λ = $λ,\n\tpurity = $f1, power = $f2"
end


## SDP Bound
using JuMP, COSMO
filename = "bound"
P, Q, A = PhysicalBounds.construct_matrices(g_functions, c)
Av = vcat(A, [Q])
b = vcat(zeros(length(A)), [1.0])
ineq = vcat(trues(length(A)), [false])
tt = redirect_to_log_files(joinpath(save_folder_data, prefix * "_" * filename * ".log"), joinpath(@__DIR__, prefix * ".err")) do
    @timed PhysicalBounds.primal_problem_solve(
        -P, Av, b; ineq, optimizer=COSMO.Optimizer()
    )
end
model, X = tt.value
purity_bound = -objective_value(model)
bound_solve_time = solve_time(model)
@show solution_summary(model)


## Figures & metrics
@inline function rectangle(start_idx, end_idx, height)
    return Shape(
        start_idx .+ [0,end_idx-start_idx,end_idx-start_idx,0], 
        [0,0,height,height]
    )
end

input_plt = plot(
    dpi=300, ylabel="Power", title="Input", xaxis=nothing,
)
output_plt = plot(
    dpi=300, ylabel="Power", title="Output", xaxis=nothing,
)
for λ in [0.0, Inf]
    filename = iszero(λ) ? "purity" : "power"
    @load joinpath(save_folder_data, "$(prefix)_$filename.bson") save_dict

    # Design plot
    design = save_dict["design"]
    design_plt = heatmap(
        repeat(design.contrast, inner=(8,8)),
        interpolate=false,
        title="Optimal Design (opt for $filename)",
        dpi=300,
        xaxis=nothing,
        yaxis=nothing
    )
    display(design_plt)
    savefig(design_plt, joinpath(save_folder_figures, "design_$filename.pdf"))


    # Power plot
    sol_design = save_dict["sol"]
    field_plt = heatmap(
        interpolate(
            Matrix(abs.(sol_design)) / maximum(abs.(sol_design)); 
            xlims=(0, width), 
            ylims=(0, height)
        ), 
        interpolate=true,
        dpi=300,
        title="Realized Field (opt for $filename)",
        xaxis=nothing,
        yaxis=nothing,
    )
    display(field_plt)
    savefig(field_plt, joinpath(save_folder_figures, "field_$filename.pdf"))

    # Add to input and output plots
    plot!(
        output_plt, 
        abs.(sol_design[:, end]),
        axis=:left,
        lw=3,
        label="opt for $filename",
    )
    plot!(
        input_plt, 
        abs.(sol_design[:, 1]),
        axis=:left,
        lw=3,
        label="opt for $filename",
    )
    
    # Metrics
    purity = power = save_dict["efficiency"]
    power = save_dict["power"]
    solve_time = save_dict["solvetime"]
    @info "Optimizing for $filename:\n\t\tpurity = $purity" * 
            "\n\t\tpower = $power\n\t\tsolve time = $(solve_time)"
end

# Add waveguide to input and output plots
waveguide_idx = findall(idx -> g.contrast[idx] > 0, getindices(g, target_line))
plot!(input_plt, 
    rectangle(waveguide_idx[1], waveguide_idx[end], ylims(input_plt)[end]),
    opacity=0.3,
    label="waveguide",
    color=:gray,
)
plot!(output_plt, 
    rectangle(waveguide_idx[1], waveguide_idx[end], ylims(output_plt)[end]),
    opacity=0.3,
    label="waveguide",
    color=:gray,
    left_margin=2Plots.mm
)

display(input_plt)
display(output_plt)
savefig(input_plt, joinpath(save_folder_figures, "input.pdf"))
savefig(output_plt, joinpath(save_folder_figures, "output.pdf"))

@info "SDP Bound:\n\t\tpurity bound = $purity_bound" * 
        "\n\t\tsolve time = $bound_solve_time"
