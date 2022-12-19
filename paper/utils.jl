using Interpolations: LinearInterpolation

# function redirect_to_log_files(dofunc, outfile, errfile)
#     open(outfile, "w") do out
#         open(errfile, "w") do err
#             redirect_stdout(out) do
#                 redirect_stderr(err) do
#                     dofunc()
#                 end
#             end
#         end
#     end
# end

function redirect_to_log_files(dofunc, outfile, errfile)
    dofunc()
end

function interpolate(img; factor=8, ylims=(0,2), xlims=(0,2))
    xx = range(xlims..., size(img, 1))
    yy = range(ylims..., size(img, 2))
    itp = LinearInterpolation((xx,yy), img)
    x2 = range(xlims..., size(img, 1)*factor)
    y2 = range(ylims..., size(img, 2)*factor)
    return [itp(x, y) for x in x2, y in y2]
end