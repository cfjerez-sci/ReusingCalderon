# Print a small sample of vertices with their nominal and perturbed
# (x,y,z) coordinates side by side, for a quick sanity check of what
# postproc/shape_perturbation.jl actually does to the mesh. Writes to a
# plain text file (in addition to stdout) so the result can be read back
# reliably regardless of terminal scrollback quirks.
#
# Only exercises the `geo` target (mesh generation), fast regardless of
# mesh size.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using ReusingCalderon
using Makeitso
using DrWatson
using CompScienceMeshes
using LinearAlgebra
using Printf
ENV["DRWATSON_WARN_DIRTY"] = "false"

module Sim2
include("../problems/p10_perturbed_sphere.jl")
end

radius      = 1.0
h           = 0.1
lmin, lmax  = 2, 6
amplitude   = 0.03
realization = 1

geo0 = make(Sim2.geo; h, radius, realization=0, lmin, lmax, amplitude)
geop = make(Sim2.geo; h, radius, realization, lmin, lmax, amplitude)

verts0 = geo0.Γ.vertices
vertsp = geop.Γ.vertices
@assert length(verts0) == length(vertsp)
n = length(verts0)

# Skip index 1 (unreferenced placeholder vertex at the origin -- see
# scripts/p10_plot_perturbed_geometry.jl). Sample evenly-spaced indices
# across the rest of the vertex list.
candidates = 2:n
nsample = 10
idxs = [candidates[round(Int, 1 + (i-1)*(length(candidates)-1)/(nsample-1))] for i in 1:nsample]

outfile = projectdir("data", "figures", "p10_sample_vertices.txt")
mkpath(dirname(outfile))

open(outfile, "w") do io
    header = @sprintf("%-6s %-9s %-9s %-9s   %-9s %-9s %-9s   %-8s\n",
        "idx", "x0", "y0", "z0", "xp", "yp", "zp", "disp%%")
    print(io, header); print(stdout, header)
    for i in idxs
        v0 = verts0[i]; vp = vertsp[i]
        disp_pct = (norm(vp) - radius) / radius * 100
        line = @sprintf("%-6d %-9.5f %-9.5f %-9.5f   %-9.5f %-9.5f %-9.5f   %-8.3f\n",
            i, v0[1], v0[2], v0[3], vp[1], vp[2], vp[3], disp_pct)
        print(io, line); print(stdout, line)
    end
end
println("\n[saved] ", outfile)
