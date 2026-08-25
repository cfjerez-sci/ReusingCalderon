# Side-by-side comparison of the actual triangulated meshes (surface +
# wireframe, so the triangulation itself is visible, not just a smooth
# shaded surface) for the nominal sphere and one perturbed realization
# used in scripts/p10_PEC_sphere_perturbation_robustness.jl.
#
# Only exercises the `geo` target (mesh generation), not the EFIE
# assembly, so this is fast regardless of mesh size.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using Exp25_CJH_KC_LocalMultiTrace
using Makeitso
using PlotlyJS
using DrWatson
using CompScienceMeshes
using LinearAlgebra
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
faces  = geop.Γ.faces   # identical connectivity to geo0.Γ.faces by construction
@assert geo0.Γ.faces == faces

println("nominal:   ", length(verts0), " vertices, ", length(faces), " triangles")
println("perturbed: ", length(vertsp), " vertices, ", length(faces), " triangles")

function mesh3d_trace(verts, faces; kwargs...)
    x = [v[1] for v in verts]; y = [v[2] for v in verts]; z = [v[3] for v in verts]
    i = [f[1]-1 for f in faces]; j = [f[2]-1 for f in faces]; k = [f[3]-1 for f in faces]
    return PlotlyJS.mesh3d(; x=x, y=y, z=z, i=i, j=j, k=k, kwargs...)
end

# Explicit wireframe (triangle edges) so the mesh discretization itself
# -- not just a smooth surface -- is visible.
function wireframe_trace(verts, faces; kwargs...)
    edges = Set{Tuple{Int,Int}}()
    for f in faces
        a, b, c = f[1], f[2], f[3]
        for (p, q) in ((a,b), (b,c), (c,a))
            push!(edges, p < q ? (p,q) : (q,p))
        end
    end
    xs = Float64[]; ys = Float64[]; zs = Float64[]
    for (p, q) in edges
        vp = verts[p]; vq = verts[q]
        append!(xs, (vp[1], vq[1], NaN))
        append!(ys, (vp[2], vq[2], NaN))
        append!(zs, (vp[3], vq[3], NaN))
    end
    return PlotlyJS.scatter3d(; x=xs, y=ys, z=zs, mode="lines", kwargs...)
end

trace_nom_surf = mesh3d_trace(verts0, faces;
    color="lightblue", opacity=1.0, flatshading=true, showscale=false, scene="scene1",
    lighting=attr(ambient=0.6, diffuse=0.6, specular=0.1), name="nominal")
trace_nom_wire = wireframe_trace(verts0, faces;
    line=attr(color="rgba(0,0,0,0.35)", width=1), scene="scene1", showlegend=false, hoverinfo="skip")

trace_pert_surf = mesh3d_trace(vertsp, faces;
    color="lightsalmon", opacity=1.0, flatshading=true, showscale=false, scene="scene2",
    lighting=attr(ambient=0.6, diffuse=0.6, specular=0.1), name="perturbed")
trace_pert_wire = wireframe_trace(vertsp, faces;
    line=attr(color="rgba(0,0,0,0.35)", width=1), scene="scene2", showlegend=false, hoverinfo="skip")

layout = PlotlyJS.Layout(
    title="Nominal mesh (left, h=$h) vs. perturbed mesh (right, l=$lmin..$lmax, amplitude=$(100*amplitude)%, realization=$realization)",
    scene=attr(domain=attr(x=[0.0, 0.48]), aspectmode="data",
        xaxis=attr(title="x"), yaxis=attr(title="y"), zaxis=attr(title="z")),
    scene2=attr(domain=attr(x=[0.52, 1.0]), aspectmode="data",
        xaxis=attr(title="x"), yaxis=attr(title="y"), zaxis=attr(title="z")),
    showlegend=false)

plt = PlotlyJS.plot([trace_nom_surf, trace_nom_wire, trace_pert_surf, trace_pert_wire], layout)

outfile = projectdir("data", "figures", "p10_mesh_side_by_side.html")
mkpath(dirname(outfile))
savefig(plt, outfile)
println("[saved] ", outfile)
display(plt)
