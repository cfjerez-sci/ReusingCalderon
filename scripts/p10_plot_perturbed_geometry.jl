# Visual sanity check for the shape-perturbation robustness study
# (scripts/p10_PEC_sphere_perturbation_robustness.jl): plot the nominal
# sphere overlaid with the perturbed geometry used in the smoke test
# (realization=1, seed=1001, l=2..6, amplitude=3%), colored by radial
# displacement, so the actual size of the perturbation can be judged by
# eye before interpreting the frozen-vs-fresh preconditioner results.
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
using Printf
ENV["DRWATSON_WARN_DIRTY"] = "false"

module Sim2
include("../problems/p10_perturbed_sphere.jl")
end

radius     = 1.0
h          = 0.1
lmin, lmax = 2, 6
amplitude  = 0.03
realization = 1   # matches the smoke test (seed = 1000+realization = 1001)

geo0 = make(Sim2.geo; h, radius, realization=0, lmin, lmax, amplitude)
geop = make(Sim2.geo; h, radius, realization, lmin, lmax, amplitude)

verts0 = geo0.Γ.vertices
vertsp = geop.Γ.vertices
faces  = geop.Γ.faces   # identical connectivity to geo0.Γ.faces by construction

@assert length(verts0) == length(vertsp)
@assert geo0.Γ.faces == faces

norms0 = norm.(verts0)
@printf("nominal vertex norms: min=%.6f max=%.6f\n", minimum(norms0), maximum(norms0))
nzero = count(r -> r < 1e-8, norms0)
println("nominal vertices with norm < 1e-8: ", nzero)
if nzero > 0
    idxs = findall(r -> r < 1e-8, norms0)
    println("  indices: ", idxs, "  values: ", verts0[idxs])
end

normsp = norm.(vertsp)
nnan = count(isnan, normsp)
println("perturbed vertices with NaN norm: ", nnan)
if nnan > 0
    idxs = findall(isnan, normsp)
    println("  indices: ", idxs[1:min(5,end)], "  nominal verts there: ", verts0[idxs[1:min(5,end)]])
end

reldisp = [ (norm(vertsp[i]) - radius) / radius * 100 for i in eachindex(vertsp) ]

# Exclude unreferenced placeholder vertices (norm(nominal) ~ 0, e.g. index
# 1 here) from statistics/color-scale -- they are not real surface points
# (confirmed above: 1 such vertex, never used by any face), so including
# them would wash out the real -100%..100% color scale down to whatever
# the placeholder's bogus displacement value is.
valid = norms0 .>= 1e-8*radius
reldisp_valid = reldisp[valid]
@printf("radial displacement (%% of radius, %d real surface vertices): min=%.2f  max=%.2f  mean=%.3f  std=%.3f\n",
    count(valid), minimum(reldisp_valid), maximum(reldisp_valid), sum(reldisp_valid)/length(reldisp_valid),
    sqrt(sum((reldisp_valid .- sum(reldisp_valid)/length(reldisp_valid)).^2)/length(reldisp_valid)))
clim = maximum(abs.(reldisp_valid))

function mesh3d_trace(verts, faces; kwargs...)
    x = [v[1] for v in verts]; y = [v[2] for v in verts]; z = [v[3] for v in verts]
    i = [f[1]-1 for f in faces]; j = [f[2]-1 for f in faces]; k = [f[3]-1 for f in faces]
    return PlotlyJS.mesh3d(; x=x, y=y, z=z, i=i, j=j, k=k, kwargs...)
end

trace_nominal = mesh3d_trace(verts0, faces;
    color="lightgray", opacity=0.25, flatshading=false, name="nominal (h=$h)",
    lighting=attr(ambient=0.7, diffuse=0.5, specular=0.05))

trace_perturbed = mesh3d_trace(vertsp, faces;
    intensity=reldisp, colorscale="RdBu", reversescale=true, cmin=-clim, cmax=clim,
    showscale=true, colorbar=attr(title="radial<br>disp. (%)"),
    opacity=1.0, flatshading=false, name="perturbed (realization=$realization)",
    lighting=attr(ambient=0.7, diffuse=0.6, specular=0.1))

plt = PlotlyJS.plot(
    [trace_nominal, trace_perturbed],
    PlotlyJS.Layout(
        title="Nominal sphere (translucent) vs. l=$lmin..$lmax, amplitude=$(100*amplitude)% perturbation (realization $realization)",
        scene=attr(aspectmode="data")))

outfile = projectdir("data", "figures", "p10_perturbed_geometry.html")
mkpath(dirname(outfile))
savefig(plt, outfile)
println("[saved] ", outfile)
display(plt)
