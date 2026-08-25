# Cheap sanity check for geos/fichera.geo + postproc/shape_perturbation_fichera.jl
# BEFORE committing to the expensive full BEM amplitude sweep
# (scripts/p10_fichera_amplitude_sweep.jl). Just loads the geometry,
# reports vertex/face counts, checks the Euler characteristic (should be
# V - E + F = 2 for a closed genus-0 polyhedron -- the Fichera cube is
# topologically a sphere despite being non-convex), checks watertightness
# (every edge shared by exactly 2 triangles), and checks that the
# two-face bump perturbation leaves connectivity untouched and produces a
# valid (non-degenerate) mesh.
#
# Run directly: F5 in VS Code, or
# `julia scripts/p10_fichera_smoketest.jl` from the project root.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using CompScienceMeshes
using DrWatson
using LinearAlgebra
using Printf

include("../postproc/shape_perturbation_fichera.jl")

h = 0.3
println("Loading geos/fichera.geo at h=$h ...")
Γ0 = mesh_fichera(h)

nv = length(Γ0.vertices)
nf = length(Γ0.faces)

# Build edge -> incident-triangle count map to check watertightness &
# compute the number of unique edges (E) for the Euler check.
edgecount = Dict{Tuple{Int,Int},Int}()
for f in Γ0.faces
    for (a, b) in ((f[1], f[2]), (f[2], f[3]), (f[3], f[1]))
        key = a < b ? (a, b) : (b, a)
        edgecount[key] = get(edgecount, key, 0) + 1
    end
end
ne = length(edgecount)
nonmanifold = count(==(1), values(edgecount)) + count(>(2), values(edgecount))
euler = nv - ne + nf

println()
@printf("  vertices V = %d\n", nv)
@printf("  edges    E = %d\n", ne)
@printf("  faces    F = %d\n", nf)
@printf("  Euler char V-E+F = %d   (expect 2 for closed genus-0 surface)\n", euler)
@printf("  non-manifold edges (not shared by exactly 2 triangles) = %d   (expect 0)\n", nonmanifold)

function triangle_area(v1, v2, v3)
    norm(cross(v2 .- v1, v3 .- v1)) / 2
end
minarea0 = minimum(triangle_area(Γ0.vertices[f[1]], Γ0.vertices[f[2]], Γ0.vertices[f[3]]) for f in Γ0.faces)
@printf("  nominal min triangle area = %.4e   (expect > 0)\n", minarea0)

println("\nApplying two-face bump perturbation (x=-1, y=-1), amplitude=0.1 ...")
Γp = perturb_fichera_mesh(Γ0, 0.1)
@assert Γp.faces == Γ0.faces "connectivity changed by perturbation -- DOF alignment would break!"
minareap = minimum(triangle_area(Γp.vertices[f[1]], Γp.vertices[f[2]], Γp.vertices[f[3]]) for f in Γp.faces)
maxdisp = maximum(norm(Γp.vertices[i] .- Γ0.vertices[i]) for i in eachindex(Γ0.vertices))
@printf("  connectivity preserved: true\n")
@printf("  max vertex displacement = %.4f\n", maxdisp)
@printf("  perturbed min triangle area = %.4e   (expect > 0)\n", minareap)

ok = (euler == 2) && (nonmanifold == 0) && (minarea0 > 0) && (minareap > 0)
println()
println(ok ? "SMOKE TEST PASSED" : "SMOKE TEST FAILED -- see above")
