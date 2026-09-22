# [mesh-consistency fix, step 1 of 3] Export the nominal meshes behind the
# published results to plain CSV, so that they can be shipped with the
# repository and reloaded instead of re-meshed.
#
# WHY. `meshsphere(radius, h)` is not reproducible across sessions: the same
# call has returned 1224, 1206 and (at h=0.1) both 4827 and 4791 degrees of
# freedom on different occasions. Any from-scratch recomputation therefore
# drifts away from the numbers in the paper. Shipping the meshes removes the
# mesher from the reproduction path entirely.
#
# IMPORTANT. Run this with the ORIGINAL, unpatched problem files, so that
# Makeitso serves the CACHED nominal geometries -- the ones the published
# tables were computed on -- rather than meshing afresh. Restore them first:
#
#     cp ~/prob_backup/*.jl problems/
#     julia +1.11 --project scripts/p28_export_base_meshes.jl
#
# then apply the patched problem files, which load these CSVs.
#
# Writes data/meshes/<name>_verts.csv and <name>_faces.csv (faces 1-based),
# the same plain format already used by data/figexport/meshes.

import Pkg
Pkg.activate((@__DIR__) * "/..")

using ReusingCalderon
using Makeitso
using DrWatson
using CompScienceMeshes
using BEAST
using Random
using DelimitedFiles
using Printf
ENV["DRWATSON_WARN_DIRTY"] = "false"

module SimS
include("../problems/p10_perturbed_sphere.jl")
include("../methods/EFIE.jl")
end

module SimF
include("../problems/p10_perturbed_fichera.jl")
include("../methods/EFIE.jl")
end

outdir = projectdir("data", "meshes")
mkpath(outdir)

function export_mesh(name, Γ)
    V = Γ.vertices
    F = Γ.faces
    # The mesh object itself is what `basemesh` reloads: storing it avoids
    # reconstructing CompScienceMeshes' vertex and face types by hand, which
    # needs StaticArrays -- a transitive dependency that is not on the
    # project's load path. The CSVs beside it are the human-readable copy.
    wsave(joinpath(outdir, name * ".jld2"), Dict("mesh" => Γ))
    writedlm(joinpath(outdir, name * "_verts.csv"),
             [v[i] for v in V, i in 1:3], ',')
    writedlm(joinpath(outdir, name * "_faces.csv"),
             [f[i] for f in F, i in 1:3], ',')
    @printf("  %-28s %5d vertices %5d faces   (RWG dof = %d)\n",
            name, length(V), length(F), 3 * length(F) ÷ 2)
end

println("="^74)
println("[p28] exporting nominal meshes from the cached geometries")
println("="^74)

# The sphere levels used by the published tables. h = 0.2 and 0.15 exist in
# two variants in the cache (the Table 4 campaign and the Figure 3 campaign);
# amplitude = 0.0 selects the Table 4 family, which spans all four levels.
println("\nsphere (radius 1.0), amplitude = 0.0, realization = 0:")
for h in [0.2, 0.15, 0.1, 0.075]
    try
        g = make(SimS.geo; h, radius=1.0, realization=0, lmin=2, lmax=6, amplitude=0.0)
        export_mesh("sphere_h$(h)_r1.0", g.Γ)
    catch err
        println("  [skip] h=$h: ", err)
    end
end

println("\nFichera corner, amplitude = 0.0, realization = 0:")
for h in [0.3, 0.15]
    try
        g = make(SimF.geo; h, realization=0, amplitude=0.0)
        export_mesh("fichera_h$(h)", g.Γ)
    catch err
        println("  [skip] h=$h: ", err)
    end
end

println("\n" * "="^74)
println("Written to ", outdir)
println("Check the dof column against the paper: sphere 1224 / 2292 / 4827 /")
println("8172 at h = 0.2 / 0.15 / 0.1 / 0.075, Fichera 1212 at h = 0.3 and")
println("4170 at h = 0.15. A mismatch means the cache was served from a")
println("fresh mesh rather than the published one -- stop and say so.")
println("="^74)
