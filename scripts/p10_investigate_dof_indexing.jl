# Check whether raviartthomas(Γ)/buffachristiansen(Γ) assign a STABLE
# DOF-index -> edge (vertex-index pair) correspondence when only vertex
# COORDINATES change (connectivity/faces identical). This is the
# hidden assumption behind reusing Tyy0/Nxy0 as a frozen preconditioner
# for the perturbed system: if DOF i in the nominal space and DOF i in
# the perturbed space refer to DIFFERENT edges, comparing/reusing
# matrix entries index-by-index is meaningless, and would explain wild
# entry-by-entry differences (scripts/p10_compare_matrix_entries.jl)
# without needing any real geometric-sensitivity argument.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using Exp25_CJH_KC_LocalMultiTrace
using Makeitso
using DrWatson
using CompScienceMeshes
using BEAST
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

X0 = raviartthomas(geo0.Γ)
Xp = raviartthomas(geop.Γ)
Y0 = buffachristiansen(geo0.Γ)
Yp = buffachristiansen(geop.Γ)

println("typeof(X0) = ", typeof(X0))
println("propertynames(X0) = ", propertynames(X0))
println("fieldnames(typeof(X0)) = ", fieldnames(typeof(X0)))

# RWG basis functions in BEAST typically store, per DOF, a list of
# (cellid, refid, coeff) triples referencing mesh cells -- try to
# recover, for each DOF, the SET OF VERTEX INDICES of the cells it
# touches, as a topology-only fingerprint independent of coordinates.
fns0 = X0.fns
fnsp = Xp.fns
println("\ntypeof(X0.fns) = ", typeof(fns0))
println("length(X0.fns) = ", length(fns0))
println("X0.fns[1] = ", fns0[1])

faces0 = geo0.Γ.faces
facesp = geop.Γ.faces
@assert faces0 == facesp

function dof_cell_fingerprint(fns, faces)
    # For each DOF, collect the sorted set of vertex indices from every
    # cell (face) referenced by that DOF's shape-function support.
    fps = Vector{Vector{Int}}(undef, length(fns))
    for (i, shapes) in enumerate(fns)
        vs = Set{Int}()
        for sh in shapes
            cellid = sh.cellid
            for v in faces[cellid]
                push!(vs, v)
            end
        end
        fps[i] = sort(collect(vs))
    end
    return fps
end

fp0 = dof_cell_fingerprint(fns0, faces0)
fpp = dof_cell_fingerprint(fnsp, facesp)

nmismatch = count(i -> fp0[i] != fpp[i], eachindex(fp0))
println("\nRWG (X) DOF count: ", length(fp0))
println("RWG (X) DOFs with mismatched cell/vertex fingerprint (nominal vs perturbed): ", nmismatch, " / ", length(fp0))

if nmismatch > 0
    mismatches = findall(i -> fp0[i] != fpp[i], eachindex(fp0))
    for i in mismatches[1:min(5,end)]
        println("  DOF $i: nominal verts=$(fp0[i])   perturbed verts=$(fpp[i])")
    end
else
    println("  -> RWG DOF-to-cell/vertex mapping is IDENTICAL between nominal and perturbed (as expected from shared connectivity).")
end

# ---- Precise EDGE fingerprint for X (RWG): the 2 vertices SHARED
# between the (exactly 2, for an interior edge) cells referenced by a
# DOF's shapes. This is the canonical, collision-free identity of an
# RWG basis function (one per interior mesh edge).
function dof_edge_fingerprint(fns, faces)
    fps = Vector{Tuple{Int,Int}}(undef, length(fns))
    for (i, shapes) in enumerate(fns)
        cellids = unique(sh.cellid for sh in shapes)
        if length(cellids) == 2
            v1 = Set(faces[cellids[1]])
            v2 = Set(faces[cellids[2]])
            shared = sort(collect(intersect(v1, v2)))
            fps[i] = length(shared) == 2 ? (shared[1], shared[2]) : (-1, -i)
        else
            # boundary/irregular DOF (shouldn't occur on a closed sphere
            # mesh) -- fall back to a unique negative placeholder so it
            # never spuriously matches another DOF.
            fps[i] = (-1, -i)
        end
    end
    return fps
end

edgefp0 = dof_edge_fingerprint(fns0, faces0)
edgefpp = dof_edge_fingerprint(fnsp, facesp)

dictp = Dict{Tuple{Int,Int},Int}()
for (j, fp) in enumerate(edgefpp)
    dictp[fp] = j
end
permX = [get(dictp, edgefp0[i], -1) for i in eachindex(edgefp0)]
nunmatched = count(==(-1), permX)
println("\nEdge-based RWG (X) permutation: unmatched DOFs = ", nunmatched, " / ", length(permX))
println("permX is a valid permutation of 1:n? ", nunmatched == 0 && sort(permX) == collect(1:length(permX)))

# Same investigation for the BC (Y) dual space -- but first check what
# mesh/geometry Y0.fns's cellid actually indexes into (likely a
# barycentrically-refined submesh with MORE cells than the original
# faces list, since the earlier attempt crashed with an out-of-bounds
# cellid against `faces0`).
println("\n" * "="^70)
println("Investigating BC (Y) space internal structure")
println("="^70)
println("propertynames(Y0) = ", propertynames(Y0))
println("fieldnames(typeof(Y0)) = ", fieldnames(typeof(Y0)))
for fld in propertynames(Y0)
    try
        g = getproperty(Y0, fld)
        if fld != :fns
            println("Y0.$fld :: ", typeof(g))
        end
    catch err
        println("Y0.$fld : ERROR ", err)
    end
end

maxcellid = maximum(sh.cellid for shapes in Y0.fns for sh in shapes)
println("\nmax cellid referenced by Y0.fns = ", maxcellid)
println("length(faces0) (original mesh)   = ", length(faces0))

println("\npropertynames(Y0.geo) = ", propertynames(Y0.geo))
println("fieldnames(typeof(Y0.geo)) = ", fieldnames(typeof(Y0.geo)))
for fld in propertynames(Y0.geo)
    g = getproperty(Y0.geo, fld)
    println("Y0.geo.$fld :: ", typeof(g), sizeof(g) < 10000 ? "" : "  (large)")
    if g isa CompScienceMeshes.Mesh
        println("    -> numvertices=", length(vertices(g)), "  numfaces=", length(g))
    end
end

# Y0.geo.parent matches the ORIGINAL (unrefined) mesh; Y0.geo.mesh is
# the barycentric refinement (more vertices/faces). Barycentric
# refinement is expected to preserve original vertex indices as a
# stable prefix (1:n_orig) and append new edge-midpoint/centroid
# vertices afterward -- so filtering a refined cell's vertex indices
# down to the ORIGINAL-vertex range gives a fingerprint comparable
# between the nominal and perturbed refined meshes, even though the
# full refined meshes have different (also-reordered) extra vertices.
n_orig_verts = length(vertices(Y0.geo.parent))
println("\nn_orig_verts (from Y0.geo.parent) = ", n_orig_verts)

refined_faces0 = Y0.geo.mesh.faces
refined_facesp = Yp.geo.mesh.faces

function dof_cell_fingerprint_filtered(fns, refined_faces, n_orig_verts)
    fps = Vector{Vector{Int}}(undef, length(fns))
    for (i, shapes) in enumerate(fns)
        vs = Set{Int}()
        for sh in shapes
            cellid = sh.cellid
            for v in refined_faces[cellid]
                v <= n_orig_verts && push!(vs, v)
            end
        end
        fps[i] = sort(collect(vs))
    end
    return fps
end

fpY0 = dof_cell_fingerprint_filtered(Y0.fns, refined_faces0, n_orig_verts)
fpYp = dof_cell_fingerprint_filtered(Yp.fns, refined_facesp, n_orig_verts)

println("\nsample fpY0[1:5] = ", fpY0[1:5])
println("sample fpYp[1:5] = ", fpYp[1:5])

dictYp = Dict{Vector{Int},Int}()
for (j, fp) in enumerate(fpYp)
    dictYp[fp] = j
end
permY = [get(dictYp, fpY0[i], -1) for i in eachindex(fpY0)]
nunmatchedY = count(==(-1), permY)
println("\nBC (Y) permutation: unmatched DOFs = ", nunmatchedY, " / ", length(permY))
println("permY is a valid permutation of 1:n? ", nunmatchedY == 0 && sort(permY) == collect(1:length(permY)))

if nunmatchedY > 0 || sort(permY) != collect(1:length(permY))
    # If fingerprints collide (multiple DOFs sharing the same filtered
    # vertex set) this naive approach won't produce a clean bijection --
    # report how bad the collision problem is.
    println("  -> fingerprint not unique per DOF; ", length(unique(fpY0)), " unique nominal fingerprints out of ", length(fpY0), " DOFs")
end

# Save both permutations to disk (plain text, one index per line) for
# reuse by the frozen-comparison script -- avoids recomputing
# spaces/fns (and the fingerprint matching) every time.
permfile = projectdir("data", "sweep", "p10_dof_permutations_realization$(realization).txt")
mkpath(dirname(permfile))
open(permfile, "w") do io
    println(io, "# h=$h amplitude=$amplitude lmin=$lmin lmax=$lmax realization=$realization")
    println(io, "# permX (RWG X-space): line i gives the perturbed-space DOF index for nominal DOF i")
    for v in permX
        println(io, v)
    end
    println(io, "# permY (BC Y-space): line i gives the perturbed-space DOF index for nominal DOF i")
    for v in permY
        println(io, v)
    end
end
println("\n[saved] ", permfile)
