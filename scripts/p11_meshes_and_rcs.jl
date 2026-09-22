# Illustrative-figure data generator for the SIAM manuscript: for each of
# the three geometries (sphere, ellipsoid, Fichera cube), export
#   (a) nominal + a highly-perturbed mesh (vertices+faces, as CSV), for a
#       side-by-side mesh comparison figure, and
#   (b) bistatic RCS(theta) curves for the SAME two shapes, where the
#       nominal shape is solved with a fresh Calderon preconditioner and
#       the highly-perturbed shape is solved with the FROZEN (nominal)
#       Calderon preconditioner via the DOF-aligned strategy -- i.e. the
#       exact same preconditioner matrix is reused, yet the resulting
#       scattered field correctly differs, demonstrating that
#       preconditioner reuse affects only GMRES iteration count, not
#       solution accuracy/physics.
#
# "Highly perturbed" = the LARGEST amplitude already tested in the
# corresponding amplitude-sweep table (sphere 50%, ellipsoid 30%, Fichera
# 15%), so the RCS/mesh figures are consistent with the tables and reuse
# already-cached Makeitso assembly targets (fast).
#
# Registers CSVs under data/figures/. Run directly: F5 in VS Code, or
# `julia scripts/p11_meshes_and_rcs.jl` from the project root.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using ReusingCalderon
using Makeitso
using DrWatson
using CompScienceMeshes
using BEAST
using LinearAlgebra
using Printf
using Dates
using DelimitedFiles
ENV["DRWATSON_WARN_DIRTY"] = "false"

module SimS
include("../problems/p10_perturbed_sphere.jl")
include("../methods/EFIE.jl")
end
module SimE
include("../problems/p10_perturbed_ellipsoid.jl")
include("../methods/EFIE.jl")
end
module SimF
include("../problems/p10_perturbed_fichera.jl")
include("../methods/EFIE.jl")
end

include("../methods/EFIE_manual_solves.jl")

outdir_mesh = projectdir("data", "figures", "meshes")
outdir_rcs  = projectdir("data", "figures", "rcs")
mkpath(outdir_mesh)
mkpath(outdir_rcs)

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
            fps[i] = (-1, -i)
        end
    end
    return fps
end

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

function build_permutation(fp0, fpp)
    dictp = Dict{eltype(fp0),Int}()
    for (j, fp) in enumerate(fpp)
        dictp[fp] = j
    end
    perm = [get(dictp, fp0[i], -1) for i in eachindex(fp0)]
    ok = count(==(-1), perm) == 0 && sort(perm) == collect(1:length(perm))
    return perm, ok
end

function export_mesh(name, verts, faces)
    V = reduce(vcat, [ [v[1] v[2] v[3]] for v in verts ])
    F = reduce(vcat, [ [f[1] f[2] f[3]] for f in faces ])
    writedlm(joinpath(outdir_mesh, "$(name)_verts.csv"), V, ',')
    writedlm(joinpath(outdir_mesh, "$(name)_faces.csv"), F, ',')
    println("  [saved] $(name): $(size(V,1)) verts, $(size(F,1)) faces")
end

function rcs_curve(κ, u, X, θ)
    T = Maxwell3D.singlelayer(wavenumber=κ)
    Tfar = BEAST.MWFarField3D(T)
    dirs = [point(sin(t), 0.0, cos(t)) for t in θ]
    F = potential(Tfar, dirs, Vector(u), X)
    return κ^2/(4π) .* abs2.(norm.(F))
end

θ = range(0, π, length=181)

println("="^70)
println("Mesh + RCS export for illustrative figures -- started ", now())
println("="^70)

# ---------------------------------------------------------------- sphere
println("\n--- SPHERE ---")
h_s, κ_s, radius_s = 0.1, 2.0, 1.0
amp_s, real_s, lmin_s, lmax_s = 0.50, 1, 2, 6

geo0 = make(SimS.geo; h=h_s, radius=radius_s, realization=0, lmin=lmin_s, lmax=lmax_s, amplitude=0.0)
geop = make(SimS.geo; h=h_s, radius=radius_s, realization=real_s, lmin=lmin_s, lmax=lmax_s, amplitude=amp_s)
export_mesh("sphere_nominal", geo0.Γ.vertices, geo0.Γ.faces)
export_mesh("sphere_perturbed50pct", geop.Γ.vertices, geop.Γ.faces)

disc0 = make(SimS.discretization; h=h_s, κ=κ_s, radius=radius_s, realization=0, lmin=lmin_s, lmax=lmax_s, amplitude=0.0)
spaces0 = make(SimS.spaces; h=h_s, radius=radius_s, realization=0, lmin=lmin_s, lmax=lmax_s, amplitude=0.0)
discp = make(SimS.discretization; h=h_s, κ=κ_s, radius=radius_s, realization=real_s, lmin=lmin_s, lmax=lmax_s, amplitude=amp_s)
spacesp = make(SimS.spaces; h=h_s, radius=radius_s, realization=real_s, lmin=lmin_s, lmax=lmax_s, amplitude=amp_s)

u0, _, _ = solve_calderon_gmres(disc0.matrices.Zxx, disc0.vectors.bx, disc0.matrices.Tyy, disc0.matrices.Nxy)

edgefp0 = dof_edge_fingerprint(spaces0.X.fns, geo0.Γ.faces)
n0v = length(vertices(spaces0.Y.geo.parent))
fpY0 = dof_cell_fingerprint_filtered(spaces0.Y.fns, spaces0.Y.geo.mesh.faces, n0v)
edgefpp = dof_edge_fingerprint(spacesp.X.fns, geop.Γ.faces)
npv = length(vertices(spacesp.Y.geo.parent))
fpYp = dof_cell_fingerprint_filtered(spacesp.Y.fns, spacesp.Y.geo.mesh.faces, npv)
permX, okX = build_permutation(edgefp0, edgefpp)
permY, okY = build_permutation(fpY0, fpYp)
@assert okX && okY "sphere DOF alignment failed"
Tyy0_al = Matrix{ComplexF64}(disc0.matrices.Tyy)[invperm(permY), invperm(permY)]
Nxy0_al = Matrix{ComplexF64}(disc0.matrices.Nxy)[invperm(permX), invperm(permY)]
up, chp, _ = solve_calderon_gmres(discp.matrices.Zxx, discp.vectors.bx, Tyy0_al, Nxy0_al)
@printf("  frozen-preconditioner solve on 50%%-perturbed sphere: %d iters, converged=%s\n", chp.iters, chp.isconverged)

rcs0 = rcs_curve(κ_s, u0, spaces0.X, θ)
rcsp = rcs_curve(κ_s, up, spacesp.X, θ)
writedlm(joinpath(outdir_rcs, "sphere_rcs.csv"), [collect(θ) rcs0 rcsp], ',')
println("  [saved] sphere RCS curves")

# -------------------------------------------------------------- ellipsoid
println("\n--- ELLIPSOID ---")
h_e, κ_e = 0.1, 2.0
amp_e, real_e, lmin_e, lmax_e = 0.30, 1, 2, 6

geo0 = make(SimE.geo; h=h_e, realization=0, lmin=lmin_e, lmax=lmax_e, amplitude=0.0)
geop = make(SimE.geo; h=h_e, realization=real_e, lmin=lmin_e, lmax=lmax_e, amplitude=amp_e)
export_mesh("ellipsoid_nominal", geo0.Γ.vertices, geo0.Γ.faces)
export_mesh("ellipsoid_perturbed30pct", geop.Γ.vertices, geop.Γ.faces)

disc0 = make(SimE.discretization; h=h_e, κ=κ_e, realization=0, lmin=lmin_e, lmax=lmax_e, amplitude=0.0)
spaces0 = make(SimE.spaces; h=h_e, realization=0, lmin=lmin_e, lmax=lmax_e, amplitude=0.0)
discp = make(SimE.discretization; h=h_e, κ=κ_e, realization=real_e, lmin=lmin_e, lmax=lmax_e, amplitude=amp_e)
spacesp = make(SimE.spaces; h=h_e, realization=real_e, lmin=lmin_e, lmax=lmax_e, amplitude=amp_e)

u0, _, _ = solve_calderon_gmres(disc0.matrices.Zxx, disc0.vectors.bx, disc0.matrices.Tyy, disc0.matrices.Nxy)

edgefp0 = dof_edge_fingerprint(spaces0.X.fns, geo0.Γ.faces)
n0v = length(vertices(spaces0.Y.geo.parent))
fpY0 = dof_cell_fingerprint_filtered(spaces0.Y.fns, spaces0.Y.geo.mesh.faces, n0v)
edgefpp = dof_edge_fingerprint(spacesp.X.fns, geop.Γ.faces)
npv = length(vertices(spacesp.Y.geo.parent))
fpYp = dof_cell_fingerprint_filtered(spacesp.Y.fns, spacesp.Y.geo.mesh.faces, npv)
permX, okX = build_permutation(edgefp0, edgefpp)
permY, okY = build_permutation(fpY0, fpYp)
@assert okX && okY "ellipsoid DOF alignment failed"
Tyy0_al = Matrix{ComplexF64}(disc0.matrices.Tyy)[invperm(permY), invperm(permY)]
Nxy0_al = Matrix{ComplexF64}(disc0.matrices.Nxy)[invperm(permX), invperm(permY)]
up, chp, _ = solve_calderon_gmres(discp.matrices.Zxx, discp.vectors.bx, Tyy0_al, Nxy0_al)
@printf("  frozen-preconditioner solve on 30%%-perturbed ellipsoid: %d iters, converged=%s\n", chp.iters, chp.isconverged)

rcs0 = rcs_curve(κ_e, u0, spaces0.X, θ)
rcsp = rcs_curve(κ_e, up, spacesp.X, θ)
writedlm(joinpath(outdir_rcs, "ellipsoid_rcs.csv"), [collect(θ) rcs0 rcsp], ',')
println("  [saved] ellipsoid RCS curves")

# ---------------------------------------------------------------- fichera
println("\n--- FICHERA CUBE ---")
h_f, κ_f = 0.3, 2.0
amp_f, real_f = 0.15, 1

geo0 = make(SimF.geo; h=h_f, realization=0, amplitude=0.0)
geop = make(SimF.geo; h=h_f, realization=real_f, amplitude=amp_f)
export_mesh("fichera_nominal", geo0.Γ.vertices, geo0.Γ.faces)
export_mesh("fichera_perturbed15pct", geop.Γ.vertices, geop.Γ.faces)

disc0 = make(SimF.discretization; h=h_f, κ=κ_f, realization=0, amplitude=0.0)
spaces0 = make(SimF.spaces; h=h_f, realization=0, amplitude=0.0)
discp = make(SimF.discretization; h=h_f, κ=κ_f, realization=real_f, amplitude=amp_f)
spacesp = make(SimF.spaces; h=h_f, realization=real_f, amplitude=amp_f)

u0, _, _ = solve_calderon_gmres(disc0.matrices.Zxx, disc0.vectors.bx, disc0.matrices.Tyy, disc0.matrices.Nxy)

edgefp0 = dof_edge_fingerprint(spaces0.X.fns, geo0.Γ.faces)
n0v = length(vertices(spaces0.Y.geo.parent))
fpY0 = dof_cell_fingerprint_filtered(spaces0.Y.fns, spaces0.Y.geo.mesh.faces, n0v)
edgefpp = dof_edge_fingerprint(spacesp.X.fns, geop.Γ.faces)
npv = length(vertices(spacesp.Y.geo.parent))
fpYp = dof_cell_fingerprint_filtered(spacesp.Y.fns, spacesp.Y.geo.mesh.faces, npv)
permX, okX = build_permutation(edgefp0, edgefpp)
permY, okY = build_permutation(fpY0, fpYp)
@assert okX && okY "fichera DOF alignment failed"
Tyy0_al = Matrix{ComplexF64}(disc0.matrices.Tyy)[invperm(permY), invperm(permY)]
Nxy0_al = Matrix{ComplexF64}(disc0.matrices.Nxy)[invperm(permX), invperm(permY)]
up, chp, _ = solve_calderon_gmres(discp.matrices.Zxx, discp.vectors.bx, Tyy0_al, Nxy0_al)
@printf("  frozen-preconditioner solve on 15%%-perturbed Fichera cube: %d iters, converged=%s\n", chp.iters, chp.isconverged)

rcs0 = rcs_curve(κ_f, u0, spaces0.X, θ)
rcsp = rcs_curve(κ_f, up, spacesp.X, θ)
writedlm(joinpath(outdir_rcs, "fichera_rcs.csv"), [collect(θ) rcs0 rcsp], ',')
println("  [saved] fichera RCS curves")

println("\n" * "="^70)
println("Mesh + RCS export complete -- ", now())
println("Output dirs: $outdir_mesh , $outdir_rcs")
println("="^70)
