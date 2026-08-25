# Shape-perturbation robustness study for Calderon preconditioning --
# CORRECTED version.
#
# scripts/p10_PEC_sphere_perturbation_robustness.jl (v1) found that
# reusing the nominal Tyy0/Nxy0 matrices unchanged as a "frozen"
# preconditioner for a perturbed shape's EFIE system completely failed
# to converge (1500-iteration cap, true residual stuck around 1e-2).
# scripts/p10_investigate_dof_indexing.jl then found the real cause:
# raviartthomas(Γ)/buffachristiansen(Γ) do NOT assign a
# coordinate-independent DOF ordering -- 91% of RWG DOFs referred to a
# DIFFERENT physical edge between two independently-built spaces on
# meshes that share identical connectivity but different vertex
# coordinates. So v1's "frozen" preconditioner was really an
# (effectively scrambled) mis-indexed operator, not a legitimate test
# of whether the nominal Calderon operator itself transfers to a
# perturbed shape.
#
# scripts/p10_PEC_perturbed_kappa_frozen_aligned.jl fixed this for a
# single realization by building an explicit edge/vertex-set
# fingerprint per DOF, matching nominal DOF i to whichever perturbed
# DOF represents the same physical edge, and permuting Tyy0/Nxy0 into
# the perturbed ordering before using them as the preconditioner --
# which converged in 9 iterations, matching the freshly-rebuilt
# Calderon preconditioner almost exactly.
#
# This script reruns the full N-realization campaign with that fix
# applied at every realization, comparing three solve strategies:
#   (a) plain        -- plain GMRES, no preconditioner
#   (b) frozen_aligned -- Calderon GMRES with the nominal Tyy0/Nxy0,
#                         correctly re-indexed via the per-realization
#                         DOF permutation, as left preconditioner
#   (c) fresh         -- Calderon GMRES with a preconditioner rebuilt
#                        from the perturbed geometry's own Tyy'/Nxy'
#                        (gold standard)
#
# Registers to data/sweep/p10_perturbation_summary_v2.csv. Seeded
# (realization -> seed 1000+realization, matching
# postproc/shape_perturbation.jl's convention) for full
# reproducibility; robust to interruption (each realization's row is
# appended to disk immediately); wrapped in try/catch per realization.
#
# Run directly: F5 in VS Code, or
# `julia scripts/p10_PEC_sphere_perturbation_robustness_v2.jl` from the
# project root.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using Exp25_CJH_KC_LocalMultiTrace
using Makeitso
using DrWatson
using CompScienceMeshes
using BEAST
using LinearAlgebra
using Printf
using Dates
ENV["DRWATSON_WARN_DIRTY"] = "false"

module Sim2
include("../problems/p10_perturbed_sphere.jl")
include("../methods/EFIE.jl")
end

include("../methods/EFIE_manual_solves.jl")

radius     = 1.0
h          = 0.1
κ          = 2.0
lmin, lmax = 2, 6
amplitude  = 0.03

nominal_lmin, nominal_lmax, nominal_amplitude = 2, 6, 0.03

N = 20   # <-- full campaign

outdir = projectdir("data", "sweep")
mkpath(outdir)
summary_file = joinpath(outdir, "p10_perturbation_summary_v2.csv")

if !isfile(summary_file)
    open(summary_file, "w") do io
        println(io, "realization,seed,amplitude,dof,assembly_time_s,solve_time_plain_s,iters_plain,converged_plain,solve_time_frozen_aligned_s,iters_frozen_aligned,converged_frozen_aligned,solve_time_fresh_s,iters_fresh,converged_fresh,trueres_plain,trueres_frozen_aligned,trueres_fresh,reldiff_plain_vs_fresh,reldiff_frozen_aligned_vs_fresh")
    end
end

# --- Edge/vertex-set DOF fingerprints (see
# scripts/p10_investigate_dof_indexing.jl for derivation/validation).
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

println("="^70)
println("PEC sphere shape-perturbation robustness study v2 (DOF-aligned) -- started ", now())
println("h=$h  kappa=$κ  realizations=$N  lmin=$lmin  lmax=$lmax  amplitude=$amplitude")
println("="^70)

# --- Nominal (unperturbed) discretization + spaces: built once.
println("\n--- building/retrieving nominal (unperturbed) discretization+spaces ---")
t0 = time()
disc0 = make(Sim2.discretization; h, κ, radius, realization=0,
    lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)
spaces0 = make(Sim2.spaces; h, radius, realization=0,
    lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)
geo0 = make(Sim2.geo; h, radius, realization=0,
    lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)
t_assembly0 = time() - t0
dof0 = length(disc0.vectors.bx)
@printf("  nominal assembly: %.1f s   (dof = %d)\n", t_assembly0, dof0)

Tyy0_dense = Matrix{ComplexF64}(disc0.matrices.Tyy)
Nxy0_dense = Matrix{ComplexF64}(disc0.matrices.Nxy)

faces0 = geo0.Γ.faces
edgefp0 = dof_edge_fingerprint(spaces0.X.fns, faces0)
n_orig_verts0 = length(vertices(spaces0.Y.geo.parent))
fpY0 = dof_cell_fingerprint_filtered(spaces0.Y.fns, spaces0.Y.geo.mesh.faces, n_orig_verts0)
println("  nominal DOF fingerprints built (X: edge-based, Y: filtered-vertex-set).")

for r in 2:N
    # NOTE: realization 1 was already run (smoke test) with a genuine
    # (non-cache-hit) timing, and its row is already in summary_file --
    # starting the loop at 2 avoids re-running it and appending a
    # duplicate row with a bogus fast "assembly_time_s" from Makeitso's
    # on-disk cache (the same restart-timing-corruption pitfall hit
    # earlier in this session's kappa sweep).
    println("\n--- realization $r / $N ---")
    try
        t0 = time()
        discp = make(Sim2.discretization; h, κ, radius, realization=r, lmin, lmax, amplitude)
        spacesp = make(Sim2.spaces; h, radius, realization=r, lmin, lmax, amplitude)
        geop = make(Sim2.geo; h, radius, realization=r, lmin, lmax, amplitude)
        t_assembly = time() - t0
        dof = length(discp.vectors.bx)
        @assert dof == dof0 "DOF mismatch: perturbed=$dof vs nominal=$dof0"
        @printf("  perturbed assembly: %.1f s   (dof = %d)\n", t_assembly, dof)

        # --- Build the DOF permutation for THIS realization.
        facesp = geop.Γ.faces
        edgefpp = dof_edge_fingerprint(spacesp.X.fns, facesp)
        permX, okX = build_permutation(edgefp0, edgefpp)
        n_orig_vertsp = length(vertices(spacesp.Y.geo.parent))
        fpYp = dof_cell_fingerprint_filtered(spacesp.Y.fns, spacesp.Y.geo.mesh.faces, n_orig_vertsp)
        permY, okY = build_permutation(fpY0, fpYp)
        @printf("  permutation valid?  X: %s   Y: %s\n", okX, okY)
        if !(okX && okY)
            error("DOF permutation construction failed for realization $r (okX=$okX, okY=$okY) -- cannot build an aligned frozen preconditioner.")
        end
        invpermX = invperm(permX)
        invpermY = invperm(permY)

        Tyy0_aligned = Tyy0_dense[invpermY, invpermY]
        Nxy0_aligned = Nxy0_dense[invpermX, invpermY]

        Zxxp = discp.matrices.Zxx
        Tyyp = discp.matrices.Tyy
        Nxyp = discp.matrices.Nxy
        bxp  = discp.vectors.bx

        u_p, ch_p, t_p = solve_plain_gmres(Zxxp, bxp)
        @printf("  [plain]          solve: %.2f s  (%d iters, converged=%s)\n", t_p, ch_p.iters, ch_p.isconverged)

        u_a, ch_a, t_a = solve_calderon_gmres(Zxxp, bxp, Tyy0_aligned, Nxy0_aligned)
        @printf("  [frozen-aligned] solve: %.2f s  (%d iters, converged=%s)\n", t_a, ch_a.iters, ch_a.isconverged)

        u_g, ch_g, t_g = solve_calderon_gmres(Zxxp, bxp, Tyyp, Nxyp)
        @printf("  [fresh]          solve: %.2f s  (%d iters, converged=%s)\n", t_g, ch_g.iters, ch_g.isconverged)

        uvec_p = Vector(u_p); uvec_a = Vector(u_a); uvec_g = Vector(u_g)
        bxpvec = Vector(bxp)
        nb = norm(bxpvec)
        trures_p = norm(bxpvec .- Zxxp*uvec_p) / nb
        trures_a = norm(bxpvec .- Zxxp*uvec_a) / nb
        trures_g = norm(bxpvec .- Zxxp*uvec_g) / nb
        reldiff_p = norm(uvec_p .- uvec_g) / norm(uvec_g)
        reldiff_a = norm(uvec_a .- uvec_g) / norm(uvec_g)
        @printf("  true residuals: plain=%.3e  frozen-aligned=%.3e  fresh=%.3e\n", trures_p, trures_a, trures_g)

        open(summary_file, "a") do io
            @printf(io, "%d,%d,%.6g,%d,%.4f,%.4f,%d,%s,%.4f,%d,%s,%.4f,%d,%s,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                r, 1000+r, amplitude, dof, t_assembly,
                t_p, ch_p.iters, ch_p.isconverged,
                t_a, ch_a.iters, ch_a.isconverged,
                t_g, ch_g.iters, ch_g.isconverged,
                trures_p, trures_a, trures_g,
                reldiff_p, reldiff_a)
        end
        println("  [saved] appended to $summary_file")
    catch err
        println("  [ERROR] realization $r failed: ", err)
        println("  ...continuing to next realization")
    end
end

println("\n" * "="^70)
println("Perturbation robustness study v2 complete -- ", now())
println("Summary: ", summary_file)
println("="^70)
