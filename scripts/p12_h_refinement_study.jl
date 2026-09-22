# Mesh-refinement (h-) study for the SIAM manuscript revision: the
# theoretical headline of Lemma "shapepert" + Theorem "main" is that the
# shape-perturbation bound's constant is independent of h, so the frozen
# preconditioner's iteration counts should be essentially FLAT under mesh
# refinement (h-robustness), not just at one mesh size. This script
# validates that directly: sphere, FIXED moderate deformation (amplitude
# 30%, realization 1), kappa=2, sweeping h in {0.2, 0.15, 0.1, 0.075},
# comparing plain GMRES / frozen DOF-aligned Calderon / fresh Calderon.
#
# Also prints full versioninfo() and system stats at startup, for the
# manuscript's reproducibility statement.
#
# Registers to data/sweep/p12_h_refinement_summary.csv.
# Run directly: F5 in VS Code, or Julia: Execute active File in REPL.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using InteractiveUtils
println("="^70)
versioninfo(verbose=false)
println("CPU: ", Sys.cpu_info()[1].model, "  (", Sys.CPU_THREADS, " threads)")
println("RAM: ", round(Sys.total_memory()/2^30, digits=1), " GiB")
println("Julia threads: ", Threads.nthreads())
import Pkg
for (uuid, dep) in Pkg.dependencies()
    dep.name in ("BEAST", "CompScienceMeshes", "Makeitso") &&
        println(dep.name, " v", dep.version)
end
println("="^70)

using ReusingCalderon
using Makeitso
using DrWatson
using CompScienceMeshes
using BEAST
using LinearAlgebra
using Printf
using Dates
ENV["DRWATSON_WARN_DIRTY"] = "false"

module SimH
include("../problems/p10_perturbed_sphere.jl")
include("../methods/EFIE.jl")
end

include("../methods/EFIE_manual_solves.jl")

κ           = 2.0
radius      = 1.0
lmin, lmax  = 2, 6
realization = 1
amplitude   = 0.30

hs = [0.2, 0.15, 0.1, 0.075]

outdir = projectdir("data", "sweep")
mkpath(outdir)
summary_file = joinpath(outdir, "p12_h_refinement_summary.csv")
if !isfile(summary_file)
    open(summary_file, "w") do io
        println(io, "h,dof,assembly_time_nominal_s,assembly_time_perturbed_s,iters_plain,converged_plain,iters_frozen_aligned,converged_frozen_aligned,iters_fresh,converged_fresh,trueres_plain,trueres_frozen_aligned,trueres_fresh")
    end
end

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

println("H-REFINEMENT study -- started ", now())
println("kappa=$κ  amplitude=$amplitude  realization=$realization  hs=$hs")

for h in hs
    println("\n--- h = $h ---")
    try
        t0 = time()
        disc0 = make(SimH.discretization; h, κ, radius, realization=0, lmin, lmax, amplitude=0.0)
        spaces0 = make(SimH.spaces; h, radius, realization=0, lmin, lmax, amplitude=0.0)
        geo0 = make(SimH.geo; h, radius, realization=0, lmin, lmax, amplitude=0.0)
        t_asm0 = time() - t0
        dof = length(disc0.vectors.bx)
        @printf("  nominal assembly:   %.1f s  (dof = %d)\n", t_asm0, dof)

        t0 = time()
        discp = make(SimH.discretization; h, κ, radius, realization, lmin, lmax, amplitude)
        spacesp = make(SimH.spaces; h, radius, realization, lmin, lmax, amplitude)
        geop = make(SimH.geo; h, radius, realization, lmin, lmax, amplitude)
        t_asmp = time() - t0
        @printf("  perturbed assembly: %.1f s\n", t_asmp)

        edgefp0 = dof_edge_fingerprint(spaces0.X.fns, geo0.Γ.faces)
        n0v = length(vertices(spaces0.Y.geo.parent))
        fpY0 = dof_cell_fingerprint_filtered(spaces0.Y.fns, spaces0.Y.geo.mesh.faces, n0v)
        edgefpp = dof_edge_fingerprint(spacesp.X.fns, geop.Γ.faces)
        npv = length(vertices(spacesp.Y.geo.parent))
        fpYp = dof_cell_fingerprint_filtered(spacesp.Y.fns, spacesp.Y.geo.mesh.faces, npv)
        permX, okX = build_permutation(edgefp0, edgefpp)
        permY, okY = build_permutation(fpY0, fpYp)
        @printf("  permutation valid?  X: %s   Y: %s\n", okX, okY)
        if !(okX && okY)
            error("DOF permutation failed at h=$h")
        end

        Tyy0_al = Matrix{ComplexF64}(disc0.matrices.Tyy)[invperm(permY), invperm(permY)]
        Nxy0_al = Matrix{ComplexF64}(disc0.matrices.Nxy)[invperm(permX), invperm(permY)]

        Zxxp = discp.matrices.Zxx
        bxp  = discp.vectors.bx

        u_p, ch_p, t_p = solve_plain_gmres(Zxxp, bxp)
        @printf("  [plain]          %d iters (converged=%s)\n", ch_p.iters, ch_p.isconverged)
        u_a, ch_a, t_a = solve_calderon_gmres(Zxxp, bxp, Tyy0_al, Nxy0_al)
        @printf("  [frozen-aligned] %d iters (converged=%s)\n", ch_a.iters, ch_a.isconverged)
        u_g, ch_g, t_g = solve_calderon_gmres(Zxxp, bxp, discp.matrices.Tyy, discp.matrices.Nxy)
        @printf("  [fresh]          %d iters (converged=%s)\n", ch_g.iters, ch_g.isconverged)

        bxpvec = Vector(bxp); nb = norm(bxpvec)
        tr_p = norm(bxpvec .- Zxxp*Vector(u_p)) / nb
        tr_a = norm(bxpvec .- Zxxp*Vector(u_a)) / nb
        tr_g = norm(bxpvec .- Zxxp*Vector(u_g)) / nb
        @printf("  true residuals: plain=%.3e frozen=%.3e fresh=%.3e\n", tr_p, tr_a, tr_g)

        open(summary_file, "a") do io
            @printf(io, "%.6g,%d,%.2f,%.2f,%d,%s,%d,%s,%d,%s,%.6e,%.6e,%.6e\n",
                h, dof, t_asm0, t_asmp,
                ch_p.iters, ch_p.isconverged,
                ch_a.iters, ch_a.isconverged,
                ch_g.iters, ch_g.isconverged,
                tr_p, tr_a, tr_g)
        end
        println("  [saved]")
    catch err
        println("  [ERROR] h=$h failed: ", err)
        println("  ...continuing")
    end
end

println("\nH-refinement study complete -- ", now())
println("Summary: ", summary_file)
