# Third-geometry validation for the SIAM manuscript: does the DOF-aligned
# frozen Calderon preconditioner's near-zero robustness to shape
# perturbation (established on a sphere and a triaxial ellipsoid, both
# smooth/convex) also hold on a non-convex, non-smooth (single reentrant
# corner) polyhedron under a qualitatively different, LOCALIZED flat-face
# perturbation -- or is the earlier near-zero-penalty result an artifact
# of smooth/global perturbations of smooth/convex shapes?
#
# Nominal shape: Fichera cube = [-1,1]^3 \ [0,1]^3
# (problems/p10_perturbed_fichera.jl, geos/fichera.geo). Perturbation:
# smooth bump/dent, tapered to exactly zero at face boundaries, applied to
# the two full square faces x=-1 and y=-1 only (postproc/
# shape_perturbation_fichera.jl) -- deliberately far from the reentrant
# corner, so the perturbation's effect can be cleanly separated from the
# geometry's intrinsic non-convex singularity.
#
# Registers to data/sweep/p10_fichera_amplitude_sweep_summary.csv.
# Wrapped in try/catch per amplitude; robust to interruption.
#
# Run directly: F5 in VS Code, or
# `julia scripts/p10_fichera_amplitude_sweep.jl` from the project root.

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
ENV["DRWATSON_WARN_DIRTY"] = "false"

module SimF
include("../problems/p10_perturbed_fichera.jl")
include("../methods/EFIE.jl")
end

include("../methods/EFIE_manual_solves.jl")

# [CJ-15] h, kappa and the amplitude list are now arguments, so that this
# flat-face sweep can be run at the same resolution as the corner sweep of
# scripts/p26_fichera_corner_sweep.jl and serve as its matched control.
# Defaults reproduce the published run exactly. Output goes to the original
# file name only at the original h = 0.3; any other h writes to its own
# file, so the data behind the manuscript's Fichera table cannot be mixed
# with control runs.
function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    (i === nothing || i == length(ARGS)) && return default
    return ARGS[i+1]
end

h   = parse(Float64, argval("--h", "0.3"))
κ   = parse(Float64, argval("--kappa", "2.0"))

amplitudes = parse.(Float64, split(argval("--amps", "0.02,0.05,0.10,0.15"), ","))

outdir = projectdir("data", "sweep")
mkpath(outdir)
summary_file = joinpath(outdir,
    h == 0.3 ? "p10_fichera_amplitude_sweep_summary.csv"
             : "p10_fichera_amplitude_sweep_summary_h$(h).csv")
if !isfile(summary_file)
    open(summary_file, "w") do io
        println(io, "amplitude,dof,maxdisp,min_triangle_area,valid_mesh,assembly_time_s,solve_time_plain_s,iters_plain,converged_plain,solve_time_frozen_aligned_s,iters_frozen_aligned,converged_frozen_aligned,solve_time_fresh_s,iters_fresh,converged_fresh,trueres_plain,trueres_frozen_aligned,trueres_fresh")
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

function triangle_area(v1, v2, v3)
    return norm(cross(v2 .- v1, v3 .- v1)) / 2
end

function min_triangle_area(verts, faces)
    m = Inf
    for f in faces
        a = triangle_area(verts[f[1]], verts[f[2]], verts[f[3]])
        m = min(m, a)
    end
    return m
end

println("="^70)
println("FICHERA CUBE perturbation amplitude sweep (DOF-aligned frozen strategy) -- started ", now())
println("h=$h  kappa=$κ  perturbed faces: x=-1, y=-1  amplitudes=$amplitudes")
println("="^70)

println("\n--- building/retrieving nominal (unperturbed) Fichera discretization+spaces ---")
t0 = time()
disc0 = make(SimF.discretization; h, κ, realization=0, amplitude=0.0)
spaces0 = make(SimF.spaces; h, realization=0, amplitude=0.0)
geo0 = make(SimF.geo; h, realization=0, amplitude=0.0)
t_assembly0 = time() - t0
dof0 = length(disc0.vectors.bx)
@printf("  nominal assembly: %.1f s   (dof = %d)\n", t_assembly0, dof0)

Tyy0_dense = Matrix{ComplexF64}(disc0.matrices.Tyy)
Nxy0_dense = Matrix{ComplexF64}(disc0.matrices.Nxy)

faces0 = geo0.Γ.faces
edgefp0 = dof_edge_fingerprint(spaces0.X.fns, faces0)
n_orig_verts0 = length(vertices(spaces0.Y.geo.parent))
fpY0 = dof_cell_fingerprint_filtered(spaces0.Y.fns, spaces0.Y.geo.mesh.faces, n_orig_verts0)
println("  nominal DOF fingerprints built.")

verts0 = geo0.Γ.vertices

for amp in amplitudes
    println("\n--- amplitude = $amp ---")
    try
        t0 = time()
        discp = make(SimF.discretization; h, κ, realization=1, amplitude=amp)
        spacesp = make(SimF.spaces; h, realization=1, amplitude=amp)
        geop = make(SimF.geo; h, realization=1, amplitude=amp)
        t_assembly = time() - t0
        dof = length(discp.vectors.bx)
        @assert dof == dof0 "DOF mismatch: perturbed=$dof vs nominal=$dof0"
        @printf("  perturbed assembly: %.1f s   (dof = %d)\n", t_assembly, dof)

        vertsp = geop.Γ.vertices
        maxdisp = maximum(norm(vertsp[i] .- verts0[i]) for i in eachindex(verts0))
        minarea = min_triangle_area(vertsp, geop.Γ.faces)
        valid_mesh = minarea > 0
        @printf("  actual displacement: max=%.4f   min_triangle_area=%.3e  valid_mesh=%s\n",
            maxdisp, minarea, valid_mesh)

        facesp = geop.Γ.faces
        edgefpp = dof_edge_fingerprint(spacesp.X.fns, facesp)
        permX, okX = build_permutation(edgefp0, edgefpp)
        n_orig_vertsp = length(vertices(spacesp.Y.geo.parent))
        fpYp = dof_cell_fingerprint_filtered(spacesp.Y.fns, spacesp.Y.geo.mesh.faces, n_orig_vertsp)
        permY, okY = build_permutation(fpY0, fpYp)
        @printf("  permutation valid?  X: %s   Y: %s\n", okX, okY)
        if !(okX && okY)
            error("DOF permutation construction failed at amplitude=$amp (okX=$okX, okY=$okY)")
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
        @printf("  true residuals: plain=%.3e  frozen-aligned=%.3e  fresh=%.3e\n", trures_p, trures_a, trures_g)

        open(summary_file, "a") do io
            @printf(io, "%.6g,%d,%.6f,%.6e,%s,%.4f,%.4f,%d,%s,%.4f,%d,%s,%.4f,%d,%s,%.6e,%.6e,%.6e\n",
                amp, dof, maxdisp, minarea, valid_mesh, t_assembly,
                t_p, ch_p.iters, ch_p.isconverged,
                t_a, ch_a.iters, ch_a.isconverged,
                t_g, ch_g.iters, ch_g.isconverged,
                trures_p, trures_a, trures_g)
        end
        println("  [saved] appended to $summary_file")
    catch err
        println("  [ERROR] amplitude=$amp failed: ", err)
        println("  ...continuing to next amplitude")
    end
end

println("\n" * "="^70)
println("Fichera amplitude sweep complete -- ", now())
println("Summary: ", summary_file)
println("="^70)
