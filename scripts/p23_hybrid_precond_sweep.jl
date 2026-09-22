# [CJ-08] HYBRID PRECONDITIONER SWEEP
#
# Question: Table 1 of the manuscript reports t_N = 0.19 s against
# t_T = 430 s, yet the "frozen" strategy freezes BOTH T_YY and N_XY.
# Section 5.3 then reports that the frozen inner solves cost ~3.2x more
# per OUTER iteration than the fresh ones, precisely because the frozen
# N_XY,0 is mismatched to the current geometry.
#
# Refreshing N_XY at every realization costs 0.04% of total assembly.
# This script tests whether the HYBRID strategy
#
#       P_hybrid = N_XY(y)^{-T} * T_YY,0(aligned) * N_XY(y)^{-1}
#
# i.e. FROZEN dual-mesh operator + FRESH duality pairing, removes the
# per-iteration penalty while keeping the amortization.
#
# Three preconditioned strategies are compared at each amplitude:
#   frozen : Tyy0_aligned, Nxy0_aligned   (the manuscript's method)
#   hybrid : Tyy0_aligned, Nxyp           (this test)
#   fresh  : Tyyp,         Nxyp           (gold standard)
#
# The outer iteration count is the scientific quantity; the per-outer-
# iteration wall time is the practical one, and is what CJ-08 predicts
# will change. Both are recorded, along with t_N for the refresh cost.
#
# Everything is read from the Makeitso cache: at h=0.1, kappa=2,
# realization=1 the discretizations for these amplitudes are ALREADY
# cached by scripts/p10_PEC_sphere_perturbation_amplitude_sweep.jl, so
# this run should take minutes, not hours. No new assembly is expected.
#
# Plain GMRES is OFF by default (272-290 iters, already tabulated in
# p10_amplitude_sweep_summary.csv). Pass --with-plain to include it.
#
# Run from the repo root:
#   julia +1.11 --project scripts/p23_hybrid_precond_sweep.jl
#   julia +1.11 --project scripts/p23_hybrid_precond_sweep.jl --with-plain
#
# The t_Nxy_refresh_s column is written as NaN by design: the refresh cost
# is one assemble() of the duality pairing, already tabulated as t_N in
# Table 1 of the manuscript. Nothing here re-measures it.
#
# Repo conventions (CLAUDE.md): run from repo root; data/ is gitignored;
# do not commit or push without asking Carlos.

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

module Sim2
include("../problems/p10_perturbed_sphere.jl")
include("../methods/EFIE.jl")
end

include("../methods/EFIE_manual_solves.jl")

# ---------------------------------------------------------------- config
const WITH_PLAIN = "--with-plain" in ARGS

radius      = 1.0
h           = 0.1
κ           = 2.0
lmin, lmax  = 2, 6
realization = 1                      # FIXED: same direction, amplitude scales

nominal_lmin, nominal_lmax, nominal_amplitude = 2, 6, 0.03

amplitudes = [0.03, 0.10, 0.20, 0.30, 0.40, 0.50]

outdir = projectdir("data", "sweep")
mkpath(outdir)
summary_file = joinpath(outdir, "p23_hybrid_precond_summary.csv")
if !isfile(summary_file)
    open(summary_file, "w") do io
        println(io, "amplitude,dof,maxdisp_pct,min_triangle_area,valid_mesh," *
                    "t_Nxy_refresh_s," *
                    "iters_plain,t_plain_s,converged_plain,trueres_plain," *
                    "iters_frozen,t_frozen_s,t_per_iter_frozen_s,converged_frozen,trueres_frozen," *
                    "iters_hybrid,t_hybrid_s,t_per_iter_hybrid_s,converged_hybrid,trueres_hybrid," *
                    "iters_fresh,t_fresh_s,t_per_iter_fresh_s,converged_fresh,trueres_fresh," *
                    "reldiff_hybrid_vs_fresh,reldiff_frozen_vs_fresh")
    end
end

# ------------------------------------------------- DOF fingerprinting
# (identical to scripts/p10_PEC_sphere_perturbation_amplitude_sweep.jl)

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

triangle_area(v1, v2, v3) = norm(cross(v2 .- v1, v3 .- v1)) / 2

function min_triangle_area(verts, faces)
    m = Inf
    for f in faces
        m = min(m, triangle_area(verts[f[1]], verts[f[2]], verts[f[3]]))
    end
    return m
end

# ------------------------------------------------------------- banner
println("="^72)
println("[CJ-08] HYBRID preconditioner sweep -- started ", now())
@printf("h=%.4g  kappa=%.4g  realization=%d  with_plain=%s\n", h, κ, realization, WITH_PLAIN)
println("amplitudes = ", amplitudes)
println("strategies : frozen(T0,N0)  hybrid(T0,Np)  fresh(Tp,Np)")
println("="^72)

println("\n--- nominal (unperturbed) discretization + spaces ---")
t0 = time()
disc0 = make(Sim2.discretization; h, κ, radius, realization=0,
    lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)
spaces0 = make(Sim2.spaces; h, radius, realization=0,
    lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)
geo0 = make(Sim2.geo; h, radius, realization=0,
    lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)
t_assembly0 = time() - t0
dof0 = length(disc0.vectors.bx)
@printf("  nominal retrieved in %.1f s   (dof = %d)\n", t_assembly0, dof0)

Tyy0_dense = Matrix{ComplexF64}(disc0.matrices.Tyy)
Nxy0_dense = Matrix{ComplexF64}(disc0.matrices.Nxy)

faces0 = geo0.Γ.faces
edgefp0 = dof_edge_fingerprint(spaces0.X.fns, faces0)
n_orig_verts0 = length(vertices(spaces0.Y.geo.parent))
fpY0 = dof_cell_fingerprint_filtered(spaces0.Y.fns, spaces0.Y.geo.mesh.faces, n_orig_verts0)
println("  nominal DOF fingerprints built.")

# --------------------------------------------------------------- sweep
for amp in amplitudes
    @printf("\n--- amplitude = %.4g%% ---\n", 100*amp)
    try
        t0 = time()
        discp   = make(Sim2.discretization; h, κ, radius, realization, lmin, lmax, amplitude=amp)
        spacesp = make(Sim2.spaces; h, radius, realization, lmin, lmax, amplitude=amp)
        geop    = make(Sim2.geo; h, radius, realization, lmin, lmax, amplitude=amp)
        t_retrieve = time() - t0
        dof = length(discp.vectors.bx)
        @assert dof == dof0 "DOF mismatch: perturbed=$dof vs nominal=$dof0"
        @printf("  retrieved in %.1f s   (dof = %d)\n", t_retrieve, dof)

        vertsp = geop.Γ.vertices
        norms0 = norm.(geo0.Γ.vertices)
        valid_idx = norms0 .>= 1e-8*radius
        reldisp = [(norm(vertsp[i]) - radius)/radius*100 for i in eachindex(vertsp)]
        maxdisp = maximum(abs.(reldisp[valid_idx]))
        minarea = min_triangle_area(vertsp, geop.Γ.faces)
        valid_mesh = minarea > 0
        @printf("  max |displacement| = %.2f%%   min_triangle_area = %.3e   valid_mesh = %s\n",
            maxdisp, minarea, valid_mesh)

        facesp = geop.Γ.faces
        edgefpp = dof_edge_fingerprint(spacesp.X.fns, facesp)
        permX, okX = build_permutation(edgefp0, edgefpp)
        n_orig_vertsp = length(vertices(spacesp.Y.geo.parent))
        fpYp = dof_cell_fingerprint_filtered(spacesp.Y.fns, spacesp.Y.geo.mesh.faces, n_orig_vertsp)
        permY, okY = build_permutation(fpY0, fpYp)
        @printf("  permutation valid?  X: %s   Y: %s\n", okX, okY)
        (okX && okY) || error("DOF permutation failed at amplitude=$amp")
        invpermX = invperm(permX)
        invpermY = invperm(permY)

        Tyy0_aligned = Tyy0_dense[invpermY, invpermY]
        Nxy0_aligned = Nxy0_dense[invpermX, invpermY]

        Zxxp = discp.matrices.Zxx
        Tyyp = discp.matrices.Tyy
        Nxyp = discp.matrices.Nxy
        bxp  = discp.vectors.bx

        # The refresh cost the hybrid strategy pays is one assemble() of the
        # X-Y duality pairing on the perturbed geometry. That is already
        # timed in the manuscript's Table 1 (t_N = 0.19 s at this h and dof,
        # against t_T = 430 s), so it is not re-measured here; Nxyp below is
        # the cached perturbed-geometry pairing.
        t_Nxy = NaN

        bxpvec = Vector(bxp); nb = norm(bxpvec)
        trueres(u) = norm(bxpvec .- Zxxp*Vector(u)) / nb

        iters_p, t_p, conv_p, tr_p = -1, NaN, false, NaN
        if WITH_PLAIN
            u_p, ch_p, t_p = solve_plain_gmres(Zxxp, bxp)
            iters_p, conv_p, tr_p = ch_p.iters, ch_p.isconverged, trueres(u_p)
            @printf("  [plain ] %4d iters  %8.2f s               trueres=%.3e\n", iters_p, t_p, tr_p)
        end

        u_f, ch_f, t_f = solve_calderon_gmres(Zxxp, bxp, Tyy0_aligned, Nxy0_aligned)
        tr_f = trueres(u_f)
        @printf("  [frozen] %4d iters  %8.2f s  %8.4f s/iter  trueres=%.3e\n",
            ch_f.iters, t_f, t_f/max(ch_f.iters,1), tr_f)

        u_h, ch_h, t_h = solve_calderon_gmres(Zxxp, bxp, Tyy0_aligned, Nxyp)
        tr_h = trueres(u_h)
        @printf("  [hybrid] %4d iters  %8.2f s  %8.4f s/iter  trueres=%.3e\n",
            ch_h.iters, t_h, t_h/max(ch_h.iters,1), tr_h)

        u_g, ch_g, t_g = solve_calderon_gmres(Zxxp, bxp, Tyyp, Nxyp)
        tr_g = trueres(u_g)
        @printf("  [fresh ] %4d iters  %8.2f s  %8.4f s/iter  trueres=%.3e\n",
            ch_g.iters, t_g, t_g/max(ch_g.iters,1), tr_g)

        ug = Vector(u_g); ng = norm(ug)
        d_hg = norm(Vector(u_h) .- ug) / ng
        d_fg = norm(Vector(u_f) .- ug) / ng
        @printf("  agreement vs fresh:  hybrid=%.3e   frozen=%.3e\n", d_hg, d_fg)

        open(summary_file, "a") do io
            @printf(io, "%.6g,%d,%.4f,%.6e,%s,%.6f,%d,%.4f,%s,%.6e,%d,%.4f,%.6f,%s,%.6e,%d,%.4f,%.6f,%s,%.6e,%d,%.4f,%.6f,%s,%.6e,%.6e,%.6e\n",
                amp, dof, maxdisp, minarea, valid_mesh,
                t_Nxy,
                iters_p, t_p, conv_p, tr_p,
                ch_f.iters, t_f, t_f/max(ch_f.iters,1), ch_f.isconverged, tr_f,
                ch_h.iters, t_h, t_h/max(ch_h.iters,1), ch_h.isconverged, tr_h,
                ch_g.iters, t_g, t_g/max(ch_g.iters,1), ch_g.isconverged, tr_g,
                d_hg, d_fg)
        end
        println("  [saved] appended to ", summary_file)
    catch err
        println("  [ERROR] amplitude=$amp failed: ", err)
        println("  ...continuing to next amplitude (row NOT written)")
    end
end

println("\n" * "="^72)
println("[CJ-08] hybrid sweep complete -- ", now())
println("Summary: ", summary_file)
println("="^72)
