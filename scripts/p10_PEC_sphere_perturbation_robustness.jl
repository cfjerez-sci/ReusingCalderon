# Shape-perturbation robustness study for Calderon preconditioning.
#
# Nominal domain: PEC unit sphere, h=0.1, kappa=2.0 (radius=1, ka=2).
# For N random realizations, the nominal sphere is perturbed by a
# low-degree (l=2..6) random real-spherical-harmonic vertex
# displacement (amplitude ~3%, see postproc/shape_perturbation.jl),
# keeping mesh connectivity IDENTICAL to the nominal mesh -- only
# vertex coordinates move. Because RWG/BC basis functions and their DOF
# indexing are built purely from mesh topology (edges), the nominal
# Calderon preconditioner matrices (Tyy0, Nxy0) act on exactly the same
# index sets as the perturbed system Zxx', so they can be reused
# UNCHANGED as an approximate/frozen preconditioner for the perturbed
# problem.
#
# Three solve strategies are compared per perturbed realization:
#   (a) plain    -- plain GMRES, no preconditioner
#   (b) frozen   -- Calderon GMRES with the FROZEN nominal
#                   preconditioner (Tyy0, Nxy0) -- the practical
#                   proposal being tested
#   (c) fresh    -- Calderon GMRES with a preconditioner rebuilt from
#                   the perturbed geometry's own Tyy'/Nxy' -- gold
#                   standard upper bound on what preconditioning could
#                   achieve here
#
# Registers to data/sweep/p10_perturbation_summary.csv. Seeded
# (realization -> seed 1000+realization) for full reproducibility;
# robust to interruption (each realization's row is appended to disk
# immediately); wrapped in try/catch per realization so one bad point
# doesn't kill the run.
#
# SMOKE TEST: N is currently set to 1 -- bump to 20 once this has been
# confirmed to run cleanly end-to-end (dof match, all 3 solves
# converge, timings sane).
#
# Run directly: F5 in VS Code, or
# `julia scripts/p10_PEC_sphere_perturbation_robustness.jl` from the
# project root.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using Exp25_CJH_KC_LocalMultiTrace
using Makeitso
using DrWatson
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
amplitude  = 0.01   # <-- retry at 1% (3% was confirmed -- not a bug -- to
                    # break the frozen-preconditioner strategy; user chose
                    # to retry at 0.5-1%)

# Nominal (realization=0) geo/discretization genuinely doesn't depend on
# lmin/lmax/amplitude (see problems/p10_perturbed_sphere.jl), but
# Makeitso's cache key is keyed on ALL args passed to `make`, so using
# the swept `amplitude` value for the nominal call would force a
# pointless full reassembly every time `amplitude` changes. Pin the
# nominal call to fixed sentinel values instead, so the (expensive)
# nominal assembly is computed once and reused across every amplitude
# tested in this study.
nominal_lmin, nominal_lmax, nominal_amplitude = 2, 6, 0.03

N = 1   # <-- SMOKE TEST: set to 20 for the full campaign

outdir = projectdir("data", "sweep")
mkpath(outdir)
summary_file = joinpath(outdir, "p10_perturbation_summary.csv")

if !isfile(summary_file)
    open(summary_file, "w") do io
        println(io, "realization,seed,amplitude,dof,assembly_time_s,solve_time_plain_s,iters_plain,converged_plain,solve_time_frozen_s,iters_frozen,converged_frozen,solve_time_fresh_s,iters_fresh,converged_fresh,reldiff_plain_vs_fresh,reldiff_frozen_vs_fresh,trueres_plain,trueres_frozen,trueres_fresh")
    end
end

println("="^70)
println("PEC sphere shape-perturbation robustness study -- started ", now())
println("h=$h  kappa=$κ  realizations=$N  lmin=$lmin  lmax=$lmax  amplitude=$amplitude")
println("="^70)

# --- Nominal (unperturbed) discretization: built once, provides
# Tyy0/Nxy0 for the frozen-preconditioner strategy.
println("\n--- building nominal (unperturbed) discretization ---")
t0 = time()
disc0 = make(Sim2.discretization; h, κ, radius, realization=0,
    lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)
t_assembly0 = time() - t0
dof0 = length(disc0.vectors.bx)
Tyy0 = disc0.matrices.Tyy
Nxy0 = disc0.matrices.Nxy
@printf("  nominal assembly: %.1f s   (dof = %d)\n", t_assembly0, dof0)

for r in 1:N
    println("\n--- realization $r / $N ---")
    try
        t0 = time()
        discp = make(Sim2.discretization; h, κ, radius, realization=r, lmin, lmax, amplitude)
        t_assembly = time() - t0
        dof = length(discp.vectors.bx)
        @assert dof == dof0 "DOF mismatch: perturbed=$dof vs nominal=$dof0 -- connectivity should be identical"
        @printf("  perturbed assembly: %.1f s   (dof = %d)\n", t_assembly, dof)

        Zxxp = discp.matrices.Zxx
        Tyyp = discp.matrices.Tyy
        Nxyp = discp.matrices.Nxy
        bxp  = discp.vectors.bx

        u_p, ch_p, t_p = solve_plain_gmres(Zxxp, bxp)
        @printf("  [plain]  solve: %.2f s  (%d iters, converged=%s)\n", t_p, ch_p.iters, ch_p.isconverged)

        u_f, ch_f, t_f = solve_calderon_gmres(Zxxp, bxp, Tyy0, Nxy0)
        @printf("  [frozen] solve: %.2f s  (%d iters, converged=%s)\n", t_f, ch_f.iters, ch_f.isconverged)
        try
            rn = collect(ch_f.data[:resnorm])
            diagfile = joinpath(outdir, "p10_frozen_resnorm_diag_realization$(r)_amp$(amplitude).csv")
            open(diagfile, "w") do io
                println(io, "iteration,resnorm")
                for (i, v) in enumerate(rn)
                    @printf(io, "%d,%.8e\n", i, v)
                end
            end
            println("    [frozen diag] wrote full resnorm history (length=$(length(rn))) to $diagfile")
            nshow = min(10, length(rn))
            @printf("    [frozen diag] resnorm length=%d  first %d: %s\n", length(rn), nshow, rn[1:nshow])
            @printf("    [frozen diag] last %d: %s\n", nshow, rn[end-nshow+1:end])
        catch err
            println("    [frozen diag] could not extract resnorm: ", err)
        end

        u_g, ch_g, t_g = solve_calderon_gmres(Zxxp, bxp, Tyyp, Nxyp)
        @printf("  [fresh]  solve: %.2f s  (%d iters, converged=%s)\n", t_g, ch_g.iters, ch_g.isconverged)

        uvec_p = Vector(u_p); uvec_f = Vector(u_f); uvec_g = Vector(u_g)
        reldiff_p = norm(uvec_p .- uvec_g) / norm(uvec_g)
        reldiff_f = norm(uvec_f .- uvec_g) / norm(uvec_g)
        @printf("  ||u_plain-u_fresh||/||u_fresh|| = %.3e   ||u_frozen-u_fresh||/||u_fresh|| = %.3e\n",
            reldiff_p, reldiff_f)

        # TRUE (unpreconditioned) residual check: GMRES convergence is
        # measured on the *preconditioned* residual, so a converged flag
        # for the frozen-preconditioner run does not by itself guarantee
        # a small TRUE residual ||bxp - Zxxp*u|| if the frozen P0 is a
        # poor approximation of the perturbed operator's inverse on some
        # modes. Compute all three directly to tell a genuine
        # preconditioning-quality effect apart from a bookkeeping bug.
        bxpvec = Vector(bxp)
        nb = norm(bxpvec)
        trures_p = norm(bxpvec .- Zxxp*uvec_p) / nb
        trures_f = norm(bxpvec .- Zxxp*uvec_f) / nb
        trures_g = norm(bxpvec .- Zxxp*uvec_g) / nb
        @printf("  true relative residuals:  plain=%.3e  frozen=%.3e  fresh=%.3e\n",
            trures_p, trures_f, trures_g)

        open(summary_file, "a") do io
            @printf(io, "%d,%d,%.6g,%d,%.4f,%.4f,%d,%s,%.4f,%d,%s,%.4f,%d,%s,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                r, 1000+r, amplitude, dof, t_assembly,
                t_p, ch_p.iters, ch_p.isconverged,
                t_f, ch_f.iters, ch_f.isconverged,
                t_g, ch_g.iters, ch_g.isconverged,
                reldiff_p, reldiff_f,
                trures_p, trures_f, trures_g)
        end
        println("  [saved] appended to $summary_file")
    catch err
        println("  [ERROR] realization $r failed: ", err)
        println("  ...continuing to next realization")
    end
end

println("\n" * "="^70)
println("Perturbation robustness study complete -- ", now())
println("Summary: ", summary_file)
println("="^70)
