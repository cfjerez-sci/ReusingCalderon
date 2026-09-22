# Sweep version of scripts/p10_kappa_mismatch_precond_test.jl.
#
# The Calderon preconditioner is ALWAYS built at kappa_precond = 1.0
# (fixed), while the actual EFIE problem's wavenumber kappa_solve is
# swept over {1.0, 2.0, 4.0, 8.0}, to see how the "penalty" for using
# a wavenumber-mismatched preconditioner grows as the solve wavenumber
# moves further away from the (fixed) preconditioner wavenumber.
#
# kappa_solve=1.0 is the degenerate/matched case (kappa_precond ==
# kappa_solve): mismatched and fresh should coincide exactly there,
# giving a useful sweep anchor. kappa_solve=4.0 was already run as a
# single test in p10_kappa_mismatch_precond_test.jl and is skipped here
# to avoid a duplicate row (rerun manually if you want to regenerate
# it).
#
# No shape perturbation (realization=0, exact nominal sphere) -- no
# DOF permutation is needed since `spaces` depends only on `geo`, not
# kappa (see methods/EFIE.jl), so X/Y indexing is identical across all
# kappa_solve values tested here.
#
# Registers to data/sweep/p10_kappa_mismatch_precond_summary.csv (same
# file as the single-run test -- same schema). Wrapped in try/catch per
# kappa_solve; robust to interruption.
#
# Run directly: F5 in VS Code, or
# `julia scripts/p10_kappa_mismatch_precond_sweep.jl` from the project
# root.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using ReusingCalderon
using Makeitso
using DrWatson
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

radius      = 1.0
h           = 0.1
realization = 0          # NO shape perturbation -- exact nominal sphere
lmin, lmax  = 2, 6
amplitude   = 0.03       # unused when realization=0, kept for target signature

κ_precond   = 1.0                    # FIXED -- preconditioner always built at kappa=1.0
κ_solve_list = [1.0, 2.0, 8.0]
# NOTE: kappa_solve=4.0 already run/saved by p10_kappa_mismatch_precond_test.jl

outdir = projectdir("data", "sweep")
mkpath(outdir)
summary_file = joinpath(outdir, "p10_kappa_mismatch_precond_summary.csv")
if !isfile(summary_file)
    open(summary_file, "w") do io
        println(io, "kappa_solve,kappa_precond,dof,assembly_time_solve_s,assembly_time_precond_s,solve_time_plain_s,iters_plain,converged_plain,solve_time_mismatched_s,iters_mismatched,converged_mismatched,solve_time_fresh_s,iters_fresh,converged_fresh,trueres_plain,trueres_mismatched,trueres_fresh")
    end
end

println("="^70)
println("Kappa-mismatch preconditioner SWEEP -- started ", now())
println("h=$h  realization=$realization (nominal sphere, no perturbation)")
println("kappa_precond=$κ_precond (fixed)   kappa_solve sweep=$κ_solve_list")
println("="^70)

# --- Preconditioner ingredients (Tyy, Nxy) built ONCE at kappa_precond=1.0
println("\n--- building/retrieving preconditioner discretization at kappa_precond=$κ_precond ---")
t0 = time()
disc_precond = make(Sim2.discretization; h, κ=κ_precond, radius, realization, lmin, lmax, amplitude)
t_assembly_precond = time() - t0
dof_precond = length(disc_precond.vectors.bx)
@printf("  kappa_precond=%.1f assembly: %.1f s   (dof = %d)\n", κ_precond, t_assembly_precond, dof_precond)

Tyy_mismatched = disc_precond.matrices.Tyy
Nxy_mismatched = disc_precond.matrices.Nxy

for κ_solve in κ_solve_list
    println("\n--- kappa_solve = $κ_solve ---")
    try
        t0 = time()
        disc_solve = make(Sim2.discretization; h, κ=κ_solve, radius, realization, lmin, lmax, amplitude)
        t_assembly_solve = time() - t0
        dof = length(disc_solve.vectors.bx)
        @assert dof == dof_precond "DOF mismatch between kappa_solve=$κ_solve and kappa_precond discretizations"
        @printf("  kappa_solve=%.1f assembly: %.1f s   (dof = %d)\n", κ_solve, t_assembly_solve, dof)

        Zxx = disc_solve.matrices.Zxx
        bx  = disc_solve.vectors.bx
        Tyy_fresh = disc_solve.matrices.Tyy
        Nxy_fresh = disc_solve.matrices.Nxy

        u_p, ch_p, t_p = solve_plain_gmres(Zxx, bx)
        @printf("  [plain]              solve: %.2f s  (%d iters, converged=%s)\n", t_p, ch_p.iters, ch_p.isconverged)

        u_m, ch_m, t_m = solve_calderon_gmres(Zxx, bx, Tyy_mismatched, Nxy_mismatched)
        @printf("  [kappa-mismatched]   solve: %.2f s  (%d iters, converged=%s)\n", t_m, ch_m.iters, ch_m.isconverged)

        u_g, ch_g, t_g = solve_calderon_gmres(Zxx, bx, Tyy_fresh, Nxy_fresh)
        @printf("  [fresh, correct κ]   solve: %.2f s  (%d iters, converged=%s)\n", t_g, ch_g.iters, ch_g.isconverged)

        uvec_p = Vector(u_p); uvec_m = Vector(u_m); uvec_g = Vector(u_g)
        bxvec = Vector(bx)
        nb = norm(bxvec)
        trures_p = norm(bxvec .- Zxx*uvec_p) / nb
        trures_m = norm(bxvec .- Zxx*uvec_m) / nb
        trures_g = norm(bxvec .- Zxx*uvec_g) / nb
        @printf("  true residuals: plain=%.3e  kappa-mismatched=%.3e  fresh=%.3e\n", trures_p, trures_m, trures_g)

        open(summary_file, "a") do io
            @printf(io, "%.6g,%.6g,%d,%.4f,%.4f,%.4f,%d,%s,%.4f,%d,%s,%.4f,%d,%s,%.6e,%.6e,%.6e\n",
                κ_solve, κ_precond, dof, t_assembly_solve, t_assembly_precond,
                t_p, ch_p.iters, ch_p.isconverged,
                t_m, ch_m.iters, ch_m.isconverged,
                t_g, ch_g.iters, ch_g.isconverged,
                trures_p, trures_m, trures_g)
        end
        println("  [saved] appended to $summary_file")
    catch err
        println("  [ERROR] kappa_solve=$κ_solve failed: ", err)
        println("  ...continuing to next kappa_solve")
    end
end

println("\n" * "="^70)
println("Kappa-mismatch sweep complete -- ", now())
println("Summary: ", summary_file)
println("="^70)
