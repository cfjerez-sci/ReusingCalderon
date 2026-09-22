# NEW EXPERIMENT: wavenumber-mismatched Calderon preconditioner.
#
# All previous p10_* experiments froze the Calderon preconditioner
# (Tyy, Nxy) built on the NOMINAL geometry and reused it (via a DOF
# permutation) on a shape-perturbed geometry at the SAME wavenumber.
# This script instead freezes the WAVENUMBER used to build the
# preconditioner, with NO shape perturbation at all (realization=0,
# exact nominal sphere for both):
#
#   - the actual problem (Zxx, excitation bx) is assembled at
#     kappa_solve = 4.0
#   - the Calderon preconditioner ingredients (Tyy, Nxy) are instead
#     taken from a discretization assembled at kappa_precond = 1.0
#     -- a 4x "wrong" wavenumber -- and used unchanged (no DOF
#     permutation needed here: with realization=0 in both cases, the
#     `spaces` target depends only on `geo`, not kappa, so X/Y and
#     their DOF indexing are IDENTICAL between the two discretizations
#     by construction; see methods/EFIE.jl).
#
# Three solves are compared, all against the SAME kappa=4.0 problem
# (Zxx, bx):
#   (a) plain        -- plain GMRES, no preconditioner
#   (b) kappa_mismatched -- Calderon GMRES using Tyy/Nxy assembled at
#                           kappa_precond=1.0 (the new experiment)
#   (c) fresh         -- Calderon GMRES using Tyy/Nxy assembled at the
#                        correct kappa_solve=4.0 (gold standard)
#
# Single run (N=1), no perturbation -- a quick, simple test case.
# Registers to data/sweep/p10_kappa_mismatch_precond_summary.csv.
#
# Run directly: F5 in VS Code, or
# `julia scripts/p10_kappa_mismatch_precond_test.jl` from the project
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

κ_solve    = 4.0   # the actual problem's wavenumber
κ_precond  = 1.0   # the (mismatched) wavenumber used to build the frozen preconditioner

outdir = projectdir("data", "sweep")
mkpath(outdir)
summary_file = joinpath(outdir, "p10_kappa_mismatch_precond_summary.csv")
if !isfile(summary_file)
    open(summary_file, "w") do io
        println(io, "kappa_solve,kappa_precond,dof,assembly_time_solve_s,assembly_time_precond_s,solve_time_plain_s,iters_plain,converged_plain,solve_time_mismatched_s,iters_mismatched,converged_mismatched,solve_time_fresh_s,iters_fresh,converged_fresh,trueres_plain,trueres_mismatched,trueres_fresh")
    end
end

println("="^70)
println("NEW EXPERIMENT: wavenumber-mismatched Calderon preconditioner -- started ", now())
println("h=$h  realization=$realization (nominal sphere, no perturbation)")
println("kappa_solve=$κ_solve   kappa_precond=$κ_precond  (preconditioner built at the WRONG wavenumber)")
println("="^70)

t0 = time()
disc_solve = make(Sim2.discretization; h, κ=κ_solve, radius, realization, lmin, lmax, amplitude)
t_assembly_solve = time() - t0
dof = length(disc_solve.vectors.bx)
@printf("  kappa_solve=%.1f assembly: %.1f s   (dof = %d)\n", κ_solve, t_assembly_solve, dof)

t0 = time()
disc_precond = make(Sim2.discretization; h, κ=κ_precond, radius, realization, lmin, lmax, amplitude)
t_assembly_precond = time() - t0
dof_precond = length(disc_precond.vectors.bx)
@printf("  kappa_precond=%.1f assembly: %.1f s   (dof = %d)\n", κ_precond, t_assembly_precond, dof_precond)

@assert dof == dof_precond "DOF mismatch between kappa_solve and kappa_precond discretizations (should be identical: spaces depend only on geo, not kappa)"

Zxx = disc_solve.matrices.Zxx
bx  = disc_solve.vectors.bx
Tyy_fresh = disc_solve.matrices.Tyy
Nxy_fresh = disc_solve.matrices.Nxy

Tyy_mismatched = disc_precond.matrices.Tyy
Nxy_mismatched = disc_precond.matrices.Nxy

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
println("\n[saved] appended to ", summary_file)

println("\n" * "="^70)
println("Complete -- ", now())
println("="^70)
