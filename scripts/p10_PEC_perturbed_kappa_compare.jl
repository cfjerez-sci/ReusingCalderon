# For a single perturbed-sphere realization, compare plain GMRES vs.
# Calderon-preconditioned GMRES (freshly built from that same perturbed
# geometry's own Tyy'/Nxy' -- NOT the frozen-nominal-preconditioner
# strategy from scripts/p10_PEC_sphere_perturbation_robustness.jl).
#
# This mirrors the very first (kappa=2.0) plain-vs-Calderon comparison
# done for the nominal sphere, but now at kappa=1.0 and on a randomly
# perturbed shape (same perturbation model as the robustness study:
# l=2..6 real spherical harmonics, i.i.d. Uniform(-1,1) coefficients,
# amplitude=3%, vertex-only displacement with identical mesh
# connectivity -- see postproc/shape_perturbation.jl).
#
# Registers to data/sweep/p10_perturbed_kappa_compare.csv (append mode,
# so multiple kappa/realization combinations can accumulate over
# separate runs of this script).
#
# Run directly: F5 in VS Code, or
# `julia scripts/p10_PEC_perturbed_kappa_compare.jl` from the project
# root.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using ReusingCalderon
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

radius      = 1.0
h           = 0.1
κ           = 1.0
lmin, lmax  = 2, 6
amplitude   = 0.03
realization = 1

outdir = projectdir("data", "sweep")
mkpath(outdir)
summary_file = joinpath(outdir, "p10_perturbed_kappa_compare.csv")

if !isfile(summary_file)
    open(summary_file, "w") do io
        println(io, "kappa,realization,seed,amplitude,dof,assembly_time_s,solve_time_plain_s,iters_plain,converged_plain,solve_time_calderon_s,iters_calderon,converged_calderon")
    end
end

println("="^70)
println("Perturbed-sphere plain-vs-Calderon comparison -- started ", now())
println("kappa=$κ  h=$h  realization=$realization  lmin=$lmin  lmax=$lmax  amplitude=$amplitude")
println("="^70)

t0 = time()
disc = make(Sim2.discretization; h, κ, radius, realization, lmin, lmax, amplitude)
t_assembly = time() - t0
dof = length(disc.vectors.bx)
@printf("assembly: %.1f s   (dof = %d)\n", t_assembly, dof)

Zxx = disc.matrices.Zxx
Tyy = disc.matrices.Tyy
Nxy = disc.matrices.Nxy
bx  = disc.vectors.bx

u_p, ch_p, t_p = solve_plain_gmres(Zxx, bx)
@printf("[plain]    solve: %.2f s   (%d iters, converged=%s)\n", t_p, ch_p.iters, ch_p.isconverged)

u_c, ch_c, t_c = solve_calderon_gmres(Zxx, bx, Tyy, Nxy)
@printf("[calderon] solve: %.2f s   (%d iters, converged=%s)\n", t_c, ch_c.iters, ch_c.isconverged)

open(summary_file, "a") do io
    @printf(io, "%.6g,%d,%d,%.6g,%d,%.4f,%.4f,%d,%s,%.4f,%d,%s\n",
        κ, realization, 1000+realization, amplitude, dof, t_assembly,
        t_p, ch_p.iters, ch_p.isconverged,
        t_c, ch_c.iters, ch_c.isconverged)
end
println("[saved] appended to ", summary_file)

println("\n" * "="^70)
println("Complete -- ", now())
println("="^70)
