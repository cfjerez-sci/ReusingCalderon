# For the same kappa=1.0, h=0.1, realization=1 perturbed sphere used in
# scripts/p10_PEC_perturbed_kappa_compare.jl, now solve using the FROZEN
# preconditioner strategy: build P0 = NXY0*Tyy0*NYX0 from the ORIGINAL
# (nominal, unperturbed) sphere's own Tyy0/Nxy0 matrices, and apply that
# unchanged as the left preconditioner for the perturbed system Zxx'/bx'.
#
# This mirrors the frozen-vs-fresh comparison already done at kappa=2.0
# in scripts/p10_PEC_sphere_perturbation_robustness.jl (where frozen
# reuse was found to converge only very slowly -- smooth monotonic
# decrease but nowhere near 1e-8 within 1500 iterations), now repeated
# at kappa=1.0 to see whether the same behavior holds at a different
# wavenumber.
#
# Appends a `frozen` row to the same
# data/sweep/p10_perturbed_kappa_compare.csv used by the plain/fresh
# comparison script, plus writes the full resnorm history to a separate
# file for direct inspection (terminal scrollback for a 1500-iteration
# solve is unreliable to read back).
#
# Run directly: F5 in VS Code, or
# `julia scripts/p10_PEC_perturbed_kappa_frozen.jl` from the project
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

# Nominal (realization=0) geo/discretization doesn't depend on
# lmin/lmax/amplitude, but Makeitso's cache key does -- reuse the same
# sentinel values (2,6,0.03) used throughout this study so the nominal
# assembly at kappa=1.0 (if already computed) is retrieved from cache
# rather than redone.
nominal_lmin, nominal_lmax, nominal_amplitude = 2, 6, 0.03

outdir = projectdir("data", "sweep")
mkpath(outdir)
summary_file = joinpath(outdir, "p10_perturbed_kappa_frozen.csv")
if !isfile(summary_file)
    open(summary_file, "w") do io
        println(io, "kappa,realization,seed,amplitude,dof,assembly_time_nominal_s,assembly_time_perturbed_s,solve_time_frozen_s,iters_frozen,converged_frozen,trueres_frozen")
    end
end

println("="^70)
println("Perturbed-sphere FROZEN-preconditioner comparison -- started ", now())
println("kappa=$κ  h=$h  realization=$realization  lmin=$lmin  lmax=$lmax  amplitude=$amplitude")
println("="^70)

println("\n--- building/retrieving nominal (unperturbed) discretization ---")
t0 = time()
disc0 = make(Sim2.discretization; h, κ, radius, realization=0,
    lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)
t_assembly0 = time() - t0
dof0 = length(disc0.vectors.bx)
Tyy0 = disc0.matrices.Tyy
Nxy0 = disc0.matrices.Nxy
@printf("  nominal assembly: %.1f s   (dof = %d)\n", t_assembly0, dof0)

println("\n--- retrieving perturbed discretization ---")
t0 = time()
discp = make(Sim2.discretization; h, κ, radius, realization, lmin, lmax, amplitude)
t_assembly = time() - t0
dof = length(discp.vectors.bx)
@assert dof == dof0 "DOF mismatch: perturbed=$dof vs nominal=$dof0"
@printf("  perturbed assembly: %.1f s   (dof = %d)\n", t_assembly, dof)

Zxxp = discp.matrices.Zxx
bxp  = discp.vectors.bx

u_f, ch_f, t_f = solve_calderon_gmres(Zxxp, bxp, Tyy0, Nxy0)
@printf("\n[frozen] solve: %.2f s   (%d iters, converged=%s)\n", t_f, ch_f.iters, ch_f.isconverged)

# True (unpreconditioned) residual -- GMRES convergence is measured on
# the *preconditioned* residual, so `converged` alone can be misleading
# for a mismatched preconditioner.
uvec_f = Vector(u_f)
bxpvec = Vector(bxp)
trures_f = norm(bxpvec .- Zxxp*uvec_f) / norm(bxpvec)
@printf("true relative residual (frozen): %.3e\n", trures_f)

# Full residual history, for inspecting whether convergence stalls,
# diverges, or is just slow (see the kappa=2.0 case for reference).
rn = try
    collect(ch_f.data[:resnorm])
catch err
    println("[warn] could not extract resnorm: ", err)
    Float64[]
end
diagfile = joinpath(outdir, "p10_frozen_resnorm_diag_kappa$(κ)_realization$(realization).csv")
open(diagfile, "w") do io
    println(io, "iteration,resnorm")
    for (i, v) in enumerate(rn)
        @printf(io, "%d,%.8e\n", i, v)
    end
end
println("[saved] frozen resnorm history (length=$(length(rn))) to ", diagfile)

open(summary_file, "a") do io
    @printf(io, "%.6g,%d,%d,%.6g,%d,%.4f,%.4f,%.4f,%d,%s,%.6e\n",
        κ, realization, 1000+realization, amplitude, dof, t_assembly0, t_assembly,
        t_f, ch_f.iters, ch_f.isconverged, trures_f)
end
println("[saved] appended frozen row to ", summary_file)

println("\n" * "="^70)
println("Complete -- ", now())
println("="^70)
