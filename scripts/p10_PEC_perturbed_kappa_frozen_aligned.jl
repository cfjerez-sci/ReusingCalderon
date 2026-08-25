# Redo the frozen-preconditioner comparison
# (scripts/p10_PEC_perturbed_kappa_frozen.jl), but this time correctly
# ALIGNED: scripts/p10_investigate_dof_indexing.jl found that
# raviartthomas(Γ)/buffachristiansen(Γ) do NOT assign a
# coordinate-independent DOF ordering -- 91% of RWG DOFs referred to a
# DIFFERENT physical edge between the nominal and perturbed spaces,
# even though mesh connectivity is identical. The earlier "frozen
# reuse fails to converge" result was therefore measuring an indexing
# artifact (an effectively scrambled preconditioner), not genuine
# preconditioner-quality degradation under shape perturbation.
#
# This script loads the exact DOF permutations built in
# p10_investigate_dof_indexing.jl (data/sweep/p10_dof_permutations_realizationN.txt)
# and uses them to re-index the nominal Tyy0/Nxy0 matrices into the
# perturbed space's DOF ordering BEFORE using them as the frozen
# preconditioner -- i.e. this tests the actual mathematical question
# (does reusing the nominal Calderon operator, correctly aligned, work
# as a preconditioner for the perturbed system), rather than an
# accidental permutation problem.
#
# Run directly: F5 in VS Code, or
# `julia scripts/p10_PEC_perturbed_kappa_frozen_aligned.jl` from the
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

radius      = 1.0
h           = 0.1
κ           = 1.0
lmin, lmax  = 2, 6
amplitude   = 0.03
realization = 1

nominal_lmin, nominal_lmax, nominal_amplitude = 2, 6, 0.03

outdir = projectdir("data", "sweep")
mkpath(outdir)

# --- Load the DOF permutations built in p10_investigate_dof_indexing.jl
permfile = joinpath(outdir, "p10_dof_permutations_realization$(realization).txt")
lines = readlines(permfile)
datalines = filter(l -> !startswith(l, "#"), lines)
ndof = length(datalines) ÷ 2
permX = parse.(Int, datalines[1:ndof])
permY = parse.(Int, datalines[ndof+1:2*ndof])
@assert sort(permX) == collect(1:ndof)
@assert sort(permY) == collect(1:ndof)
invpermX = invperm(permX)
invpermY = invperm(permY)
println("Loaded permutations from ", permfile, "  (ndof=", ndof, ")")

println("="^70)
println("Perturbed-sphere ALIGNED-frozen-preconditioner comparison -- started ", now())
println("kappa=$κ  h=$h  realization=$realization  lmin=$lmin  lmax=$lmax  amplitude=$amplitude")
println("="^70)

println("\n--- building/retrieving nominal (unperturbed) discretization ---")
t0 = time()
disc0 = make(Sim2.discretization; h, κ, radius, realization=0,
    lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)
t_assembly0 = time() - t0
dof0 = length(disc0.vectors.bx)
@printf("  nominal assembly: %.1f s   (dof = %d)\n", t_assembly0, dof0)
@assert dof0 == ndof "permutation file dof count ($ndof) doesn't match nominal discretization dof ($dof0)"

println("\n--- retrieving perturbed discretization ---")
t0 = time()
discp = make(Sim2.discretization; h, κ, radius, realization, lmin, lmax, amplitude)
t_assembly = time() - t0
dof = length(discp.vectors.bx)
@assert dof == dof0
@printf("  perturbed assembly: %.1f s   (dof = %d)\n", t_assembly, dof)

println("\nmaterializing dense Tyy0/Nxy0 and permuting into perturbed DOF ordering...")
Tyy0_dense = Matrix{ComplexF64}(disc0.matrices.Tyy)
Nxy0_dense = Matrix{ComplexF64}(disc0.matrices.Nxy)
Tyy0_aligned = Tyy0_dense[invpermY, invpermY]
Nxy0_aligned = Nxy0_dense[invpermX, invpermY]
println("done.")

Zxxp = discp.matrices.Zxx
bxp  = discp.vectors.bx

u_a, ch_a, t_a = solve_calderon_gmres(Zxxp, bxp, Tyy0_aligned, Nxy0_aligned)
@printf("\n[frozen-ALIGNED] solve: %.2f s   (%d iters, converged=%s)\n", t_a, ch_a.iters, ch_a.isconverged)

uvec_a = Vector(u_a)
bxpvec = Vector(bxp)
trures_a = norm(bxpvec .- Zxxp*uvec_a) / norm(bxpvec)
@printf("true relative residual (frozen-aligned): %.3e\n", trures_a)

rn = try
    collect(ch_a.data[:resnorm])
catch err
    println("[warn] could not extract resnorm: ", err)
    Float64[]
end
diagfile = joinpath(outdir, "p10_frozen_aligned_resnorm_diag_kappa$(κ)_realization$(realization).csv")
open(diagfile, "w") do io
    println(io, "iteration,resnorm")
    for (i, v) in enumerate(rn)
        @printf(io, "%d,%.8e\n", i, v)
    end
end
println("[saved] frozen-aligned resnorm history (length=$(length(rn))) to ", diagfile)

summary_file = joinpath(outdir, "p10_perturbed_kappa_frozen_aligned.csv")
if !isfile(summary_file)
    open(summary_file, "w") do io
        println(io, "kappa,realization,seed,amplitude,dof,assembly_time_nominal_s,assembly_time_perturbed_s,solve_time_frozen_aligned_s,iters_frozen_aligned,converged_frozen_aligned,trueres_frozen_aligned")
    end
end
open(summary_file, "a") do io
    @printf(io, "%.6g,%d,%d,%.6g,%d,%.4f,%.4f,%.4f,%d,%s,%.6e\n",
        κ, realization, 1000+realization, amplitude, dof, t_assembly0, t_assembly,
        t_a, ch_a.iters, ch_a.isconverged, trures_a)
end
println("[saved] appended aligned-frozen row to ", summary_file)

println("\n" * "="^70)
println("Complete -- ", now())
println("="^70)
