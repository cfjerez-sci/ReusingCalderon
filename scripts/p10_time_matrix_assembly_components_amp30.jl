# Same as p10_time_matrix_assembly_components.jl (breaks the combined
# "discretization" assembly into its Zxx/Tyy/Nxy/bx pieces) but at
# amplitude=30% instead of the original 3%, to build the assembly-time-
# savings figure for the 30% campaign
# (scripts/p10_PEC_sphere_perturbation_robustness_v2_amp30.jl).
#
# Run directly: F5 in VS Code, or
# `julia scripts/p10_time_matrix_assembly_components_amp30.jl` from the
# project root.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using Exp25_CJH_KC_LocalMultiTrace
using Makeitso
using DrWatson
using BEAST
using Printf
using Dates
ENV["DRWATSON_WARN_DIRTY"] = "false"

module Sim2
include("../problems/p10_perturbed_sphere.jl")
include("../methods/EFIE.jl")
end

radius      = 1.0
h           = 0.1
κ           = 2.0
lmin, lmax  = 2, 6
amplitude   = 0.30
realization = 1

outdir = projectdir("data", "sweep")
mkpath(outdir)
summary_file = joinpath(outdir, "p10_matrix_assembly_component_times_amp30.csv")
if !isfile(summary_file)
    open(summary_file, "w") do io
        println(io, "realization,kappa,h,amplitude,dof,t_Zxx_s,t_Tyy_s,t_Nxy_s,t_bx_s,t_total_components_s")
    end
end

println("="^70)
println("Matrix-assembly component timing @ amplitude=30% -- started ", now())
println("kappa=$κ  h=$h  realization=$realization  amplitude=$amplitude")
println("="^70)

formulation = make(Sim2.formulation; κ)
spaces = make(Sim2.spaces; h, radius, realization, lmin, lmax, amplitude)

(;bilforms, linforms) = formulation
(;A, Nform) = bilforms
(;b) = linforms
(;X, Y) = spaces

dof = length(X.fns)
@printf("dof = %d\n", dof)

println("\nassembling Zxx = assemble(A, X, X)  (primal EFIE -- needed by ALL strategies)...")
t0 = time(); Zxx = assemble(A, X, X); t_Zxx = time() - t0
@printf("  t_Zxx = %.1f s\n", t_Zxx)

println("\nassembling Tyy = assemble(A, Y, Y)  (dual/BC discretisation -- needed ONLY for a fresh rebuild)...")
t0 = time(); Tyy = assemble(A, Y, Y); t_Tyy = time() - t0
@printf("  t_Tyy = %.1f s\n", t_Tyy)

println("\nassembling Nxy = assemble(Nform, X, Y)  (duality pairing -- needed ONLY for a fresh rebuild)...")
t0 = time(); Nxy = assemble(Nform, X, Y); t_Nxy = time() - t0
@printf("  t_Nxy = %.1f s\n", t_Nxy)

println("\nassembling bx = assemble(b, X)  (excitation vector -- needed by ALL strategies)...")
t0 = time(); bx = assemble(b, X); t_bx = time() - t0
@printf("  t_bx  = %.1f s\n", t_bx)

t_total = t_Zxx + t_Tyy + t_Nxy + t_bx
@printf("\ntotal (Zxx+Tyy+Nxy+bx) = %.1f s\n", t_total)
@printf("Zxx+bx only (what frozen-reuse actually needs)   = %.1f s  (%.1f%% of total)\n",
    t_Zxx + t_bx, 100*(t_Zxx+t_bx)/t_total)
@printf("Tyy+Nxy (what frozen-reuse SKIPS)                = %.1f s  (%.1f%% of total)\n",
    t_Tyy + t_Nxy, 100*(t_Tyy+t_Nxy)/t_total)

open(summary_file, "a") do io
    @printf(io, "%d,%.6g,%.6g,%.6g,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n",
        realization, κ, h, amplitude, dof, t_Zxx, t_Tyy, t_Nxy, t_bx, t_total)
end
println("\n[saved] appended to ", summary_file)

println("\n" * "="^70)
println("Complete -- ", now())
println("="^70)
