# [CJ-15] Fichera corner sweep: perturb AT the reentrant corner, and at a
# finer mesh than the flat-face study.
#
# The flat-face sweep (scripts/p10_fichera_amplitude_sweep.jl, h=0.3,
# 1212 dof) returns a flat 26 iterations with a zero frozen-minus-fresh
# gap at every amplitude, with a realized displacement of 0.48h -- below
# one mesh width, and the only one of the three geometries for which that
# is true. It is therefore open to the reading that the perturbation is
# not binding. This script removes both objections at once: the
# perturbation now reaches the reentrant corner
# (postproc/shape_perturbation_fichera_corner.jl) and h is a free
# argument.
#
# If the gap stays at zero here, that is evidence that the geometric
# hypothesis extends beyond C^{1,1}. If it does not, the flat-face result
# was indeed measuring a perturbation that never touched the singularity,
# and that is worth knowing before a referee finds it.
#
# Usage, from the project root:
#   julia +1.11 --project scripts/p26_fichera_corner_sweep.jl
#   julia +1.11 --project scripts/p26_fichera_corner_sweep.jl --h 0.2
#   julia +1.11 --project scripts/p26_fichera_corner_sweep.jl --h 0.15 --amps 0.02,0.05,0.10,0.15
#
# Appends to data/sweep/p26_fichera_corner_sweep_summary.csv, one row per
# amplitude, with the same columns as the flat-face sweep plus `h` and
# `corner_disp` (the displacement of the reentrant corner itself, which
# is sqrt(3)*amplitude by construction and is reported as a check).
# Wrapped in try/catch per amplitude; safe to interrupt and resume.

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

module SimFC
include("../problems/p10_perturbed_fichera_corner.jl")
include("../methods/EFIE.jl")
end

include("../methods/EFIE_manual_solves.jl")
include(joinpath(@__DIR__, "p23_fingerprints.jl"))   # dof_edge_fingerprint,
                                                     # dof_cell_fingerprint_filtered,
                                                     # build_permutation

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    (i === nothing || i == length(ARGS)) && return default
    return ARGS[i+1]
end

h          = parse(Float64, argval("--h", "0.15"))
κ          = parse(Float64, argval("--kappa", "2.0"))
amplitudes = parse.(Float64, split(argval("--amps", "0.02,0.05,0.10,0.15"), ","))

# triangle_area / min_triangle_area come from p23_fingerprints.jl

# index of the vertex sitting at the reentrant corner (0,0,0)
function corner_index(verts; tol=1e-6)
    for (i, v) in enumerate(verts)
        if abs(v[1]) < tol && abs(v[2]) < tol && abs(v[3]) < tol
            return i
        end
    end
    return -1
end

outdir = projectdir("data", "sweep")
mkpath(outdir)
summary_file = joinpath(outdir, "p26_fichera_corner_sweep_summary.csv")
if !isfile(summary_file)
    open(summary_file, "w") do io
        println(io, "h,kappa,amplitude,dof,maxdisp,corner_disp,min_triangle_area,valid_mesh," *
                    "assembly_time_s,solve_time_plain_s,iters_plain,converged_plain," *
                    "solve_time_frozen_aligned_s,iters_frozen_aligned,converged_frozen_aligned," *
                    "solve_time_fresh_s,iters_fresh,converged_fresh," *
                    "trueres_plain,trueres_frozen_aligned,trueres_fresh")
    end
end

println("="^74)
println("[CJ-15] FICHERA CORNER sweep -- started ", now())
println("h = $h   kappa = $κ   amplitudes = $amplitudes")
println("perturbed faces: the three reentrant notch squares x=0, y=0, z=0")
println("="^74)

println("\n--- nominal (unperturbed) Fichera discretization + spaces ---")
t0 = time()
disc0   = make(SimFC.discretization; h, κ, realization=0, amplitude=0.0)
spaces0 = make(SimFC.spaces;         h,    realization=0, amplitude=0.0)
geo0    = make(SimFC.geo;            h,    realization=0, amplitude=0.0)
t_assembly0 = time() - t0
dof0  = length(disc0.vectors.bx)
verts0 = geo0.Γ.vertices
ic = corner_index(verts0)
@printf("  nominal assembly: %.1f s   (dof = %d)\n", t_assembly0, dof0)
if ic < 0
    @warn "no mesh vertex sits exactly at the reentrant corner (0,0,0); " *
          "corner_disp will be reported as NaN"
else
    @printf("  reentrant corner is vertex %d\n", ic)
end

Tyy0_dense = Matrix{ComplexF64}(disc0.matrices.Tyy)
Nxy0_dense = Matrix{ComplexF64}(disc0.matrices.Nxy)

edgefp0 = dof_edge_fingerprint(spaces0.X.fns, geo0.Γ.faces)
n_orig_verts0 = length(vertices(spaces0.Y.geo.parent))
fpY0 = dof_cell_fingerprint_filtered(spaces0.Y.fns, spaces0.Y.geo.mesh.faces, n_orig_verts0)
println("  nominal DOF fingerprints built.")

for amp in amplitudes
    println("\n--- amplitude = $amp ---")
    try
        t0 = time()
        discp   = make(SimFC.discretization; h, κ, realization=1, amplitude=amp)
        spacesp = make(SimFC.spaces;         h,    realization=1, amplitude=amp)
        geop    = make(SimFC.geo;            h,    realization=1, amplitude=amp)
        t_assembly = time() - t0
        dof = length(discp.vectors.bx)
        @assert dof == dof0 "DOF mismatch: perturbed=$dof vs nominal=$dof0"
        @printf("  perturbed assembly: %.1f s   (dof = %d)\n", t_assembly, dof)

        vertsp  = geop.Γ.vertices
        maxdisp = maximum(norm(vertsp[i] .- verts0[i]) for i in eachindex(verts0))
        cdisp   = ic > 0 ? norm(vertsp[ic] .- verts0[ic]) : NaN
        minarea = min_triangle_area(vertsp, geop.Γ.faces)
        valid_mesh = minarea > 0
        @printf("  max vertex displacement %.4f   corner displacement %.4f  (sqrt(3)*amp = %.4f)\n",
                maxdisp, cdisp, sqrt(3) * abs(amp))
        @printf("  min_triangle_area = %.3e   valid_mesh = %s\n", minarea, valid_mesh)

        edgefpp = dof_edge_fingerprint(spacesp.X.fns, geop.Γ.faces)
        permX, okX = build_permutation(edgefp0, edgefpp)
        n_orig_vertsp = length(vertices(spacesp.Y.geo.parent))
        fpYp = dof_cell_fingerprint_filtered(spacesp.Y.fns, spacesp.Y.geo.mesh.faces, n_orig_vertsp)
        permY, okY = build_permutation(fpY0, fpYp)
        @printf("  permutation valid?  X: %s   Y: %s\n", okX, okY)
        (okX && okY) || error("DOF permutation construction failed at amplitude=$amp")

        invpermX = invperm(permX); invpermY = invperm(permY)
        Tyy0_aligned = Tyy0_dense[invpermY, invpermY]
        Nxy0_aligned = Nxy0_dense[invpermX, invpermY]

        Zxxp = discp.matrices.Zxx
        Tyyp = discp.matrices.Tyy
        Nxyp = discp.matrices.Nxy
        bxp  = discp.vectors.bx

        u_p, ch_p, t_p = solve_plain_gmres(Zxxp, bxp)
        @printf("  [plain]          %.2f s  (%d iters, converged=%s)\n", t_p, ch_p.iters, ch_p.isconverged)
        u_a, ch_a, t_a = solve_calderon_gmres(Zxxp, bxp, Tyy0_aligned, Nxy0_aligned)
        @printf("  [frozen-aligned] %.2f s  (%d iters, converged=%s)\n", t_a, ch_a.iters, ch_a.isconverged)
        u_g, ch_g, t_g = solve_calderon_gmres(Zxxp, bxp, Tyyp, Nxyp)
        @printf("  [fresh]          %.2f s  (%d iters, converged=%s)\n", t_g, ch_g.iters, ch_g.isconverged)
        @printf("  >>> frozen - fresh = %d iterations\n", ch_a.iters - ch_g.iters)

        bxpvec = Vector(bxp); nb = norm(bxpvec)
        trures_p = norm(bxpvec .- Zxxp*Vector(u_p)) / nb
        trures_a = norm(bxpvec .- Zxxp*Vector(u_a)) / nb
        trures_g = norm(bxpvec .- Zxxp*Vector(u_g)) / nb
        @printf("  true residuals: plain=%.3e  frozen=%.3e  fresh=%.3e\n", trures_p, trures_a, trures_g)

        open(summary_file, "a") do io
            @printf(io, "%.6g,%.6g,%.6g,%d,%.6f,%.6f,%.6e,%s,%.4f,%.4f,%d,%s,%.4f,%d,%s,%.4f,%d,%s,%.6e,%.6e,%.6e\n",
                h, κ, amp, dof, maxdisp, cdisp, minarea, valid_mesh, t_assembly,
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

println("\n" * "="^74)
println("[CJ-15] Fichera corner sweep complete -- ", now())
println("Summary: ", summary_file)
println("="^74)
