# [CJ-08 follow-up] WHY does the frozen path cost ~4x more per outer iteration?
#
# p23 found: hybrid (frozen T_YY + fresh N_XY) gives EXACTLY the same outer
# iteration counts and the same solution as fully frozen, at every amplitude,
# but runs ~4x faster per outer iteration. Two explanations are possible:
#
#   (A) CONDITIONING. The frozen N_XY,0 is mismatched to the perturbed
#       geometry, so the inner GMRES solves against it need more iterations.
#       This is what Section 5.3 of the manuscript currently asserts.
#
#   (B) DATA STRUCTURE. To apply the DOF permutation, the frozen path
#       densifies: Nxy0_dense = Matrix{ComplexF64}(disc0.matrices.Nxy), then
#       permutes. The fresh path passes BEAST's native (sparse/structured)
#       Nxy. N_XY enters through TWO inner GMRES solves per outer iteration,
#       so a dense N_XY costs 2 x (inner iters) dense matvecs per outer step,
#       while T_YY is applied once. A dense-vs-sparse gap would show up here
#       and nowhere else.
#
# (A) and (B) have different consequences. Under (A) the hybrid is a better
# ALGORITHM and belongs in the paper. Under (B) the penalty is an artifact of
# how we store the frozen matrices, it applies equally to the numbers already
# reported in Section 5.3, and the fix is to keep N_XY in its native format
# (or apply the permutation lazily) rather than to refresh it.
#
# Three measurements, all at one amplitude, all from cache:
#   1. types of the matrices actually being applied
#   2. ||Nxy0_aligned - Nxyp|| / ||Nxyp||  -- is the pairing geometry-invariant?
#   3. frozen T_YY with a DENSIFIED fresh N_XY. This is the decisive run:
#        as slow as frozen  => (B) data structure
#        as fast as hybrid  => (A) conditioning
#
# Run from the repo root:
#   julia +1.11 --project scripts/p24_nxy_invariance_check.jl
#
# Takes a couple of minutes; everything comes from the Makeitso cache.

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

radius      = 1.0
h           = 0.1
κ           = 2.0
lmin, lmax  = 2, 6
realization = 1
amp         = 0.30                    # mid-range, well away from JIT warmup
nominal_lmin, nominal_lmax, nominal_amplitude = 2, 6, 0.03

include(joinpath(@__DIR__, "p23_fingerprints.jl"))   # see note at end of file

println("="^72)
println("[CJ-08 follow-up] N_XY invariance / dense-vs-native check -- ", now())
@printf("h=%.4g  kappa=%.4g  realization=%d  amplitude=%.4g\n", h, κ, realization, amp)
println("="^72)

disc0 = make(Sim2.discretization; h, κ, radius, realization=0,
    lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)
spaces0 = make(Sim2.spaces; h, radius, realization=0,
    lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)
geo0 = make(Sim2.geo; h, radius, realization=0,
    lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)

discp   = make(Sim2.discretization; h, κ, radius, realization, lmin, lmax, amplitude=amp)
spacesp = make(Sim2.spaces; h, radius, realization, lmin, lmax, amplitude=amp)
geop    = make(Sim2.geo; h, radius, realization, lmin, lmax, amplitude=amp)

# ---------------------------------------------------- 1. what are we applying?
println("\n--- 1. matrix types actually passed to the preconditioner ---")
println("  nominal  Nxy : ", typeof(disc0.matrices.Nxy))
println("  nominal  Tyy : ", typeof(disc0.matrices.Tyy))
println("  perturbed Nxy: ", typeof(discp.matrices.Nxy))
println("  perturbed Tyy: ", typeof(discp.matrices.Tyy))
n0 = disc0.matrices.Nxy
@printf("  nominal Nxy: %d x %d", size(n0,1), size(n0,2))
try
    @printf("   nnz = %d  (density %.3f%%)\n", count(!iszero, Matrix(n0)),
        100*count(!iszero, Matrix(n0))/length(Matrix(n0)))
catch; println(); end

Tyy0_dense = Matrix{ComplexF64}(disc0.matrices.Tyy)
Nxy0_dense = Matrix{ComplexF64}(disc0.matrices.Nxy)

faces0  = geo0.Γ.faces
edgefp0 = dof_edge_fingerprint(spaces0.X.fns, faces0)
fpY0    = dof_cell_fingerprint_filtered(spaces0.Y.fns, spaces0.Y.geo.mesh.faces,
                                        length(vertices(spaces0.Y.geo.parent)))
edgefpp = dof_edge_fingerprint(spacesp.X.fns, geop.Γ.faces)
fpYp    = dof_cell_fingerprint_filtered(spacesp.Y.fns, spacesp.Y.geo.mesh.faces,
                                        length(vertices(spacesp.Y.geo.parent)))
permX, okX = build_permutation(edgefp0, edgefpp)
permY, okY = build_permutation(fpY0, fpYp)
(okX && okY) || error("permutation failed")
invpermX, invpermY = invperm(permX), invperm(permY)

Tyy0_aligned = Tyy0_dense[invpermY, invpermY]
Nxy0_aligned = Nxy0_dense[invpermX, invpermY]

Zxxp = discp.matrices.Zxx
Tyyp = discp.matrices.Tyy
Nxyp = discp.matrices.Nxy
bxp  = discp.vectors.bx

# ------------------------------------- 2. is the RWG-BC pairing geometry-free?
println("\n--- 2. invariance of the duality pairing under the deformation ---")
Nxyp_dense = Matrix{ComplexF64}(Nxyp)
relN = norm(Nxy0_aligned - Nxyp_dense) / norm(Nxyp_dense)
absN = maximum(abs.(Nxy0_aligned - Nxyp_dense))
@printf("  ||Nxy0_aligned - Nxyp||_F / ||Nxyp||_F = %.6e\n", relN)
@printf("  max |entrywise difference|             = %.6e\n", absN)
println("  (near machine precision would mean the RWG-BC pairing is exactly")
println("   invariant under this connectivity-preserving deformation, which is")
println("   the discrete counterpart of the Piola/Nanson identity in Sec. 3.2.)")

relT = norm(Tyy0_aligned - Matrix{ComplexF64}(Tyyp)) / norm(Matrix{ComplexF64}(Tyyp))
@printf("  for contrast, ||Tyy0_aligned - Tyyp||_F / ||Tyyp||_F = %.6e\n", relT)

# --------------------------------------------- 3. the decisive timing control
println("\n--- 3. dense-vs-native control (same matrix content, different type) ---")
bxpvec = Vector(bxp); nb = norm(bxpvec)
trueres(u) = norm(bxpvec .- Zxxp*Vector(u)) / nb

# warm up JIT so the first timed solve is not penalised
solve_calderon_gmres(Zxxp, bxp, Tyy0_aligned, Nxyp)

u_f, ch_f, t_f = solve_calderon_gmres(Zxxp, bxp, Tyy0_aligned, Nxy0_aligned)
@printf("  frozen        (dense T0, dense N0 ): %3d iters  %7.3f s  %7.4f s/iter\n",
    ch_f.iters, t_f, t_f/max(ch_f.iters,1))

u_h, ch_h, t_h = solve_calderon_gmres(Zxxp, bxp, Tyy0_aligned, Nxyp)
@printf("  hybrid        (dense T0, native Np): %3d iters  %7.3f s  %7.4f s/iter\n",
    ch_h.iters, t_h, t_h/max(ch_h.iters,1))

u_d, ch_d, t_d = solve_calderon_gmres(Zxxp, bxp, Tyy0_aligned, Nxyp_dense)
@printf("  CONTROL       (dense T0, DENSE  Np): %3d iters  %7.3f s  %7.4f s/iter\n",
    ch_d.iters, t_d, t_d/max(ch_d.iters,1))

u_g, ch_g, t_g = solve_calderon_gmres(Zxxp, bxp, Tyyp, Nxyp)
@printf("  fresh         (native Tp, native Np): %3d iters  %7.3f s  %7.4f s/iter\n",
    ch_g.iters, t_g, t_g/max(ch_g.iters,1))

println("\n--- verdict ---")
tpi_f = t_f/max(ch_f.iters,1); tpi_h = t_h/max(ch_h.iters,1); tpi_d = t_d/max(ch_d.iters,1)
if tpi_d > 0.5*(tpi_f + tpi_h)
    println("  CONTROL tracks FROZEN => (B) the penalty is dense-vs-native storage,")
    println("  not conditioning. Section 5.3's explanation needs correcting, and the")
    println("  remedy is lazy permutation / native storage, not refreshing N_XY.")
else
    println("  CONTROL tracks HYBRID => (A) the penalty is genuine mismatch of the")
    println("  frozen N_XY,0 to the perturbed geometry. The hybrid is a better")
    println("  algorithm and belongs in the paper.")
end
@printf("  true residuals: frozen=%.3e hybrid=%.3e control=%.3e fresh=%.3e\n",
    trueres(u_f), trueres(u_h), trueres(u_d), trueres(u_g))

println("\n" * "="^72)
println("done -- ", now())
println("="^72)

# NOTE: this script expects the three fingerprint/permutation helpers
#   dof_edge_fingerprint, dof_cell_fingerprint_filtered, build_permutation
# in scripts/p23_fingerprints.jl. If that file does not exist, create it by
# copying those three function definitions verbatim out of
# scripts/p23_hybrid_precond_sweep.jl (they are identical there and in
# scripts/p10_PEC_sphere_perturbation_amplitude_sweep.jl).
