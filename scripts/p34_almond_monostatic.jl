# [CJ-almond 4/4] Monostatic RCS azimuth sweep on the NASA almond.
#
# The benchmark's canonical presentation: backscatter cross-section against
# azimuth in the body's symmetry plane, for the two principal polarisations.
# Here it serves one purpose only -- to show that the frozen preconditioner
# returns the SAME far field as a freshly assembled one, over the whole
# angular sweep and at the top perturbation amplitude. Remark 7.3 states the
# scope: the bound of Theorem 5.1 is on the solver, and an observable
# quadratic in the solution inherits it only through the solution error,
# which is what this figure exhibits rather than proves.
#
# COST. The system matrix does not depend on the incidence direction, so a
# whole azimuth sweep is ONE assembly and N right-hand sides. That is why a
# monostatic sweep is affordable here at all, and it is worth saying in the
# paper: the frozen preconditioner is reused across both the shape samples
# and the incidence angles.
#
# Convention: azimuth phi is measured from the tip. The wave propagates along
#   dhat(phi) = (-cos phi, -sin phi, 0),
# so phi = 0 is nose-on (arriving from the sharp tip, +x) and phi = 180 deg
# is tail-on (the blunt end). Polarisations: V is out of the sweep plane
# (zhat), H is in it (dhat x zhat). Backscatter direction is -dhat.
#
# Usage, from the project root:
#   julia +1.11 --project scripts/p34_almond_monostatic.jl
#   julia +1.11 --project scripts/p34_almond_monostatic.jl --nphi 73 --seed 5000
#
# Writes data/sweep/p34_almond_monostatic.csv.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using Exp25_CJH_KC_LocalMultiTrace
using Makeitso
using DrWatson
using CompScienceMeshes
using BEAST
using LinearAlgebra
using Printf
using Dates
ENV["DRWATSON_WARN_DIRTY"] = "false"

module SimAL
include("../problems/p11_perturbed_almond.jl")
include("../methods/EFIE.jl")
end

include("../methods/EFIE_manual_solves.jl")
include(joinpath(@__DIR__, "p23_fingerprints.jl"))
include("../postproc/almond_geometry.jl")

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    (i === nothing || i == length(ARGS)) && return default
    return ARGS[i+1]
end

meshname = argval("--mesh", "almond_lam12")
ellnm    = argval("--ell", "d5")
case     = argval("--case", "U")
κ        = parse(Float64, argval("--kappa", string(ALMOND_KAPPA)))
seed     = parse(Int, argval("--seed", "5000"))
lvl      = parse(Int, argval("--level", "5"))
nphi     = parse(Int, argval("--nphi", "73"))
L        = almond_levels()[lvl]
phis     = range(0, stop=pi, length=nphi)

Top  = Maxwell3D.singlelayer(wavenumber=κ)
Tfar = BEAST.MWFarField3D(Top)

"""Right-hand side for one incidence direction and polarisation."""
function rhs(X, dhat, phat)
    @hilbertspace k
    Einc = Maxwell3D.planewave(direction=dhat, polarization=phat, wavenumber=κ)
    e = (n × Einc) × n
    return assemble(e[k], X)
end

"""Backscatter cross-section from a solution vector."""
function backscatter(u, X, dhat)
    F = potential(Tfar, [-dhat], u, X)
    return κ^2/(4pi) * abs2(norm(F[1]))
end

println("="^78)
println("[CJ-almond 4/4] MONOSTATIC AZIMUTH SWEEP -- ", now())
@printf("mesh %s  kappa %.4f  (%.2f lambda)  %d azimuths, 0 to 180 deg\n",
        meshname, κ, ALMOND_D/(2pi/κ), nphi)
@printf("perturbed sample: case %s, ell %s, seed %d, level %d (eps = %.3f mm = lambda/%.1f)\n", case, ellnm, seed, L.level,
        L.eps_mm, L.lambda_over)
println("="^78)

# nominal: assemble once, keep Tyy/Nxy for freezing
println("\n--- nominal ---")
d0 = make(SimAL.discretization; meshname, case, ell_name=ellnm, seed=0,
          eps=0.0, κ)
s0 = make(SimAL.spaces; meshname, case, ell_name=ellnm, seed=0, eps=0.0)
g0 = make(SimAL.geo; meshname, case, ell_name=ellnm, seed=0, eps=0.0)
X0 = s0.X
Tyy0d = Matrix{ComplexF64}(d0.matrices.Tyy)
Nxy0d = Matrix{ComplexF64}(d0.matrices.Nxy)
edgefp0 = dof_edge_fingerprint(X0.fns, g0.Γ.faces)
nv0 = length(vertices(s0.Y.geo.parent))
fpY0 = dof_cell_fingerprint_filtered(s0.Y.fns, s0.Y.geo.mesh.faces, nv0)

# perturbed: full assembly, so that fresh is available as the reference
println("--- perturbed ---")
gp = make(SimAL.geo; meshname, case, ell_name=ellnm, seed, eps=L.eps)
sp = make(SimAL.spaces; meshname, case, ell_name=ellnm, seed, eps=L.eps)
dp = make(SimAL.discretization; meshname, case, ell_name=ellnm, seed,
          eps=L.eps, κ)
Xp = sp.X
@printf("  max displacement %.3f mm, eps*sup|DV| = %.3f\n",
        1e3*gp.maxdisp, gp.cert)
fpp = dof_edge_fingerprint(Xp.fns, gp.Γ.faces)
nvp = length(vertices(sp.Y.geo.parent))
fpYp = dof_cell_fingerprint_filtered(sp.Y.fns, sp.Y.geo.mesh.faces, nvp)
permX, okX = build_permutation(edgefp0, fpp)
permY, okY = build_permutation(fpY0, fpYp)
(okX && okY) || error("permutation failed")
ipX, ipY = invperm(permX), invperm(permY)
Tyy0a, Nxy0a = Tyy0d[ipY, ipY], Nxy0d[ipX, ipY]

outdir = projectdir("data", "sweep"); mkpath(outdir)
csv = joinpath(outdir, "p34_almond_monostatic.csv")
open(csv, "w") do io
    println(io, "phi_deg,pol,geometry,strategy,sigma_m2,iters,seconds")
end

for (ip, phi) in enumerate(phis)
    dhat = point(-cos(phi), -sin(phi), 0.0)
    zhat = point(0.0, 0.0, 1.0)
    hhat = cross(dhat, zhat)
    for (polname, phat) in (("V", zhat), ("H", hhat))
        b0 = rhs(X0, dhat, phat)
        u0, c0, t0 = solve_calderon_gmres(d0.matrices.Zxx, b0,
                                          d0.matrices.Tyy, d0.matrices.Nxy)
        s_nom = backscatter(Vector(u0), X0, dhat)

        bp = rhs(Xp, dhat, phat)
        ua, ca, ta = solve_calderon_gmres(dp.matrices.Zxx, bp, Tyy0a, Nxy0a)
        ug, cg, tg = solve_calderon_gmres(dp.matrices.Zxx, bp,
                                          dp.matrices.Tyy, dp.matrices.Nxy)
        s_a = backscatter(Vector(ua), Xp, dhat)
        s_g = backscatter(Vector(ug), Xp, dhat)

        open(csv, "a") do io
            @printf(io,"%.4f,%s,nominal,fresh,%.8e,%d,%.3f\n",
                    rad2deg(phi), polname, s_nom, c0.iters, t0)
            @printf(io,"%.4f,%s,perturbed,frozen,%.8e,%d,%.3f\n",
                    rad2deg(phi), polname, s_a, ca.iters, ta)
            @printf(io,"%.4f,%s,perturbed,fresh,%.8e,%d,%.3f\n",
                    rad2deg(phi), polname, s_g, cg.iters, tg)
        end
        @printf("  phi %6.1f deg  %s : nominal %.3e | perturbed frozen %.3e (%d it) fresh %.3e (%d it)  rel.diff %.2e\n",
                rad2deg(phi), polname, s_nom, s_a, ca.iters, s_g, cg.iters,
                abs(s_a - s_g) / max(s_g, eps(Float64)))
    end
end

println("\nSaved ", csv)
println("="^78)
