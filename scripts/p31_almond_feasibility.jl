# [CJ-almond 1/4] Feasibility, calibration and resonance screening.
#
# RUN THIS FIRST, AND READ ITS OUTPUT, before launching p32 or p33. It is the
# only script here that is cheap (one nominal assembly plus one perturbed
# sample), and it produces the MEASURED per-stage timings from which the
# wall-clock estimates for the two campaigns are computed. No number in the
# paper may come from an estimate; the estimates exist only to decide whether
# to start a run.
#
# What it does:
#   1. loads and checks the shipped lambda/12 mesh (dof, orientation, volume);
#   2. assembles Zxx, Tyy, Nxy and bx SEPARATELY and times each, giving the
#      ratio t_T/t_Z that the cost model of Section 5 needs;
#   3. screens for an interior resonance by solving the nominal problem at
#      0.97 kappa, kappa and 1.03 kappa;
#   4. runs one perturbed sample at the top amplitude through plain, frozen
#      and fresh, timing each;
#   5. prints the projected wall clock for the sweep and the campaign, each
#      clearly labelled as an extrapolation from the measurements above.
#
# Usage, from the project root:
#   julia +1.11 --project scripts/p31_almond_feasibility.jl
#   julia +1.11 --project scripts/p31_almond_feasibility.jl --mesh almond_lam20
#
# Writes data/sweep/p31_almond_feasibility.csv.

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
κ0 = ALMOND_KAPPA
levels = almond_levels()
epstop = levels[end].eps
seed = parse(Int, argval("--seed", "1000"))
case = argval("--case", "U")
ellnm = argval("--ell", "d5")

println("="^78)
println("[CJ-almond 1/4] FEASIBILITY -- ", now())
println("="^78)
@printf("geometry     : NASA almond, d = %.6f m (9.936 in)\n", ALMOND_D)
@printf("wavelength   : %.4f mm  -> electrical size %.2f lambda, kappa*d = %.3f\n",
        1e3*ALMOND_LAMBDA, ALMOND_D/ALMOND_LAMBDA, κ0*ALMOND_D)
@printf("kappa        : %.4f m^-1   (f = %.4f GHz)\n", κ0, 299792458*κ0/(2pi)/1e9)
println("mesh         : $meshname")
println("\namplitude ladder (data/almond/levels.csv):")
for L in levels
    @printf("  level %d : eps = %6.3f mm = lambda/%.1f = %.2f h = %.1f%% of the seam half-thickness\n", L.level, L.eps_mm, L.lambda_over,
            L.eps_over_h, L.pct_thick)
end

# ---------------------------------------------------------------- 1. mesh
println("\n--- 1. nominal mesh ---")
bm = make(SimAL.basemesh; meshname)
Γ0 = bm.Γ0
@printf("  %d vertices, %d faces, RWG dof = %d, enclosed volume %.6e m^3\n",
        length(Γ0.vertices), length(Γ0.faces), 3*length(Γ0.faces)÷2, bm.vol)

# ------------------------------------------------- 2. assembly, component-wise
println("\n--- 2. assembly, timed component by component ---")
sp0 = make(SimAL.spaces; meshname, case, ell_name=ellnm, seed=0, eps=0.0)
fm0 = make(SimAL.formulation; κ=κ0)
X0, Y0 = sp0.X, sp0.Y
(;A, Nform) = fm0.bilforms
(;b) = fm0.linforms

t = time(); Zxx0 = assemble(A, X0, X0);     tZ = time() - t
t = time(); Tyy0 = assemble(A, Y0, Y0);     tT = time() - t
t = time(); Nxy0 = assemble(Nform, X0, Y0); tN = time() - t
t = time(); bx0  = assemble(b, X0);         tb = time() - t
dof = length(bx0)
@printf("  Zxx  %8.2f s      (primal EFIE, %d x %d)\n", tZ, dof, dof)
@printf("  Tyy  %8.2f s      (dual/BC)\n", tT)
@printf("  Nxy  %8.2f s      (duality pairing)\n", tN)
@printf("  bx   %8.2f s\n", tb)
@printf("  t_T / t_Z = %.2f      full assembly %.2f s, frozen-only (Zxx + bx) %.2f s, saving %.1f%%\n",
        tT/tZ, tZ+tT+tN+tb, tZ+tb, 100*(tT+tN)/(tZ+tT+tN+tb))

Tyy0d = Matrix{ComplexF64}(Tyy0)
Nxy0d = Matrix{ComplexF64}(Nxy0)

# ------------------------------------------------- 3. resonance screening
println("\n--- 3. resonance screening at 0.97 kappa, kappa, 1.03 kappa ---")
res_rows = []
for fac in (0.97, 1.0, 1.03)
    κ = fac * κ0
    fm = make(SimAL.formulation; κ)
    Aκ = fm.bilforms.A
    Z = assemble(Aκ, X0, X0)
    T_ = assemble(Aκ, Y0, Y0)
    N_ = assemble(fm.bilforms.Nform, X0, Y0)
    bb = assemble(fm.linforms.b, X0)
    _, chp, tp = solve_plain_gmres(Z, bb)
    _, chc, tc = solve_calderon_gmres(Z, bb, T_, N_)
    @printf("  %.2f kappa : plain %4d iters (%.1f s), Calderon %3d iters (%.1f s)\n",
            fac, chp.iters, tp, chc.iters, tc)
    push!(res_rows, (fac=fac, iters_plain=chp.iters, iters_cald=chc.iters))
end
med = sort([r.iters_cald for r in res_rows])[2]
flag = any(r -> r.iters_cald > 1.5*med, res_rows)
if flag
    println("  *** one of the three counts is more than 1.5x the median: " *
            "treat this frequency as a resonance candidate and shift kappa.")
else
    println("  no resonance signature: the three counts are within 1.5x of " *
            "each other.")
end

# ------------------------------------------------- 4. one perturbed sample
println("\n--- 4. one perturbed sample at the top amplitude ---")
fld = almond_field(case, ellnm, seed)
@printf("  case %s, ell = %s, seed %d, eps = %.3f mm, sup|DV| = %.2f m^-1, eps*sup|DV| = %.3f\n", case, ellnm, seed, 1e3*epstop, fld.sup_DV,
        almond_certificate(fld, epstop))

t0 = time()
gp = make(SimAL.geo; meshname, case, ell_name=ellnm, seed, eps=epstop)
spp = make(SimAL.spaces; meshname, case, ell_name=ellnm, seed, eps=epstop)
Xp, Yp = spp.X, spp.Y
t = time(); Zxxp = assemble(A, Xp, Xp);     tZp = time() - t
t = time(); bxp  = assemble(b, Xp);         tbp = time() - t
t = time(); Tyyp = assemble(A, Yp, Yp);     tTp = time() - t
t = time(); Nxyp = assemble(Nform, Xp, Yp); tNp = time() - t
@printf("  max displacement %.4f mm (%.2f%% of eps -- the reference sampling is finer than this mesh)\n", 1e3*gp.maxdisp, 100*gp.maxdisp/epstop)
@printf("  assembly: Zxx %.2f s, bx %.2f s, Tyy %.2f s, Nxy %.2f s\n",
        tZp, tbp, tTp, tNp)

edgefp0 = dof_edge_fingerprint(X0.fns, Γ0.faces)
nv0 = length(vertices(Y0.geo.parent))
fpY0 = dof_cell_fingerprint_filtered(Y0.fns, Y0.geo.mesh.faces, nv0)
edgefpp = dof_edge_fingerprint(Xp.fns, gp.Γ.faces)
nvp = length(vertices(Yp.geo.parent))
fpYp = dof_cell_fingerprint_filtered(Yp.fns, Yp.geo.mesh.faces, nvp)
permX, okX = build_permutation(edgefp0, edgefpp)
permY, okY = build_permutation(fpY0, fpYp)
println("  permutation valid?  X: $okX   Y: $okY")
(okX && okY) || error("DOF permutation construction failed on the almond")
ipX, ipY = invperm(permX), invperm(permY)
Tyy0a = Tyy0d[ipY, ipY]
Nxy0a = Nxy0d[ipX, ipY]

_, chp, tp = solve_plain_gmres(Zxxp, bxp)
_, cha, ta = solve_calderon_gmres(Zxxp, bxp, Tyy0a, Nxy0a)
_, chg, tg = solve_calderon_gmres(Zxxp, bxp, Tyyp, Nxyp)
@printf("  [plain]  %4d iters (%.1f s)\n", chp.iters, tp)
@printf("  [frozen] %4d iters (%.1f s)\n", cha.iters, ta)
@printf("  [fresh]  %4d iters (%.1f s)\n", chg.iters, tg)
@printf("  frozen - fresh = %d iterations   (ratio %.3f)\n",
        cha.iters - chg.iters, cha.iters / chg.iters)

# ------------------------------------------------- 5. projection
println("\n--- 5. projected wall clock  [EXTRAPOLATION, not a measurement] ---")
t_full = tZp + tbp + tTp + tNp
t_froz = tZp + tbp
t_sw = t_full + tp + ta + tg
t_cp = t_froz + ta
@printf("  per sweep sample    (full assembly + 3 solves) : %7.1f s\n", t_sw)
@printf("  per campaign sample (Zxx+bx + frozen solve)    : %7.1f s\n", t_cp)
@printf("  sweep      100 samples : %6.2f h\n", 100*t_sw/3600)
@printf("  campaign   100 frozen  : %6.2f h\n", 100*t_cp/3600)
@printf("  campaign    20 fresh   : %6.2f h  (subset for the reference)\n",
        20*t_sw/3600)
@printf("  lambda/20 check, 3 samples, dof x %.2f, assembly ~ dof^2 : %6.2f h\n",
        7269/2673, 3*t_sw*(7269/2673)^2/3600)
println("\n  These are projections from the single sample above. Stop and")
println("  reconsider if any campaign exceeds 12 h.")

outdir = projectdir("data", "sweep"); mkpath(outdir)
csv = joinpath(outdir, "p31_almond_feasibility.csv")
open(csv, "w") do io
    println(io, "quantity,value")
    for (k, v) in (("dof", dof), ("t_Zxx_s", tZ), ("t_Tyy_s", tT),
                   ("t_Nxy_s", tN), ("t_bx_s", tb), ("t_T_over_t_Z", tT/tZ),
                   ("kappa", κ0), ("eps_top_m", epstop),
                   ("maxdisp_m", gp.maxdisp), ("cert", almond_certificate(fld, epstop)),
                   ("iters_plain", chp.iters), ("iters_frozen", cha.iters),
                   ("iters_fresh", chg.iters),
                   ("t_solve_plain_s", tp), ("t_solve_frozen_s", ta),
                   ("t_solve_fresh_s", tg))
        println(io, "$k,$v")
    end
    # The resonance screen is the one result that decides whether the whole
    # campaign is run at this kappa, so it belongs in the file and not only
    # in the terminal.
    for r in res_rows
        println(io, "iters_plain_at_$(r.fac)_kappa,$(r.iters_plain)")
        println(io, "iters_calderon_at_$(r.fac)_kappa,$(r.iters_cald)")
    end
    println(io, "resonance_median_calderon,$med")
    println(io, "resonance_flag,$flag")
end
println("\nSaved ", csv)
println("="^78)
