# [CJ-almond 3/4] Monte Carlo shape-uncertainty campaign on the NASA almond.
#
# 100 independent draws of the admissible vector field at the top amplitude,
# case U, ell = d/5. Every draw is solved with the FROZEN preconditioner --
# the nominal Calderon operator assembled once and reused -- which is the
# configuration the paper is about, and the only one that is affordable at
# this sample count. A subset is additionally solved FRESH, to anchor the
# frozen counts against the gold standard.
#
# WHY IT IS CHEAP. A frozen sample needs Zxx and bx on the perturbed
# geometry, and nothing else: Tyy and Nxy come from the nominal geometry
# through the DOF permutation. Those are exactly the two matrices the cost
# model of Section 5 identifies as dominant, so the campaign costs roughly
# what the primal EFIE alone costs. That ratio is measured by p31 on this
# machine and printed again here.
#
# NO SAMPLE IS EVER DROPPED. Validity is certified in advance by
# eps*sup|DV| <= 0.5, which `geo` re-checks and refuses to proceed without;
# a sample that fails to converge is recorded as non-converged, with its
# iteration count and true residual, and stays in the statistics.
#
# Usage, from the project root:
#   julia +1.11 --project scripts/p33_almond_campaign.jl
#   julia +1.11 --project scripts/p33_almond_campaign.jl --n 20 --nfresh 5
#
# Appends to data/sweep/p33_almond_campaign_summary.csv (one row per sample)
# and data/sweep/p33_almond_campaign_rcs.csv (one row per sample, the full
# bistatic cut). Safe to interrupt and resume.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using Exp25_CJH_KC_LocalMultiTrace
using Makeitso
using DrWatson
using CompScienceMeshes
using BEAST
using LinearAlgebra
using Statistics
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
κ0       = parse(Float64, argval("--kappa", string(ALMOND_KAPPA)))
nsamp    = parse(Int, argval("--n", "100"))
nfresh   = parse(Int, argval("--nfresh", "20"))
lvl      = parse(Int, argval("--level", "5"))
L        = almond_levels()[lvl]
seeds    = [5000 + i for i in 0:(nsamp-1)]
θ        = range(0, stop=pi, length=181)

outdir = projectdir("data", "sweep"); mkpath(outdir)
sumfile = joinpath(outdir, "p33_almond_campaign_summary.csv")
rcsfile = joinpath(outdir, "p33_almond_campaign_rcs.csv")
if !isfile(sumfile)
    open(sumfile, "w") do io
        println(io, "mesh,ell,kappa,level,case,seed,dof,eps_m,maxdisp_m,sup_DV," *
                    "cert,min_area,t_assembly_frozen_s,t_assembly_extra_s," *
                    "t_frozen_s,iters_frozen,conv_frozen,res_frozen," *
                    "t_fresh_s,iters_fresh,conv_fresh,res_fresh,has_fresh")
    end
end
if !isfile(rcsfile)
    open(rcsfile, "w") do io
        println(io, "seed,source," * join([@sprintf("%.6f", t) for t in θ], ','))
    end
end
done = Set{Int}()
for ln in readlines(sumfile)[2:end]
    p = split(ln, ','); length(p) < 6 && continue
    push!(done, parse(Int, p[6]))
end

println("="^78)
println("[CJ-almond 3/4] MONTE CARLO CAMPAIGN -- ", now())
@printf("mesh %s  ell %s  case %s  level %d (eps = %.3f mm = lambda/%.1f = %.2f h = %.1f%% thickness)\n", meshname, ellnm, case, L.level,
        L.eps_mm, L.lambda_over, L.eps_over_h, L.pct_thick)
println("samples $nsamp (frozen), of which $nfresh also solved fresh")
println("already done: $(length(done))")
println("="^78)

println("\n--- nominal discretisation (assembled once, reused by every sample) ---")
t0 = time()
d0 = make(SimAL.discretization; meshname, case, ell_name=ellnm, seed=0,
          eps=0.0, κ=κ0)
s0 = make(SimAL.spaces; meshname, case, ell_name=ellnm, seed=0, eps=0.0)
g0 = make(SimAL.geo; meshname, case, ell_name=ellnm, seed=0, eps=0.0)
fm = make(SimAL.formulation; κ=κ0)
t_nom = time() - t0
dof0 = length(d0.vectors.bx)
@printf("  %.1f s, dof = %d\n", t_nom, dof0)
Tyy0d = Matrix{ComplexF64}(d0.matrices.Tyy)
Nxy0d = Matrix{ComplexF64}(d0.matrices.Nxy)
edgefp0 = dof_edge_fingerprint(s0.X.fns, g0.Γ.faces)
nv0 = length(vertices(s0.Y.geo.parent))
fpY0 = dof_cell_fingerprint_filtered(s0.Y.fns, s0.Y.geo.mesh.faces, nv0)
Abil, Nbil, blin = fm.bilforms.A, fm.bilforms.Nform, fm.linforms.b

Tfar = BEAST.MWFarField3D(Maxwell3D.singlelayer(wavenumber=κ0))
dirs = [point(sin(t), 0.0, cos(t)) for t in θ]
rcs_of(u, X) = κ0^2/(4pi) .* abs2.(norm.(potential(Tfar, dirs, u, X)))

# nominal RCS, for the reference curve of Figure C
u0, ch0, t0s = solve_calderon_gmres(d0.matrices.Zxx, d0.vectors.bx,
                                    d0.matrices.Tyy, d0.matrices.Nxy)
rcs0 = rcs_of(Vector(u0), s0.X)
@printf("  nominal solve: %d iters (%.1f s); sigma(0) = %.4e m^2\n",
        ch0.iters, t0s, rcs0[1])
if !any(startswith(ln, "0,nominal") for ln in readlines(rcsfile))
    open(rcsfile, "a") do io
        println(io, "0,nominal," * join([@sprintf("%.8e", r) for r in rcs0], ','))
    end
end

for (i, sd) in enumerate(seeds)
    if sd in done
        println("sample $i/$nsamp seed $sd: already done, skipped"); continue
    end
    try
        gp = make(SimAL.geo; meshname, case, ell_name=ellnm, seed=sd, eps=L.eps)
        sp = make(SimAL.spaces; meshname, case, ell_name=ellnm, seed=sd, eps=L.eps)
        Xp, Yp = sp.X, sp.Y

        t = time()
        Zxx = assemble(Abil, Xp, Xp)
        bx  = assemble(blin, Xp)
        t_frz_asm = time() - t

        fpp  = dof_edge_fingerprint(Xp.fns, gp.Γ.faces)
        nvp  = length(vertices(Yp.geo.parent))
        fpYp = dof_cell_fingerprint_filtered(Yp.fns, Yp.geo.mesh.faces, nvp)
        permX, okX = build_permutation(edgefp0, fpp)
        permY, okY = build_permutation(fpY0, fpYp)
        (okX && okY) || error("permutation failed at seed $sd")
        ipX, ipY = invperm(permX), invperm(permY)

        u_a, ch_a, t_a = solve_calderon_gmres(Zxx, bx, Tyy0d[ipY, ipY],
                                              Nxy0d[ipX, ipY])
        bv = Vector(bx); nb = norm(bv)
        res_a = norm(bv .- Zxx * Vector(u_a)) / nb

        has_fresh = i <= nfresh
        t_extra = 0.0; t_g = NaN; it_g = -1; cv_g = false; res_g = NaN
        if has_fresh
            t = time()
            Tyyp = assemble(Abil, Yp, Yp)
            Nxyp = assemble(Nbil, Xp, Yp)
            t_extra = time() - t
            u_g, ch_g, t_g = solve_calderon_gmres(Zxx, bx, Tyyp, Nxyp)
            it_g = ch_g.iters; cv_g = ch_g.isconverged
            res_g = norm(bv .- Zxx * Vector(u_g)) / nb
        end

        rcs = rcs_of(Vector(u_a), Xp)
        open(rcsfile, "a") do io
            println(io, "$sd,frozen," * join([@sprintf("%.8e", r) for r in rcs], ','))
        end

        freshmsg = ""
        if has_fresh
            freshmsg = @sprintf(" | fresh %3d (%.0f s asm %.0f s)",
                                it_g, t_g, t_extra)
        end
        @printf("sample %3d/%d seed %d: disp %.3f mm cert %.3f | frozen %3d (%.0f s asm %.0f s)%s\n",
                i, nsamp, sd, 1e3*gp.maxdisp, gp.cert, ch_a.iters, t_a,
                t_frz_asm, freshmsg)

        open(sumfile, "a") do io
            @printf(io,"%s,%s,%.8g,%d,%s,%d,%d,%.8g,%.8g,%.6f,%.6f,%.6e,%.3f,%.3f,%.3f,%d,%s,%.4e,%.3f,%d,%s,%.4e,%s\n",
                meshname, ellnm, κ0, L.level, case, sd, length(bx), L.eps,
                gp.maxdisp, gp.sup_DV, gp.cert,
                min_triangle_area(gp.Γ.vertices, gp.Γ.faces),
                t_frz_asm, t_extra, t_a, ch_a.iters, ch_a.isconverged, res_a,
                t_g, it_g, cv_g, res_g, has_fresh)
        end
    catch err
        println("sample $i seed $sd [ERROR] ", err)
    end
end

println("\n" * "="^78)
println("[CJ-almond 3/4] campaign complete -- ", now())
println("Summary: ", sumfile)
println("RCS:     ", rcsfile)
println("="^78)
