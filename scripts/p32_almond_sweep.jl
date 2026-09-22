# [CJ-almond 2/4] Amplitude sweep on the NASA almond: 5 levels x 2 cases
# x 10 seeds, plain / frozen / fresh.
#
# This is the almond analogue of p26 (Fichera corner) and p10 (sphere), with
# three differences that matter:
#
#   * the deformation is the paper's admissible class, T(x) = x + eps V(x)
#     with V a smooth vector field on R^3, not a normal displacement;
#   * every sample carries an injectivity certificate eps*sup|DV| <= 0.5,
#     checked in `geo`, so no sample is ever rejected and the statistics are
#     unbiased by construction;
#   * case U lets the sharp tip move with its neighbourhood, case T holds it
#     fixed with a C^1 cutoff. The U/T contrast is the almond's version of
#     the question Section 5.8 asks at the Fichera corner: does the frozen
#     preconditioner degrade because the SHAPE is non-smooth, or because the
#     perturbation acts where the shape is non-smooth?
#
# RESONANCE SCREENING (per Carlos's amendment 2). After each (level, case)
# cell completes, any sample whose FRESH iteration count exceeds 1.5x the
# cell median is flagged as a resonance candidate and RE-SOLVED at 1.03
# kappa. Both results are reported. Nothing is dropped.
#
# Usage, from the project root:
#   julia +1.11 --project scripts/p32_almond_sweep.jl
#   julia +1.11 --project scripts/p32_almond_sweep.jl --levels 1,2 --seeds 3
#   julia +1.11 --project scripts/p32_almond_sweep.jl --ell d10 --levels 1,2
#   julia +1.11 --project scripts/p32_almond_sweep.jl --nfresh 3   # ~2.1 h
#
# Appends to data/sweep/p32_almond_sweep_summary.csv; safe to interrupt and
# resume (a sample already present in the CSV is skipped).

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

meshname  = argval("--mesh", "almond_lam12")
ellnm     = argval("--ell", "d5")
κ0        = parse(Float64, argval("--kappa", string(ALMOND_KAPPA)))
nseeds    = parse(Int, argval("--seeds", "10"))
# Fresh solves need Tyy on every sample -- 204 s of the 220 s a sample costs,
# i.e. 93% of the sweep, spent on the reference rather than on the result.
# --nfresh n solves fresh on the first n seeds of each cell only; the frozen
# counts, which are what the paper claims, still come from every seed.
cases     = String.(split(argval("--cases", "U,T"), ","))
levels    = almond_levels()
wanted    = parse.(Int, split(argval("--levels", "1,2,3,4,5"), ","))
levels    = [L for L in levels if L.level in wanted]
seeds     = [1000 + i for i in 0:(nseeds-1)]

# --seedlist 1004,1007 solves exactly those seeds instead of the first
# nseeds. It exists to top up a cell whose worst sample was never solved
# fresh: --nfresh leaves the tail of each cell without a reference, and the
# tail is what a referee asks about. With --seedlist, --nfresh defaults to
# all of the listed seeds. --redo ignores the resume set so an existing
# frozen-only row can be replaced; the caller must then dedupe the summary
# file on (mesh, ell, level, case, seed), keeping the row that has a fresh
# solve. dedupe_sweep_csv.py does exactly that.
seedlist  = argval("--seedlist", "")
if !isempty(seedlist)
    seeds = parse.(Int, split(seedlist, ","))
end
nfresh    = parse(Int, argval("--nfresh", string(length(seeds))))
redo      = "--redo" in ARGS

outdir = projectdir("data", "sweep"); mkpath(outdir)
summary_file = joinpath(outdir, "p32_almond_sweep_summary.csv")
const HDR = "mesh,ell,kappa,level,case,seed,dof,eps_m,maxdisp_m,sup_DV,cert," *
            "min_area,assembly_time_s," *
            "t_plain_s,iters_plain,conv_plain," *
            "t_frozen_s,iters_frozen,conv_frozen," *
            "t_fresh_s,iters_fresh,conv_fresh," *
            "res_plain,res_frozen,res_fresh,resonance_flag,iters_fresh_1p03"
if !isfile(summary_file)
    open(summary_file, "w") do io; println(io, HDR); end
end

done = Set{Tuple{String,String,Int,String,Int}}()
for ln in readlines(summary_file)[2:end]
    p = String.(split(ln, ','))
    length(p) < 6 && continue
    push!(done, (p[1], p[2], parse(Int, p[4]), p[5], parse(Int, p[6])))
end

println("="^78)
println("[CJ-almond 2/4] ALMOND AMPLITUDE SWEEP -- ", now())
println("mesh $meshname   ell $ellnm   kappa $(round(κ0, digits=4))   ",
        "levels $(join([L.level for L in levels], ','))   cases $(join(cases, ','))",
        "   seeds $(length(seeds))")
println("already in the summary: $(length(done)) samples (they will be skipped)")
println("="^78)

# ------------------------------------------------------------- nominal
println("\n--- nominal discretisation ---")
t0 = time()
d0 = make(SimAL.discretization; meshname, case=cases[1], ell_name=ellnm,
          seed=0, eps=0.0, κ=κ0)
s0 = make(SimAL.spaces; meshname, case=cases[1], ell_name=ellnm, seed=0, eps=0.0)
g0 = make(SimAL.geo; meshname, case=cases[1], ell_name=ellnm, seed=0, eps=0.0)
@printf("  assembled in %.1f s, dof = %d\n", time()-t0, length(d0.vectors.bx))
dof0 = length(d0.vectors.bx)
Tyy0d = Matrix{ComplexF64}(d0.matrices.Tyy)
Nxy0d = Matrix{ComplexF64}(d0.matrices.Nxy)
edgefp0 = dof_edge_fingerprint(s0.X.fns, g0.Γ.faces)
nv0 = length(vertices(s0.Y.geo.parent))
fpY0 = dof_cell_fingerprint_filtered(s0.Y.fns, s0.Y.geo.mesh.faces, nv0)
println("  nominal fingerprints built.")

"""One sample, end to end. Returns a NamedTuple of everything logged.

`dofresh = false` skips the freshly assembled Calderon reference, and with it
the assembly of Tyy and Nxy on the perturbed geometry -- which is the entire
reason a sweep sample is expensive. The matrices are assembled here rather
than through the `discretization` target precisely so the dual pair can be
left out.
"""
function run_sample(case, seed, L, dofresh::Bool; κ = κ0)
    eps = L.eps
    t0 = time()
    sp = make(SimAL.spaces; meshname, case, ell_name=ellnm, seed, eps)
    gp = make(SimAL.geo;    meshname, case, ell_name=ellnm, seed, eps)
    fm = make(SimAL.formulation; κ)
    Abil, Nbil, blin = fm.bilforms.A, fm.bilforms.Nform, fm.linforms.b
    Zxx = assemble(Abil, sp.X, sp.X)
    bx  = assemble(blin, sp.X)
    Tyy = dofresh ? assemble(Abil, sp.Y, sp.Y) : nothing
    Nxy = dofresh ? assemble(Nbil, sp.X, sp.Y) : nothing
    t_asm = time() - t0
    dof = length(bx)
    dof == dof0 || error("dof mismatch $dof vs $dof0")

    fpp = dof_edge_fingerprint(sp.X.fns, gp.Γ.faces)
    nvp = length(vertices(sp.Y.geo.parent))
    fpYp = dof_cell_fingerprint_filtered(sp.Y.fns, sp.Y.geo.mesh.faces, nvp)
    permX, okX = build_permutation(edgefp0, fpp)
    permY, okY = build_permutation(fpY0, fpYp)
    (okX && okY) || error("permutation failed: case=$case seed=$seed level=$(L.level)")
    ipX, ipY = invperm(permX), invperm(permY)

    u_p, ch_p, t_p = solve_plain_gmres(Zxx, bx)
    u_a, ch_a, t_a = solve_calderon_gmres(Zxx, bx, Tyy0d[ipY, ipY], Nxy0d[ipX, ipY])

    bv = Vector(bx); nb = norm(bv)
    r(u) = norm(bv .- Zxx * Vector(u)) / nb

    t_g = NaN; it_g = -1; cv_g = false; res_g = NaN
    if dofresh
        u_g, ch_g, t_g = solve_calderon_gmres(Zxx, bx, Tyy, Nxy)
        it_g = ch_g.iters; cv_g = ch_g.isconverged; res_g = r(u_g)
    end
    return (; dof, t_asm, gp, dofresh,
            minarea = min_triangle_area(gp.Γ.vertices, gp.Γ.faces),
            t_p, ch_p, t_a, ch_a, t_g, it_g, cv_g, res_g,
            res_p = r(u_p), res_a = r(u_a))
end

for L in levels, case in cases
    println("\n" * "-"^70)
    @printf("level %d (eps = %.3f mm = lambda/%.1f = %.2f h = %.1f%% thickness), case %s\n", L.level, L.eps_mm, L.lambda_over, L.eps_over_h,
            L.pct_thick, case)
    println("-"^70)
    cell = []                       # (seed, iters_frozen, row-as-string)
    for (j, sd) in enumerate(seeds)
        key = (meshname, ellnm, L.level, case, sd)
        if key in done && !redo
            println("  seed $sd: already done, skipped")
            continue
        end
        try
            R = run_sample(case, sd, L, j <= nfresh)
            freshmsg = "  fresh not solved"
            if R.dofresh
                freshmsg = @sprintf("  fresh %3d (%.0f s)  |  f-f = %d",
                                    R.it_g, R.t_g, R.ch_a.iters - R.it_g)
            end
            @printf("  seed %d: disp %.3f mm  cert %.3f  |  plain %4d (%.0f s)  frozen %3d (%.0f s)%s\n",
                    sd, 1e3*R.gp.maxdisp, R.gp.cert, R.ch_p.iters, R.t_p,
                    R.ch_a.iters, R.t_a, freshmsg)
            row = @sprintf("%s,%s,%.8g,%d,%s,%d,%d,%.8g,%.8g,%.6f,%.6f,%.6e,%.3f,%.3f,%d,%s,%.3f,%d,%s,%.3f,%d,%s,%.4e,%.4e,%.4e,%s,%d",
                    meshname, ellnm, κ0, L.level, case, sd, R.dof, L.eps,
                    R.gp.maxdisp, R.gp.sup_DV, R.gp.cert, R.minarea, R.t_asm,
                    R.t_p, R.ch_p.iters, R.ch_p.isconverged,
                    R.t_a, R.ch_a.iters, R.ch_a.isconverged,
                    R.t_g, R.it_g, R.cv_g,
                    R.res_p, R.res_a, R.res_g, false, -1)
            push!(cell, (sd, R.ch_a.iters, row))
        catch err
            println("  [ERROR] seed $sd: ", err)
        end
    end
    isempty(cell) && continue

    # ---- resonance screening within the cell.
    # Screened on the FROZEN count, not the fresh one: it is available for
    # every sample whatever --nfresh is, and a resonance shows in both.
    m = median([c[2] for c in cell])
    flagged = [c for c in cell if c[2] > 1.5 * m]
    @printf("  cell median frozen count %.1f; %d resonance candidate(s)\n",
            m, length(flagged))
    reflag = Dict{Int,Int}()
    for (sd, it, _) in flagged
        @printf("    seed %d: frozen %d > 1.5 x %.1f -- re-solving at 1.03 kappa\n",
                sd, it, m)
        try
            R2 = run_sample(case, sd, L, true; κ = 1.03 * κ0)
            @printf("      at 1.03 kappa: frozen %d, fresh %d (was %d)\n",
                    R2.ch_a.iters, R2.it_g, it)
            reflag[sd] = R2.ch_a.iters
        catch err
            println("      [ERROR] ", err)
            reflag[sd] = -2
        end
    end

    open(summary_file, "a") do io
        for (sd, it, row) in cell
            isflag = haskey(reflag, sd)
            parts = String.(split(row, ','))
            parts[end-1] = string(isflag)
            parts[end] = string(get(reflag, sd, -1))
            println(io, join(parts, ','))
        end
    end
    println("  [saved] $(length(cell)) rows appended")
end

println("\n" * "="^78)
println("[CJ-almond 2/4] sweep complete -- ", now())
println("Summary: ", summary_file)
println("="^78)
