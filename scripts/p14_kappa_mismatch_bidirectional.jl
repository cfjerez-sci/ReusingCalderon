# Bidirectional wavenumber-mismatch experiment (Parts A and D).
#
# Extends scripts/p10_kappa_mismatch_precond_sweep.jl, which froze the
# Calderón preconditioner at kappa_precond = 1.0 and swept kappa_solve
# upward only over {1,2,4,8}. That establishes behaviour for INCREASING
# kappa. Here the sweep is deliberately BIDIRECTIONAL.
#
#   Part A   kappa_precond = 2.0 (fixed)
#            kappa_solve   in {0.25, 0.5, 1, 2, 4, 8}
#            i.e. d = log2(kappa_solve/kappa_precond) in {-3,-2,-1,0,1,2}
#
#   Part D   kappa_precond in {1, 2, 4}  x  kappa_solve in {0.5, 1, 2, 4, 8}
#            giving R_ij = iters_frozen(i,j) / iters_fresh(j)
#
#   --detune adds resonance-control solves at kappa = 3.52 and 8.4661
#            (see the "interior resonance" note below).
#
# NUMERICAL CONVENTIONS are inherited unchanged from methods/EFIE.jl and
# methods/EFIE_manual_solves.jl: same nominal sphere, same RWG/BC spaces,
# same plane wave (d = ẑ, p = x̂), same outer/inner GMRES with abstol =
# reltol = 1e-8 and maxiter = 1500, same true-residual definition. The
# ONLY thing that differs between the "frozen" and "fresh" strategies is
# which (Tyy, Nxy) pair is handed to solve_calderon_gmres.
#
# NO SHAPE PERTURBATION (realization = 0). `spaces` depends only on `geo`
# and not on kappa (methods/EFIE.jl), so X/Y indexing is identical for
# every kappa and NO DOF permutation is involved. This isolates wavenumber
# mismatch from geometry.
#
# THIS IS A NUMERICAL CHARACTERISATION ONLY. It supports no theorem about
# wavenumber reuse: changing kappa rescales the EFIE's vector- and
# scalar-potential terms (the -iκ and 1/(iκ) weights) relative to one
# another, so the shape-perturbation argument gives no uniform (h,ν)
# bound here. See Remark 11 of the manuscript.
#
# INTERIOR RESONANCE. The interior Maxwell spectrum of the unit ball
# begins at kappa = 2.743707 (TM, l=1). Every kappa <= 2 in this sweep is
# therefore EXACTLY resonance-free, while kappa = 4 lies 3.2% from the
# TM(l=2) eigenvalue 3.870239 and kappa = 8 lies 2.3% from the TE(l=4)
# eigenvalue 8.182561. The downward arm is clean and the upward arm is
# not; the --detune controls exist to separate mismatch from resonance
# proximity. Every row carries its own rel_resonance_gap.
#
# Writes data/sweep/p14_kappa_mismatch_bidirectional.csv, consumed by
# kappa_mismatch/analyze_kappa_mismatch.py. Does NOT touch
# data/sweep/p10_kappa_mismatch_precond_summary.csv (Section 5.6).
#
# Run:
#   julia scripts/p14_kappa_mismatch_bidirectional.jl --smoke     # ~minutes, h=0.4
#   julia scripts/p14_kappa_mismatch_bidirectional.jl --detune    # the real run

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using Exp25_CJH_KC_LocalMultiTrace
using Makeitso
using DrWatson
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

smoke   = "--smoke"  in ARGS
detune  = "--detune" in ARGS
partsel = let i = findfirst(==("--part"), ARGS)
    i === nothing ? "AD" : uppercase(ARGS[i+1])
end

radius      = 1.0
h           = smoke ? 0.4 : 0.1
realization = 0
lmin, lmax  = 2, 6
amplitude   = 0.03          # unused at realization=0, kept for the target signature

KAPPA0_A  = 2.0
KSOLVE_A  = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
KPREC_D   = [1.0, 2.0, 4.0]
KSOLVE_D  = [0.5, 1.0, 2.0, 4.0, 8.0]
KDETUNE   = [3.52, 8.4661]

# Interior Maxwell eigenvalues of the unit ball below 10.
const EIG = [2.743707, 3.870239, 4.493409, 4.973420, 5.763459, 6.061949,
             6.116764, 6.987932, 7.140227, 7.443087, 7.725252, 8.182561,
             8.210842, 8.721751, 9.095011, 9.275463, 9.316616, 9.355812,
             9.967547]
relgap(κ) = minimum(abs.(κ .- EIG)) / κ

κ_solve_all = sort(unique(vcat(KSOLVE_A, KSOLVE_D, detune ? KDETUNE : Float64[])))
κ_prec_all  = sort(unique(vcat([KAPPA0_A], KPREC_D)))
κ_all       = sort(unique(vcat(κ_solve_all, κ_prec_all)))

outdir = projectdir("data", "sweep")
mkpath(outdir)
csvfile = joinpath(outdir, smoke ? "p14_kappa_mismatch_bidirectional_SMOKE.csv" :
                                   "p14_kappa_mismatch_bidirectional.csv")

const COLS = ["part", "kappa_precond", "kappa_solve", "strategy", "iters",
              "converged", "true_res", "solve_time_s", "matrix_assembly_s",
              "precond_assembly_s", "sol_l2", "rel_diff_vs_fresh",
              "dof", "h", "geometry", "seed", "tol_outer", "tol_inner",
              "maxit", "rel_resonance_gap", "notes"]

open(csvfile, "w") do io
    println(io, join(COLS, ","))
end

println("="^74)
println("BIDIRECTIONAL kappa-mismatch experiment -- started ", now())
println("h = $h   realization = $realization (nominal sphere, no perturbation)")
println("parts = $partsel   detune = $detune   smoke = $smoke")
println("Part A: kappa_precond = $KAPPA0_A, kappa_solve = $KSOLVE_A")
println("Part D: kappa_precond = $KPREC_D x kappa_solve = $KSOLVE_D")
detune && println("Detune controls: kappa = $KDETUNE")
println("Output: $csvfile")
if smoke
    println()
    println("*** SMOKE TEST (h = 0.4). The iteration counts are NOT physically")
    println("*** meaningful -- kappa = 8 at h = 0.4 is about 2 elements per")
    println("*** wavelength. The point is to prove the plumbing and the matched")
    println("*** diagonal. These numbers must never reach the manuscript.")
end
println()
println("Interior-resonance proximity (unit ball, spectrum starts at 2.743707):")
for κ in κ_all
    g = relgap(κ)
    tag = g < 0.05 ? "  <<< FLAG: near-resonant, iteration count is not clean" :
          g < 0.10 ? "  <<< watch" : ""
    @printf("   kappa = %-8.4g rel. gap = %8.2f%%%s\n", κ, 100g, tag)
end
println("="^74)

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

csvcell(x::AbstractFloat) = isnan(x) ? "" : @sprintf("%.10g", x)
csvcell(x) = string(x)

function record!(; part, κp, κs, strategy, u, ch, tsolve, Zxx, bx,
                 t_mat, t_pre, u_fresh=nothing, notes="")
    uv = Vector(u); bv = Vector(bx)
    true_res = norm(bv .- Zxx * uv) / norm(bv)
    rel_diff = u_fresh === nothing ? NaN : norm(uv .- u_fresh) / norm(u_fresh)
    row = Dict(
        "part" => part,
        "kappa_precond" => (κp === nothing ? "" : @sprintf("%.10g", κp)),
        "kappa_solve" => @sprintf("%.10g", κs),
        "strategy" => strategy,
        "iters" => ch.iters,
        "converged" => ch.isconverged,
        "true_res" => true_res,
        "solve_time_s" => tsolve,
        "matrix_assembly_s" => t_mat,
        "precond_assembly_s" => t_pre,
        "sol_l2" => norm(uv),
        "rel_diff_vs_fresh" => rel_diff,
        "dof" => length(bv), "h" => h,
        "geometry" => "sphere", "seed" => 1000 + realization,
        "tol_outer" => 1e-8, "tol_inner" => 1e-8, "maxit" => 1500,
        "rel_resonance_gap" => relgap(κs),
        "notes" => notes)
    open(csvfile, "a") do io
        println(io, join([csvcell(get(row, c, "")) for c in COLS], ","))
    end
    @printf("  [%-6s] part %s  kp=%-8s ks=%-8.4g iters=%-5d conv=%-5s true_res=%.3e  %.2f s%s\n",
            strategy, part, (κp === nothing ? "-" : @sprintf("%.4g", κp)), κs,
            ch.iters, ch.isconverged, true_res, tsolve,
            ch.isconverged ? "" : "   <<< DID NOT CONVERGE")
    ch.isconverged || @warn "NON-CONVERGED RUN RETAINED (not discarded)" part κp κs strategy
    uv
end

function record_failure!(; part, κp, κs, strategy, err)
    row = Dict("part" => part,
               "kappa_precond" => (κp === nothing ? "" : @sprintf("%.10g", κp)),
               "kappa_solve" => @sprintf("%.10g", κs),
               "strategy" => strategy, "iters" => -1, "converged" => false,
               "true_res" => NaN, "solve_time_s" => NaN,
               "matrix_assembly_s" => NaN, "precond_assembly_s" => NaN,
               "sol_l2" => NaN, "rel_diff_vs_fresh" => NaN,
               "dof" => -1, "h" => h, "geometry" => "sphere",
               "seed" => 1000 + realization, "tol_outer" => 1e-8,
               "tol_inner" => 1e-8, "maxit" => 1500,
               "rel_resonance_gap" => relgap(κs),
               "notes" => "FAILED: " * replace(string(err), ","=>";", "\n"=>" "))
    open(csvfile, "a") do io
        println(io, join([csvcell(get(row, c, "")) for c in COLS], ","))
    end
    println("  [ERROR] part $part kp=$κp ks=$κs $strategy -- recorded, not discarded")
end

# ---------------------------------------------------------------------------
# Discretizations. Makeitso caches on (h, κ, radius, realization, ...), so a
# kappa assembled for Part A is free for Part D and for the preconditioners.
# ---------------------------------------------------------------------------

disc  = Dict{Float64,Any}()
t_asm = Dict{Float64,Float64}()

function get_disc!(κ)
    haskey(disc, κ) && return disc[κ]
    println("\n--- discretization at kappa = $κ ---")
    t0 = time()
    d = make(Sim2.discretization; h, κ, radius, realization, lmin, lmax, amplitude)
    t = time() - t0
    disc[κ] = d; t_asm[κ] = t
    @printf("    assembled/retrieved in %.1f s  (dof = %d)\n", t, length(d.vectors.bx))
    d
end

# ---------------------------------------------------------------------------
# Baselines: plain and fresh, once per kappa_solve
# ---------------------------------------------------------------------------

fresh_sol = Dict{Float64,Any}()
dof_ref = Ref(-1)

println("\n" * "="^74)
println("BASELINES (plain + fresh) over kappa_solve = $κ_solve_all")
println("="^74)
for κ in κ_solve_all
    try
        d = get_disc!(κ)
        Zxx = d.matrices.Zxx; bx = d.vectors.bx
        dof_ref[] < 0 && (dof_ref[] = length(bx))
        @assert length(bx) == dof_ref[] "DOF changed between kappa values -- mesh drift"

        u_p, ch_p, t_p = solve_plain_gmres(Zxx, bx)
        record!(part="A", κp=nothing, κs=κ, strategy="plain", u=u_p, ch=ch_p,
                tsolve=t_p, Zxx=Zxx, bx=bx, t_mat=t_asm[κ], t_pre=NaN)

        u_f, ch_f, t_f = solve_calderon_gmres(Zxx, bx, d.matrices.Tyy, d.matrices.Nxy)
        uf = record!(part="A", κp=κ, κs=κ, strategy="fresh", u=u_f, ch=ch_f,
                     tsolve=t_f, Zxx=Zxx, bx=bx, t_mat=t_asm[κ], t_pre=t_asm[κ])
        fresh_sol[κ] = uf
    catch err
        record_failure!(part="A", κp=nothing, κs=κ, strategy="plain", err=err)
        record_failure!(part="A", κp=κ, κs=κ, strategy="fresh", err=err)
    end
end

# ---------------------------------------------------------------------------
# Part A: one frozen preconditioner at KAPPA0_A, swept both directions
# ---------------------------------------------------------------------------

if occursin("A", partsel)
    println("\n" * "="^74)
    println("PART A -- preconditioner frozen at kappa_0 = $KAPPA0_A")
    println("="^74)
    d0 = get_disc!(KAPPA0_A)
    Tyy0, Nxy0 = d0.matrices.Tyy, d0.matrices.Nxy
    for κ in vcat(KSOLVE_A, detune ? KDETUNE : Float64[])
        try
            d = get_disc!(κ)
            u, ch, t = solve_calderon_gmres(d.matrices.Zxx, d.vectors.bx, Tyy0, Nxy0)
            record!(part="A", κp=KAPPA0_A, κs=κ, strategy="frozen", u=u, ch=ch,
                    tsolve=t, Zxx=d.matrices.Zxx, bx=d.vectors.bx,
                    t_mat=t_asm[κ], t_pre=0.0,
                    u_fresh=get(fresh_sol, κ, nothing),
                    notes=(κ in KDETUNE ? "resonance control" : ""))
        catch err
            record_failure!(part="A", κp=KAPPA0_A, κs=κ, strategy="frozen", err=err)
        end
    end
end

# ---------------------------------------------------------------------------
# Part D: the kappa_precond x kappa_solve matrix
# ---------------------------------------------------------------------------

if occursin("D", partsel)
    println("\n" * "="^74)
    println("PART D -- preconditioner matrix (every assembly already cached)")
    println("="^74)

    # mirror the baseline rows so the analysis can form R_ij within part D
    for line in readlines(csvfile)[2:end]
        f = split(line, ",")
        if f[1] == "A" && (f[4] == "plain" || f[4] == "fresh") &&
           parse(Float64, f[3]) in KSOLVE_D
            open(csvfile, "a") do io
                println(io, join(vcat("D", f[2:end]), ","))
            end
        end
    end

    for κp in KPREC_D
        dp = get_disc!(κp)
        Tyy, Nxy = dp.matrices.Tyy, dp.matrices.Nxy
        for κs in KSOLVE_D
            try
                d = get_disc!(κs)
                u, ch, t = solve_calderon_gmres(d.matrices.Zxx, d.vectors.bx, Tyy, Nxy)
                record!(part="D", κp=κp, κs=κs, strategy="frozen", u=u, ch=ch,
                        tsolve=t, Zxx=d.matrices.Zxx, bx=d.vectors.bx,
                        t_mat=t_asm[κs], t_pre=t_asm[κp],
                        u_fresh=get(fresh_sol, κs, nothing))
            catch err
                record_failure!(part="D", κp=κp, κs=κs, strategy="frozen", err=err)
            end
        end
    end
end

# ---------------------------------------------------------------------------

println("\n" * "="^74)
println("complete -- ", now())
@printf("assembly time over %d wavenumbers: %.1f s\n", length(t_asm), sum(values(t_asm)))
println("CSV: $csvfile")
println()
println("Next:")
println("  python3 kappa_mismatch/analyze_kappa_mismatch.py \\")
println("      $csvfile --k0 $KAPPA0_A --outdir kappa_mismatch/results")
println("Read kappa_mismatch_qc.txt FIRST. If the matched diagonal")
println("(kappa_solve == kappa_precond) does not reproduce the fresh count to")
println("within one iteration, something is wrong -- do not use the numbers.")
println("="^74)
