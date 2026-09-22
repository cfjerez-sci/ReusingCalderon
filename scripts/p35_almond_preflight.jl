# [CJ-almond 0/4] Preflight. Seconds, not minutes.
#
# Run before committing a machine to eight or twelve hours. It touches only
# the cached geometry targets -- no assembly, no solve -- and checks the
# things that, if wrong, make every downstream hour worthless:
#
#   1. both shipped meshes load, are consistently oriented, enclose a positive
#      volume and give the expected dof;
#   2. the exported field index is present and complete, and the correlation
#      length asked for actually exists in it;
#   3. the amplitude ladder's top level satisfies the injectivity certificate
#      for EVERY field that will be drawn at it -- so a re-export at a longer
#      correlation length cannot silently push a sample over 0.5;
#   4. the perturbation applies, moves the mesh by what the ladder says, and
#      leaves the connectivity alone.
#
# Exits non-zero on the first failure, so a runner script can gate on it.
#
# Usage, from the project root:
#   julia +1.11 --project scripts/p35_almond_preflight.jl
#   julia +1.11 --project scripts/p35_almond_preflight.jl --ell d4

import Pkg
Pkg.activate((@__DIR__) * "/..")

using Exp25_CJH_KC_LocalMultiTrace
using Makeitso
using DrWatson
using CompScienceMeshes
using LinearAlgebra
using DelimitedFiles
using Printf
using Dates
ENV["DRWATSON_WARN_DIRTY"] = "false"

module SimAL
include("../problems/p11_perturbed_almond.jl")
include("../methods/EFIE.jl")
end

include("../postproc/almond_geometry.jl")

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    (i === nothing || i == length(ARGS)) && return default
    return ARGS[i+1]
end

ellnm  = argval("--ell", "d5")
nseeds = parse(Int, argval("--seeds", "10"))
ncamp  = parse(Int, argval("--n", "100"))

fails = String[]
ok(msg)   = println("  [ok]   ", msg)
bad(msg)  = (push!(fails, msg); println("  [FAIL] ", msg))

println("="^78)
println("[CJ-almond 0/4] PREFLIGHT -- ", now())
println("correlation length: $ellnm")
println("="^78)

# ---------------------------------------------------------------- 1. meshes
println("\n--- meshes ---")
for (name, want) in (("almond_lam12", 2673), ("almond_lam20", 7269))
    try
        bm = make(SimAL.basemesh; meshname=name)
        dof = 3 * length(bm.Γ0.faces) ÷ 2
        oriented = CompScienceMeshes.isoriented(bm.Γ0)
        if dof == want && oriented && bm.vol > 0
            @printf("  [ok]   %s: dof %d, oriented, volume %.6e m^3\n",
                    name, dof, bm.vol)
        else
            bad("$name: dof $dof (want $want), oriented=$oriented, vol=$(bm.vol)")
        end
    catch err
        bad("$name did not load: $err")
    end
end

# -------------------------------------------------------- 2. exported fields
println("\n--- exported fields ---")
idxpath = joinpath(almond_dir(), "fields_index.csv")
if !isfile(idxpath)
    bad("missing $idxpath -- run almond/export_for_julia.py")
else
    raw, h = almond_field_index()
    names = unique(String.(strip.(string.(raw[:, h["ell_name"]]))))
    ok("index has $(size(raw, 1)) fields, correlation lengths $(join(names, ", "))")
    ellnm in names || bad("no fields at ell = $ellnm; export writes " *
                          "$(join(names, ", "))")
end

# ------------------------------------------------- 3. the ladder is certified
println("\n--- amplitude ladder ---")
levels = almond_levels()
for L in levels
    @printf("  level %d: eps = %6.3f mm = lambda/%.1f = %.2f h = %.1f%% thickness\n",
            L.level, L.eps_mm, L.lambda_over, L.eps_over_h, L.pct_thick)
end
epstop = levels[end].eps

"""
    worst_certificate(ellnm, epstop, seeds)

The largest eps*sup|DV| over every exported field that the campaign will draw
at the top amplitude, with the sample that attains it, and a count of the
fields actually examined.

This lives in a FUNCTION for a reason. Written as a bare `for` loop at top
level it silently returned zero: a top-level loop opens a soft scope, so
`worst = c` created a new local that died each iteration while the outer
`worst` stayed at 0.0, and the check passed by comparing nothing against 0.5.
Julia warns about it, and the warning is easy to scroll past. Inside a
function there is no soft scope and no ambiguity.
"""
function worst_certificate(ellnm, epstop, seeds)
    worst = 0.0
    worstid = ""
    seen = 0
    for case in ("U", "T"), sd in seeds
        fld = try
            almond_field(case, ellnm, sd)
        catch
            # not every (case, seed) pair is exported: the campaign seeds are
            # case U only. A missing pair is not a failure.
            continue
        end
        seen += 1
        c = almond_certificate(fld, epstop)
        if c > worst
            worst = c
            worstid = "case $case seed $sd"
        end
    end
    return worst, worstid, seen
end

if isempty(fails)
    seeds = vcat([1000 + i for i in 0:(nseeds-1)], [5000 + i for i in 0:(ncamp-1)])
    worst, worstid, seen = worst_certificate(ellnm, epstop, seeds)
    @printf("  [ok]   examined %d exported fields at ell = %s\n", seen, ellnm)
    if seen == 0
        bad("no exported fields were examined -- the certificate was not checked")
    elseif worst <= 0.5 + 1e-12
        @printf("  [ok]   worst eps*sup|DV| at the top level: %.4f (%s)\n",
                worst, worstid)
    else
        bad(@sprintf("worst eps*sup|DV| = %.4f > 0.5 at %s -- the ladder and the fields disagree; re-export", worst, worstid))
    end
end

# ---------------------------------------------------- 4. the perturbation runs
println("\n--- perturbation ---")
if isempty(fails)
    try
        g0 = make(SimAL.geo; meshname="almond_lam12", case="U",
                  ell_name=ellnm, seed=0, eps=0.0)
        gp = make(SimAL.geo; meshname="almond_lam12", case="U",
                  ell_name=ellnm, seed=1000, eps=epstop)
        samefaces = g0.Γ.faces == gp.Γ.faces
        frac = gp.maxdisp / epstop
        if samefaces && 0.9 < frac <= 1.0
            @printf("  [ok]   max displacement %.3f mm (%.1f%% of eps), connectivity unchanged\n", 1e3*gp.maxdisp, 100*frac)
            @printf("  [ok]   certificate %.4f\n", gp.cert)
        else
            bad("displacement $(1e3*gp.maxdisp) mm is $(100*frac)% of eps, " *
                "faces unchanged = $samefaces")
        end
    catch err
        bad("perturbation failed: $err")
    end
end

println("\n" * "="^78)
if isempty(fails)
    println("PREFLIGHT PASSED -- safe to start the campaign.")
else
    println("PREFLIGHT FAILED:")
    for f in fails
        println("  - ", f)
    end
    exit(1)
end
println("="^78)
