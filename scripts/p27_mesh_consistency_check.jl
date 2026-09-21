# [mesh-consistency fix] Cheap verification that the nominal and perturbed
# geometries of a problem share one triangulation.
#
# Before the fix, `geo` called the mesher itself, so every (realization,
# amplitude) pair meshed independently and two campaigns at the same h could
# receive different triangulations. That is what produced the 1224-vs-1206
# and 2292-vs-2298 discrepancy disclosed in Section 5.3, and what aborted
# the amplitude-0.3 rows of the linearity study. With the mesher promoted to
# its own `basemesh` target, keyed on the mesh parameters alone, every geo
# at a given h draws the same mesh.
#
# Geometry only: no assembly, no solves, so this runs in seconds. At each
# mesh level it checks that the perturbed geometries agree with the nominal
# one in vertex count, face count and the face array itself -- connectivity
# being exactly what the DOF-alignment permutation relies on.
#
# Usage, from the project root:
#   julia +1.11 --project scripts/p27_mesh_consistency_check.jl
#   julia +1.11 --project scripts/p27_mesh_consistency_check.jl --levels 0.2,0.15
#   julia +1.11 --project scripts/p27_mesh_consistency_check.jl --problem fichera_corner --levels 0.15
#
# --problem is one of: sphere (default), ellipsoid, fichera, fichera_corner.
# Exits 0 if every check passes, 1 otherwise, so it doubles as a regression
# test.

import Pkg
Pkg.activate((@__DIR__) * "/..")

using Exp25_CJH_KC_LocalMultiTrace
using Makeitso
using DrWatson
using CompScienceMeshes
using BEAST
using Random
using Printf
ENV["DRWATSON_WARN_DIRTY"] = "false"

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    (i === nothing || i == length(ARGS)) && return default
    return ARGS[i+1]
end

const PROBLEM = argval("--problem", "sphere")
const PROBFILE = Dict(
    "sphere"         => "../problems/p10_perturbed_sphere.jl",
    "ellipsoid"      => "../problems/p10_perturbed_ellipsoid.jl",
    "fichera"        => "../problems/p10_perturbed_fichera.jl",
    "fichera_corner" => "../problems/p10_perturbed_fichera_corner.jl",
)[PROBLEM]

levels = parse.(Float64, split(argval("--levels", "0.2,0.15,0.1"), ","))
amps   = parse.(Float64, split(argval("--amps", "0.03,0.1,0.2,0.3,0.4,0.5"), ","))

module SimC
include(Main.PROBFILE)
include("../methods/EFIE.jl")
end

# the three problem families take different keyword sets
geo_at(h, realization, amplitude) =
    PROBLEM == "sphere"    ? make(SimC.geo; h, radius=1.0, realization, lmin=2, lmax=6, amplitude) :
    PROBLEM == "ellipsoid" ? make(SimC.geo; h, realization, lmin=2, lmax=6, amplitude) :
                             make(SimC.geo; h, realization, amplitude)

println("="^74)
println("[mesh consistency] problem = ", PROBLEM, "   levels = ", levels)
println("   amplitudes = ", amps)
println("="^74)

failures = 0
for h in levels
    g0 = nothing
    try
        g0 = geo_at(h, 0, 0.0)
    catch err
        println("  [ERROR] nominal geometry failed at h=$h: ", err)
        global failures += 1
        continue
    end
    nv0 = length(g0.Γ.vertices)
    nf0 = length(g0.Γ.faces)
    @printf("\nh = %-7g nominal: %5d vertices, %5d faces   (RWG dof = %d)\n",
            h, nv0, nf0, 3 * nf0 ÷ 2)
    for a in amps
        try
            gp = geo_at(h, 1, a)
            nv = length(gp.Γ.vertices)
            nf = length(gp.Γ.faces)
            same_conn = (nf == nf0) && all(gp.Γ.faces[i] == g0.Γ.faces[i] for i in 1:nf0)
            ok = (nv == nv0) && same_conn
            note = ok ? "" :
                   nf != nf0 ? "   <-- MESH DIFFERS from nominal" :
                   nv != nv0 ? "   <-- same faces, different vertex count" :
                               "   <-- same counts, DIFFERENT face array"
            @printf("   amp %-6g %s %5d vertices, %5d faces%s\n",
                    a, ok ? "OK  " : "FAIL", nv, nf, note)
            ok || (global failures += 1)
        catch err
            println("   amp $a  [ERROR] ", err)
            global failures += 1
        end
    end
end

println("\n" * "="^74)
if failures == 0
    println("PASS -- every perturbed geometry shares the nominal triangulation.")
else
    println("FAIL -- ", failures, " check(s) did not pass.")
    println("A cached geometry written before the fix can still disagree; clear")
    println("that problem's geo cache under data/problems/ and re-run.")
end
println("="^74)
exit(failures == 0 ? 0 : 1)
