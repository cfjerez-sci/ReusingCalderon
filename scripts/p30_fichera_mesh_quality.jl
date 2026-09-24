# [CJ 2.1(d)] Shape-regularity of the corner-perturbed Fichera meshes.
#
# Table 9 currently reports the minimum triangle area A_min as its mesh
# health column. Area is the wrong diagnostic: it conflates "small" with
# "badly shaped", and a uniformly scaled-down but well-proportioned
# triangle is harmless while a thin sliver of the same area is not. The
# referee question behind Section 5.8 -- whether the frozen preconditioner
# degrades because the operator is perturbed or merely because the mesh
# deteriorates -- needs a SHAPE measure.
#
# This script recomputes, for each amplitude already in the corner sweep,
#
#   min_angle_deg   the smallest interior angle over all triangles
#   max_aspect      the largest ratio (longest edge)/(2*inradius), which
#                   is the standard shape-regularity constant: it equals
#                   sqrt(3) for an equilateral triangle and diverges for a
#                   sliver
#   min_area        kept, so the new columns can be compared against the
#                   published ones
#
# on exactly the meshes the sweep used. It runs no solves and assembles
# nothing: it only calls the `geo` target, which is cached, so it is cheap
# (seconds) and cannot perturb any published number.
#
# Usage, from the project root:
#   julia +1.11 --project scripts/p30_fichera_mesh_quality.jl
#   julia +1.11 --project scripts/p30_fichera_mesh_quality.jl --h 0.3 --amps 0.02,0.05,0.10,0.15
#
# Writes data/sweep/p30_fichera_mesh_quality.csv.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using ReusingCalderon
using Makeitso
using DrWatson
using CompScienceMeshes
using LinearAlgebra
using Printf
using Dates
ENV["DRWATSON_WARN_DIRTY"] = "false"

module SimQ
include("../problems/p10_perturbed_fichera_corner.jl")
include("../methods/EFIE.jl")
end

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    (i === nothing || i == length(ARGS)) && return default
    return ARGS[i+1]
end

h    = parse(Float64, argval("--h", "0.15"))
amps = parse.(Float64, split(argval("--amps", "0.0,0.02,0.05,0.10,0.15"), ","))

"""
    tri_quality(a, b, c)

Smallest interior angle (degrees) and the shape-regularity ratio
(longest edge)/(2*inradius) of the triangle with vertices a, b, c.
The ratio is sqrt(3) for an equilateral triangle (inradius a/(2*sqrt(3)) for
side a) and grows without bound as the triangle degenerates, in either the
sliver or the needle direction.
"""
function tri_quality(a, b, c)
    ea = norm(c .- b); eb = norm(a .- c); ec = norm(b .- a)
    area = norm(cross(b .- a, c .- a)) / 2
    (area <= 0 || min(ea, eb, ec) <= 0) && return (0.0, Inf, area)
    # law of cosines, clamped against round-off
    cl(x) = clamp(x, -1.0, 1.0)
    A = acosd(cl((eb^2 + ec^2 - ea^2) / (2 * eb * ec)))
    B = acosd(cl((ea^2 + ec^2 - eb^2) / (2 * ea * ec)))
    C = 180.0 - A - B
    s = (ea + eb + ec) / 2
    inradius = area / s
    return (min(A, B, C), maximum((ea, eb, ec)) / (2 * inradius), area)
end

function mesh_quality(verts, faces)
    minang = Inf; maxasp = 0.0; minarea = Inf; nbad = 0
    for f in faces
        ang, asp, ar = tri_quality(verts[f[1]], verts[f[2]], verts[f[3]])
        minang = min(minang, ang); maxasp = max(maxasp, asp)
        minarea = min(minarea, ar); ar <= 0 && (nbad += 1)
    end
    return (minang, maxasp, minarea, nbad)
end

outdir = projectdir("data", "sweep"); mkpath(outdir)
csv = joinpath(outdir, "p30_fichera_mesh_quality.csv")
open(csv, "w") do io
    println(io, "h,amplitude,n_faces,min_angle_deg,max_aspect,min_area,n_degenerate")
end

println("="^72)
println("[CJ 2.1(d)] Fichera corner mesh quality -- ", now())
println("h = $h   amplitudes = $amps   (geo target only; no assembly, no solves)")
println("="^72)
@printf("%10s %8s %14s %12s %14s %6s\n",
        "amplitude", "faces", "min_angle_deg", "max_aspect", "min_area", "bad")

for amp in amps
    realization = amp == 0.0 ? 0 : 1
    geo = make(SimQ.geo; h, realization, amplitude=amp)
    verts = geo.Γ.vertices; faces = geo.Γ.faces
    minang, maxasp, minarea, nbad = mesh_quality(verts, faces)
    @printf("%10.3g %8d %14.3f %12.3f %14.4e %6d\n",
            amp, length(faces), minang, maxasp, minarea, nbad)
    open(csv, "a") do io
        @printf(io, "%.6g,%.6g,%d,%.6f,%.6f,%.6e,%d\n",
                h, amp, length(faces), minang, maxasp, minarea, nbad)
    end
end

println("\nSaved ", csv)
println("Read max_aspect against amplitude: if it is flat while the frozen")
println("iteration count rises, mesh degradation is not the cause. If both")
println("rise, they are confounded and only the frozen/fresh ratio separates")
println("them. Report the numbers either way.")
