# Compare the actual Calderon-preconditioner ingredient matrices
# (Tyy, Nxy) between the nominal (unperturbed) sphere and the perturbed
# realization used throughout scripts/p10_PEC_*.jl, entry by entry and
# in aggregate (relative Frobenius-norm difference). This is meant to
# explain WHY the frozen-preconditioner strategy (reusing Tyy0/Nxy0
# unchanged) performs so poorly even at a small (3%) shape perturbation:
# if these matrices already differ substantially entry-wise despite the
# small geometric change, that's the direct cause.
#
# Only retrieves already-cached discretizations (kappa=1.0, h=0.1,
# realization=0 and 1) -- no fresh assembly, fast.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using ReusingCalderon
using Makeitso
using DrWatson
using LinearAlgebra
using Printf
ENV["DRWATSON_WARN_DIRTY"] = "false"

module Sim2
include("../problems/p10_perturbed_sphere.jl")
include("../methods/EFIE.jl")
end

radius      = 1.0
h           = 0.1
κ           = 1.0
lmin, lmax  = 2, 6
amplitude   = 0.03
realization = 1

nominal_lmin, nominal_lmax, nominal_amplitude = 2, 6, 0.03

disc0 = make(Sim2.discretization; h, κ, radius, realization=0,
    lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)
discp = make(Sim2.discretization; h, κ, radius, realization, lmin, lmax, amplitude)

Tyy0 = disc0.matrices.Tyy
Nxy0 = disc0.matrices.Nxy
Tyyp = discp.matrices.Tyy
Nxyp = discp.matrices.Nxy

println("typeof(Tyy0) = ", typeof(Tyy0))
println("typeof(Nxy0) = ", typeof(Nxy0))
println("size(Tyy0)   = ", size(Tyy0))
println("size(Nxy0)   = ", size(Nxy0))

# Tyy/Nxy come back as LinearMaps.jl lazy maps (LiftedMap), which don't
# support scalar getindex(i,j) -- materialize to plain dense arrays once
# up front (each is only 4827x4827 here) and index into those instead.
println("materializing dense matrices (one-off cost)...")
Tyy0 = Matrix{ComplexF64}(Tyy0)
Nxy0 = Matrix{ComplexF64}(Nxy0)
Tyyp = Matrix{ComplexF64}(Tyyp)
Nxyp = Matrix{ComplexF64}(Nxyp)
println("done.")

outfile = projectdir("data", "figures", "p10_matrix_entry_comparison.txt")
mkpath(dirname(outfile))
io = open(outfile, "w")

function logboth(str)
    println(str)
    println(io, str)
end

logboth("="^78)
logboth("Sample diagonal entries: Tyy0[i,i] (nominal) vs Tyy'[i,i] (perturbed)")
logboth("="^78)
n = size(Tyy0, 1)
idxs = round.(Int, range(1, n, length=10))
header = @sprintf("%-6s %-24s %-24s %-10s", "i", "Tyy0[i,i]", "Tyy'[i,i]", "rel.diff")
logboth(header)
for i in idxs
    a = Tyy0[i,i]; b = Tyyp[i,i]
    reld = abs(a-b) / abs(a)
    logboth(@sprintf("%-6d %-24s %-24s %-10.4f", i, string(round(a, digits=6)), string(round(b, digits=6)), reld))
end

logboth("")
logboth("="^78)
logboth("Sample diagonal entries: Nxy0[i,i] (nominal) vs Nxy'[i,i] (perturbed)")
logboth("="^78)
logboth(header)
for i in idxs
    a = Nxy0[i,i]; b = Nxyp[i,i]
    reld = abs(a-b) / abs(a)
    logboth(@sprintf("%-6d %-24s %-24s %-10.4f", i, string(round(a, digits=6)), string(round(b, digits=6)), reld))
end

logboth("")
logboth("="^78)
logboth("Sample off-diagonal entries (i, i+1): Tyy and Nxy, nominal vs perturbed")
logboth("="^78)
logboth(header)
for i in idxs[1:end-1]
    j = i+1
    a = Tyy0[i,j]; b = Tyyp[i,j]
    reld = abs(a-b) / abs(a)
    logboth(@sprintf("Tyy  (%d,%d): %-24s %-24s %-10.4f", i, j, string(round(a, digits=6)), string(round(b, digits=6)), reld))
end
for i in idxs[1:end-1]
    j = i+1
    a = Nxy0[i,j]; b = Nxyp[i,j]
    reld = abs(a-b) / abs(a)
    logboth(@sprintf("Nxy  (%d,%d): %-24s %-24s %-10.4f", i, j, string(round(a, digits=6)), string(round(b, digits=6)), reld))
end

logboth("")
logboth("="^78)
logboth("Aggregate (whole-matrix) relative Frobenius-norm differences")
logboth("="^78)

diffTyy = norm(Tyy0 .- Tyyp) / norm(Tyy0)
logboth(@sprintf("||Tyy0 - Tyy'||_F / ||Tyy0||_F = %.4f  (%.1f%%)", diffTyy, 100*diffTyy))

diffNxy = norm(Nxy0 .- Nxyp) / norm(Nxy0)
logboth(@sprintf("||Nxy0 - Nxy'||_F / ||Nxy0||_F = %.4f  (%.1f%%)", diffNxy, 100*diffNxy))

logboth("")
logboth(@sprintf("max|Tyy0-Tyy'| = %.4e   max|Tyy0| = %.4e", maximum(abs.(Tyy0 .- Tyyp)), maximum(abs.(Tyy0))))
logboth(@sprintf("max|Nxy0-Nxy'| = %.4e   max|Nxy0| = %.4e", maximum(abs.(Nxy0 .- Nxyp)), maximum(abs.(Nxy0))))

close(io)
println("\n[saved] ", outfile)
