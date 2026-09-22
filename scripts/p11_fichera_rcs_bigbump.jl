# The tested Fichera amplitudes (<=15%, Table~tab:fichera) give a
# frozen-vs-nominal RCS difference too small to see clearly on a plot
# (the two-face localized bump is a small perturbation relative to the
# cube's overall backscatter, which is dominated by the untouched
# reentrant corner and the other seven faces). For the illustrative
# "large perturbation" RCS figure only, redo the frozen-preconditioner
# solve + RCS at 40% amplitude (matching the mesh-comparison figure's
# illustrative bump size) so the physical difference is visible.

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
using DelimitedFiles

module SimF2
include("../problems/p10_perturbed_fichera.jl")
include("../methods/EFIE.jl")
end

include("../methods/EFIE_manual_solves.jl")

function dof_edge_fingerprint(fns, faces)
    fps = Vector{Tuple{Int,Int}}(undef, length(fns))
    for (i, shapes) in enumerate(fns)
        cellids = unique(sh.cellid for sh in shapes)
        if length(cellids) == 2
            v1 = Set(faces[cellids[1]])
            v2 = Set(faces[cellids[2]])
            shared = sort(collect(intersect(v1, v2)))
            fps[i] = length(shared) == 2 ? (shared[1], shared[2]) : (-1, -i)
        else
            fps[i] = (-1, -i)
        end
    end
    return fps
end

function dof_cell_fingerprint_filtered(fns, refined_faces, n_orig_verts)
    fps = Vector{Vector{Int}}(undef, length(fns))
    for (i, shapes) in enumerate(fns)
        vs = Set{Int}()
        for sh in shapes
            cellid = sh.cellid
            for v in refined_faces[cellid]
                v <= n_orig_verts && push!(vs, v)
            end
        end
        fps[i] = sort(collect(vs))
    end
    return fps
end

function build_permutation(fp0, fpp)
    dictp = Dict{eltype(fp0),Int}()
    for (j, fp) in enumerate(fpp)
        dictp[fp] = j
    end
    perm = [get(dictp, fp0[i], -1) for i in eachindex(fp0)]
    ok = count(==(-1), perm) == 0 && sort(perm) == collect(1:length(perm))
    return perm, ok
end

function rcs_curve(κ, u, X, θ)
    T = Maxwell3D.singlelayer(wavenumber=κ)
    Tfar = BEAST.MWFarField3D(T)
    dirs = [point(sin(t), 0.0, cos(t)) for t in θ]
    F = potential(Tfar, dirs, Vector(u), X)
    return κ^2/(4π) .* abs2.(norm.(F))
end

h_f, κ_f = 0.3, 2.0
amp_f, real_f = 0.4, 1
θ = range(0, π, length=181)

outdir_rcs = projectdir("data", "figures", "rcs")
mkpath(outdir_rcs)

geo0 = make(SimF2.geo; h=h_f, realization=0, amplitude=0.0)
geop = make(SimF2.geo; h=h_f, realization=real_f, amplitude=amp_f)

disc0 = make(SimF2.discretization; h=h_f, κ=κ_f, realization=0, amplitude=0.0)
spaces0 = make(SimF2.spaces; h=h_f, realization=0, amplitude=0.0)
discp = make(SimF2.discretization; h=h_f, κ=κ_f, realization=real_f, amplitude=amp_f)
spacesp = make(SimF2.spaces; h=h_f, realization=real_f, amplitude=amp_f)

u0, _, _ = solve_calderon_gmres(disc0.matrices.Zxx, disc0.vectors.bx, disc0.matrices.Tyy, disc0.matrices.Nxy)

edgefp0 = dof_edge_fingerprint(spaces0.X.fns, geo0.Γ.faces)
n0v = length(vertices(spaces0.Y.geo.parent))
fpY0 = dof_cell_fingerprint_filtered(spaces0.Y.fns, spaces0.Y.geo.mesh.faces, n0v)
edgefpp = dof_edge_fingerprint(spacesp.X.fns, geop.Γ.faces)
npv = length(vertices(spacesp.Y.geo.parent))
fpYp = dof_cell_fingerprint_filtered(spacesp.Y.fns, spacesp.Y.geo.mesh.faces, npv)
permX, okX = build_permutation(edgefp0, edgefpp)
permY, okY = build_permutation(fpY0, fpYp)
@assert okX && okY "fichera (40%) DOF alignment failed"
Tyy0_al = Matrix{ComplexF64}(disc0.matrices.Tyy)[invperm(permY), invperm(permY)]
Nxy0_al = Matrix{ComplexF64}(disc0.matrices.Nxy)[invperm(permX), invperm(permY)]
up, chp, _ = solve_calderon_gmres(discp.matrices.Zxx, discp.vectors.bx, Tyy0_al, Nxy0_al)
@printf("frozen-preconditioner solve on 40%%-perturbed Fichera cube: %d iters, converged=%s\n", chp.iters, chp.isconverged)

rcs0 = rcs_curve(κ_f, u0, spaces0.X, θ)
rcsp = rcs_curve(κ_f, up, spacesp.X, θ)
writedlm(joinpath(outdir_rcs, "fichera_rcs_40pct.csv"), [collect(θ) rcs0 rcsp], ',')
println("[saved] fichera_rcs_40pct.csv")
