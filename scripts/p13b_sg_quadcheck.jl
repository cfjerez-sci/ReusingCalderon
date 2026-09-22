# Quadrature-convergence check for the SG p-refinement experiment
# (scripts/p13_sg_prefinement.jl): repeat the p = 4 computation with a
# tensorized 7-point Gauss-Legendre rule (343 nodes) instead of 5-point
# (125 nodes), to confirm that the quadrature-assembled stochastic
# Galerkin approximation's iteration counts are quadrature-converged.
# Everything else identical to p13. Registers to
# data/sweep/p13b_sg_quadcheck_summary.csv.

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

module SimSGb
include("../problems/p10_perturbed_sphere.jl")
include("../methods/EFIE.jl")
end

include("../postproc/shape_perturbation.jl")

h, κ, radius = 0.2, 2.0, 1.0
ε            = 0.20
modes        = [(2,-2), (2,-1), (2,0)]
p            = 4

outdir = projectdir("data", "sweep")
summary_file = joinpath(outdir, "p13b_sg_quadcheck_summary.csv")
open(summary_file, "w") do io
    println(io, "quadpts,p,dimS,sg_dofs,iters_plain,converged_plain,iters_IkronP0,converged_IkronP0,trueres_plain,trueres_IkronP0")
end

println("="^70)
println("SG quadrature check (7-pt GL, p=4) -- started ", now())
println("="^70)

disc0   = make(SimSGb.discretization; h, κ, radius, realization=0, lmin=2, lmax=6, amplitude=0.0)
spaces0 = make(SimSGb.spaces; h, radius, realization=0, lmin=2, lmax=6, amplitude=0.0)
geo0    = make(SimSGb.geo; h, radius, realization=0, lmin=2, lmax=6, amplitude=0.0)
Γ0      = geo0.Γ
ndof    = length(disc0.vectors.bx)

Tyy0 = Matrix{ComplexF64}(disc0.matrices.Tyy)
Nxy0 = Matrix{ComplexF64}(disc0.matrices.Nxy)
FN   = lu(Nxy0)
applyP0(x) = FN' \ (Tyy0 * (FN \ x))

function edge_fps(fns, faces)
    fps = Vector{Tuple{Int,Int}}(undef, length(fns))
    for (i, shapes) in enumerate(fns)
        cellids = unique(sh.cellid for sh in shapes)
        if length(cellids) == 2
            shared = sort(collect(intersect(Set(faces[cellids[1]]), Set(faces[cellids[2]]))))
            fps[i] = length(shared) == 2 ? (shared[1], shared[2]) : (-1, -i)
        else
            fps[i] = (-1, -i)
        end
    end
    fps
end
fps0 = edge_fps(spaces0.X.fns, Γ0.faces)

# 7-point Gauss-Legendre on [-1,1]
gl_nodes = [-0.9491079123427585, -0.7415311855993945, -0.4058451513773972,
             0.0,
             0.4058451513773972,  0.7415311855993945,  0.9491079123427585]
gl_w     = [ 0.1294849661688697,  0.2797053914892766,  0.3818300505051189,
             0.4179591836734694,
             0.3818300505051189,  0.2797053914892766,  0.1294849661688697] ./ 2

nodes3 = vec([(a,b,c) for a in gl_nodes, b in gl_nodes, c in gl_nodes])
w3     = vec([wa*wb*wc for wa in gl_w, wb in gl_w, wc in gl_w])
Q      = length(nodes3)
@assert abs(sum(w3) - 1) < 1e-12

formu = make(SimSGb.formulation; κ)
Aform = formu.bilforms.A
bform = formu.linforms.b

Aq = Vector{Matrix{ComplexF64}}(undef, Q)
bq = Vector{Vector{ComplexF64}}(undef, Q)

println("assembling $Q nodal EFIE matrices (h=$h) ...")
t0 = time()
for q in 1:Q
    y = nodes3[q]
    coeffs = Dict{Tuple{Int,Int},Float64}()
    for (i, (l,m)) in enumerate(modes)
        coeffs[(l,m)] = ε * y[i] / l^2
    end
    Γq = perturb_sphere_mesh(Γ0, radius, coeffs; lmax=2)
    Xq = raviartthomas(Γq)
    fpsq = edge_fps(Xq.fns, Γq.faces)
    dict = Dict{Tuple{Int,Int},Int}(fp => j for (j,fp) in enumerate(fpsq))
    perm = [dict[fps0[i]] for i in 1:ndof]
    @assert sort(perm) == collect(1:ndof)
    Zfull = Matrix{ComplexF64}(assemble(Aform, Xq, Xq))
    bfull = Vector{ComplexF64}(assemble(bform, Xq))
    Aq[q] = Zfull[perm, perm]
    bq[q] = bfull[perm]
    q % 50 == 0 && @printf("  %d/%d nodes (%.0f s)\n", q, Q, time()-t0)
end
@printf("nodal assembly done: %.0f s\n", time()-t0)

function legendre_upto(t::Float64, kmax::Int)
    P = zeros(kmax+1); P[1] = 1.0
    kmax >= 1 && (P[2] = t)
    for k in 1:kmax-1
        P[k+2] = ((2k+1)*t*P[k+1] - k*P[k]) / (k+1)
    end
    [sqrt(2k+1) * P[k+1] for k in 0:kmax]
end

function multiindices(p::Int)
    idx = NTuple{3,Int}[]
    for i in 0:p, j in 0:p-i, k in 0:p-i-j
        push!(idx, (i,j,k))
    end
    sort(idx, by = t -> (sum(t), t))
end

function gmres_full(applyA!, b::Vector{ComplexF64}; M = identity,
                    reltol = 1e-8, maxiter = 600)
    N  = length(b)
    Mb = M(b)
    β  = norm(Mb)
    V  = Vector{Vector{ComplexF64}}()
    push!(V, Mb ./ β)
    H  = zeros(ComplexF64, maxiter+1, maxiter)
    cs = zeros(ComplexF64, maxiter); sn = zeros(ComplexF64, maxiter)
    g  = zeros(ComplexF64, maxiter+1); g[1] = β
    tmp = similar(b)
    k = 0; converged = false
    for j in 1:maxiter
        k = j
        applyA!(tmp, V[j])
        w = M(tmp)
        for i in 1:j
            H[i,j] = dot(V[i], w)
            w .-= H[i,j] .* V[i]
        end
        H[j+1,j] = norm(w)
        push!(V, w ./ H[j+1,j])
        for i in 1:j-1
            t        = cs[i]*H[i,j] + sn[i]*H[i+1,j]
            H[i+1,j] = -conj(sn[i])*H[i,j] + conj(cs[i])*H[i+1,j]
            H[i,j]   = t
        end
        δ = sqrt(abs2(H[j,j]) + abs2(H[j+1,j]))
        cs[j] = conj(H[j,j])/δ; sn[j] = conj(H[j+1,j])/δ
        H[j,j] = δ; H[j+1,j] = 0
        g[j+1] = -conj(sn[j])*g[j]; g[j] = cs[j]*g[j]
        if abs(g[j+1])/β < reltol
            converged = true; break
        end
    end
    ykr = H[1:k,1:k] \ g[1:k]
    x = zeros(ComplexF64, N)
    for i in 1:k
        x .+= ykr[i] .* V[i]
    end
    return x, k, converged
end

Λ    = multiindices(p)
dimS = length(Λ)
Ψ = zeros(Q, dimS)
for q in 1:Q
    y = nodes3[q]
    L1 = legendre_upto(y[1], p); L2 = legendre_upto(y[2], p); L3 = legendre_upto(y[3], p)
    for (a, α) in enumerate(Λ)
        Ψ[q,a] = L1[α[1]+1] * L2[α[2]+1] * L3[α[3]+1]
    end
end
@assert opnorm(Ψ' * Diagonal(w3) * Ψ - I) < 1e-10

Nsg = ndof * dimS
function applySG!(out::Vector{ComplexF64}, x::Vector{ComplexF64})
    Xm = reshape(x, ndof, dimS)
    Om = reshape(out, ndof, dimS)
    fill!(Om, 0)
    for q in 1:Q
        ψq = @view Ψ[q, :]
        u  = Xm * ψq
        v  = Aq[q] * u
        BLAS.ger!(ComplexF64(w3[q]), v, conj.(complex.(ψq)), Om)
    end
    out
end
bsg = zeros(ComplexF64, Nsg)
Bm  = reshape(bsg, ndof, dimS)
for q in 1:Q, a in 1:dimS
    Bm[:, a] .+= (w3[q] * Ψ[q,a]) .* bq[q]
end
function MP0(x::Vector{ComplexF64})
    Xm = reshape(x, ndof, dimS)
    out = similar(x); Om = reshape(out, ndof, dimS)
    for a in 1:dimS
        Om[:, a] = applyP0(Xm[:, a])
    end
    out
end

println("\n--- p = $p, $Q-node quadrature (dimS = $dimS, SG dofs = $Nsg) ---")
xpl, itpl, cvpl = gmres_full(applySG!, bsg; reltol=1e-8, maxiter=600)
@printf("  [plain]     %d iters (converged=%s)\n", itpl, cvpl)
xpc, itpc, cvpc = gmres_full(applySG!, bsg; M=MP0, reltol=1e-8, maxiter=600)
@printf("  [I kron P0] %d iters (converged=%s)\n", itpc, cvpc)
r1 = similar(bsg); applySG!(r1, xpl); tr_pl = norm(bsg - r1)/norm(bsg)
r2 = similar(bsg); applySG!(r2, xpc); tr_pc = norm(bsg - r2)/norm(bsg)
@printf("  true residuals: plain=%.3e  IkronP0=%.3e\n", tr_pl, tr_pc)

open(summary_file, "a") do io
    @printf(io, "%d,%d,%d,%d,%d,%s,%d,%s,%.6e,%.6e\n",
        7, p, dimS, Nsg, itpl, cvpl, itpc, cvpc, tr_pl, tr_pc)
end
println("quadcheck complete -- ", now())
