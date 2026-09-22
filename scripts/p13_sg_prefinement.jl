# Stochastic Galerkin p-refinement experiment for the SIAM manuscript
# (Corollary "Bochner-space and stochastic-Galerkin robustness"):
# numerically verify that GMRES on the COUPLED stochastic Galerkin
# system, left-preconditioned by the block-diagonal lift I ⊗ P0 of the
# frozen nominal Calderon preconditioner, has iteration counts that
# remain essentially FLAT as the stochastic polynomial dimension grows.
#
# Setup: sphere (radius 1), h = 0.2, kappa = 2, amplitude eps = 20%,
# d = 3 random modes: the (l,m) = (2,-2), (2,-1), (2,0) coefficients of
# the spherical-harmonic family (eq. (27) of the manuscript), each
# scaled by eps * y_i / l^2 with y in [-1,1]^3 (uniform product
# measure). Legendre polynomial chaos, total degree p = 1..4, dims
# |Lambda_p| = 4, 10, 20, 35. Stochastic integrals via tensorized
# 5-point Gauss-Legendre quadrature (125 nodes; exact for the
# psi_alpha*psi_beta polynomial factor up to per-dim degree 9).
#
# The SG operator is applied MATRIX-FREE from the 125 stored nodal EFIE
# matrices A(eps, y_q); P0 is applied exactly via an LU factorization
# of N_XY,0 (the same operator N^{-adj} Tyy N^{-1}, applied directly).
# Dependency-free full-memory left-preconditioned GMRES below.
#
# Registers to data/sweep/p13_sg_prefinement_summary.csv.
# Run: Julia: Execute active File in REPL.

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

module SimSG
include("../problems/p10_perturbed_sphere.jl")
include("../methods/EFIE.jl")
end

include("../postproc/shape_perturbation.jl")

# ------------------------------------------------------------------ setup
h, κ, radius = 0.2, 2.0, 1.0
ε            = 0.20
modes        = [(2,-2), (2,-1), (2,0)]     # d = 3 random directions
ps           = [1, 2, 3, 4]

outdir = projectdir("data", "sweep")
mkpath(outdir)
summary_file = joinpath(outdir, "p13_sg_prefinement_summary.csv")
open(summary_file, "w") do io
    println(io, "p,dimS,sg_dofs,iters_plain,converged_plain,iters_IkronP0,converged_IkronP0,trueres_plain,trueres_IkronP0")
end

println("="^70)
println("SG p-refinement (I kron P0) -- started ", now())
println("h=$h kappa=$κ eps=$ε modes=$modes p=$ps")
println("="^70)

# nominal ingredients (cached from earlier runs at h=0.2)
disc0   = make(SimSG.discretization; h, κ, radius, realization=0, lmin=2, lmax=6, amplitude=0.0)
spaces0 = make(SimSG.spaces; h, radius, realization=0, lmin=2, lmax=6, amplitude=0.0)
geo0    = make(SimSG.geo; h, radius, realization=0, lmin=2, lmax=6, amplitude=0.0)
Γ0      = geo0.Γ
ndof    = length(disc0.vectors.bx)
println("nominal ready: ndof = $ndof")

Tyy0 = Matrix{ComplexF64}(disc0.matrices.Tyy)
Nxy0 = Matrix{ComplexF64}(disc0.matrices.Nxy)
FN   = lu(Nxy0)
applyP0(x) = FN' \ (Tyy0 * (FN \ x))

# nominal RWG ordering fingerprints (ordering-consistency check)
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

# --------------------------------------------------- quadrature (5-pt GL)
gl_nodes = [-0.9061798459386640, -0.5384693101056831, 0.0,
             0.5384693101056831,  0.9061798459386640]
gl_w     = [ 0.2369268850561891,  0.4786286704993665, 0.5688888888888889,
             0.4786286704993665,  0.2369268850561891] ./ 2   # prob. weights

nodes3 = vec([(a,b,c) for a in gl_nodes, b in gl_nodes, c in gl_nodes])
w3     = vec([wa*wb*wc for wa in gl_w, wb in gl_w, wc in gl_w])
Q      = length(nodes3)
@assert abs(sum(w3) - 1) < 1e-12

# -------------------------------------- nodal assemblies A(eps,y_q), b(y_q)
formu = make(SimSG.formulation; κ)      # (;bilforms=(;A,Nform), linforms=(;b))
Aform = formu.bilforms.A
bform = formu.linforms.b

Aq = Vector{Matrix{ComplexF64}}(undef, Q)
bq = Vector{Vector{ComplexF64}}(undef, Q)

# Per-node permutation aligning the perturbed-mesh DOF ordering to the
# nominal one, built combinatorially from edge fingerprints (as in
# scripts/p10_*): perm[i] = index, in the perturbed ordering, of
# nominal RWG dof i. (The raviartthomas enumeration is geometry
# dependent, so this is rebuilt for every quadrature node.)
println("\nassembling $Q nodal EFIE matrices (h=$h) ...")
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
    @assert sort(perm) == collect(1:ndof) "DOF permutation not bijective at node $q!"
    Zfull = Matrix{ComplexF64}(assemble(Aform, Xq, Xq))
    bfull = Vector{ComplexF64}(assemble(bform, Xq))
    Aq[q] = Zfull[perm, perm]
    bq[q] = bfull[perm]
    q % 25 == 0 && @printf("  %d/%d nodes (%.0f s elapsed)\n", q, Q, time()-t0)
end
@printf("nodal assembly done: %.0f s total\n", time()-t0)

# ------------------------------------- Legendre chaos (orthonormal, uniform)
function legendre_upto(t::Float64, kmax::Int)
    P = zeros(kmax+1); P[1] = 1.0
    kmax >= 1 && (P[2] = t)
    for k in 1:kmax-1
        P[k+2] = ((2k+1)*t*P[k+1] - k*P[k]) / (k+1)
    end
    [sqrt(2k+1) * P[k+1] for k in 0:kmax]   # orthonormal wrt uniform prob.
end

function multiindices(d::Int, p::Int)
    idx = NTuple{3,Int}[]
    for i in 0:p, j in 0:p-i, k in 0:p-i-j
        push!(idx, (i,j,k))
    end
    sort(idx, by = t -> (sum(t), t))
end

# ---------------------------------------------- dependency-free GMRES (full)
"""
    gmres_full(applyA!, b; M=identity, reltol=1e-8, maxiter=600)

Left-preconditioned full-memory GMRES for the system A x = b, where
`applyA!(out, x)` computes out = A*x and `M(x)` applies the (left)
preconditioner. Stopping on the PRECONDITIONED relative residual
(matching the manuscript's convention). Returns (x, iters, converged).
"""
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
    k = 0
    converged = false
    for j in 1:maxiter
        k = j
        applyA!(tmp, V[j])
        w = M(tmp)
        for i in 1:j                       # modified Gram-Schmidt
            H[i,j] = dot(V[i], w)
            w .-= H[i,j] .* V[i]
        end
        H[j+1,j] = norm(w)
        push!(V, w ./ H[j+1,j])
        for i in 1:j-1                     # apply stored Givens
            t        = cs[i]*H[i,j] + sn[i]*H[i+1,j]
            H[i+1,j] = -conj(sn[i])*H[i,j] + conj(cs[i])*H[i+1,j]
            H[i,j]   = t
        end
        δ = sqrt(abs2(H[j,j]) + abs2(H[j+1,j]))
        cs[j] = conj(H[j,j])/δ; sn[j] = conj(H[j+1,j])/δ
        H[j,j] = δ; H[j+1,j] = 0
        g[j+1] = -conj(sn[j])*g[j]; g[j] = cs[j]*g[j]
        if abs(g[j+1])/β < reltol
            converged = true
            break
        end
    end
    ykr = H[1:k,1:k] \ g[1:k]
    x = zeros(ComplexF64, N)
    for i in 1:k
        x .+= ykr[i] .* V[i]
    end
    return x, k, converged
end

# --------------------------------------------------- deterministic self-test
# single-node (center, y = 0 -> nominal geometry) sanity check of the
# GMRES implementation against the known deterministic behavior
let qc = findfirst(t -> t == (0.0,0.0,0.0), nodes3)
    applyC!(out, x) = (out .= Aq[qc]*x; out)
    xs, its, cvs = gmres_full(applyC!, bq[qc]; reltol=1e-8, maxiter=600)
    tr = norm(bq[qc] - Aq[qc]*xs)/norm(bq[qc])
    @printf("\nself-test (nominal node): plain %d iters (conv=%s, trueres=%.2e; expect ~168, <1e-7)\n",
            its, cvs, tr)
    xsp, itsp, cvsp = gmres_full(applyC!, bq[qc]; M=applyP0, reltol=1e-8, maxiter=600)
    trp = norm(bq[qc] - Aq[qc]*xsp)/norm(bq[qc])
    @printf("self-test (nominal node): P0    %d iters (conv=%s, trueres=%.2e; expect ~13)\n",
            itsp, cvsp, trp)
end

# ------------------------------------------------------------- experiments
for p in ps
    Λ    = multiindices(3, p)
    dimS = length(Λ)
    # basis values at quadrature nodes: Ψ[q, α]
    Ψ = zeros(Q, dimS)
    for q in 1:Q
        y = nodes3[q]
        L1 = legendre_upto(y[1], p); L2 = legendre_upto(y[2], p); L3 = legendre_upto(y[3], p)
        for (a, α) in enumerate(Λ)
            Ψ[q,a] = L1[α[1]+1] * L2[α[2]+1] * L3[α[3]+1]
        end
    end
    G = Ψ' * Diagonal(w3) * Ψ
    @assert opnorm(G - I) < 1e-10 "chaos basis not orthonormal (max dev $(opnorm(G-I)))"

    Nsg = ndof * dimS
    # SG matvec: out = sum_q w_q (psi_q psi_q^T ⊗ A_q) x
    function applySG!(out::Vector{ComplexF64}, x::Vector{ComplexF64})
        Xm = reshape(x, ndof, dimS)
        Om = reshape(out, ndof, dimS)
        fill!(Om, 0)
        for q in 1:Q
            ψq = @view Ψ[q, :]
            u  = Xm * ψq                    # ndof
            v  = Aq[q] * u                  # ndof
            BLAS.ger!(ComplexF64(w3[q]), v, conj.(complex.(ψq)), Om)
        end
        out
    end
    # SG rhs
    bsg = zeros(ComplexF64, Nsg)
    Bm  = reshape(bsg, ndof, dimS)
    for q in 1:Q
        for a in 1:dimS
            Bm[:, a] .+= (w3[q] * Ψ[q,a]) .* bq[q]
        end
    end
    # blockwise preconditioner
    function MP0(x::Vector{ComplexF64})
        Xm = reshape(x, ndof, dimS)
        out = similar(x); Om = reshape(out, ndof, dimS)
        for a in 1:dimS
            Om[:, a] = applyP0(Xm[:, a])
        end
        out
    end

    println("\n--- p = $p  (dimS = $dimS, SG dofs = $Nsg) ---")
    tA = time()
    xpl, itpl, cvpl = gmres_full(applySG!, bsg; reltol=1e-8, maxiter=600)
    @printf("  [plain]     %d iters (converged=%s, %.0f s)\n", itpl, cvpl, time()-tA)
    tB = time()
    xpc, itpc, cvpc = gmres_full(applySG!, bsg; M=MP0, reltol=1e-8, maxiter=600)
    @printf("  [I kron P0] %d iters (converged=%s, %.0f s)\n", itpc, cvpc, time()-tB)

    r1 = similar(bsg); applySG!(r1, xpl); tr_pl = norm(bsg - r1)/norm(bsg)
    r2 = similar(bsg); applySG!(r2, xpc); tr_pc = norm(bsg - r2)/norm(bsg)
    @printf("  true residuals: plain=%.3e  IkronP0=%.3e\n", tr_pl, tr_pc)

    open(summary_file, "a") do io
        @printf(io, "%d,%d,%d,%d,%s,%d,%s,%.6e,%.6e\n",
            p, dimS, Nsg, itpl, cvpl, itpc, cvpc, tr_pl, tr_pc)
    end
    println("  [saved]")
end

println("\nSG p-refinement complete -- ", now())
println("Summary: ", summary_file)
