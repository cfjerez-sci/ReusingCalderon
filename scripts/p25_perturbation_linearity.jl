# [CJ-12] DOES THE THEORY DO ANY WORK? Measuring what Theorem 8 predicts.
#
# Nothing in Section 5 currently tests Section 3. Corollary 7 says the
# (h,nu)-perturbation parameter grows LINEARLY in epsilon with a constant
# independent of h; Theorem 8 turns that into
#
#     kappa_S(P_0 A_eps,h)  <=  K_star * (1 + nu) / (1 - nu),
#     nu = C_geo * eps / gamma_a0,h,     both bounds h-uniform.
#
# Two things are therefore predicted and both are directly measurable at
# these problem sizes:
#
#   (P1) LINEARITY.  ||A_eps,h - A_0,h|| grows like eps. Because the
#        amplitude sweep rescales the SAME deformation direction, we have
#        A_eps - A_0 = eps*D + O(eps^2) for a fixed D, so the slope of
#        log||A_eps - A_0|| against log(eps) should be 1 -- and that slope
#        is the same in ANY norm, which is why the Frobenius norm is used
#        here and no discrete H^{-1/2}(div) Gram matrix is needed.
#
#   (P2) h-UNIFORMITY.  kappa_S(P_0 A_eps,h) as a function of eps should be
#        essentially the same curve at every mesh level. This is the
#        quantity Theorem 8 actually bounds, so no norm ambiguity enters.
#
# NOTE ON WHAT IS *NOT* MEASURED. The sharp constant in Corollary 7 is the
# sup of |a_0(u,v) - a_y(u,v)| / (||u||_X ||v||_Y) over the discrete spaces,
# with the X = H^{-1/2}(div_Gamma) norm. That norm is fractional and its
# Gram matrix is not available here, so this script does NOT claim to
# measure C_geo or nu_h themselves. It measures (P1) and (P2), which are the
# two falsifiable consequences, plus a Euclidean surrogate
#
#     nu_tilde := ||A_eps - A_0||_F / sigma_min(A_0)
#
# which has the shape of nu_h with the Euclidean norm standing in for the
# trace-space norms. Its eps-slope is meaningful; its absolute value and its
# h-scaling are NOT the theory's nu_h and are reported only for reference.
#
# Everything is compared in the NOMINAL DOF ordering: the perturbed matrix
# is transported by the permutation of Section 3.5, A_eps := Zxx_p[permX,permX].
#
# COST. Only Z_XX is needed per (h,eps); T_YY is needed once per mesh level
# (nominal only) to build P_0. Dense eigenvalues of P_0*A_eps dominate:
# roughly O(dof^3). At h=0.1 (4827 dof) expect ~1 min per amplitude.
# Pass --no-kappa to skip the eigenvalue part and get (P1) alone in minutes.
#
# Run from the repo root:
#   julia +1.11 --project scripts/p25_perturbation_linearity.jl
#   julia +1.11 --project scripts/p25_perturbation_linearity.jl --no-kappa
#   julia +1.11 --project scripts/p25_perturbation_linearity.jl --levels 0.2,0.15,0.1,0.075
#
# Writes data/sweep/p25_perturbation_linearity.csv (one row per (h,eps)).

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using Exp25_CJH_KC_LocalMultiTrace
using Makeitso
using DrWatson
using CompScienceMeshes
using BEAST
using LinearAlgebra
using Printf
using Dates
ENV["DRWATSON_WARN_DIRTY"] = "false"

module Sim2
include("../problems/p10_perturbed_sphere.jl")
include("../methods/EFIE.jl")
end

include(joinpath(@__DIR__, "p23_fingerprints.jl"))

# ------------------------------------------------------------------ config
const DO_KAPPA = !("--no-kappa" in ARGS)

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    (i === nothing || i == length(ARGS)) && return default
    return ARGS[i+1]
end

levels = parse.(Float64, split(argval("--levels", "0.2,0.15,0.1"), ","))
amps   = parse.(Float64, split(argval("--amps", "0.03,0.10,0.20,0.30,0.40,0.50"), ","))

radius      = 1.0
κ           = 2.0
lmin, lmax  = 2, 6
realization = 1
nominal_lmin, nominal_lmax, nominal_amplitude = 2, 6, 0.03

outdir = projectdir("data", "sweep")
mkpath(outdir)
csv = joinpath(outdir, "p25_perturbation_linearity.csv")
if !isfile(csv)
    open(csv, "w") do io
        println(io, "h,dof,amplitude,maxdisp_pct,normF_dA,normF_A0,rel_normF," *
                    "sigma_min_A0,nu_tilde,kappaS_P0Aeps,kappaS_P0A0,lam_max_abs,lam_min_abs")
    end
end

println("="^74)
println("[CJ-12] perturbation linearity and h-uniformity -- started ", now())
println("levels   = ", levels)
println("amps     = ", amps)
println("kappa_S  = ", DO_KAPPA ? "computed (dense eigenvalues)" : "SKIPPED (--no-kappa)")
println("="^74)

for h in levels
    @printf("\n########## mesh level h = %.4g ##########\n", h)
    try
        disc0 = make(Sim2.discretization; h, κ, radius, realization=0,
            lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)
        spaces0 = make(Sim2.spaces; h, radius, realization=0,
            lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)
        geo0 = make(Sim2.geo; h, radius, realization=0,
            lmin=nominal_lmin, lmax=nominal_lmax, amplitude=nominal_amplitude)

        A0 = Matrix{ComplexF64}(disc0.matrices.Zxx)
        dof = size(A0, 1)
        nF_A0 = norm(A0)
        @printf("  dof = %d   ||A_0||_F = %.6e\n", dof, nF_A0)

        # sigma_min(A_0): one SVD per mesh level, not per amplitude.
        t0 = time()
        smin = minimum(svdvals(A0))
        @printf("  sigma_min(A_0) = %.6e   (%.1f s)\n", smin, time()-t0)

        # P_0 from the NOMINAL geometry only: P_0 = N^{-T} T N^{-1}.
        P0 = nothing
        kappaS_P0A0 = NaN
        if DO_KAPPA
            t0 = time()
            T0 = Matrix{ComplexF64}(disc0.matrices.Tyy)
            N0 = Matrix{ComplexF64}(disc0.matrices.Nxy)
            P0 = (N0') \ (T0 / N0)
            @printf("  P_0 formed (%.1f s)\n", time()-t0)
            t0 = time()
            ev0 = eigvals(P0 * A0)
            a0 = abs.(ev0)
            kappaS_P0A0 = maximum(a0) / minimum(a0)
            @printf("  kappa_S(P_0 A_0) = %.4f   (nominal baseline, %.1f s)\n",
                kappaS_P0A0, time()-t0)
            T0 = nothing; N0 = nothing; GC.gc()
        end

        edgefp0 = dof_edge_fingerprint(spaces0.X.fns, geo0.Γ.faces)

        for amp in amps
            @printf("  --- eps = %.4g%% ---\n", 100*amp)
            try
                discp   = make(Sim2.discretization; h, κ, radius, realization, lmin, lmax, amplitude=amp)
                spacesp = make(Sim2.spaces; h, radius, realization, lmin, lmax, amplitude=amp)
                geop    = make(Sim2.geo; h, radius, realization, lmin, lmax, amplitude=amp)
                @assert length(discp.vectors.bx) == dof "DOF mismatch at h=$h amp=$amp"

                edgefpp = dof_edge_fingerprint(spacesp.X.fns, geop.Γ.faces)
                permX, okX = build_permutation(edgefp0, edgefpp)
                okX || error("X permutation failed at h=$h amp=$amp")

                # transport the perturbed matrix into the NOMINAL ordering
                Aeps = Matrix{ComplexF64}(discp.matrices.Zxx)[permX, permX]

                vertsp = geop.Γ.vertices
                nrm0 = norm.(geo0.Γ.vertices)
                vi = nrm0 .>= 1e-8*radius
                maxdisp = maximum(abs.([(norm(vertsp[i]) - radius)/radius*100
                                        for i in eachindex(vertsp)][vi]))

                dA = Aeps .- A0
                nF_dA = norm(dA)
                rel = nF_dA / nF_A0
                nu_tilde = nF_dA / smin
                @printf("     ||A_eps-A_0||_F = %.6e   rel = %.6e   nu_tilde = %.6e\n",
                    nF_dA, rel, nu_tilde)

                kS = NaN; lmax_a = NaN; lmin_a = NaN
                if DO_KAPPA
                    t0 = time()
                    ev = eigvals(P0 * Aeps)
                    av = abs.(ev)
                    lmax_a = maximum(av); lmin_a = minimum(av)
                    kS = lmax_a / lmin_a
                    @printf("     kappa_S(P_0 A_eps) = %.4f   (ratio to nominal %.4f, %.1f s)\n",
                        kS, kS/kappaS_P0A0, time()-t0)
                end

                open(csv, "a") do io
                    @printf(io, "%.6g,%d,%.6g,%.4f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                        h, dof, amp, maxdisp, nF_dA, nF_A0, rel, smin, nu_tilde,
                        kS, kappaS_P0A0, lmax_a, lmin_a)
                end
                Aeps = nothing; dA = nothing; GC.gc()
            catch err
                println("     [ERROR] h=$h amp=$amp: ", err)
            end
        end
        A0 = nothing; P0 = nothing; GC.gc()
    catch err
        println("  [ERROR] mesh level h=$h failed: ", err)
    end
end

println("\n" * "="^74)
println("[CJ-12] complete -- ", now())
println("CSV: ", csv)
println("Expected if the theory holds:")
println("  (P1) slope of log||A_eps-A_0||_F vs log(eps) == 1 at every h")
println("  (P2) kappa_S(P_0 A_eps) vs eps: the same curve at every h")
println("="^74)
