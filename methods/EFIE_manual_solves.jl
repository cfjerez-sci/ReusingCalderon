# Standalone (non-@target) GMRES solve helpers for the shape-perturbation
# robustness study (scripts/p10_PEC_sphere_perturbation_robustness.jl).
#
# Mirrors the two solve branches already used in methods/EFIE.jl
# (plain GMRES, and Calderon-preconditioned GMRES with
# P = NXY * Tyy * NYX), but exposed as plain functions that take the
# (Zxx, bx) system and (Tyy, Nxy) preconditioner ingredients as
# SEPARATE arguments. This is what lets the same
# `solve_calderon_gmres` function be used for both:
#   - the FROZEN strategy: Zxx/bx from a perturbed discretization,
#     Tyy/Nxy from the nominal (unperturbed) discretization
#   - the FRESH strategy:  Zxx/bx/Tyy/Nxy all from the same perturbed
#     discretization (gold-standard baseline)
# by simply passing different Tyy/Nxy arguments.

using BEAST

function solve_plain_gmres(Zxx, bx)
    t0 = time()
    u, ch = BEAST.solve(
        BEAST.GMRESSolver(Zxx; restart=1500, abstol=1e-8, reltol=1e-8, maxiter=1500),
        bx)
    return u, ch, time() - t0
end

function build_calderon_precond(Tyy, Nxy)
    NYX = BEAST.GMRESSolver(Nxy;  restart=1500, abstol=1e-8, reltol=1e-8, maxiter=1500)
    NXY = BEAST.GMRESSolver(Nxy'; restart=1500, abstol=1e-8, reltol=1e-8, maxiter=1500)
    return NXY * Tyy * NYX
end

function solve_calderon_gmres(Zxx, bx, Tyy, Nxy)
    t0 = time()
    P = build_calderon_precond(Tyy, Nxy)
    u, ch = BEAST.solve(
        BEAST.GMRESSolver(Zxx; restart=1500, abstol=1e-8, reltol=1e-8, maxiter=1500,
            left_preconditioner=P),
        bx)
    return u, ch, time() - t0
end
