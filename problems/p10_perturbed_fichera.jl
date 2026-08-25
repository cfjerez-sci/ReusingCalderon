# Fichera-cube (non-convex, single reentrant corner, piecewise-flat)
# analogue of problems/p10_perturbed_ellipsoid.jl. Third geometry for the
# SIAM manuscript's robustness study: does the DOF-aligned frozen
# Calderon preconditioner's near-zero penalty (established on the sphere
# and the triaxial ellipsoid, both smooth convex shapes) also survive on a
# genuinely non-convex, non-smooth domain, under a qualitatively different
# perturbation (a localized flat-face bump/dent rather than a global
# spherical-harmonic radial field)?
#
# realization = 0 gives the exact nominal Fichera mesh; realization = 1
# (and only 1 -- this is a single fixed deterministic perturbation
# pattern, not a random family) gives the two-face-bumped perturbation at
# the given `amplitude` (see postproc/shape_perturbation_fichera.jl).
# Mesh connectivity (faces) is identical between nominal and perturbed
# cases by construction (taper vanishes at each perturbed face's
# boundary), so the DOF-alignment strategy from the sphere/ellipsoid
# studies applies unchanged.
#
# Reused unchanged from methods/EFIE.jl: formulation, spaces,
# discretization, solution targets.

using BEAST
using CompScienceMeshes
using DrWatson
using Makeitso

include("../postproc/shape_perturbation_fichera.jl")

@target geo (;h, realization, amplitude) -> begin
    Γ0 = mesh_fichera(h)
    if realization == 0
        return (;Γ=Γ0, amplitude=0.0)
    else
        Γ = perturb_fichera_mesh(Γ0, amplitude)
        return (;Γ, amplitude)
    end
end

@target excitation (;κ) -> begin
    Einc = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ)
    Hinc = -1/(im*κ)*curl(Einc)
    return (;Einc, Hinc)
end
