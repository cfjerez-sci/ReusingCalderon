# Triaxial-ellipsoid analogue of problems/p10_perturbed_sphere.jl: a
# genuinely non-spherical nominal PEC scatterer (semi-axes a != b != c,
# fixed below at (1.5, 1.0, 0.7)), with the same reproducibly-seeded
# low-degree real-spherical-harmonic radial perturbation family used for
# the sphere study -- see postproc/shape_perturbation_ellipsoid.jl.
#
# realization = 0 gives the exact nominal ellipsoid mesh; realization =
# 1,2,... gives a reproducibly-seeded perturbation of it. Mesh
# connectivity (faces) is identical between nominal and perturbed cases
# by construction, so the DOF-alignment strategy from the sphere study
# (scripts/p10_*.jl) applies unchanged.
#
# Reused unchanged from methods/EFIE.jl: formulation, spaces,
# discretization, solution targets.

using BEAST
using CompScienceMeshes
using DrWatson
using Makeitso
using Random

include("../postproc/shape_perturbation_ellipsoid.jl")

const ELLIPSOID_A = 1.5
const ELLIPSOID_B = 1.0
const ELLIPSOID_C = 0.7

@target geo (;h, realization, lmin, lmax, amplitude) -> begin
    Γ0, unitverts = mesh_ellipsoid(ELLIPSOID_A, ELLIPSOID_B, ELLIPSOID_C, h)
    if realization == 0
        return (;Γ=Γ0, coeffs=nothing, amplitude=0.0)
    else
        rng    = MersenneTwister(1000 + realization)
        coeffs = random_perturbation_coeffs(rng; lmin, lmax, amplitude)
        Γ      = perturb_ellipsoid_mesh(ELLIPSOID_A, ELLIPSOID_B, ELLIPSOID_C,
                                         unitverts, Γ0.faces, coeffs; lmax)
        return (;Γ, coeffs, amplitude)
    end
end

@target excitation (;κ) -> begin
    Einc = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ)
    Hinc = -1/(im*κ)*curl(Einc)
    return (;Einc, Hinc)
end
