# PEC sphere problem with an additional `realization` parameter: 0 gives
# the exact nominal sphere mesh (identical to problems/p9.jl's geo
# target); realization = 1,2,... gives a reproducibly-seeded random
# low-degree real-spherical-harmonic perturbation of that same mesh,
# built by displacing vertex positions only (connectivity/faces
# unchanged), so the resulting discretization has identical DOF
# indexing to the nominal case -- see postproc/shape_perturbation.jl.
#
# Reused unchanged from methods/EFIE.jl: formulation, spaces,
# discretization, solution targets. Makeitso threads h/κ/radius (and,
# for this problem, realization/lmin/lmax/amplitude) down to whichever
# targets declare them as keyword arguments.

using BEAST
using CompScienceMeshes
using DrWatson
using Makeitso
using Random

include("../postproc/shape_perturbation.jl")

@target geo (;h, radius, realization, lmin, lmax, amplitude) -> begin
    Γ0 = meshsphere(radius, h)
    if realization == 0
        return (;Γ=Γ0, coeffs=nothing, amplitude=0.0)
    else
        rng    = MersenneTwister(1000 + realization)
        coeffs = random_perturbation_coeffs(rng; lmin, lmax, amplitude)
        Γ      = perturb_sphere_mesh(Γ0, radius, coeffs; lmax)
        return (;Γ, coeffs, amplitude)
    end
end

@target excitation (;κ) -> begin
    Einc = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ)
    Hinc = -1/(im*κ)*curl(Einc)
    return (;Einc, Hinc)
end
