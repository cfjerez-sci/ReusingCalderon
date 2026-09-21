# [CJ-15] Corner-reaching Fichera problem definition. Identical to
# problems/p10_perturbed_fichera.jl except that the perturbation is
# applied to the three reentrant notch faces rather than to two outer
# faces, so that it reaches the reentrant corner itself -- which is where
# the Lipschitz question that Assumption G leaves open actually lives.
# See postproc/shape_perturbation_fichera_corner.jl for the construction.
#
# The nominal mesh is the same mesh_fichera(h) used by the flat-face
# study, so the two sweeps are directly comparable at equal h.
#
# This file lives beside the flat-face one rather than replacing it, so
# that each gets its own Makeitso target cache and neither invalidates
# the other's stored results.
#
# Reused unchanged from methods/EFIE.jl: formulation, spaces,
# discretization, solution targets.

using BEAST
using CompScienceMeshes
using DrWatson
using Makeitso

include("../postproc/shape_perturbation_fichera_corner.jl")

@target geo (;h, realization, amplitude) -> begin
    Γ0 = mesh_fichera(h)
    if realization == 0
        return (;Γ=Γ0, amplitude=0.0)
    else
        Γ = perturb_fichera_mesh_corner(Γ0, amplitude)
        return (;Γ, amplitude)
    end
end

@target excitation (;κ) -> begin
    Einc = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ)
    Hinc = -1/(im*κ)*curl(Einc)
    return (;Einc, Hinc)
end
