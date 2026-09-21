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

# [mesh-consistency fix] The nominal mesh is now its own Makeitso target,
# keyed on the mesh parameters alone, and `geo` takes it as a dependency
# instead of calling the mesher itself. Previously every (realization,
# amplitude) combination re-meshed independently, so two campaigns at the
# same h could silently receive different triangulations -- which is what
# produced the dof discrepancy disclosed in Section 5.3 and the aborted
# amplitude-0.3 rows of the linearity study. `geo` still declares the same
# keyword arguments, so its cache key is unchanged and existing cached
# results stay valid.

# [mesh-consistency fix, step 2 of 3] `basemesh` prefers a mesh shipped with
# the repository over calling the mesher. gmsh is not reproducible across
# sessions -- the same meshsphere(1.0, 0.1) call has returned both 4827 and
# 4791 degrees of freedom -- so a from-scratch recomputation drifts away from
# the published numbers unless the mesh itself is fixed. Export them once
# with scripts/p28_export_base_meshes.jl; if a file is absent the mesher is
# used and a warning says so.
function _shipped_mesh(name)
    path = projectdir("data", "meshes", name * ".jld2")
    isfile(path) || return nothing
    return wload(path)["mesh"]
end

@target basemesh (;h) -> begin
    name = "fichera_h$(h)"
    Γ0 = _shipped_mesh(name)
    if Γ0 === nothing
        @warn "no shipped mesh $name; meshing with gmsh (not reproducible across sessions)"
        Γ0 = mesh_fichera(h)
    end
    return (;Γ0)
end

@target geo (basemesh,; h, realization, amplitude) -> begin
    (;Γ0) = basemesh
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
