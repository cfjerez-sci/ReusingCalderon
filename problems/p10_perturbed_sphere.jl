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

@target basemesh (;h, radius) -> begin
    name = "sphere_h$(h)_r$(radius)"
    Γ0 = _shipped_mesh(name)
    if Γ0 === nothing
        @warn "no shipped mesh $name; meshing with gmsh (not reproducible across sessions)"
        Γ0 = meshsphere(radius, h)
    end
    return (;Γ0)
end

@target geo (basemesh,; h, radius, realization, lmin, lmax, amplitude) -> begin
    (;Γ0) = basemesh
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
