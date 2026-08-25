# Ellipsoid analogue of shape_perturbation.jl: a genuinely non-spherical
# nominal geometry (triaxial ellipsoid, semi-axes a != b != c), built by
# anisotropically scaling a unit-sphere mesh, then perturbed by the SAME
# family of random low-degree real-spherical-harmonic radial deformations
# used for the sphere study -- evaluated on the underlying UNIT-SPHERE
# angular coordinates (theta,phi) of each vertex (i.e. before anisotropic
# scaling), and then applied multiplicatively to the already-scaled
# ellipsoid vertex position. This keeps mesh connectivity (faces)
# identical to the nominal ellipsoid mesh, so the DOF-alignment strategy
# from postproc/shape_perturbation.jl (edge/vertex-set fingerprints, mesh
# topology only) applies completely unchanged.
#
# Nominal ellipsoid vertex:   v0 = (a*u1, b*u2, c*u3),  u = unit-sphere vertex
# Perturbed vertex:           v  = (1 + delta(theta,phi)) * v0
# where (theta,phi) are the angular coordinates of u (NOT of v0), so the
# perturbation field's angular structure is identical to the sphere case
# -- only the base geometry it is displacing is now an ellipsoid.

using CompScienceMeshes
using LinearAlgebra
using Random

include("shape_perturbation.jl")   # reuses assoc_legendre_table, real_sphharm,
                                    # random_perturbation_coeffs, perturbation_radius_factor

"""
    mesh_ellipsoid(a, b, c, h)

Build the nominal (unperturbed) triaxial ellipsoid mesh with semi-axes
`(a,b,c)` by anisotropically scaling a unit-radius sphere mesh generated
at resolution `h` (same `h` convention as `meshsphere`). Also returns the
underlying unit-sphere vertices (needed to evaluate the perturbation
field's angular coordinates consistently -- see module docstring).
"""
function mesh_ellipsoid(a::Real, b::Real, c::Real, h::Real)
    Γunit = meshsphere(1.0, h)
    unitverts = Γunit.vertices
    T = eltype(unitverts)
    newverts = [T(a*u[1], b*u[2], c*u[3]) for u in unitverts]
    Γ0 = Mesh(newverts, Γunit.faces)
    return Γ0, unitverts
end

"""
    perturb_ellipsoid_mesh(a, b, c, unitverts, faces, coeffs; lmax)

Return a new mesh with identical connectivity (`faces`) to the nominal
ellipsoid mesh but with vertex positions displaced multiplicatively by
    v_new = (1 + delta(theta,phi)) * (a*u1, b*u2, c*u3)
where `(theta,phi)` are the angular coordinates of the corresponding
UNIT-SPHERE vertex `u = (u1,u2,u3)` in `unitverts`.
"""
function perturb_ellipsoid_mesh(a::Real, b::Real, c::Real, unitverts, faces,
                                 coeffs::Dict{Tuple{Int,Int},Float64}; lmax::Int=6)
    T = eltype(unitverts)
    newverts = similar(unitverts)
    for (i, u) in enumerate(unitverts)
        r = norm(u)
        if r < 1e-8
            newverts[i] = T(a*u[1], b*u[2], c*u[3])   # degenerate/origin vertex: leave at nominal (scaled) position
            continue
        end
        θ = acos(clamp(u[3]/r, -1.0, 1.0))
        φ = atan(u[2], u[1])
        δ = perturbation_radius_factor(θ, φ, coeffs, lmax)
        newverts[i] = T((1+δ)*a*u[1], (1+δ)*b*u[2], (1+δ)*c*u[3])
    end
    return Mesh(newverts, faces)
end
