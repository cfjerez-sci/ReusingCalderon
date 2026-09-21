# [CJ-15] Corner-reaching variant of shape_perturbation_fichera.jl.
#
# The sweep in scripts/p10_fichera_amplitude_sweep.jl perturbs the two
# full outer faces x=-1 and y=-1, deliberately AWAY from the reentrant
# corner. That test returns a flat iteration count with a zero
# frozen-minus-fresh gap at every amplitude, and the realized displacement
# stays below a single mesh width -- so it is open to the objection that
# the perturbation is simply not binding, and that what is being measured
# is that a bump on a flat face far from the singularity does not matter.
#
# Here the perturbation is moved ONTO the singularity. The Fichera solid
# [-1,1]^3 \ [0,1]^3 has three "notch" squares,
#
#     {x=0, 0<=y,z<=1},  {y=0, 0<=x,z<=1},  {z=0, 0<=x,y<=1},
#
# which meet pairwise along the three reentrant edges and all three at the
# reentrant corner (0,0,0). On the solid, the outward normals there are
# +x, +y and +z respectively. We displace each notch square along its own
# outward normal by
#
#     amplitude * g(u) * g(v),      g(t) = cos(pi*t/2)^2,
#
# with (u,v) the two in-face coordinates. Since g(0)=1 and g(1)=g'(1)=0,
# the displacement is MAXIMAL at the reentrant corner and vanishes, to
# machine precision and with vanishing slope, on the two edges where each
# notch square meets an L-shaped outer face. Hence:
#
#   * the six outer faces stay exactly flat (their vertices never move);
#   * a vertex on a reentrant edge, say {x=0,y=0,z}, receives
#     a*g(0)*g(z) from the x-notch and a*g(0)*g(z) from the y-notch, i.e.
#     the two contributions are consistent and lie in orthogonal
#     directions;
#   * the reentrant corner itself moves to amplitude*(1,1,1), a
#     displacement of magnitude sqrt(3)*amplitude along the body diagonal;
#   * vertex positions change but `faces` does not, so mesh connectivity
#     is preserved exactly and the DOF-alignment permutation of
#     Section "Implementation" applies unchanged, as for every other
#     geometry in the study.
#
# amplitude > 0 pushes the corner outward into the removed octant (a
# shallower notch); amplitude < 0 deepens it. It is a physical length, in
# the same units as the cube's half-width 1. Note that the maximum vertex
# displacement is sqrt(3)*|amplitude|, NOT |amplitude| as in the flat-face
# variant; the sweep script reports the measured value.

using CompScienceMeshes
using DrWatson
using LinearAlgebra

include(joinpath(@__DIR__, "shape_perturbation_fichera.jl"))  # mesh_fichera, _taper

"""
    perturb_fichera_mesh_corner(Γ0, amplitude; tol=1e-6)

Return a new mesh with connectivity identical to `Γ0` in which the three
reentrant notch faces of the Fichera cube are displaced along their
outward normals with a taper that is maximal at the reentrant corner and
vanishes where each notch face meets an outer face. See the file header.
"""
function perturb_fichera_mesh_corner(Γ0, amplitude::Real; tol::Real=1e-6)
    verts = Γ0.vertices
    T = eltype(verts)
    newverts = similar(verts)
    for (i, v) in enumerate(verts)
        x, y, z = v[1], v[2], v[3]
        dx = 0.0; dy = 0.0; dz = 0.0
        # notch face x=0, spanned by y,z in [0,1]; outward normal +x
        if abs(x) < tol && y > -tol && z > -tol && y < 1 + tol && z < 1 + tol
            dx += amplitude * _taper(y) * _taper(z)
        end
        # notch face y=0, spanned by x,z in [0,1]; outward normal +y
        if abs(y) < tol && x > -tol && z > -tol && x < 1 + tol && z < 1 + tol
            dy += amplitude * _taper(x) * _taper(z)
        end
        # notch face z=0, spanned by x,y in [0,1]; outward normal +z
        if abs(z) < tol && x > -tol && y > -tol && x < 1 + tol && y < 1 + tol
            dz += amplitude * _taper(x) * _taper(y)
        end
        newverts[i] = T(x + dx, y + dy, z + dz)
    end
    return Mesh(newverts, Γ0.faces)
end
