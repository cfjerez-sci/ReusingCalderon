# Fichera-cube analogue of shape_perturbation_ellipsoid.jl: a non-convex,
# piecewise-FLAT polyhedron (one reentrant corner) rather than the smooth
# sphere/ellipsoid. The sphere/ellipsoid studies perturb the WHOLE surface
# with a global low-degree spherical-harmonic radial field; here we instead
# perturb only "a couple" of the cube's flat faces with a smooth bump/dent
# normal displacement that is tapered to exactly zero at that face's own
# boundary (the face is a flat axis-aligned square, so a
# cos(pi*x/2)^2-type taper vanishes to machine precision at the edges).
#
# Because the taper vanishes exactly at each perturbed face's boundary,
# vertices shared with neighbouring faces never move, so mesh connectivity
# (faces array) is IDENTICAL between nominal and perturbed meshes -- the
# same DOF-alignment strategy (edge/vertex-set fingerprints, mesh topology
# only) used for the sphere/ellipsoid studies applies unchanged.
#
# Nominal solid: cube [-1,1]^3 minus octant [0,1]^3 (see geos/fichera.geo).
# Perturbed faces (fixed choice, not random): the two full (un-notched)
# square faces x=-1 and y=-1. These are chosen precisely because they are
# far from the reentrant corner, so the perturbation's effect can be
# cleanly separated from the geometry's intrinsic non-convex singularity.

using CompScienceMeshes
using DrWatson
using LinearAlgebra

# Signed volume via the divergence theorem, assuming a closed,
# consistently-oriented triangle mesh: V = (1/6) sum_faces v1.(v2 x v3).
# Positive iff the orientation is outward-pointing (standard right-hand-
# rule convention: CCW as seen from outside each triangle).
function _signed_volume(verts, faces)
    V = 0.0
    for f in faces
        v1, v2, v3 = verts[f[1]], verts[f[2]], verts[f[3]]
        V += dot(v1, cross(v2, v3))
    end
    return V / 6
end

"""
    mesh_fichera(h)

Load the nominal Fichera-cube mesh (cube [-1,1]^3 minus octant [0,1]^3)
from `geos/fichera.geo` at target mesh size `h`, via Gmsh/OpenCASCADE
boolean subtraction (see that file for the CSG construction).

Gmsh meshes each CAD surface entity independently, so the resulting
global triangle soup is generally NOT globally-orientation-consistent
(`CompScienceMeshes.isoriented` fails, which trips up
`BEAST.buffachristiansen`'s internal assertion). We fix this here:
`CompScienceMeshes.orient(Γ0)` propagates a single consistent orientation
by breadth-first search over face adjacency (mutates `Γ0.faces` in
place), then we check the resulting signed volume and flip the whole
mesh if it came out inward-pointing.
"""
function mesh_fichera(h::Real)
    geofile = projectdir("geos/fichera.geo")
    Γ0 = meshgeo(geofile; physical="Gamma", h=h)
    CompScienceMeshes.orient(Γ0)
    @assert CompScienceMeshes.isoriented(Γ0) "orient() failed to produce a globally consistent orientation"
    if _signed_volume(Γ0.vertices, Γ0.faces) < 0
        Γ0 = -Γ0   # flip: orient() only guarantees consistency, not outward-ness
    end
    return Γ0
end

# Smooth taper on [-1,1], exactly 0 at the endpoints, exactly 1 at 0.
_taper(t::Real) = cos(pi * clamp(t, -1.0, 1.0) / 2)^2

"""
    perturb_fichera_mesh(Γ0, amplitude; tol=1e-6)

Return a new mesh with identical connectivity (`faces`) to the nominal
Fichera mesh but with vertex positions on the x=-1 and y=-1 faces bumped
along their outward normal by
    amplitude * taper(y)*taper(z)   (on the x=-1 face, taper over y,z in [-1,1])
    amplitude * taper(x)*taper(z)   (on the y=-1 face, taper over x,z in [-1,1])
A vertex lying on the shared edge x=-1,y=-1 gets contributions from both
faces (both tapers vanish there individually, so this is consistent and
the edge stays fixed). `amplitude` may be negative (dent) or positive
(bulge); it is a physical length (same units as the cube's half-width 1).
"""
function perturb_fichera_mesh(Γ0, amplitude::Real; tol::Real=1e-6)
    verts = Γ0.vertices
    T = eltype(verts)
    newverts = similar(verts)
    for (i, v) in enumerate(verts)
        x, y, z = v[1], v[2], v[3]
        dx = 0.0
        dy = 0.0
        if abs(x - (-1.0)) < tol
            dx += amplitude * _taper(y) * _taper(z)   # outward normal (-1,0,0)
        end
        if abs(y - (-1.0)) < tol
            dy += amplitude * _taper(x) * _taper(z)   # outward normal (0,-1,0)
        end
        newverts[i] = T(x - dx, y - dy, z)
    end
    return Mesh(newverts, Γ0.faces)
end
