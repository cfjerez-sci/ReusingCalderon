# Random shape perturbations of a reference sphere mesh via low-degree
# real spherical harmonics, keeping mesh connectivity (faces) identical
# to the nominal mesh -- only vertex positions move. This lets a
# Calderon preconditioner built once on the nominal sphere be reused,
# unchanged, on the perturbed geometry's linear system (same DOF
# indexing, since RWG/BC bases are built from mesh topology alone --
# confirmed via scripts/p10_investigate_mesh_api.jl: Mesh(newverts,
# Γ0.faces) reconstructs a mesh with identical connectivity).
#
# Real spherical harmonics Y_l^m(θ,φ), orthonormal on the unit sphere,
# built from a self-contained stable associated-Legendre recursion (same
# style as postproc/mie_pec_sphere.jl -- no external special-function
# package needed):
#   P_m^m(x)     = -(2m-1) sqrt(1-x^2) P_{m-1}^{m-1}(x)        (seed)
#   P_{m+1}^m(x) = x(2m+1) P_m^m(x)
#   (l-m) P_l^m(x) = x(2l-1) P_{l-1}^m(x) - (l+m-1) P_{l-2}^m(x)
# Normalization: N_l^m = sqrt((2l+1)/(4π) * (l-m)!/(l+m)!)
#   Y_l^0        = N_l^0 P_l^0(cosθ)
#   Y_l^m (m>0)  = √2 N_l^m P_l^m(cosθ) cos(mφ)
#   Y_l^m (m<0)  = √2 N_l^m P_l^|m|(cosθ) sin(|m|φ)

using CompScienceMeshes
using LinearAlgebra
using Random

function assoc_legendre_table(lmax::Int, x::Float64)
    P = zeros(Float64, lmax+1, lmax+1)     # P[l+1,m+1] = P_l^m(x)
    somx2 = sqrt(max(0.0, (1-x)*(1+x)))
    P[1,1] = 1.0                            # P_0^0
    for m in 1:lmax
        P[m+1,m+1] = -P[m,m] * (2m-1) * somx2
    end
    for m in 0:lmax-1
        P[m+2,m+1] = x*(2m+1)*P[m+1,m+1]
    end
    for m in 0:lmax
        for l in m+2:lmax
            P[l+1,m+1] = (x*(2l-1)*P[l,m+1] - (l+m-1)*P[l-1,m+1]) / (l-m)
        end
    end
    return P
end

function real_sphharm(l::Int, m::Int, φ::Float64, P::AbstractMatrix)
    am  = abs(m)
    Nlm = sqrt((2l+1)/(4π) * factorial(l-am)/factorial(l+am))
    Plm = P[l+1, am+1]
    if m == 0
        return Nlm * Plm
    elseif m > 0
        return sqrt(2) * Nlm * Plm * cos(m*φ)
    else
        return sqrt(2) * Nlm * Plm * sin(am*φ)
    end
end

"""
    random_perturbation_coeffs(rng; lmin=2, lmax=6, amplitude=0.03)

Random i.i.d. Uniform(-1,1) coefficients a_lm for l=lmin:lmax, m=-l:l,
with per-mode scale decaying like 1/l^2 (so low-degree modes dominate,
keeping the perturbed shape smooth/star-shaped) and overall scaled by
`amplitude`.
"""
function random_perturbation_coeffs(rng::AbstractRNG; lmin::Int=2, lmax::Int=6, amplitude::Float64=0.03)
    coeffs = Dict{Tuple{Int,Int},Float64}()
    for l in lmin:lmax
        decay = 1.0 / l^2
        for m in -l:l
            coeffs[(l,m)] = amplitude * decay * (2*rand(rng) - 1)   # Uniform(-1,1)
        end
    end
    return coeffs
end

function perturbation_radius_factor(θ::Float64, φ::Float64, coeffs::Dict{Tuple{Int,Int},Float64}, lmax::Int)
    x = cos(θ)
    P = assoc_legendre_table(lmax, x)
    s = 0.0
    for ((l,m), a) in coeffs
        s += a * real_sphharm(l, m, φ, P)
    end
    return s
end

"""
    perturb_sphere_mesh(Γ0, radius, coeffs; lmax)

Return a new mesh with identical connectivity (`faces`) to `Γ0` but with
vertex positions displaced radially by
    r(θ,φ) = radius * (1 + Σ a_lm Y_l^m(θ,φ))
where (θ,φ) are the angular coordinates of each nominal vertex direction
(measured from the origin, which is assumed to be the sphere's center).
"""
function perturb_sphere_mesh(Γ0, radius::Real, coeffs::Dict{Tuple{Int,Int},Float64}; lmax::Int=6)
    verts0    = Γ0.vertices
    newverts  = similar(verts0)
    for (i, v) in enumerate(verts0)
        r = norm(v)
        if r < 1e-8*radius
            # Not a real surface point (e.g. an unreferenced/placeholder
            # vertex some mesh backends carry at the origin -- confirmed
            # via scripts/p10_plot_perturbed_geometry.jl: index 1 of
            # CompScienceMeshes' meshsphere output sits exactly at the
            # origin and is not referenced by any face). Leave it
            # untouched rather than dividing by ~0 and poisoning the
            # perturbed mesh with a NaN vertex.
            newverts[i] = v
            continue
        end
        θ = acos(clamp(v[3]/r, -1.0, 1.0))
        φ = atan(v[2], v[1])
        δ = perturbation_radius_factor(θ, φ, coeffs, lmax)
        newr = radius * (1 + δ)
        newverts[i] = (newr/r) * v
    end
    return Mesh(newverts, Γ0.faces)
end
