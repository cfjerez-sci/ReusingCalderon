# NASA almond problem definition, with the admissible vector-field shape
# perturbation of Section 3.1.
#
# Mirrors problems/p10_perturbed_sphere.jl: `basemesh` serves a SHIPPED
# nominal mesh (no mesher in the reproduction path), `geo` applies the
# perturbation by moving vertices only, so connectivity -- and therefore RWG
# and BC degree-of-freedom indexing -- is identical to the nominal case, and
# `excitation` supplies the incident plane wave.
#
# Keyword arguments threaded by Makeitso:
#   meshname  "almond_lam12" (2673 dof) or "almond_lam20" (7269 dof)
#   case      "U" (no cutoff: the tip moves) or "T" (tip held fixed)
#   ell_name  "d5" or "d10", the correlation length of the random field
#   seed      0 for the nominal geometry; otherwise an exported field's seed
#   eps       amplitude, in metres; see data/almond/levels.csv
#
# INCIDENCE. The plane wave travels along +z with the electric field along
# +x. The almond's long axis is x, so this is broadside incidence with E
# along the body -- the configuration in which the tip and the seam both
# radiate, and the one the NASA almond benchmark is usually shown in.

using BEAST
using CompScienceMeshes
using DrWatson
using Makeitso

include("../postproc/almond_geometry.jl")

@target basemesh (;meshname) -> begin
    Γ0 = almond_orient(almond_mesh(meshname))
    vol = almond_check_orientation(Γ0)
    return (;Γ0, vol)
end

@target geo (basemesh,; meshname, case, ell_name, seed, eps) -> begin
    (;Γ0) = basemesh
    if seed == 0 || eps == 0.0
        return (;Γ=Γ0, maxdisp=0.0, cert=0.0, sup_DV=0.0, eps=0.0)
    end
    fld = almond_field(case, ell_name, seed)
    cert = almond_certificate(fld, eps)
    cert <= 0.5 + 1e-12 || error(
        "sample (case=$case, ell=$ell_name, seed=$seed) has " *
        "eps*sup|DV| = $(round(cert, digits=4)) > 0.5: the injectivity " *
        "certificate is violated. This is a specification error -- fix the " *
        "amplitude ladder; do NOT drop the sample.")
    Γ, dmax = perturb_almond_mesh(Γ0, fld, eps)
    return (;Γ, maxdisp=dmax, cert, sup_DV=fld.sup_DV, eps)
end

@target excitation (;κ) -> begin
    Einc = Maxwell3D.planewave(direction=ẑ, polarization=x̂, wavenumber=κ)
    Hinc = -1/(im*κ)*curl(Einc)
    return (;Einc, Hinc)
end
