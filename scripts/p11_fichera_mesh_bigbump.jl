# Purely for the illustrative mesh-comparison figure: the Fichera cube's
# tested amplitudes (up to 15%, Table~tab:fichera) produce a fairly
# subtle bump relative to the cube's size. This script exports a larger,
# visually-clearer bump (40%) for the FIGURE ONLY -- it does not touch
# the EFIE discretization/RCS pipeline and is not a new numerical data
# point (no GMRES solve, no assembly; pure mesh geometry, seconds to run).

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using CompScienceMeshes
using DrWatson
using DelimitedFiles

include("../postproc/shape_perturbation_fichera.jl")

Γ0 = mesh_fichera(0.3)
Γp = perturb_fichera_mesh(Γ0, 0.4)

outdir = projectdir("data", "figures", "meshes")
mkpath(outdir)

V = reduce(vcat, [ [v[1] v[2] v[3]] for v in Γp.vertices ])
F = reduce(vcat, [ [f[1] f[2] f[3]] for f in Γp.faces ])
writedlm(joinpath(outdir, "fichera_perturbed40pct_verts.csv"), V, ',')
writedlm(joinpath(outdir, "fichera_perturbed40pct_faces.csv"), F, ',')
println("[saved] fichera_perturbed40pct: $(size(V,1)) verts, $(size(F,1)) faces")
