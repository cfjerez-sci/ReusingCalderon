# One-off investigation script: inspect CompScienceMeshes' Mesh object
# structure (how vertices/faces are stored, how to construct a mesh with
# perturbed vertex positions but identical connectivity) before writing
# the shape-perturbation robustness study.

import Pkg
Pkg.activate((@__DIR__) * "/..")
Pkg.instantiate()

using CompScienceMeshes
using Exp25_CJH_KC_LocalMultiTrace

radius = 1.0
h = 0.3
Γ = meshsphere(radius, h)

println("typeof(Γ) = ", typeof(Γ))
println("propertynames(Γ) = ", propertynames(Γ))
println("fieldnames(typeof(Γ)) = ", fieldnames(typeof(Γ)))

println("\nnumvertices(Γ) = ", length(vertices(Γ)))
println("numcells/faces = ", length(Γ))

v1 = vertices(Γ)[1]
println("\ntypeof(vertex) = ", typeof(v1))
println("vertex[1] = ", v1)

c1 = cells(Γ)[1]
println("\ntypeof(cell) = ", typeof(c1))
println("cell[1] = ", c1)
println("Γ.faces[1] = ", Γ.faces[1])
println("typeof(Γ.faces) = ", typeof(Γ.faces))
println("Γ.vertices[1] = ", Γ.vertices[1])
println("typeof(Γ.vertices) = ", typeof(Γ.vertices))

# try constructing a new mesh with modified vertices, same cells
newverts = [v .* 1.05 for v in vertices(Γ)]
println("\ntypeof(newverts) = ", typeof(newverts))

try
    Γ2 = Mesh(newverts, cells(Γ))
    println("Mesh(newverts, cells(Γ)) WORKED: ", typeof(Γ2), " nv=", length(vertices(Γ2)))
catch err
    println("Mesh(newverts, cells(Γ)) FAILED: ", err)
end

try
    Γ2 = Mesh(newverts, Γ.faces)
    println("Mesh(newverts, Γ.faces) WORKED: ", typeof(Γ2), " nv=", length(vertices(Γ2)))
catch err
    println("Mesh(newverts, Γ.faces) FAILED: ", err)
end

println("\ncells(Γ) type = ", typeof(cells(Γ)))
println("cells(Γ)[1] = ", cells(Γ)[1])

println("\nmethods(CompScienceMeshes.Mesh):")
for m in methods(CompScienceMeshes.Mesh)
    println("  ", m)
end

# Does simply mutating vertices in place work, since faces/dict are
# reused unchanged? This would be the simplest possible perturbation
# strategy if Mesh is mutable or if we can construct via same type.
println("\nismutabletype(typeof(Γ)) = ", ismutabletype(typeof(Γ)))
