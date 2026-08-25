# Classic Electric Field Integral Equation (EFIE) for scattering off a
# perfect electric conductor (PEC). Single domain, single unknown current j.
#
#   T[k,j] = e[k],   e = (n × Einc) × n
#
# where T is the Maxwell single-layer (EFIE) operator on the exterior
# wavenumber κ. This mirrors the structure of methods/LMT.jl (formulation
# -> spaces -> discretization -> solution) but without the multi-trace
# transmission/jump conditions needed for dielectric interfaces.

using BEAST
using Makeitso

@target formulation (excitation,; κ) -> begin

    @hilbertspace j
    @hilbertspace k

    (;Einc) = excitation
    e = (n × Einc) × n

    T = Maxwell3D.singlelayer(wavenumber=κ)
    N = BEAST.NCross()

    # same abstract biform, later assembled against different discrete
    # spaces (X,X for the primal EFIE, Y,Y for the dual/Calderón system)
    A = T[k,j]
    Nform = N[k,j]
    b = e[k]

    return (;bilforms=(;A, Nform), linforms=(;b))
end


@target spaces (geo) -> begin

    (;Γ) = geo
    X = raviartthomas(Γ)
    Y = buffachristiansen(Γ)

    return (;X, Y)
end


@target discretization (formulation, spaces) -> begin

    (;bilforms, linforms) = formulation
    (;A, Nform) = bilforms
    (;b) = linforms
    (;X, Y) = spaces

    Zxx = assemble(A, X, X)       # primal EFIE system
    Tyy = assemble(A, Y, Y)       # dual (BC) discretisation, for Calderón preconditioning
    Nxy = assemble(Nform, X, Y)   # duality pairing between X and Y
    bx = assemble(b, X)

    return (;matrices=(;Zxx, Tyy, Nxy), vectors=(;bx))
end


@target solution (discretization, spaces; s) -> begin

    (;matrices, vectors) = discretization
    (;Zxx, Tyy, Nxy) = matrices
    (;bx) = vectors
    (;X) = spaces

    @show s

    function solve(Zxx, Tyy, Nxy, bx, ::Type{Val{:lu}})
        u = Matrix(Zxx) \ Vector(bx)
        return u, -1
    end

    function solve(Zxx, Tyy, Nxy, bx, ::Type{Val{:gmres}})
        u, ch = BEAST.solve(
            BEAST.GMRESSolver(Zxx; restart=1500, abstol=1e-8, reltol=1e-8, maxiter=1500),
            bx)
        return u, ch
    end

    # Calderón-preconditioned EFIE: P = N⁻ᵀ Tyy N⁻¹, applied as a left
    # preconditioner to the primal system. N⁻¹ and N⁻ᵀ are themselves
    # applied through (inner) GMRES solves on the duality pairing Nxy,
    # following the standard BEAST recipe for EFIE Calderón preconditioning.
    function solve(Zxx, Tyy, Nxy, bx, ::Type{Val{:gmres_calderon}})
        NYX = BEAST.GMRESSolver(Nxy;  restart=1500, abstol=1e-8, reltol=1e-8, maxiter=1500)
        NXY = BEAST.GMRESSolver(Nxy'; restart=1500, abstol=1e-8, reltol=1e-8, maxiter=1500)
        P = NXY * Tyy * NYX
        u, ch = BEAST.solve(
            BEAST.GMRESSolver(Zxx; restart=1500, abstol=1e-8, reltol=1e-8, maxiter=1500,
                left_preconditioner=P),
            bx)
        return u, ch
    end

    u, ch = solve(Zxx, Tyy, Nxy, bx, Val{s})

    return (;u=BEAST.FEMFunction(u, X), ch)
end
