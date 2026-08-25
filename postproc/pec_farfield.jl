# Bistatic RCS computed from the numerical EFIE solution, for comparison
# against the analytic Mie series (postproc/mie_pec_sphere.jl).
#
# BEAST's MWFarField3D potential returns F_BEAST(y) = (ŷ×J̃(y))×ŷ, i.e. the
# transverse-projected radiation integral WITHOUT the α=-iκ weighting that
# the near-field potential applies to the current term (see
# maxwell/nearfield.jl vs maxwell/farfield.jl). Matching BEAST's near-field
# asymptotics (Green's function e^{-iκR}/(4πR), stationary phase, and
# integration by parts on the charge-continuity term) shows the physical
# far-field pattern, defined via Escat(r) ~ F_true(ŷ) e^{-iκr}/r, is
#   F_true(ŷ) = (α/4π) F_BEAST(ŷ) = (-iκ/4π) F_BEAST(ŷ)
# so the standard bistatic RCS σ(θ) = 4π|F_true|²/|E0|² becomes
#   σ(θ) = (κ²/4π) |F_BEAST(ŷ)|² / |E0|²
#
# E-plane cut (φ=0, the x-z plane containing k̂=ẑ and E0∥x̂): direction
# vectors x̂(θ) = (sin θ, 0, cos θ).

using Makeitso
using BEAST
using LinearAlgebra

@target farfield (solution,; κ, θ) -> begin

    T = Maxwell3D.singlelayer(wavenumber=κ)
    Tfar = BEAST.MWFarField3D(T)

    dirs = [point(sin(t), 0.0, cos(t)) for t in θ]

    X = solution.u.space
    j = solution.u.coeffs

    F = potential(Tfar, dirs, j, X)

    # σ(θ) = (κ²/4π) |F(θ)|² / |E0|² , with |E0| = 1 for BEAST's planewave
    rcs = κ^2/(4π) .* abs2.(norm.(F))

    return (;θ, rcs)
end
