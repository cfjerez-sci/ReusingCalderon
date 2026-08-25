# Analytic Mie series bistatic RCS for a perfectly conducting (PEC) sphere,
# used as a reference to validate the EFIE solution in problems/p9.jl.
#
# Convention (Bohren & Huffman, "Absorption and Scattering of Light by
# Small Particles", ch. 4): incident plane wave e^{ikz} x̂ along +z,
# polarized along x. Size parameter x = k*radius. Mie coefficients for a
# PEC sphere are built from the Riccati-Bessel functions
#   ψ_n(x) = x j_n(x),   χ_n(x) = -x y_n(x),   ξ_n(x) = ψ_n(x) - i χ_n(x)
# via simple upward recursions (no external special-function package
# needed):
#   ψ_0 = sin(x),              ψ_1 = sin(x)/x - cos(x)
#   χ_0 = cos(x),              χ_1 = cos(x)/x + sin(x)
#   f_{n+1} = (2n+1)/x f_n - f_{n-1}
#   ψ_n' = ψ_{n-1} - (n/x) ψ_n,   χ_n' = χ_{n-1} - (n/x) χ_n
#   a_n = ψ_n'(x) / ξ_n'(x),      b_n = ψ_n(x) / ξ_n(x)
#
# Scattering amplitudes (B&H eq. 4.74), with μ = cos(θ) and angular
# functions π_n, τ_n built from the associated Legendre recursion:
#   S1(θ) = Σ (2n+1)/(n(n+1)) [a_n π_n(μ) + b_n τ_n(μ)]   (H-plane / ⊥ pol.)
#   S2(θ) = Σ (2n+1)/(n(n+1)) [a_n τ_n(μ) + b_n π_n(μ)]   (E-plane / ∥ pol.)
#
# Bistatic RCS: σ(θ) = (4π/k²) |S(θ)|². We use S2 for the E-plane cut
# (φ=0, the x-z plane, which contains both k̂ and the incident E field),
# matching the excitation in problems/p9.jl (E0 ∥ x̂, k̂ = ẑ).

using Makeitso

function mie_pec_coefficients(x::Real, nmax::Integer)

    ψ = zeros(Float64, nmax+1)
    χ = zeros(Float64, nmax+1)

    ψ[1] = sin(x)                    # ψ_0
    ψ[2] = sin(x)/x - cos(x)         # ψ_1
    χ[1] = cos(x)                    # χ_0
    χ[2] = cos(x)/x + sin(x)         # χ_1

    for n in 1:nmax-1
        ψ[n+2] = (2n+1)/x * ψ[n+1] - ψ[n]
        χ[n+2] = (2n+1)/x * χ[n+1] - χ[n]
    end

    a = zeros(ComplexF64, nmax)
    b = zeros(ComplexF64, nmax)

    for n in 1:nmax
        ψn, ψnm1 = ψ[n+1], ψ[n]
        χn, χnm1 = χ[n+1], χ[n]

        dψn = ψnm1 - (n/x) * ψn
        dχn = χnm1 - (n/x) * χn

        ξn  = ψn  - im*χn
        dξn = dψn - im*dχn

        a[n] = dψn / dξn
        b[n] = ψn  / ξn
    end

    return a, b
end

function mie_pec_S1S2(θ::Real, a::Vector, b::Vector)

    μ = cos(θ)
    nmax = length(a)

    p0 = 0.0   # π_0
    p1 = 1.0   # π_1 (base case, used directly at n=1)

    S1 = zero(ComplexF64)
    S2 = zero(ComplexF64)

    for n in 1:nmax
        πn = p1
        τn = n*μ*πn - (n+1)*p0

        c = (2n+1)/(n*(n+1))
        S1 += c * (a[n]*πn + b[n]*τn)
        S2 += c * (a[n]*τn + b[n]*πn)

        # advance (p0,p1) from (π_{n-1}, π_n) to (π_n, π_{n+1})
        p0, p1 = p1, ((2n+1)/n)*μ*p1 - ((n+1)/n)*p0
    end

    return S1, S2
end

@target mie_rcs (;κ, radius, θ) -> begin

    x = κ * radius
    nmax = ceil(Int, x + 4*cbrt(x) + 10)

    a, b = mie_pec_coefficients(x, nmax)

    rcs = Vector{Float64}(undef, length(θ))
    for (i,t) in enumerate(θ)
        _, S2 = mie_pec_S1S2(t, a, b)
        rcs[i] = 4π/κ^2 * abs2(S2)
    end

    return (;θ, rcs)
end
