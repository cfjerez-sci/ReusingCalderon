#!/usr/bin/env python3
"""
Interior Maxwell (PEC cavity) eigenvalues of the unit ball, and their distance
from the wavenumbers requested for the bidirectional wavenumber-mismatch
experiment.

Background
----------
The EFIE on a PEC scatterer D loses injectivity exactly when kappa^2 is an
interior Maxwell eigenvalue of D (manuscript Section 2.1).  For the unit ball
these eigenvalues are known in closed form:

    TE_{l,n} modes :  j_l(kappa) = 0,                 l >= 1
    TM_{l,n} modes :  d/dx [ x j_l(x) ] |_{x=kappa} = 0,   l >= 1

with j_l the spherical Bessel function of the first kind.  Each has geometric
multiplicity 2l+1.  (l = 0 gives no nontrivial Maxwell mode.)

Equivalently, TE eigenvalues are zeros of the Riccati-Bessel function
psi_l(x) = x j_l(x) (excluding x = 0) and TM eigenvalues are zeros of
psi_l'(x).

This script finds every such eigenvalue in a requested range by dense bracketing
plus Brent refinement, and cross-checks the TE values against the known
identity j_l zeros == Bessel J_{l+1/2} zeros.

Usage
-----
    python3 resonance_check.py                 # default range and kappa list
    python3 resonance_check.py --kmax 12

Output: a table of eigenvalues, and for each requested kappa_solve the nearest
eigenvalue, the gap, and the gap measured in units of the mesh resolution.
"""

import argparse
import numpy as np
from scipy.special import spherical_jn
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Mode functions
# ---------------------------------------------------------------------------


def te_fun(x, l):
    """psi_l(x)/x = j_l(x); zeros (x>0) are the TE cavity eigenvalues."""
    return spherical_jn(l, x)


def tm_fun(x, l):
    """d/dx [ x j_l(x) ] = j_l(x) + x j_l'(x); zeros are the TM eigenvalues."""
    return spherical_jn(l, x) + x * spherical_jn(l, x, derivative=True)


def find_zeros(f, l, xmin, xmax, n_scan=200000):
    """All sign changes of f(.,l) on (xmin, xmax], refined by Brent."""
    xs = np.linspace(xmin, xmax, n_scan)
    fs = f(xs, l)
    out = []
    sgn = np.sign(fs)
    idx = np.nonzero((sgn[:-1] * sgn[1:]) < 0)[0]
    for i in idx:
        r = brentq(f, xs[i], xs[i + 1], args=(l,), xtol=1e-13, rtol=1e-14)
        out.append(r)
    return out


def cavity_eigenvalues(kmax, lmax=None, kmin=1e-6):
    """All interior Maxwell eigenvalues of the unit ball in (kmin, kmax]."""
    if lmax is None:
        # j_l has its first zero above ~l, so l > kmax contributes nothing.
        lmax = int(np.ceil(kmax)) + 4
    rows = []
    for l in range(1, lmax + 1):
        for k in find_zeros(te_fun, l, kmin, kmax):
            rows.append((k, "TE", l, 2 * l + 1))
        for k in find_zeros(tm_fun, l, kmin, kmax):
            rows.append((k, "TM", l, 2 * l + 1))
    rows.sort(key=lambda r: r[0])
    return rows


def verify(rows, kmax, tol=1e-9):
    """Three independent cross-checks on the computed eigenvalues.

    1. Residual: |f(root)| must be at round-off.
    2. mpmath: zeros of j_l are the zeros of J_{l+1/2} (if mpmath present).
    3. Closed form for l = 1.  With x j_1(x) = sin x / x - cos x,
       d/dx[x j_1(x)] = 0  <=>  x cos x - sin x + x^2 sin x = 0
                          <=>  tan x = x / (1 - x^2).
       TE roots solve j_1(x) = 0 <=> tan x = x.
    """
    report = []

    # (1) residuals
    worst = 0.0
    for k, kind, l, _ in rows:
        f = te_fun if kind == "TE" else tm_fun
        worst = max(worst, abs(f(k, l)))
    report.append(("residual max|f(root)|", "%.2e" % worst, worst < 1e-10))

    # (2) mpmath half-integer Bessel zeros
    try:
        import mpmath as mp
        bad = []
        for l in range(1, int(np.ceil(kmax)) + 4):
            ours = sorted(k for k, kind, ll, _ in rows if kind == "TE" and ll == l)
            for n, a in enumerate(ours, start=1):
                b = float(mp.besseljzero(mp.mpf(l) + mp.mpf(0.5), n))
                if abs(a - b) > tol:
                    bad.append((l, n, a, b))
        report.append(("TE == zeros of J_{l+1/2} (mpmath)",
                       "all %d match" % sum(1 for r in rows if r[1] == "TE")
                       if not bad else str(bad[:3]), not bad))
    except ImportError:
        report.append(("TE == zeros of J_{l+1/2} (mpmath)", "mpmath absent", None))

    # (3) closed form, l = 1
    te1 = sorted(k for k, kind, l, _ in rows if kind == "TE" and l == 1)
    tm1 = sorted(k for k, kind, l, _ in rows if kind == "TM" and l == 1)
    ok_te1 = all(abs(np.tan(x) - x) < 1e-6 * max(1.0, abs(x)) for x in te1)
    ok_tm1 = all(abs(x * np.cos(x) - np.sin(x) + x * x * np.sin(x))
                 < 1e-6 * max(1.0, x * x) for x in tm1)
    report.append(("TE l=1 solves tan x = x", "%d roots" % len(te1), ok_te1))
    report.append(("TM l=1 solves x cos x-sin x+x^2 sin x=0",
                   "%d roots" % len(tm1), ok_tm1))
    return report


def safe_ladder(rows, exponents, k0_lo=0.8, k0_hi=4.0, n=6401):
    """Scan anchors k0 and report the worst relative gap of the dyadic ladder
    {k0 * 2^d : d in exponents} to the interior spectrum.  Returns the scan and
    the best anchor."""
    eig = np.array([r[0] for r in rows])
    k0s = np.linspace(k0_lo, k0_hi, n)
    worst = np.empty_like(k0s)
    for i, k0 in enumerate(k0s):
        ks = k0 * 2.0 ** np.array(exponents, dtype=float)
        rel = np.min(np.abs(ks[:, None] - eig[None, :]), axis=1) / ks
        worst[i] = rel.min()
    j = int(np.argmax(worst))
    return k0s, worst, k0s[j], worst[j]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=float, default=10.0)
    ap.add_argument("--h", type=float, default=0.1,
                    help="mesh size used in the experiment (unit sphere)")
    ap.add_argument("--kappas", type=float, nargs="*",
                    default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
                    help="kappa_solve values to check")
    ap.add_argument("--precond", type=float, nargs="*", default=[1.0, 2.0, 4.0],
                    help="kappa_precond values to check")
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    rows = cavity_eigenvalues(args.kmax)
    checks = verify(rows, args.kmax)

    print("=" * 74)
    print("INTERIOR MAXWELL (PEC CAVITY) EIGENVALUES OF THE UNIT BALL")
    print("=" * 74)
    print("range (0, %.1f]   -- %d eigenvalues (counted without multiplicity)"
          % (args.kmax, len(rows)))
    for name, detail, ok in checks:
        mark = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        print("  [%s] %-38s %s" % (mark, name, detail))
    print()
    print("%10s  %5s  %4s  %6s" % ("kappa", "type", "l", "mult."))
    print("-" * 34)
    for k, kind, l, m in rows:
        print("%10.6f  %5s  %4d  %6d" % (k, kind, l, m))

    allk = sorted(set(list(args.kappas) + list(args.precond)))
    print()
    print("=" * 74)
    print("PROXIMITY OF REQUESTED WAVENUMBERS TO THE INTERIOR SPECTRUM")
    print("=" * 74)
    print("h = %.3f on the unit sphere; 'el/wave' = 2*pi/(kappa*h)" % args.h)
    print()
    hdr = ("%8s  %8s  %6s  %11s  %9s  %8s  %s"
           % ("kappa", "role", "el/wave", "nearest eig", "type,l", "gap",
              "rel.gap"))
    print(hdr)
    print("-" * len(hdr))
    out = []
    for k in allk:
        role = []
        if k in args.kappas:
            role.append("solve")
        if k in args.precond:
            role.append("prec")
        role = "+".join(role)
        d = [(abs(k - kk), kk, kind, l) for kk, kind, l, _ in rows]
        gap, kk, kind, l = min(d)
        elw = 2 * np.pi / (k * args.h)
        rel = gap / k
        flag = ""
        if rel < 0.02:
            flag = "  <<< VERY CLOSE"
        elif rel < 0.05:
            flag = "  <<< close"
        print("%8.3f  %8s  %6.1f  %11.5f  %5s,%-3d  %8.4f  %6.2f%%%s"
              % (k, role, elw, kk, kind, l, gap, 100 * rel, flag))
        out.append(dict(kappa=k, role=role, el_per_wave=elw, nearest=kk,
                        kind=kind, l=l, gap=gap, rel_gap=rel))

    # ---- resonance-safe anchor search for the dyadic ladder ----------------
    exps = [-3, -2, -1, 0, 1, 2]
    k0s, worst, best_k0, best_gap = safe_ladder(rows, exps)
    k0_ref = 2.0
    i_ref = int(np.argmin(np.abs(k0s - k0_ref)))
    print()
    print("=" * 74)
    print("RESONANCE-SAFE ANCHOR FOR THE DYADIC LADDER  {k0 * 2^d}, d in %s"
          % exps)
    print("=" * 74)
    print("criterion: maximise the WORST relative gap over the six rungs")
    print()
    print("  k0 = %.3f (requested)  -> worst rel.gap %5.2f%%" %
          (k0_ref, 100 * worst[i_ref]))
    print("  k0 = %.3f (optimal)    -> worst rel.gap %5.2f%%" %
          (best_k0, 100 * best_gap))
    print()
    for label, k0 in (("requested k0=2", k0_ref), ("optimal k0=%.3f" % best_k0,
                                                   best_k0)):
        ks = [k0 * 2.0 ** d for d in exps]
        gaps = []
        for k in ks:
            g = min(abs(k - kk) for kk, _, _, _ in rows) / k
            gaps.append(g)
        print("  %-22s kappa = %s" % (label,
              ", ".join("%.3f" % k for k in ks)))
        print("  %-22s gap%%  = %s" % ("",
              ", ".join("%5.1f" % (100 * g) for g in gaps)))

    # ---- best detuned alternative within +/- band --------------------------
    eig = np.array([r[0] for r in rows])
    band = 0.12
    print()
    print("=" * 74)
    print("BEST DETUNED ALTERNATIVE WITHIN +/-%d%% OF EACH REQUESTED kappa"
          % int(100 * band))
    print("=" * 74)
    print("%8s  %8s  %10s  %10s  %s"
          % ("kappa", "rel.gap", "best alt.", "its rel.gap", "verdict"))
    print("-" * 62)
    for k in allk:
        g0 = np.min(np.abs(k - eig)) / k
        cand = np.linspace(k * (1 - band), k * (1 + band), 4001)
        gg = np.min(np.abs(cand[:, None] - eig[None, :]), axis=1) / cand
        j = int(np.argmax(gg))
        verdict = ("already clean" if g0 > 0.10 else
                   ("detune helps" if gg[j] > 2 * g0 and gg[j] > 0.05 else
                    "no clean spot nearby"))
        print("%8.3f  %7.2f%%  %10.4f  %10.2f%%  %s"
              % (k, 100 * g0, cand[j], 100 * gg[j], verdict))

    print()
    print("Note: the interior spectrum of the unit ball begins at kappa ="
          " %.6f," % eig.min())
    print("so EVERY kappa below that is exactly resonance-free.  The density of")
    print("eigenvalues grows with kappa, which is why no clean spot exists near 8.")

    print()
    print("Legend: rel.gap = |kappa - nearest eigenvalue| / kappa.")
    print("A small rel.gap means the EFIE at that kappa is near-singular for")
    print("reasons that have nothing to do with preconditioner reuse, so the")
    print("iteration count there must NOT be read as ordinary mismatch")
    print("behaviour (manuscript Part E, item 7).")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["kappa", "type", "l", "multiplicity"])
            for k, kind, l, m in rows:
                w.writerow(["%.10f" % k, kind, l, m])
        print("\neigenvalues written to %s" % args.csv)


if __name__ == "__main__":
    main()
