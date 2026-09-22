#!/usr/bin/env python3
"""Item 1: audit the intersection test, then regenerate the v1 table cleanly.

(a) BVH / SAT audit -- adjacency exclusion, tolerance, brute-force agreement.
(b) Three flagged pairs at the apex fans, with the actual intersection
    segments, so we can see they are genuine folds and not touching neighbours.
(c) The v1 amplitude table regenerated with ONE envelope and ONE fixed set of
    draws, rescaled across levels, so it is monotone by construction.
"""
import time
import numpy as np
import almond as al
import legacy as lg

LAM = al.D / 3.0                     # electrical size 3 lambda -> lambda = d/3
H12 = LAM / 12


def seg_tri(p, q, T):
    """Intersection point of segment pq with triangle T, or None."""
    a, b, c = T
    n = np.cross(b - a, c - a)
    dn = np.dot(q - p, n)
    if abs(dn) < 1e-18:
        return None
    t = np.dot(a - p, n) / dn
    if not (-1e-12 <= t <= 1 + 1e-12):
        return None
    x = p + t * (q - p)
    # barycentric inside test
    n2 = np.dot(n, n)
    u = np.dot(np.cross(c - b, x - b), n) / n2
    v = np.dot(np.cross(a - c, x - c), n) / n2
    w = 1 - u - v
    if min(u, v, w) < -1e-9:
        return None
    return x


def intersection_points(Va, Ta, Tb):
    pts = []
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        x = seg_tri(Ta[i], Ta[j], Tb)
        if x is not None:
            pts.append(x)
        x = seg_tri(Tb[i], Tb[j], Ta)
        if x is not None:
            pts.append(x)
    return pts


def main():
    print("=" * 74)
    print("(a) INTERSECTION TEST AUDIT")
    print("=" * 74)

    # coarse mesh so the brute force is affordable
    Vc, Fc = al.mesh(0.020)
    print(f"audit mesh h = 20 mm: {len(Vc)} vertices, {len(Fc)} faces")

    # -- adjacency exclusion: how many pairs are removed, and do the removed
    #    pairs really share a vertex?
    n = len(Fc)
    I, J = np.triu_indices(n, k=1)
    shared = (Fc[I][:, :, None] == Fc[J][:, None, :]).any(2).any(1)
    print(f"unordered face pairs          : {len(I)}")
    print(f"pairs sharing >=1 vertex      : {shared.sum()}  (excluded by index)")
    # sanity: every excluded pair really does share a vertex
    chk = [len(set(Fc[i]) & set(Fc[j])) for i, j in zip(I[shared][:5000],
                                                        J[shared][:5000])]
    print(f"min shared vertices among excluded (first 5000): {min(chk)}  "
          f"(must be >= 1)")
    adj_only = [len(set(Fc[i]) & set(Fc[j])) for i, j in
                zip(I[~shared][:5000], J[~shared][:5000])]
    print(f"max shared vertices among kept    (first 5000): {max(adj_only)}  "
          f"(must be 0)")

    # -- the nominal mesh must be clean under both tests
    t0 = time.time()
    hb = al.self_intersects(Vc, Fc)
    t1 = time.time()
    hf = al.verify_tri_tri(Vc, Fc)
    t2 = time.time()
    print(f"nominal, BVH   : {len(hb)} hits  ({t1-t0:.2f} s)")
    print(f"nominal, brute : {len(hf)} hits  ({t2-t1:.2f} s)")

    # -- tolerance sweep on the nominal mesh: a correct test is flat in tol
    for tol in (0.0, 1e-15, 1e-12, 1e-9, 1e-7):
        h = al.self_intersects(Vc, Fc, tol=tol)
        print(f"  nominal hits at tol = {tol:>7.0e} : {len(h)}")

    # -- a mesh that is KNOWN to fold: push one vertex through the body
    Vb = Vc.copy()
    k = int(np.argmax(Vb[:, 2]))                 # a top-of-seam vertex
    Vb[k, 2] -= 3.0 * abs(Vb[k, 2])              # drive it out the other side
    hb2 = sorted(al.self_intersects(Vb, Fc, max_report=10**6))
    hf2 = sorted(al.verify_tri_tri(Vb, Fc, max_report=10**6))
    print(f"seeded fold, BVH   : {len(hb2)} hits")
    print(f"seeded fold, brute : {len(hf2)} hits")
    print(f"BVH == brute force : {hb2 == hf2}")

    print()
    print("=" * 74)
    print("(b) FLAGGED PAIRS AT THE APEX FANS UNDER THE v1 NORMAL MODEL")
    print("=" * 74)
    V, F = al.mesh(H12)
    N = al.vertex_normals(V, F)
    u = lg.unit_displacement(V, N, ell=al.D / 5, seed=7, case="U")
    Vp = V + 0.50 * al.HALF_THICK_SEAM * u       # the 50% level Carlos specified
    hits = al.self_intersects(Vp, F, max_report=200)
    print(f"case U, ell = d/5, seed 7, level = 50% of seam half-thickness")
    print(f"flagged pairs: {len(hits)} (capped at 200)")
    tipd = lambda T: min(np.linalg.norm(T - np.array([al.X_TIP, 0, 0]), axis=1).min(),
                         np.linalg.norm(T - np.array([al.X_NOSE, 0, 0]), axis=1).min())
    shown = 0
    for (i, j) in hits:
        Ti, Tj = Vp[F[i]], Vp[F[j]]
        pts = intersection_points(Vp, Ti, Tj)
        if len(pts) < 2:
            continue
        seg = np.linalg.norm(pts[0] - pts[-1])
        d_apex = min(tipd(Ti), tipd(Tj))
        print(f"\n  pair ({i}, {j})   shared vertices: "
              f"{len(set(F[i]) & set(F[j]))}")
        print(f"    faces      {F[i]}  /  {F[j]}")
        print(f"    nearest apex distance  {1e3*d_apex:8.3f} mm "
              f"({d_apex/H12:.2f} h)")
        print(f"    crossing points        {len(pts)}, chord "
              f"{1e3*seg:.3f} mm")
        for p in pts[:2]:
            print(f"      x = ({1e3*p[0]:8.3f}, {1e3*p[1]:7.3f}, "
                  f"{1e3*p[2]:7.3f}) mm")
        # nominal positions of the same two faces: must NOT intersect
        pre = al.tri_tri(V[F[[i]]], V[F[[j]]])[0]
        print(f"    same pair on the NOMINAL mesh intersects: {bool(pre)}")
        shown += 1
        if shown == 3:
            break

    print()
    print("=" * 74)
    print("(c) v1 TABLE, ONE ENVELOPE, ONE FIXED DRAW SET, RESCALED")
    print("=" * 74)
    seeds = list(range(1, 21))
    levels = [0.05, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.375, 0.50]
    for ell_name, ell in (("d/5", al.D / 5), ("d/10", al.D / 10)):
        for case in ("U", "T"):
            U = {s: lg.unit_displacement(V, N, ell, s, case) for s in seeds}
            print(f"\n  ell = {ell_name}, case {case}   "
                  f"(same {len(seeds)} draws at every level)")
            print(f"  {'% half-thick':>12} {'mm':>7} {'lambda/':>8} "
                  f"{'h':>6} {'invalid':>9}")
            for L in levels:
                amp = L * al.HALF_THICK_SEAM
                bad = 0
                for s in seeds:
                    Vp = V + amp * U[s]
                    if al.self_intersects(Vp, F, max_report=1):
                        bad += 1
                print(f"  {100*L:11.1f}% {1e3*amp:7.2f} {LAM/amp:8.1f} "
                      f"{amp/H12:6.2f} {bad:5d}/{len(seeds)}")


if __name__ == "__main__":
    main()
