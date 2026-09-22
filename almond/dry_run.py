#!/usr/bin/env python3
"""Dry run for the v2 (vector-field) perturbation model.

Checks every geometry the campaign will use -- sweep, Monte Carlo campaign,
sensitivity and mesh check -- WITHOUT assembling anything, and reports per
sample the injectivity certificate eps*sup|DV|, the realised displacement in
four units, the element quality, and the triangle-triangle verdict.

Under this model no sample should ever be invalid: the certificate is imposed
in advance. A reported intersection is a bug, and is printed as one.

    python dry_run.py            # full run, writes dry_run.csv
    python dry_run.py --quick    # 3 seeds per cell, for a smoke test
"""
import csv
import sys
import time
import numpy as np
import almond as al

LAM = al.D / 3.0                      # electrical size 3 lambda
H12 = LAM / 12
H20 = LAM / 20
ELL = {"d/5": al.D / 5, "d/10": al.D / 10}
PAD = H12                             # sup taken over bbox(Gamma) grown by h
NGRID, REFINE = 32, 3
CERT = 0.5                            # eps * sup|DV| <= CERT

QUICK = "--quick" in sys.argv
N_SWEEP = 3 if QUICK else 10          # seeds per (level, case) cell
N_CAMP = 12 if QUICK else 100         # Monte Carlo sample count
SEED_SWEEP = lambda i: 1000 + i
SEED_CAMP = lambda i: 5000 + i


def units(disp_m):
    return dict(mm=1e3 * disp_m, lam=LAM / disp_m, h=disp_m / H12,
                pct=100 * disp_m / al.HALF_THICK_SEAM)


def main():
    t_all = time.time()
    print("=" * 78)
    print("ALMOND DRY RUN -- vector-field model  T(x) = x + eps V(x)")
    print("=" * 78)

    V12, F12 = al.mesh(H12)
    V20, F20 = al.mesh(H20)
    Vref, _ = al.mesh(LAM / 40)        # dense reference for normalisation
    for nm, (V, F, h) in (("lambda/12", (V12, F12, H12)),
                          ("lambda/20", (V20, F20, H20))):
        q = al.quality(V, F)
        print(f"{nm}: h = {1e3*h:.3f} mm, {len(V)} vertices, {len(F)} faces, "
              f"{3*len(F)//2} RWG dof")
        print(f"          min angle {q['min_angle']:.1f} deg, max aspect "
              f"{q['max_aspect']:.2f}, edges {1e3*q['edge_min']:.2f}-"
              f"{1e3*q['edge_max']:.2f} mm, self-int "
              f"{len(al.self_intersects(V, F))}")
    print(f"normalisation reference: {len(Vref)} vertices at h = lambda/40")
    print(f"lambda = {1e3*LAM:.3f} mm, seam half-thickness "
          f"{1e3*al.HALF_THICK_SEAM:.2f} mm")

    # ---------------------------------------------------------------- fields
    print("\nbuilding fields and evaluating sup|DV| over bbox(Gamma) + h ...")
    jobs = []                          # (tag, case, ellname, seed)
    for i in range(N_SWEEP):
        for case in ("U", "T"):
            jobs.append(("sweep", case, "d/5", SEED_SWEEP(i)))
    for i in range(N_CAMP):
        jobs.append(("campaign", "U", "d/5", SEED_CAMP(i)))
    for i in range(N_SWEEP):
        for case in ("U", "T"):
            jobs.append(("sens", case, "d/10", SEED_SWEEP(i)))

    fields, sup = {}, {}
    t0 = time.time()
    for k, (tag, case, en, sd) in enumerate(jobs):
        key = (case, en, sd)
        if key in fields:
            continue
        f = al.make_field(ELL[en], sd, case, Vref)
        b, s = al.sup_jacobian(f, Vref, pad=PAD, ngrid=NGRID, refine=REFINE)
        fields[key], sup[key] = f, (b, s)
    print(f"  {len(fields)} distinct fields in {time.time()-t0:.1f} s")

    for en in ("d/5", "d/10"):
        for case in ("U", "T"):
            v = [sup[k][0] for k in sup if k[0] == case and k[1] == en]
            if not v:
                continue
            print(f"  sup|DV|  case {case}, ell = {en:5s}: "
                  f"{min(v):7.2f} .. {max(v):7.2f} m^-1   over {len(v)} draws")

    # the ladder is set by the WORST draw at the primary correlation length
    s5 = max(sup[k][0] for k in sup if k[1] == "d/5")
    eps_top_raw = CERT / s5
    eps_top = np.floor(eps_top_raw * 1e3 * 10) / 1e4      # round down to 0.1 mm
    levels = [(i + 1) / 5 * eps_top for i in range(5)]
    print(f"\nworst sup|DV| at ell = d/5 : {s5:.2f} m^-1")
    print(f"eps allowed by the bound   : {1e3*eps_top_raw:.3f} mm "
          f"-> ladder top {1e3*eps_top:.1f} mm")
    print(f"levels (mm): " + ", ".join(f"{1e3*e:.1f}" for e in levels))

    s10 = max(sup[k][0] for k in sup if k[1] == "d/10")
    kmax10 = sum(1 for e in levels if e * s10 <= CERT)
    print(f"worst sup|DV| at ell = d/10: {s10:.2f} m^-1 -> certified for the "
          f"first {kmax10} of the 5 levels")

    # ------------------------------------------------------------- geometries
    rows, bad = [], []
    print("\nchecking geometries ...")

    def contraction(V, Vp, F):
        """min |T(x)-T(y)| / |x-y| over the mesh edges.

        The Lipschitz certificate says this cannot fall below 1 - eps*sup|DV|,
        for ANY pair of points, so measuring it on the edges is a direct
        empirical test of the bound rather than a restatement of it.
        """
        E = np.unique(np.sort(np.concatenate(
            [F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
        a = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        b = np.linalg.norm(Vp[E[:, 0]] - Vp[E[:, 1]], axis=1)
        return float((b / a).min())

    def check(group, case, en, sd, k, eps, V, F, mesh_name):
        f = fields[(case, en, sd)]
        Vp, d = al.perturb(V, f, eps)
        q = al.quality(Vp, F)
        contr = contraction(V, Vp, F)
        hits = al.self_intersects(Vp, F, max_report=4)
        dm = float(d.max())
        u = units(dm)
        row = dict(group=group, mesh=mesh_name, case=case, ell=en, seed=sd,
                   level=k, eps_mm=1e3 * eps, sup_DV=sup[(case, en, sd)][0],
                   cert=eps * sup[(case, en, sd)][0], disp_mm=u["mm"],
                   disp_lam=u["lam"], disp_h=u["h"], disp_pct=u["pct"],
                   min_angle=q["min_angle"], max_aspect=q["max_aspect"],
                   min_area=q["min_area"], contraction=contr,
                   tip_disp_mm=1e3 * float(d[int(np.argmax(V[:, 0]))]),
                   n_hits=len(hits))
        rows.append(row)
        if hits:
            bad.append((row, hits))
        return row

    t0 = time.time()
    for k, eps in enumerate(levels, 1):
        for case in ("U", "T"):
            for i in range(N_SWEEP):
                check("sweep", case, "d/5", SEED_SWEEP(i), k, eps,
                      V12, F12, "lambda/12")
    print(f"  sweep      {sum(1 for r in rows if r['group']=='sweep'):4d} "
          f"geometries  ({time.time()-t0:.1f} s)")

    t0 = time.time()
    for i in range(N_CAMP):
        check("campaign", "U", "d/5", SEED_CAMP(i), 5, levels[4],
              V12, F12, "lambda/12")
    print(f"  campaign   {N_CAMP:4d} geometries  ({time.time()-t0:.1f} s)")

    t0 = time.time()
    for k in range(1, kmax10 + 1):
        for case in ("U", "T"):
            for i in range(N_SWEEP):
                check("sens", case, "d/10", SEED_SWEEP(i), k, levels[k - 1],
                      V12, F12, "lambda/12")
    print(f"  sensitivity{sum(1 for r in rows if r['group']=='sens'):4d} "
          f"geometries  ({time.time()-t0:.1f} s)")

    t0 = time.time()
    for i in range(3):
        check("meshcheck", "U", "d/5", SEED_SWEEP(i), 5, levels[4],
              V20, F20, "lambda/20")
    print(f"  mesh check    3 geometries  ({time.time()-t0:.1f} s)")

    # ------------------------------------------------------------- reporting
    def block(title, sel):
        sub = [r for r in rows if sel(r)]
        if not sub:
            return
        print(f"\n{title}")
        print(f"  {'lvl':>3} {'case':>4} {'eps mm':>7} {'disp mm':>8} "
              f"{'lam/':>6} {'h':>5} {'%thk':>6} {'eps*supDV':>19} "
              f"{'min ang':>8} {'max asp':>8} {'contr':>7} {'tip mm':>8} "
              f"{'bad':>4}")
        key = lambda r: (r["level"], r["case"])
        for lv, cs in sorted({key(r) for r in sub}):
            g = [r for r in sub if key(r) == (lv, cs)]
            c = [r["cert"] for r in g]
            print(f"  {lv:3d} {cs:>4} {g[0]['eps_mm']:7.2f} "
                  f"{max(r['disp_mm'] for r in g):8.3f} "
                  f"{min(r['disp_lam'] for r in g):6.1f} "
                  f"{max(r['disp_h'] for r in g):5.2f} "
                  f"{max(r['disp_pct'] for r in g):6.1f} "
                  f"{min(c):6.3f}-{max(c):6.3f} ({len(g):3d})  "
                  f"{min(r['min_angle'] for r in g):8.2f} "
                  f"{max(r['max_aspect'] for r in g):8.2f} "
                  f"{min(r['contraction'] for r in g):7.3f} "
                  f"{max(r['tip_disp_mm'] for r in g):8.3f} "
                  f"{sum(r['n_hits'] > 0 for r in g):4d}")

    block("SWEEP  (ell = d/5, lambda/12)", lambda r: r["group"] == "sweep")
    block("CAMPAIGN  (ell = d/5, case U, level 5, lambda/12)",
          lambda r: r["group"] == "campaign")
    block("SENSITIVITY  (ell = d/10, lambda/12)", lambda r: r["group"] == "sens")
    block("MESH CHECK  (ell = d/5, case U, level 5, lambda/20)",
          lambda r: r["group"] == "meshcheck")

    worst = min(r["contraction"] - (1 - r["cert"]) for r in rows)
    print(f"\ncontraction margin  min(|Te|/|e|) - (1 - eps*sup|DV|) = "
          f"{worst:+.4f}   (must be >= 0: the certificate is not violated)")
    print(f"total geometries checked : {len(rows)}")
    print(f"invalid (self-intersecting): {len(bad)}")
    if bad:
        print("\n*** BUG: the certificate was satisfied and the mesh still "
              "folded. Offending samples:")
        for r, h in bad[:5]:
            print(f"   {r['group']} case {r['case']} seed {r['seed']} "
                  f"level {r['level']}  cert {r['cert']:.3f}  pairs {h}")

    with open("dry_run.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote dry_run.csv   ({time.time()-t_all:.1f} s total)")


if __name__ == "__main__":
    main()
