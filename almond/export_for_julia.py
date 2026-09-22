#!/usr/bin/env python3
"""Write everything the Julia drivers need: meshes, field parameters, levels.

Nothing about the deformation is recomputed on the Julia side except the
arithmetic: the random Fourier features (w_k, b_k, a_k), the normalising
scale and the cutoff lengths are all exported as numbers, and Julia evaluates

    V(x) = scale * chi(x) * sqrt(2/M) * sum_k a_k cos(w_k . x + b_k)

at whatever vertex coordinates its mesh happens to have. The field is a
function of POSITION, so nothing depends on vertex ordering and the Gmsh
reader is free to renumber. The same file therefore serves the lambda/12 and
lambda/20 meshes, and the perturbation is identical in both.

Layout, under <repo>/data/almond/:
    almond_lam12.msh, almond_lam20.msh      nominal meshes (Gmsh 2.2 ASCII)
    fields_index.csv                        one row per field, with `scale`
    fields/<case>_<ell>_<seed>.csv          M rows of w1,w2,w3,b,a1,a2,a3
    levels.csv                              the amplitude ladder
"""
import csv
import os
import sys
import numpy as np
import almond as al

LAM = al.D / 3.0
H12, H20 = LAM / 12, LAM / 20
PAD, NGRID, REFINE, CERT = H12, 32, 3, 0.5
N_SWEEP, N_CAMP = 10, 100
DEAD, RAMP = 0.10, 0.25

# The two correlation lengths, as denominators of d. The primary one sets the
# amplitude ladder, since it is the one the sweep and the campaign run at; the
# secondary is the sensitivity case. Passing them here rather than hard-coding
# them is what makes switching the ladder a command instead of an edit:
#
#     python export_for_julia.py ../data/almond        -> d/5 and d/10
#     python export_for_julia.py ../data/almond 4 10   -> d/4 and d/10
#
# The drivers then take --ell d4. A longer correlation length buys amplitude
# by making the field smoother, which is also easier for the preconditioner;
# that trade belongs in the paper, not hidden in a default.
OUT = sys.argv[1] if len(sys.argv) > 1 else "export"
DEN_P = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
DEN_S = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0


def ell_name(den):
    return "d" + (f"{den:g}".replace(".", "p"))


PRIMARY, SECONDARY = ell_name(DEN_P), ell_name(DEN_S)
ELL = {PRIMARY: al.D / DEN_P, SECONDARY: al.D / DEN_S}


def write_msh(path, V, F):
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$Nodes\n{len(V)}\n")
        for i, v in enumerate(V, 1):
            f.write(f"{i} {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{len(F)}\n")
        for i, t in enumerate(F, 1):
            f.write(f"{i} 2 2 0 1 {t[0]+1} {t[1]+1} {t[2]+1}\n")
        f.write("$EndElements\n")


def main():
    os.makedirs(os.path.join(OUT, "fields"), exist_ok=True)
    V12, F12 = al.mesh(H12)
    V20, F20 = al.mesh(H20)
    Vref, _ = al.mesh(LAM / 40)
    # A positive enclosed volume is NOT enough: BEAST asserts
    # CompScienceMeshes.isoriented before it will build the BC dual space, and
    # the ring mesher leaves eight faces in each apex fan traversing their
    # shared edges the wrong way -- too few to flip the volume's sign, enough
    # to fail the assertion. orient_consistently propagates one orientation
    # over the whole surface and verifies the result.
    F12, vol12 = al.orient_consistently(V12, F12)
    F20, vol20 = al.orient_consistently(V20, F20)
    for nm, Fx in (("lambda/12", F12), ("lambda/20", F20)):
        n, lone, dup = al.edge_consistency(Fx)
        print(f"{nm} edges: {n} directed, {lone} lone, {dup} repeated "
              f"(both must be 0)")
    exact = 0.0400            # measured surface area, for the record
    print(f"lambda/12: {len(V12)} v, {len(F12)} f, {3*len(F12)//2} dof, "
          f"volume {vol12:.6e} m^3")
    print(f"lambda/20: {len(V20)} v, {len(F20)} f, {3*len(F20)//2} dof, "
          f"volume {vol20:.6e} m^3  (ratio {vol20/vol12:.5f})")
    for nm, (Vx, Fx) in (("almond_lam12", (V12, F12)),
                         ("almond_lam20", (V20, F20))):
        # CSV pair: the format p28 already uses for the sphere and Fichera
        # meshes, and the path almond_geometry.jl loads by default. The Gmsh
        # copy beside it is the fallback and the human-readable version.
        np.savetxt(os.path.join(OUT, nm + "_verts.csv"), Vx, delimiter=",",
                   fmt="%.17g")
        np.savetxt(os.path.join(OUT, nm + "_faces.csv"), Fx + 1, delimiter=",",
                   fmt="%d")
        write_msh(os.path.join(OUT, nm + ".msh"), Vx, Fx)

    jobs = []
    for i in range(N_SWEEP):
        for case in ("U", "T"):
            jobs.append((case, PRIMARY, 1000 + i))
            jobs.append((case, SECONDARY, 1000 + i))
    for i in range(N_CAMP):
        jobs.append(("U", PRIMARY, 5000 + i))
    jobs = sorted(set(jobs))
    print(f"correlation lengths: {PRIMARY} = d/{DEN_P:g} (primary, sets the "
          f"ladder), {SECONDARY} = d/{DEN_S:g} (sensitivity)")

    rows = []
    for case, en, sd in jobs:
        f = al.make_field(ELL[en], sd, case, Vref, dead=DEAD, ramp=RAMP)
        b, s = al.sup_jacobian(f, Vref, pad=PAD, ngrid=NGRID, refine=REFINE)
        name = f"{case}_{en}_{sd}.csv"
        np.savetxt(os.path.join(OUT, "fields", name),
                   np.hstack([f.W, f.b[:, None], f.A]), delimiter=",",
                   fmt="%.17g")
        rows.append(dict(case=case, ell_name=en, ell=ELL[en], seed=sd,
                         nfeat=f.M, c=f.c, scale=f.scale,
                         dead=DEAD if case == "T" else 0.0,
                         ramp=RAMP if case == "T" else 0.0,
                         x_tip=al.X_TIP, d=al.D,
                         sup_DV=b, sup_DV_surface=s, file="fields/" + name))
    with open(os.path.join(OUT, "fields_index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    s5 = max(r["sup_DV"] for r in rows if r["ell_name"] == PRIMARY)
    s10 = max(r["sup_DV"] for r in rows if r["ell_name"] == SECONDARY)
    eps_top = np.floor(CERT / s5 * 1e4) / 1e4
    levels = [(i + 1) / 5 * eps_top for i in range(5)]
    with open(os.path.join(OUT, "levels.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "eps_m", "eps_mm", "lambda_over", "eps_over_h12",
                    "pct_seam_half_thickness", "worst_cert_primary",
                    "worst_cert_secondary", "ell_primary", "ell_secondary"])
        for k, e in enumerate(levels, 1):
            w.writerow([k, f"{e:.17g}", f"{1e3*e:.4f}", f"{LAM/e:.3f}",
                        f"{e/H12:.4f}", f"{100*e/al.HALF_THICK_SEAM:.2f}",
                        f"{e*s5:.4f}", f"{e*s10:.4f}", PRIMARY, SECONDARY])
    print(f"\n{len(rows)} fields; worst sup|DV|: {PRIMARY} {s5:.2f}, "
          f"{SECONDARY} {s10:.2f}")
    print(f"certified levels at {SECONDARY}: "
          f"{sum(1 for e in levels if e * s10 <= CERT)} of 5")
    print("levels (mm): " + ", ".join(f"{1e3*e:.2f}" for e in levels))
    print(f"wrote {OUT}/")


if __name__ == "__main__":
    main()
