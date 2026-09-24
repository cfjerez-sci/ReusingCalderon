#!/usr/bin/env python3
"""Figure B: the amplitude sweep.

(a) GMRES iteration counts against the injectivity screening quantity
    eps*sup|DV|,
    which is the x-axis Carlos asked for: it is the quantity the perturbation
    theory is written in, it varies sample to sample at a fixed level, and it
    is dimensionless. The realised maximum displacement -- the quantity a
    reader pictures -- runs along the secondary top axis.
(b) The frozen/fresh iteration ratio against the same abscissa, cases U and T
    separately. Ratio, not gap: the gap grows with the iteration count and
    says less.

Reads data/sweep/p32_almond_sweep_summary.csv, written by p32.

    python make_fig_almond_sweep.py <summary.csv> [out.pdf]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "p32_almond_sweep_summary.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "fig_almond_sweep.pdf"
C = {"plain": "#616161", "frozen": "#1565C0", "fresh": "#2E7D32"}
# Strategy -> marker shape, case -> fill (filled = U, open = T). Colour is a
# redundant third channel: SIAM prints in black and white, so identity must
# never rest on hue alone.
MK = {"plain": "^", "frozen": "o", "fresh": "s"}


def load(path):
    import csv as _csv
    with open(path) as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty -- run p32 first")
    for r in rows:
        for k in ("cert", "maxdisp_m", "eps_m", "sup_DV"):
            r[k] = float(r[k])
        for k in ("iters_plain", "iters_frozen", "iters_fresh", "level"):
            r[k] = int(r[k])
        r["flag"] = r.get("resonance_flag", "false").strip().lower() == "true"
        # p32 --nfresh leaves iters_fresh = -1 where no fresh solve was run.
        # That sentinel must never reach a ratio: it is absence, not a count
        # of one.
        r["has_fresh"] = r["iters_fresh"] > 0

    # The summary file is APPENDED to by every p32 invocation, so it also
    # holds the correlation-length sensitivity (ell = d/10) and the refined
    # mesh check (lambda/20). Figure B is the amplitude sweep on the primary
    # mesh at the primary correlation length; mixing in a mesh whose plain
    # count is 50% higher puts a false jump in the level-5 column. The
    # primary of each is the one with the most rows, as in
    # make_almond_tables.py -- never a hard-coded name.
    def commonest(key):
        n = {}
        for r in rows:
            n[r[key]] = n.get(r[key], 0) + 1
        return max(n, key=n.get)

    mesh, ell = commonest("mesh"), commonest("ell")
    keep = [r for r in rows if r["mesh"] == mesh and r["ell"] == ell]
    if len(keep) < len(rows):
        print(f"  [fig B] {len(keep)} of {len(rows)} rows: mesh {mesh}, "
              f"ell {ell} (dropped the sensitivity and refined-mesh runs)")
    return keep


def main():
    rows = load(CSV)
    lam = None
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.5))

    for case in sorted({r["case"] for r in rows}):
        s = [r for r in rows if r["case"] == case]
        x = np.array([r["cert"] for r in s])
        for key, lab in (("iters_plain", "plain"), ("iters_frozen", "frozen"),
                         ("iters_fresh", "fresh")):
            sel = [r for r in s if r["has_fresh"]] if lab == "fresh" else s
            if not sel:
                continue
            a.scatter([r["cert"] for r in sel], [r[key] for r in sel], s=19,
                      marker=MK[lab],
                      facecolor=C[lab] if case == "U" else "none",
                      edgecolor=C[lab], linewidths=0.9,
                      label=f"{lab}, case {case}")
        both = [r for r in s if r["has_fresh"]]
        if both:
            b.scatter([r["cert"] for r in both],
                      [r["iters_frozen"] / r["iters_fresh"] for r in both],
                      s=20, marker="D",
                      facecolor="#1565C0" if case == "U" else "none",
                      edgecolor="#1565C0", linewidths=0.9,
                      label=f"case {case} ({'filled' if case == 'U' else 'open'})")
        fl = [r for r in s if r["flag"]]
        if fl:
            a.scatter([r["cert"] for r in fl], [r["iters_frozen"] for r in fl],
                      s=85, facecolor="none", edgecolor="#C62828",
                      linewidths=1.2, zorder=5,
                      label="resonance candidate" if case == "U" else None)

    a.set_xlabel(r"$\varepsilon\,\sup|\mathrm{D}V|$")
    a.set_ylabel("GMRES iterations")
    a.set_yscale("log")
    a.grid(alpha=0.25, lw=0.5)
    a.legend(fontsize=7.0, ncol=2, frameon=False)

    # secondary axis: displacement. cert and displacement are proportional
    # only within a sample, so the top axis is labelled by LEVEL, using each
    # level's nominal eps -- which is exact, and honest about the scatter.
    lv = {}
    for r in rows:
        lv.setdefault(r["level"], r["eps_m"])
    xs, lbl = [], []
    for level in sorted(lv):
        s = [r["cert"] for r in rows if r["level"] == level]
        xs.append(np.mean(s))
        lbl.append(f"{1e3*lv[level]:.1f}")
    at = a.twiny()
    at.set_xlim(a.get_xlim())
    at.set_xticks(xs)
    at.set_xticklabels(lbl, fontsize=8)
    at.set_xlabel("level mean; max displacement (mm)", fontsize=9)

    b.axhline(1.0, color="0.6", lw=0.8, ls="--")
    b.set_xlabel(r"$\varepsilon\,\sup|\mathrm{D}V|$")
    b.set_ylabel("frozen / fresh iterations")
    b.grid(alpha=0.25, lw=0.5)
    b.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    plt.savefig(OUT, dpi=220, bbox_inches="tight")
    print("saved", OUT)
    for level in sorted(lv):
        s = [r for r in rows if r["level"] == level]
        both = [r for r in s if r["has_fresh"]]
        ratio = [r["iters_frozen"] / r["iters_fresh"] for r in both]
        txt = (f"frozen/fresh {min(ratio):.3f}-{max(ratio):.3f} "
               f"(median {np.median(ratio):.3f}) over {len(both)}"
               if ratio else "no fresh solves")
        print(f"  level {level} ({1e3*lv[level]:5.2f} mm): n = {len(s):3d}  "
              + txt)


if __name__ == "__main__":
    main()
