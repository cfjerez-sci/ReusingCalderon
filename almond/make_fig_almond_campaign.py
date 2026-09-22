#!/usr/bin/env python3
"""Figure C: the Monte Carlo campaign.

(a) The distribution of frozen iteration counts over the 100 shape samples,
    with the nominal count and the fresh subset marked. The claim the figure
    supports is about the whole distribution, not a best case: what matters
    is where the upper tail sits.
(b) The bistatic radar cross-section: the nominal cut, and the sample
    envelope (min-max band and mean) over the campaign. Per Remark 7.3 this
    is exhibited, not bounded: the theorem is about the solver.

    python make_fig_almond_campaign.py <summary.csv> <rcs.csv> [out.pdf]
"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUM = sys.argv[1] if len(sys.argv) > 1 else "p33_almond_campaign_summary.csv"
RCS = sys.argv[2] if len(sys.argv) > 2 else "p33_almond_campaign_rcs.csv"
OUT = sys.argv[3] if len(sys.argv) > 3 else "fig_almond_campaign.pdf"
FROZ, FRESH, NOM = "#1565C0", "#2E7D32", "#616161"


def main():
    with open(SUM) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{SUM} is empty -- run p33 first")
    it_f = np.array([int(r["iters_frozen"]) for r in rows])
    has = [r for r in rows if r["has_fresh"].strip().lower() == "true"]
    it_g = np.array([int(r["iters_fresh"]) for r in has]) if has else None

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.8, 3.5))

    bins = np.arange(it_f.min() - 1.5, it_f.max() + 2.5, 1.0)
    a.hist(it_f, bins=bins, color=FROZ, alpha=0.75, label=f"frozen, n = {len(it_f)}")
    if it_g is not None and len(it_g):
        a.hist(it_g, bins=bins, color=FRESH, alpha=0.55,
               label=f"fresh, n = {len(it_g)}")
    a.set_xlabel("GMRES iterations")
    a.set_ylabel("samples")
    a.grid(alpha=0.25, lw=0.5, axis="y")
    a.legend(fontsize=8, frameon=False)
    a.set_title(f"median {np.median(it_f):.0f}, max {it_f.max()}", fontsize=9.5)

    th, curves, nominal = None, [], None
    with open(RCS) as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        th = np.degrees(np.array([float(x) for x in hdr[2:]]))
        for r in rd:
            y = np.array([float(x) for x in r[2:]])
            if r[1] == "nominal":
                nominal = y
            else:
                curves.append(y)
    if curves:
        Cv = np.vstack(curves)
        db = lambda y: 10 * np.log10(np.maximum(y, 1e-12))
        b.fill_between(th, db(Cv.min(0)), db(Cv.max(0)), color=FROZ,
                       alpha=0.22, lw=0, label=f"sample range, n = {len(Cv)}")
        b.plot(th, db(Cv.mean(0)), color=FROZ, lw=1.4, label="sample mean")
    if nominal is not None:
        b.plot(th, 10 * np.log10(np.maximum(nominal, 1e-12)), color=NOM,
               lw=1.2, ls="--", label="nominal")
    b.set_xlabel(r"scattering angle $\theta$ (deg)")
    b.set_ylabel(r"$\sigma$ (dBsm)")
    b.set_xlim(0, 180)
    b.grid(alpha=0.25, lw=0.5)
    b.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    plt.savefig(OUT, dpi=220, bbox_inches="tight")
    print("saved", OUT)
    print(f"  frozen iterations: min {it_f.min()}, median "
          f"{np.median(it_f):.1f}, max {it_f.max()}")
    if it_g is not None and len(it_g):
        sub = it_f[:len(it_g)]
        print(f"  on the fresh subset: frozen/fresh ratio "
              f"{np.min(sub/it_g):.3f}-{np.max(sub/it_g):.3f}")


if __name__ == "__main__":
    main()
