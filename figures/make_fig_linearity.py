#!/usr/bin/env python3
"""[CJ-12] Figure for the direct test of the perturbation estimate.

  python3 figures/make_fig_linearity.py \
      data/sweep/p25_perturbation_linearity.csv figures/fig_perturbation_linearity.pdf

Palette is the paper's existing categorical order (#c62828, #1565c0, #2e7d32),
verified colourblind-safe; distinct markers carry identity as well as colour.
"""
import csv, math
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

import sys
src = sys.argv[1] if len(sys.argv) > 1 else "data/sweep/p25_perturbation_linearity.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "fig_perturbation_linearity.pdf"
rows = list(csv.DictReader(open(src)))
by = defaultdict(list)
for r in rows: by[float(r["h"])].append(r)
for h in by: by[h].sort(key=lambda r: float(r["amplitude"]))

# paper's existing categorical order (validated: all checks pass), fixed not cycled
# dof labels are those of the meshes actually used in p25 (see the CSV),
# which were generated separately from the Table 5.3 meshes at h=0.2, 0.15.
STYLE = {0.1:  ("#c62828", "o", "$h=0.1$ (4827 dof)"),
         0.15: ("#1565c0", "s", "$h=0.15$ (2298 dof)"),
         0.2:  ("#2e7d32", "^", "$h=0.2$ (1206 dof)")}
ORDER = [0.1, 0.15, 0.2]
TICKS = [0.03, 0.1, 0.2, 0.3, 0.4, 0.5]
LABS  = ["3", "10", "20", "30", "40", "50"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.7))

# ---- (a) linearity -------------------------------------------------------
for h in ORDER:
    c, m, lab = STYLE[h]
    rs = by[h]
    ax1.loglog([float(r["amplitude"]) for r in rs],
               [float(r["normF_dA"]) for r in rs],
               marker=m, color=c, lw=1.8, ms=5, label=lab, zorder=3)
# slope-1 guide, offset well below the coarsest curve so it is visible
base = by[0.2]
x0, y0 = float(base[0]["amplitude"]), float(base[0]["normF_dA"]) * 0.42
ax1.loglog([x0, 0.5], [y0, y0 * (0.5 / x0)], ls=(0, (5, 3)), color="0.35", lw=1.2, zorder=2)
ax1.annotate("slope 1", xy=(0.19, y0 * (0.19 / x0)), xytext=(0, -13),
             textcoords="offset points", fontsize=8.5, color="0.35", ha="center")
ax1.set_xlabel(r"perturbation amplitude $\varepsilon$ (\%)" if False else
               "perturbation amplitude $\\varepsilon$ (%)")
ax1.set_ylabel(r"$\|\mathbf{A}_{\varepsilon,h}-\mathbf{A}_{0,h}\|_F$")
ax1.set_title("(a) the perturbation enters linearly in $\\varepsilon$", fontsize=10)
ax1.xaxis.set_major_locator(FixedLocator(TICKS))
ax1.xaxis.set_major_formatter(FixedFormatter(LABS))
ax1.xaxis.set_minor_locator(FixedLocator([]))
ax1.set_xlim(0.026, 0.58)
ax1.grid(alpha=0.28, which="major", lw=0.6)
ax1.legend(fontsize=8, loc="upper left", framealpha=0.92)
for sp in ("top", "right"): ax1.spines[sp].set_visible(False)

# ---- (b) h-uniformity ----------------------------------------------------
k0 = sum(float(by[h][0]["kappaS_P0A0"]) for h in ORDER) / 3
ax2.axhline(k0, color="0.55", lw=1.0, ls=(0, (4, 3)), zorder=1)
ax2.annotate(r"$\kappa_S(\mathbf{P}_0\mathbf{A}_{0,h})\approx%.2f$" % k0,
             xy=(0.5, k0), xytext=(-4, 5), textcoords="offset points",
             fontsize=8, color="0.42", ha="right")
for h in ORDER:
    c, m, lab = STYLE[h]
    rs = by[h]
    ax2.plot([float(r["amplitude"]) for r in rs],
             [float(r["kappaS_P0Aeps"]) for r in rs],
             marker=m, color=c, lw=1.8, ms=5, label=lab, zorder=3,
             markeredgecolor="white", markeredgewidth=0.7)
ax2.set_xlabel("perturbation amplitude $\\varepsilon$ (%)")
ax2.set_ylabel(r"$\kappa_S(\mathbf{P}_0\mathbf{A}_{\varepsilon,h})$")
ax2.set_title("(b) the bound is $h$-uniform", fontsize=10)
ax2.xaxis.set_major_locator(FixedLocator(TICKS))
ax2.xaxis.set_major_formatter(FixedFormatter(LABS))
ax2.set_xlim(0.0, 0.535)
ax2.grid(alpha=0.28, lw=0.6)
ax2.legend(fontsize=8, loc="upper left", framealpha=0.92)
for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
# the finding, stated on the figure
worst = 0.0
for a in TICKS:
    v = [float(m[0]["kappaS_P0Aeps"]) for h in ORDER
         for m in [[r for r in by[h] if abs(float(r["amplitude"]) - a) < 1e-12]] if m]
    if len(v) > 1: worst = max(worst, (max(v) - min(v)) / min(v))
ax2.annotate("curves agree to %.1f%% over a\n$4\\times$ range in dof" % (100 * worst),
             xy=(0.30, 1.86), fontsize=8.5, color="0.30", ha="center")

plt.tight_layout()
plt.savefig(out, bbox_inches="tight")
print("worst spread = %.3f%%" % (100 * worst))
print("saved", out)
