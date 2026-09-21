import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "../data/sweep"
out = "fig_amplitude_only.pdf"

# panel (a): amplitude sweep  -- unchanged
rows = list(csv.DictReader(open(f"{base}/p10_amplitude_sweep_summary.csv")))
rows.sort(key=lambda r: float(r["amplitude"]))
amp = [100 * float(r["amplitude"]) for r in rows]
pl = [int(r["iters_plain"]) for r in rows]
fr = [int(r["iters_frozen_aligned"]) for r in rows]
fs = [int(r["iters_fresh"]) for r in rows]

# panel (b): bidirectional kappa mismatch, preconditioner frozen at kappa0 = 2.
# Source: p22_kappa_mismatch_bidirectional_LOWK.csv (scripts/p22_...jl --lowk
# --detune).  Abscissa is log2(kappa_solve/kappa_precond) so that the matched
# case sits at zero and the two directions are on opposite sides.
K0 = 2.0
krows = [r for r in csv.DictReader(
    open(f"{base}/p22_kappa_mismatch_bidirectional_LOWK.csv"))
    if r["part"] == "A"]

fresh_by_k, froz_by_k = {}, {}
for r in krows:
    ks = float(r["kappa_solve"])
    if r["strategy"] == "fresh":
        fresh_by_k[ks] = int(r["iters"])
    elif r["strategy"] == "frozen" and float(r["kappa_precond"]) == K0:
        froz_by_k[ks] = int(r["iters"])

ks_all = sorted(set(fresh_by_k) & set(froz_by_k))
d = [math.log2(k / K0) for k in ks_all]
pen = [froz_by_k[k] / fresh_by_k[k] for k in ks_all]
# rows within 5% of an interior Maxwell eigenvalue of the unit ball
gap = [float(next(r["rel_resonance_gap"] for r in krows
                  if float(r["kappa_solve"]) == k)) for k in ks_all]

fig, ax1 = plt.subplots(1, 1, figsize=(10.5, 3.3))

ax1.semilogy(amp, pl, "o-", color="#c62828", label="Plain GMRES")
ax1.semilogy(amp, fr, "s-", color="#1565c0", label="Calderón, frozen (DOF-aligned)")
ax1.semilogy(amp, fs, "^-", color="#2e7d32", label="Calderón, freshly assembled")
ax1.set_xlabel(r"nominal perturbation amplitude $\varepsilon$ (%)", fontsize=15)
ax1.set_ylabel("GMRES iterations (log scale)", fontsize=15)
ax1.set_title(r"Sphere, $\kappa=2$, $h=0.1$", fontsize=15)
ax1.legend(fontsize=13, loc="center right")
ax1.tick_params(labelsize=13)
ax1.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
