import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "../data/sweep"

# panel (a): amplitude sweep
rows = list(csv.DictReader(open(f"{base}/p10_amplitude_sweep_summary.csv")))
rows.sort(key=lambda r: float(r["amplitude"]))
amp = [100 * float(r["amplitude"]) for r in rows]
pl = [int(r["iters_plain"]) for r in rows]
fr = [int(r["iters_frozen_aligned"]) for r in rows]
fs = [int(r["iters_fresh"]) for r in rows]

# panel (b): kappa mismatch
krows = list(csv.DictReader(open(f"{base}/p10_kappa_mismatch_precond_summary.csv")))
krows.sort(key=lambda r: float(r["kappa_solve"]))
ratio = [float(r["kappa_solve"]) / float(r["kappa_precond"]) for r in krows]
pen = [int(r["iters_mismatched"]) / int(r["iters_fresh"]) for r in krows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))

ax1.semilogy(amp, pl, "o-", color="#c62828", label="Plain GMRES")
ax1.semilogy(amp, fr, "s-", color="#1565c0", label="Calderón, frozen (DOF-aligned)")
ax1.semilogy(amp, fs, "^-", color="#2e7d32", label="Calderón, freshly assembled")
ax1.set_xlabel(r"nominal perturbation amplitude $\varepsilon$ (%)")
ax1.set_ylabel("GMRES iterations (log scale)")
ax1.set_title(r"(a) Sphere, $\kappa=2$, $h=0.1$", fontsize=10)
ax1.legend(fontsize=7.5, loc="center right")
ax1.grid(alpha=0.3, which="both")

ax2.plot(ratio, pen, "D-", color="#6a4fa3")
for x, y in zip(ratio, pen):
    ax2.annotate(f"{y:.1f}×", (x, y), textcoords="offset points",
                 xytext=(0, 7), ha="center", fontsize=8)
ax2.set_xscale("log", base=2)
ax2.set_xticks(ratio)
ax2.set_xticklabels([f"{r:g}" for r in ratio])
ax2.set_xlabel(r"$\kappa_{\rm solve}/\kappa_{\rm precond}$")
ax2.set_ylabel("iters(mismatched)/iters(fresh)")
ax2.set_title("(b) Wavenumber-mismatch penalty", fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/sessions/nice-epic-fermat/mnt/outputs/fig_amplitude_and_mismatch.pdf",
            dpi=200, bbox_inches="tight")
print("saved")
