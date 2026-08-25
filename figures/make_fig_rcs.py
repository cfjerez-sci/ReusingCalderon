import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "../data/figexport/rcs"

panels = [
    ("sphere_rcs.csv", "Sphere", "50%"),
    ("ellipsoid_rcs.csv", "Ellipsoid", "30%"),
    ("fichera_rcs_40pct.csv", "Fichera corner", "40%"),
]

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

for ax, (fname, label, amp) in zip(axes, panels):
    data = np.loadtxt(f"{base}/{fname}", delimiter=",")
    theta, rcs_nom, rcs_pert = data[:, 0], data[:, 1], data[:, 2]
    theta_deg = np.degrees(theta)
    rcs_nom_db = 10 * np.log10(np.maximum(rcs_nom, 1e-16))
    rcs_pert_db = 10 * np.log10(np.maximum(rcs_pert, 1e-16))

    ax.plot(theta_deg, rcs_nom_db, color="#2166ac", lw=1.6, label="nominal")
    ax.plot(theta_deg, rcs_pert_db, color="#b2182b", lw=1.6, ls="--",
            label=f"perturbed ({amp}), frozen precond.")
    ax.set_title(label, fontsize=11)
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_xlim(0, 180)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Bistatic RCS (dB)")
axes[0].legend(fontsize=8, loc="lower center")

plt.tight_layout()
plt.savefig("/sessions/nice-epic-fermat/mnt/outputs/fig_rcs_comparison.pdf", dpi=200, bbox_inches="tight")
print("saved fig_rcs_comparison.pdf")
