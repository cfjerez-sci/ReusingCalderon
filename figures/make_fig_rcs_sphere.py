# [read-3] Single-panel RCS figure. The ellipsoid panel repeated the
# sphere's story and the Fichera curves barely separated at the tested
# amplitudes; the sphere alone carries the claim that the perturbation is
# large enough to change the far field. Drawn wide and short, with fonts
# sized for the page rather than the canvas: at width=\textwidth every
# label is scaled by roughly 0.48 on the way in.
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "../data/figexport/rcs"
out  = sys.argv[1] if len(sys.argv) > 1 else "fig_rcs_sphere.pdf"

data = np.loadtxt(f"{base}/sphere_rcs.csv", delimiter=",")
theta = np.degrees(data[:, 0])
nom  = 10 * np.log10(np.maximum(data[:, 1], 1e-16))
pert = 10 * np.log10(np.maximum(data[:, 2], 1e-16))

fig, ax = plt.subplots(1, 1, figsize=(11.5, 2.6))
ax.plot(theta, nom,  color="#2166ac", lw=2.2, label="nominal")
ax.plot(theta, pert, color="#b2182b", lw=2.2, ls="--",
        label="perturbed (50\\%), frozen preconditioner".replace("\\", ""))
ax.set_xlabel(r"$\theta$ (deg)", fontsize=15)
ax.set_ylabel("Bistatic RCS (dB)", fontsize=15)
ax.set_xlim(0, 180)
ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
ax.tick_params(labelsize=13)
ax.grid(alpha=0.3)
ax.legend(fontsize=14, loc="upper right", ncol=1, framealpha=0.95)
plt.tight_layout()
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
