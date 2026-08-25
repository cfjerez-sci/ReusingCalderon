import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

base = "../data/figexport/meshes"

rows = [
    ("Sphere", "sphere_nominal", "sphere_perturbed50pct", "50%"),
    ("Ellipsoid", "ellipsoid_nominal", "ellipsoid_perturbed30pct", "30%"),
    ("Fichera corner", "fichera_nominal", "fichera_perturbed40pct", "40%"),
]

fig = plt.figure(figsize=(7.0, 10.2))

def load(name):
    V = np.loadtxt(f"{base}/{name}_verts.csv", delimiter=",")
    F = np.loadtxt(f"{base}/{name}_faces.csv", delimiter=",").astype(int) - 1
    return V, F

def plot_mesh(ax, V, F, color, elev, azim):
    tris = V[F]
    coll = Poly3DCollection(tris, facecolor=color, edgecolor="black",
                             linewidths=0.15, alpha=1.0)
    ax.add_collection3d(coll)
    mins = V.min(axis=0)
    maxs = V.max(axis=0)
    ctr = (mins + maxs) / 2
    r = (maxs - mins).max() / 2 * 1.05
    ax.set_xlim(ctr[0]-r, ctr[0]+r)
    ax.set_ylim(ctr[1]-r, ctr[1]+r)
    ax.set_zlim(ctr[2]-r, ctr[2]+r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

views = [(20, -60), (20, -60), (20, -135)]  # Fichera: view from (-x,-y) octant to show the two bumped faces (x=-1, y=-1)

for i, (label, nom, pert, amp) in enumerate(rows):
    Vn, Fn = load(nom)
    Vp, Fp = load(pert)
    elev, azim = views[i]

    ax1 = fig.add_subplot(3, 2, 2*i+1, projection="3d")
    plot_mesh(ax1, Vn, Fn, "#a6c8e0", elev, azim)
    ax1.set_title(f"{label}: nominal", fontsize=11)

    ax2 = fig.add_subplot(3, 2, 2*i+2, projection="3d")
    plot_mesh(ax2, Vp, Fp, "#e0a6a6", elev, azim)
    ax2.set_title(f"{label}: perturbed ({amp})", fontsize=11)

plt.tight_layout()
plt.savefig("/sessions/nice-epic-fermat/mnt/outputs/fig_mesh_comparison.pdf", dpi=200, bbox_inches="tight")
print("saved fig_mesh_comparison.pdf")
