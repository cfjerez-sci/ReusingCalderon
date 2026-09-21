# [read-3] Two-by-three relayout of the mesh comparison figure: nominal on
# the top row, perturbed below, one column per geometry. The original 3x2
# version (make_fig_meshes.py) has aspect ratio 10.2/7.0 = 1.46 and at
# \textwidth therefore fills a page on its own; this one is 7.0/10.5 = 0.67,
# a little under half the height, with the same data, views and colours.
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

base = "../data/figexport/meshes"
out  = sys.argv[1] if len(sys.argv) > 1 else "fig_mesh_comparison.pdf"

cols = [
    ("Sphere",         "sphere_nominal",    "sphere_perturbed50pct",    "50\\%", (20,  -60), 1.38),
    ("Ellipsoid",      "ellipsoid_nominal", "ellipsoid_perturbed30pct", "30\\%", (20,  -60), 1.50),
    ("Fichera corner", "fichera_nominal",   "fichera_perturbed40pct",   "40\\%", (20, -135), 1.05),
]

def load(name):
    V = np.loadtxt(f"{base}/{name}_verts.csv", delimiter=",")
    F = np.loadtxt(f"{base}/{name}_faces.csv", delimiter=",").astype(int) - 1
    return V, F

def plot_mesh(ax, V, F, color, elev, azim, zoom=1.38):
    ax.add_collection3d(Poly3DCollection(V[F], facecolor=color,
                                         edgecolor="black", linewidths=0.15))
    mins, maxs = V.min(axis=0), V.max(axis=0)
    ctr = (mins + maxs) / 2
    r = (maxs - mins).max() / 2 * 1.02
    ax.set_xlim(ctr[0]-r, ctr[0]+r)
    ax.set_ylim(ctr[1]-r, ctr[1]+r)
    ax.set_zlim(ctr[2]-r, ctr[2]+r)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

fig = plt.figure(figsize=(10.5, 6.0))
for j, (label, nom, pert, amp, (elev, azim), zm) in enumerate(cols):
    Vn, Fn = load(nom)
    Vp, Fp = load(pert)
    ax = fig.add_subplot(2, 3, j + 1, projection="3d")
    plot_mesh(ax, Vn, Fn, "#a6c8e0", elev, azim, zm)
    ax.set_title(f"{label}: nominal", fontsize=11, pad=-2)
    ax = fig.add_subplot(2, 3, j + 4, projection="3d")
    plot_mesh(ax, Vp, Fp, "#e0a6a6", elev, azim, zm)
    ax.set_title(f"perturbed ({amp.replace(chr(92)+'%','%')})", fontsize=11, pad=-2)

fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.01,
                    wspace=-0.06, hspace=0.12)
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
