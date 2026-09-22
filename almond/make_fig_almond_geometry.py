#!/usr/bin/env python3
"""Figure: the NASA almond, nominal and heavily perturbed.

Two panels, one scale, one viewpoint. Palette and line weights follow
figures/fig_mesh_comparison_2x3.pdf, the figure this one sits near: a salmon
surface with black element edges, over a blue-grey wireframe of the nominal
mesh so the reader can see what moved.

WHICH DRAW. Every exported field is normalised to the same maximum
displacement, so what distinguishes one draw from another is how much of the
body moves, not how far the worst point moves. Seed 5012 is chosen because it
has both a high mean displacement (7.65 mm of a possible 11.0) and a high
spread (2.13 mm): most of the body is displaced, and unevenly, which is what
a heavy perturbation looks like. A draw with a high mean and a low spread
merely rescales the almond and looks like nothing at all. The seed is named
in the caption; it is an ordinary member of the campaign, not a hand-built
illustration.

FRAMING. The almond is 252 x 97 x 32 mm. An equal-aspect cube wastes three
quarters of the panel, so each axis is scaled by its own extent. And a 3D
axes centres its content in a SQUARE region, so a wide flat body leaves a
tall empty band that shrinking the figure cannot remove -- shrinking scales
the body down with it. Each axes is therefore given a box taller than the
figure and hung off both edges, so what shows is the middle band where the
body is. Titles are figure text for the same reason.

    python make_fig_almond_geometry.py [exportdir] [out.pdf] [seed]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import almond as al
from export_io import load_field, load_levels, primary_ell

EXPORT = sys.argv[1] if len(sys.argv) > 1 else "export"
OUT = sys.argv[2] if len(sys.argv) > 2 else "fig_almond_geometry.pdf"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 5012
LAM = al.D / 3.0

NOM = "#8aa8bf"          # nominal wireframe, as in fig_mesh_comparison_2x3
PER = "#e0a6a6"          # perturbed surface, ditto
PLAIN = "#c6d2dc"        # the nominal surface in panel (a)

ELEV, AZIM = 30, -62
# drawn at ~2.5x the size it prints, so type and line weights are set 2.5x
# larger than the values wanted on the page
TITLE_PT, EDGE_LW, WIRE_LW = 22, 0.28, 0.7


def edges_of(F):
    e = set()
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            e.add((min(a, b), max(a, b)))
    return np.array(sorted(e))


def frame(ax, V, zoom=1.0):
    lo, hi = V.min(0), V.max(0)
    ctr, ext = (lo + hi) / 2, np.maximum(hi - lo, 1e-9)
    for setlim, c, e in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), ctr, ext):
        setlim(c - 0.52 * e, c + 0.52 * e)
    ax.set_box_aspect(tuple(ext / ext.max()), zoom=zoom)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()


def main():
    V, F = al.mesh(LAM / 12)
    F, _ = al.orient_consistently(V, F)
    E = edges_of(F)
    eps = load_levels(EXPORT)[-1]["eps_m"]
    ell = primary_ell(EXPORT)
    fld, _ = load_field(EXPORT, "U", ell, SEED)
    W = V + eps * fld(V)
    BOX = np.vstack([V, W])          # one scale for both panels

    fig = plt.figure(figsize=(11.4, 2.52))
    for k, (title, X, wireframe) in enumerate(
            [("(a) nominal", V, False), ("(b) perturbed", W, True)]):
        ax = fig.add_axes([0.005 + k * 0.498, -0.575, 0.492, 2.141],
                          projection="3d")
        if wireframe:
            ax.add_collection3d(Line3DCollection(V[E], colors=NOM,
                                                 linewidths=WIRE_LW))
        ax.add_collection3d(Poly3DCollection(
            X[F], facecolor=PER if wireframe else PLAIN, edgecolor="black",
            linewidths=EDGE_LW))
        frame(ax, BOX, zoom=1.19)
        fig.text(0.251 + k * 0.498, 0.99, title, fontsize=TITLE_PT,
                 ha="center", va="top")

    plt.savefig(OUT, dpi=240, pad_inches=0.0)
    d = np.linalg.norm(W - V, axis=1)
    i = int(np.argmax(V[:, 0]))
    print(f"ell {ell}, seed {SEED}, eps = {1e3*eps:.2f} mm: displacement max "
          f"{1e3*d.max():.2f} mm, mean {1e3*d.mean():.2f} mm, "
          f"tip {1e3*d[i]:.2f} mm")
    print("saved", OUT)


if __name__ == "__main__":
    main()
