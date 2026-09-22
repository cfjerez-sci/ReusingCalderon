#!/usr/bin/env python3
"""Read the exported field/level files back.

The figure scripts read the SAME files the Julia drivers read, so a picture
cannot drift from the geometry that was actually solved on.
"""
import csv
import os
import numpy as np

import almond as al


def _num(v):
    """levels.csv carries the correlation-length NAMES as well as numbers, so
    a blanket float() breaks on 'd4'. Convert what converts, keep the rest."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def load_levels(exportdir):
    with open(os.path.join(exportdir, "levels.csv")) as fh:
        rows = [{k: int(v) if k == "level" else _num(v) for k, v in r.items()}
                for r in csv.DictReader(fh)]
    if not rows:
        raise SystemExit(f"{exportdir}/levels.csv is empty")
    return rows


def primary_ell(exportdir):
    """The correlation length the ladder was built for, as the drivers name
    it. Figures read it from the data rather than hard-coding 'd5', so a
    re-export at another length does not silently break them."""
    return load_levels(exportdir)[-1].get("ell_primary", "d5")


def load_index(exportdir):
    with open(os.path.join(exportdir, "fields_index.csv")) as fh:
        return list(csv.DictReader(fh))


def load_field(exportdir, case, ell_name, seed):
    """Return (field_fn, meta). field_fn(X) -> (n,3) displacement per unit eps."""
    rows = [r for r in load_index(exportdir)
            if r["case"] == case and r["ell_name"] == ell_name
            and int(r["seed"]) == seed]
    if not rows:
        raise KeyError(f"no field {case}/{ell_name}/{seed}")
    m = rows[0]
    P = np.loadtxt(os.path.join(exportdir, m["file"]), delimiter=",")
    W, b, A = P[:, :3], P[:, 3], P[:, 4:7]
    c, scale = float(m["c"]), float(m["scale"])
    dead, ramp = float(m["dead"]), float(m["ramp"])
    x_tip, d = float(m["x_tip"]), float(m["d"])

    def field(X):
        v = c * (np.cos(X @ W.T + b) @ A)
        if ramp > 0:
            s = np.clip((x_tip - dead * d - X[:, 0]) / (ramp * d), 0, 1)
            v = v * (s * s * (3 - 2 * s))[:, None]
        return scale * v

    return field, m
