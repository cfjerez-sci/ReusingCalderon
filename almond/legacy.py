#!/usr/bin/env python3
"""v1 perturbation model: displacement along the surface normal.

Kept ONLY to document why it was abandoned, and to produce a clean, monotone
replacement for the first (mixed) dry-run table: one envelope, one fixed set
of draws, the same draws rescaled across amplitude levels. It is not used by
the paper's experiments.

    x_i  ->  x_i + a(x_i) f(x_i) n_i,     max |a f| = level

with f a SCALAR squared-exponential random field and a the clearance-
proportional envelope a = min(1, r(x)/r_seam).
"""
import numpy as np
from almond import D, A1Z, THI


def axis_radius(V):
    """Distance from the x axis; vanishes at both apexes."""
    return np.linalg.norm(V[:, 1:], axis=1)


def envelope(V, case, frac=0.10, r_ref=None):
    """a = min(1, r/r_seam), times a C^1 tip taper in case T."""
    if r_ref is None:
        r_ref = A1Z * D
    a = np.minimum(1.0, axis_radius(V) / r_ref)
    if case == "T":
        s = np.clip((D * THI - V[:, 0]) / (frac * D), 0.0, 1.0)
        a = a * (s * s * (3 - 2 * s))
    return a


def scalar_field(V, ell, seed, nfeat=512):
    """Scalar random Fourier features, normalised to max|f| = 1 on V."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0 / ell, size=(nfeat, 3))
    b = rng.uniform(0.0, 2 * np.pi, size=nfeat)
    f = np.sqrt(2.0 / nfeat) * np.cos(V @ W.T + b).sum(1)
    m = np.abs(f).max()
    return f / m if m > 0 else f


def unit_displacement(V, N, ell, seed, case):
    """The normal displacement of unit maximum magnitude, for one draw.

    Scaling THIS by a level gives the whole amplitude sweep for that draw,
    which is what makes the resulting validity table monotone in the level.
    """
    d = envelope(V, case) * scalar_field(V, ell, seed)
    m = np.abs(d).max()
    if m > 0:
        d = d / m
    return d[:, None] * N
