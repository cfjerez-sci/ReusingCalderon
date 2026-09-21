#!/usr/bin/env python3
"""
Analysis and figures for the bidirectional wavenumber-mismatch experiment
(Parts A-D) of "One Calderon Operator to Rule Them All".

Consumes the raw CSV written by run_kappa_mismatch.jl and emits:

  Part A  kappa_mismatch_partA.tex        LaTeX table (manuscript style)
  Part B  kappa_mismatch_partB.txt        directional-asymmetry analysis
  Part C  fig_kappa_mismatch_bidir.pdf    penalty vs log2(kappa/kappa0)
          fig_kappa_mismatch_counts.pdf   absolute iteration counts
  Part D  kappa_mismatch_partD.tex        R_ij and raw frozen-count matrices
          fig_kappa_mismatch_heatmap.pdf  heatmap of R_ij
  Part E  kappa_mismatch_qc.txt           quality-control report

Raw CSV schema (one row per linear solve; see run_kappa_mismatch.jl):

    part              "A" or "D"
    kappa_precond     wavenumber at which P was assembled (NaN for plain)
    kappa_solve       wavenumber of the EFIE being solved
    strategy          "plain" | "frozen" | "fresh"
    iters             outer GMRES iteration count
    converged         true/false
    true_res          ||b - Z u|| / ||b||   (unpreconditioned)
    solve_time_s      wall-clock of the solve only
    sol_l2            ||u||_2, for cross-strategy solution comparison
    rel_diff_vs_fresh ||u - u_fresh|| / ||u_fresh||  (frozen rows; NaN otherwise)
    dof, h, geometry, seed, tol_outer, tol_inner, maxit, notes

Usage
-----
    python3 analyze_kappa_mismatch.py raw_kappa_mismatch.csv --outdir .

Every number in the outputs comes from the CSV. Nothing is filled in or
interpolated; missing cells are reported as missing.
"""

import argparse
import os
import sys
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# ---------------------------------------------------------------------------
# House style, matched to figures/fig_amplitude_and_mismatch.pdf
# ---------------------------------------------------------------------------
STYLE = {
    "plain":  dict(color="#d62728", marker="o", label="Plain GMRES"),
    "frozen": dict(color="#1f77b4", marker="s",
                   label="Calder\u00f3n, frozen at $\\kappa_0$"),
    "fresh":  dict(color="#2ca02c", marker="^",
                   label="Calderón, freshly assembled"),
}
PENALTY_COLOR = "#6a51a3"          # the purple of the existing panel (b)
SEQ_CMAP = "Blues"                 # sequential, one hue, light -> dark

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

# Interior Maxwell eigenvalues of the unit ball, kappa <= 10 (see
# resonance_check.py; the spectrum begins at 2.743707).
UNIT_BALL_EIG = np.array([
    2.743707, 3.870239, 4.493409, 4.973420, 5.763459, 6.061949, 6.116764,
    6.987932, 7.140227, 7.443087, 7.725252, 8.182561, 8.210842, 8.721751,
    9.095011, 9.275463, 9.316616, 9.355812, 9.967547,
])


def rel_resonance_gap(kappa):
    """|kappa - nearest interior eigenvalue| / kappa for the unit ball."""
    if not np.isfinite(kappa) or kappa <= 0:
        return np.nan
    return float(np.min(np.abs(kappa - UNIT_BALL_EIG)) / kappa)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

REQUIRED = ["part", "kappa_precond", "kappa_solve", "strategy", "iters",
            "converged", "true_res"]


def load(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit("ERROR: raw CSV is missing required columns: %s" % missing)
    for c in ("kappa_precond", "kappa_solve", "true_res", "iters"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df["converged"].dtype == object:
        df["converged"] = (df["converged"].astype(str).str.strip().str.lower()
                           .isin(["true", "t", "1", "yes"]))
    df["strategy"] = df["strategy"].astype(str).str.strip().str.lower()
    return df


def pick(df, **kw):
    """Single matching row, or None. Errors on ambiguity rather than guessing."""
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        if isinstance(v, float) and np.isfinite(v):
            m &= np.isclose(df[k].astype(float), v, rtol=1e-9, atol=1e-12)
        else:
            m &= (df[k] == v)
    sub = df[m]
    if len(sub) == 0:
        return None
    if len(sub) > 1:
        sys.exit("ERROR: %d duplicate rows for %s -- de-duplicate the raw CSV "
                 "rather than letting the analysis choose." % (len(sub), kw))
    return sub.iloc[0]


def fmt(x, spec="%.1e", missing="--"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return missing
    return spec % x


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------

def part_a(df, k0, outdir):
    a = df[df["part"] == "A"]
    ks = sorted(a["kappa_solve"].dropna().unique())
    rows = []
    for k in ks:
        pl = pick(a, kappa_solve=k, strategy="plain")
        fr = pick(a, kappa_solve=k, strategy="frozen", kappa_precond=k0)
        fe = pick(a, kappa_solve=k, strategy="fresh")
        R = (fr["iters"] / fe["iters"]) if (fr is not None and fe is not None
                                            and fe["iters"] > 0) else np.nan
        rows.append(dict(
            kappa=k, ratio=k / k0, d=math.log2(k / k0),
            plain_it=None if pl is None else pl["iters"],
            frozen_it=None if fr is None else fr["iters"],
            fresh_it=None if fe is None else fe["iters"],
            penalty=R,
            plain_res=None if pl is None else pl["true_res"],
            frozen_res=None if fr is None else fr["true_res"],
            fresh_res=None if fe is None else fe["true_res"],
            allconv=all(r is not None and bool(r["converged"])
                        for r in (pl, fr, fe)),
            gap=rel_resonance_gap(k),
        ))
    tab = pd.DataFrame(rows)

    lines = [
        r"\begin{table}[htbp]", r"  \centering", r"  \scriptsize",
        r"  \caption{Bidirectional wavenumber mismatch, nominal sphere,",
        r"    $h=0.1$, preconditioner frozen at $\kappa_0=%g$. Penalty is" % k0,
        r"    $\mathrm{iters}_{\mathrm{frozen}}/\mathrm{iters}_{\mathrm{fresh}}$."
        r" Last three columns:",
        r"    converged true relative residual $\|\bs b-\bs Z\bs u\|/\|\bs b\|$.}",
        r"  \label{tab:kappamismatch-bidir}",
        r"  \begin{tabular}{@{}rrrrrrrrr@{}}", r"    \toprule",
        r"    & & & & & & \multicolumn{3}{c}{True residual} \\",
        r"    $\kappa_{\mathrm{solve}}$ & $\kappa/\kappa_0$ & "
        r"$\log_2(\kappa/\kappa_0)$ & Plain it. & Frozen it. & Fresh it. "
        r"& plain & frozen & fresh \\",
        r"    \midrule",
    ]
    for _, r in tab.iterrows():
        flag = r"$^{\dagger}$" if (np.isfinite(r["gap"]) and r["gap"] < 0.05) else ""
        lines.append(
            "    %g%s & %s & %+g & %s & %s & %s & %s & %s & %s \\\\" % (
                r["kappa"], flag,
                ("1/%g" % round(1 / r["ratio"])) if r["ratio"] < 1
                else "%g" % r["ratio"],
                r["d"],
                fmt(r["plain_it"], "%d"), fmt(r["frozen_it"], "%d"),
                fmt(r["fresh_it"], "%d"),
                fmt(r["plain_res"]), fmt(r["frozen_res"]), fmt(r["fresh_res"])))
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    out = os.path.join(outdir, "kappa_mismatch_partA.tex")
    open(out, "w").write("\n".join(lines) + "\n")
    return tab, out


# ---------------------------------------------------------------------------
# Part B -- directional asymmetry
# ---------------------------------------------------------------------------

def part_b(tab, k0, outdir):
    L = []
    w = L.append
    w("=" * 70)
    w("PART B -- DIRECTIONAL ASYMMETRY OF WAVENUMBER MISMATCH")
    w("=" * 70)
    w("kappa_0 = %g. d = log2(kappa_solve/kappa_0)." % k0)
    w("")
    w("%6s %7s %9s %9s %9s %9s" %
      ("d", "kappa", "frozen", "fresh", "penalty", "res.gap"))
    w("-" * 54)
    for _, r in tab.sort_values("d").iterrows():
        w("%+6.0f %7.3f %9s %9s %9s %8.1f%%" % (
            r["d"], r["kappa"], fmt(r["frozen_it"], "%d"),
            fmt(r["fresh_it"], "%d"), fmt(r["penalty"], "%.2f"),
            100 * r["gap"] if np.isfinite(r["gap"]) else float("nan")))
    w("")

    # reciprocal-pair comparison
    w("Reciprocal pairs (down vs up at equal |d|):")
    w("")
    by_d = {int(round(r["d"])): r for _, r in tab.iterrows()
            if np.isfinite(r["d"]) and abs(r["d"] - round(r["d"])) < 1e-9}
    verdicts = []
    for m in sorted({abs(d) for d in by_d if d != 0}):
        lo, hi = by_d.get(-m), by_d.get(+m)
        if lo is None or hi is None:
            w("  |d| = %d : incomplete (need both d=%+d and d=%+d)" % (m, -m, m))
            continue
        for what, key in (("frozen iterations", "frozen_it"),
                          ("penalty frozen/fresh", "penalty")):
            a, b = lo[key], hi[key]
            if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
                w("  |d| = %d  %-22s : missing data" % (m, what))
                continue
            ratio = b / a if a else float("nan")
            w("  |d| = %d  %-22s : down(k=%.3f)=%.3g  up(k=%.3f)=%.3g  "
              "up/down=%.2f" % (m, what, lo["kappa"], a, hi["kappa"], b, ratio))
            if key == "penalty":
                verdicts.append((m, ratio, lo["gap"], hi["gap"]))
    w("")

    # classification, strictly from the numbers present
    w("Classification (a)-(d):")
    if not verdicts:
        w("  (d) INSUFFICIENT EVIDENCE -- no complete reciprocal pair present.")
    else:
        sym = [abs(math.log(r)) for _, r, _, _ in verdicts
               if np.isfinite(r) and r > 0]
        contaminated = [m for m, _, g_lo, g_hi in verdicts
                        if (np.isfinite(g_hi) and g_hi < 0.05)
                        or (np.isfinite(g_lo) and g_lo < 0.05)]
        worst = max(sym) if sym else float("nan")
        w("  max |log(up/down penalty)| over reciprocal pairs = %.3f "
          "(0 = perfect symmetry in |d|)" % worst)
        if contaminated:
            w("  NOTE: pair(s) |d| = %s involve a kappa within 5%% of an "
              "interior Maxwell eigenvalue." % contaminated)
            w("        Those pairs cannot separate direction from resonance "
              "proximity; see the QC report.")
        if not np.isfinite(worst):
            w("  (d) INSUFFICIENT EVIDENCE.")
        elif worst < 0.18:                     # within ~20% either way
            w("  (a) consistent with dependence on |log2(kappa/kappa_0)| alone.")
        else:
            w("  (b) directional asymmetry: up and down mismatch of equal |d| "
              "give penalties differing by more than 20%.")
        w("  Distinguishing (b) from (c) -- dependence on absolute kappa -- "
          "requires Part D;")
        w("  a pure-ratio law makes R_ij constant along each diagonal of the "
          "Part D matrix.")
    w("")
    w("This classification is a mechanical reading of the computed numbers, "
      "not a physical claim.")
    out = os.path.join(outdir, "kappa_mismatch_partB.txt")
    open(out, "w").write("\n".join(L) + "\n")
    return "\n".join(L), out


# ---------------------------------------------------------------------------
# Part C -- figures
# ---------------------------------------------------------------------------

def part_c(tab, k0, outdir):
    t = tab.dropna(subset=["penalty"]).sort_values("d")
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.axvline(0.0, color="0.45", lw=1.0, ls="--", zorder=1)
    ax.axhline(1.0, color="0.75", lw=0.8, ls=":", zorder=1)
    ax.plot(t["d"], t["penalty"], color=PENALTY_COLOR, marker="D",
            markersize=5, zorder=3, label=r"frozen / fresh")
    for _, r in t.iterrows():
        ax.annotate("%.1f$\\times$" % r["penalty"], (r["d"], r["penalty"]),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5, color="0.25")
    # mark near-resonant abscissae without recolouring the series
    for _, r in t.iterrows():
        if np.isfinite(r["gap"]) and r["gap"] < 0.05:
            ax.annotate("$\\dagger$", (r["d"], r["penalty"]),
                        textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=9, color="#b35806")
    ax.set_xlabel(r"$\log_2(\kappa_{\mathrm{solve}}/\kappa_{\mathrm{precond}})$")
    ax.set_ylabel(r"iters(frozen) / iters(fresh)")
    ax.set_title(r"Wavenumber-mismatch penalty, $\kappa_0=%g$" % k0)
    ax.xaxis.set_major_locator(FixedLocator(sorted(t["d"].tolist())))
    ax.annotate("matched", (0, ax.get_ylim()[1]), textcoords="offset points",
                xytext=(4, -12), fontsize=7.5, color="0.45", ha="left")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))      # headroom for the value labels
    ax.legend(frameon=True, framealpha=0.9, loc="upper center")
    f1 = os.path.join(outdir, "fig_kappa_mismatch_bidir.pdf")
    fig.savefig(f1)
    plt.close(fig)

    # diagnostic: absolute counts
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.axvline(0.0, color="0.45", lw=1.0, ls="--", zorder=1)
    for key, col in (("plain_it", "plain"), ("frozen_it", "frozen"),
                     ("fresh_it", "fresh")):
        s = tab.dropna(subset=[key]).sort_values("d")
        if s.empty:
            continue
        st = STYLE[col]
        ax.plot(s["d"], s[key], color=st["color"], marker=st["marker"],
                label=st["label"], zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_2(\kappa_{\mathrm{solve}}/\kappa_{\mathrm{precond}})$")
    ax.set_ylabel("GMRES iterations (log scale)")
    ax.set_title(r"Absolute iteration counts, $\kappa_0=%g$" % k0)
    ax.xaxis.set_major_locator(FixedLocator(sorted(tab["d"].dropna().tolist())))
    ax.legend(frameon=True, framealpha=0.9)
    f2 = os.path.join(outdir, "fig_kappa_mismatch_counts.pdf")
    fig.savefig(f2)
    plt.close(fig)
    return f1, f2


# ---------------------------------------------------------------------------
# Part D -- preconditioner matrix
# ---------------------------------------------------------------------------

def part_d(df, outdir):
    d = df[df["part"] == "D"]
    if d.empty:
        return None, None, None
    kps = sorted(d["kappa_precond"].dropna().unique())
    kss = sorted(d["kappa_solve"].dropna().unique())
    R = np.full((len(kps), len(kss)), np.nan)
    F = np.full((len(kps), len(kss)), np.nan)
    for i, kp in enumerate(kps):
        for j, ks in enumerate(kss):
            fr = pick(d, kappa_precond=kp, kappa_solve=ks, strategy="frozen")
            fe = pick(d, kappa_solve=ks, strategy="fresh")
            if fr is not None:
                F[i, j] = fr["iters"]
            if fr is not None and fe is not None and fe["iters"] > 0:
                R[i, j] = fr["iters"] / fe["iters"]

    def mat_tex(M, name, spec):
        L = [r"\begin{tabular}{@{}l%s@{}}" % ("r" * len(kss)), r"\toprule",
             r"$\kappa_{\mathrm{precond}}\backslash\kappa_{\mathrm{solve}}$ & "
             + " & ".join("%g" % k for k in kss) + r" \\", r"\midrule"]
        for i, kp in enumerate(kps):
            cells = []
            for j in range(len(kss)):
                v = M[i, j]
                s = "--" if not np.isfinite(v) else spec % v
                if np.isclose(kps[i], kss[j]):
                    s = r"\underline{%s}" % s
                cells.append(s)
            L.append("%g & %s \\\\" % (kp, " & ".join(cells)))
        L += [r"\bottomrule", r"\end{tabular}"]
        return "%% %s\n%s\n" % (name, "\n".join(L))

    tex = (mat_tex(R, "Part D: penalty R_ij = frozen/fresh", "%.2f") + "\n"
           + mat_tex(F, "Part D: raw frozen iteration counts", "%.0f"))
    out = os.path.join(outdir, "kappa_mismatch_partD.tex")
    open(out, "w").write(tex)

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    im = ax.imshow(R, cmap=SEQ_CMAP, aspect="auto", origin="upper",
                   vmin=1.0 if np.nanmin(R) >= 1.0 else np.nanmin(R))
    ax.set_xticks(range(len(kss)), ["%g" % k for k in kss])
    ax.set_yticks(range(len(kps)), ["%g" % k for k in kps])
    ax.set_xlabel(r"$\kappa_{\mathrm{solve}}$")
    ax.set_ylabel(r"$\kappa_{\mathrm{precond}}$")
    ax.set_title(r"Reuse penalty $R_{ij}$ = iters(frozen)/iters(fresh)")
    ax.grid(False)
    vmid = np.nanmean([np.nanmin(R), np.nanmax(R)])
    for i in range(len(kps)):
        for j in range(len(kss)):
            v = R[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "--", ha="center", va="center", fontsize=7.5,
                        color="0.4")
                continue
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=7.5,
                    color="white" if v > vmid else "#20304a")
            if np.isclose(kps[i], kss[j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="#d62728", lw=1.6))
    fig.colorbar(im, ax=ax, label=r"$R_{ij}$", fraction=0.046, pad=0.03)
    f = os.path.join(outdir, "fig_kappa_mismatch_heatmap.pdf")
    fig.savefig(f)
    plt.close(fig)
    return R, F, (out, f, kps, kss)


# ---------------------------------------------------------------------------
# Part D -- ratio-law test (the real discriminator between (a), (b) and (c))
# ---------------------------------------------------------------------------

def ratio_law(R, kps, kss, outdir):
    """The Part D grid replicates each log2 ratio d at several absolute kappa.

    If the penalty depends on the RATIO alone, R_ij is constant along each
    diagonal d = log2(ks/kp).  Spread within a d-group therefore measures
    dependence on absolute kappa; comparing the d and -d groups measures
    direction.
    """
    L = []
    w = L.append
    w("=" * 70)
    w("PART D -- RATIO-LAW TEST")
    w("=" * 70)
    w("Each row is one value of d = log2(kappa_solve/kappa_precond), pooled")
    w("over all (precond, solve) pairs in the Part D grid that realise it.")
    w("")
    w("%6s %4s %9s %9s %9s %9s   %s"
      % ("d", "n", "min R", "max R", "mean R", "spread", "pairs (kp->ks)"))
    w("-" * 88)
    groups = {}
    for i, kp in enumerate(kps):
        for j, ks in enumerate(kss):
            v = R[i, j]
            if not np.isfinite(v):
                continue
            d = round(math.log2(ks / kp), 6)
            groups.setdefault(d, []).append((kp, ks, v))
    for d in sorted(groups):
        g = groups[d]
        vals = np.array([v for _, _, v in g])
        spread = (vals.max() / vals.min()) if vals.min() > 0 else float("nan")
        w("%+6g %4d %9.3f %9.3f %9.3f %8.2fx   %s"
          % (d, len(vals), vals.min(), vals.max(), vals.mean(), spread,
             ", ".join("%g->%g" % (a, b) for a, b, _ in g)))
    w("")

    multi = {d: np.array([v for _, _, v in g]) for d, g in groups.items()
             if len(g) >= 2}
    if not multi:
        w("No d value is realised more than once: the ratio law cannot be")
        w("tested from this grid.  Verdict: (d) INSUFFICIENT EVIDENCE.")
    else:
        worst = max(v.max() / v.min() for v in multi.values() if v.min() > 0)
        w("Largest within-d spread over replicated ratios: %.2fx" % worst)
        if worst < 1.15:
            w("  -> R depends on the ratio alone to within 15%: supports (a)/(b),")
            w("     and argues AGAINST (c) dependence on absolute kappa.")
        elif worst < 1.5:
            w("  -> moderate dependence on absolute kappa in addition to the")
            w("     ratio: consistent with (c) as a secondary effect.")
        else:
            w("  -> strong dependence on absolute kappa: (c) dominates, and the")
            w("     penalty cannot be described by the ratio alone.")

        # direction, pooled over absolute kappa
        w("")
        w("Direction, pooled over absolute kappa:")
        any_pair = False
        for m in sorted({abs(d) for d in groups if d != 0}):
            lo, hi = groups.get(-m), groups.get(+m)
            if not lo or not hi:
                continue
            any_pair = True
            a = np.mean([v for _, _, v in lo])
            b = np.mean([v for _, _, v in hi])
            w("  |d| = %g : mean R down = %.3f (n=%d), up = %.3f (n=%d), "
              "up/down = %.2f" % (m, a, len(lo), b, len(hi), b / a))
        if not any_pair:
            w("  no reciprocal d pair is present in the grid.")
    w("")
    w("Caveat: any pair involving kappa = 4 or 8 on the unit sphere sits within")
    w("3.3% and 2.3% of an interior Maxwell eigenvalue respectively, while every")
    w("kappa below 2.743707 is exactly resonance-free.  Upward mismatch is")
    w("therefore contaminated in a way downward mismatch is not; the detuned")
    w("control runs are what separate the two.")
    out = os.path.join(outdir, "kappa_mismatch_partD_ratiolaw.txt")
    open(out, "w").write("\n".join(L) + "\n")
    return "\n".join(L), out


# ---------------------------------------------------------------------------
# Part E -- quality control
# ---------------------------------------------------------------------------

def part_e(df, k0, outdir, res_tol=1e-6, sol_tol=1e-6):
    L = []
    w = L.append
    w("=" * 70)
    w("PART E -- QUALITY CONTROL")
    w("=" * 70)
    ok = True

    nc = df[~df["converged"].astype(bool)]
    w("[%s] convergence: %d of %d solves did not converge"
      % ("PASS" if nc.empty else "FAIL", len(nc), len(df)))
    if not nc.empty:
        ok = False
        for _, r in nc.iterrows():
            w("       kp=%s ks=%s %s iters=%s" % (r["kappa_precond"],
              r["kappa_solve"], r["strategy"], r["iters"]))

    bad = df[df["true_res"] > res_tol]
    w("[%s] true residuals: %d rows exceed %.0e (max %.2e)"
      % ("PASS" if bad.empty else "FAIL", len(bad), res_tol,
         df["true_res"].max() if len(df) else float("nan")))
    if not bad.empty:
        ok = False

    if "rel_diff_vs_fresh" in df.columns:
        rd = pd.to_numeric(df["rel_diff_vs_fresh"], errors="coerce").dropna()
        if len(rd):
            w("[%s] frozen vs fresh solutions agree: max rel. diff %.2e "
              "(tol %.0e)" % ("PASS" if rd.max() <= sol_tol else "FAIL",
                              rd.max(), sol_tol))
            ok &= bool(rd.max() <= sol_tol)
        else:
            w("[SKIP] frozen-vs-fresh solution comparison: column empty")
    else:
        w("[SKIP] frozen-vs-fresh solution comparison: column absent")

    # matched diagonal
    w("")
    w("Matched diagonal (kappa_solve == kappa_precond must reproduce fresh):")
    diag_ok = True
    for part in ("A", "D"):
        sub = df[df["part"] == part]
        for kp in sorted(sub["kappa_precond"].dropna().unique()):
            fr = pick(sub, kappa_precond=kp, kappa_solve=kp, strategy="frozen")
            fe = pick(sub, kappa_solve=kp, strategy="fresh")
            if fr is None or fe is None:
                w("  part %s kappa=%g : missing" % (part, kp))
                continue
            same = abs(fr["iters"] - fe["iters"]) <= 1
            diag_ok &= bool(same)
            w("  part %s kappa=%-5g frozen=%-4d fresh=%-4d  %s"
              % (part, kp, fr["iters"], fe["iters"],
                 "OK" if same else "MISMATCH (>1 iteration)"))
    ok &= diag_ok

    # settings homogeneity
    w("")
    w("Settings homogeneity (must be constant across every row):")
    for c in ("dof", "h", "geometry", "tol_outer", "tol_inner", "maxit", "seed"):
        if c not in df.columns:
            w("  %-10s : column absent" % c)
            continue
        u = list(pd.unique(df[c].dropna()))
        flag = "OK" if len(u) <= 1 else "VARIES -- investigate"
        if len(u) > 1:
            ok = False
        w("  %-10s : %s  %s" % (c, ", ".join(str(x) for x in u[:6]), flag))

    # resonance proximity
    w("")
    w("Interior-resonance proximity (unit ball; spectrum starts at 2.743707):")
    for k in sorted(set(df["kappa_solve"].dropna()) |
                    set(df["kappa_precond"].dropna())):
        g = rel_resonance_gap(k)
        tagged = "FLAG" if g < 0.05 else ("watch" if g < 0.10 else "clean")
        w("  kappa = %-6g rel. gap = %7.2f%%   %s" % (k, 100 * g, tagged))
    w("")
    w("OVERALL: %s" % ("PASS" if ok else "ATTENTION REQUIRED"))
    out = os.path.join(outdir, "kappa_mismatch_qc.txt")
    open(out, "w").write("\n".join(L) + "\n")
    return ok, "\n".join(L), out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--k0", type=float, default=2.0,
                    help="kappa_precond of the Part A frozen preconditioner")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--res-tol", type=float, default=1e-6, dest="res_tol",
                    help="QC ceiling on the true relative residual. Keep 1e-6 "
                         "for production runs at h=0.1 (the manuscript's own "
                         "runs land at <=6.3e-7); loosen to ~5e-6 only for a "
                         "coarse smoke mesh.")
    ap.add_argument("--sol-tol", type=float, default=1e-6, dest="sol_tol",
                    help="QC ceiling on ||u_frozen - u_fresh||/||u_fresh||. "
                         "Same advice as --res-tol.")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load(args.csv)
    tab, fa = part_a(df, args.k0, args.outdir)
    txt_b, fb = part_b(tab, args.k0, args.outdir)
    f1, f2 = part_c(tab, args.k0, args.outdir)
    R, F, dout = part_d(df, args.outdir)
    txt_rl = frl = None
    if dout is not None:
        txt_rl, frl = ratio_law(R, dout[2], dout[3], args.outdir)
    okE, txt_e, fe = part_e(df, args.k0, args.outdir,
                            res_tol=args.res_tol, sol_tol=args.sol_tol)

    print(txt_b)
    if txt_rl:
        print()
        print(txt_rl)
    print()
    print(txt_e)
    print()
    print("written:")
    for f in ([fa, fb, f1, f2, fe] + ([dout[0], dout[1]] if dout else [])
              + ([frl] if frl else [])):
        print("   ", f)
    if not okE:
        print("\nQC did not fully pass -- do not put these numbers in the "
              "manuscript until the flagged items are resolved.")


if __name__ == "__main__":
    main()
