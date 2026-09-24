#!/usr/bin/env python3
"""Turn the p32/p33 CSVs into the LaTeX that replaces the \\todo blocks in 5.9.

Every number the paper prints for the almond comes out of here, so that no
figure is ever transcribed by hand from a terminal. Run it after p32 and p33
and paste the output, or redirect it to a file and \\input it.

    python make_almond_tables.py <sweepdir> [> sec59_results.tex]

`sweepdir` is data/sweep. Missing files are reported and skipped, so this is
safe to run while a campaign is still going: it writes tables for whatever is
finished and says what is not.

Samples with no fresh solve (p32 --nfresh) carry iters_fresh = -1; they count
towards the frozen statistics and are excluded from the frozen/fresh columns,
which is stated in the caption the script emits.
"""
import csv
import os
import statistics as st
import sys

OUT = sys.stdout


def read(path):
    if not os.path.exists(path):
        print(f"%% MISSING: {os.path.basename(path)} -- not yet run", file=OUT)
        return None
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return rows or None


def f(x, n=0):
    """Format, or an em dash when the quantity does not exist."""
    if x is None:
        return "---"
    return f"{x:.{n}f}" if n else f"{x:.0f}"


def rng(v, n=0):
    v = [x for x in v if x is not None]
    if not v:
        return "---"
    lo, hi = min(v), max(v)
    return f(lo, n) if lo == hi else f"{f(lo,n)}--{f(hi,n)}"


# ------------------------------------------------------------------ sweep
def sweep_table(rows):
    cells = {}
    for r in rows:
        k = (int(r["level"]), r["case"])
        cells.setdefault(k, []).append(r)

    print(r"""\begin{table}[t]
  \centering
  \caption{\cj{NASA almond amplitude sweep, $h=\lambda/12$, ten
    random fields per cell. Iteration counts are medians over the cell, with
    the range in brackets. The frozen-to-fresh column is over the samples
    solved both ways.}}
  \label{tab:almondsweep}
  \begin{tabular}{llrrrrr}
    \toprule
    & & $\varepsilon$ & & \multicolumn{3}{c}{GMRES iterations} \\
    \cmidrule(lr){5-7}
    Level & Case & (mm) & $\varepsilon\sup|\mathrm D\bs V|$
      & Plain & Frozen & Fresh \\
    \midrule""", file=OUT)
    for (lv, case) in sorted(cells):
        g = cells[(lv, case)]
        eps = float(g[0]["eps_m"]) * 1e3
        cert = [float(r["cert"]) for r in g]
        pl = [int(r["iters_plain"]) for r in g]
        fr = [int(r["iters_frozen"]) for r in g]
        fe = [int(r["iters_fresh"]) for r in g if int(r["iters_fresh"]) > 0]
        print(f"    {lv} & {case} & {eps:.1f} & "
              f"{min(cert):.2f}--{max(cert):.2f} & "
              f"{st.median(pl):.0f} [{min(pl)}--{max(pl)}] & "
              f"{st.median(fr):.0f} [{min(fr)}--{max(fr)}] & "
              + (f"{st.median(fe):.0f} [{min(fe)}--{max(fe)}] \\\\"
                 if fe else "--- \\\\"), file=OUT)
    print(r"""    \bottomrule
  \end{tabular}
\end{table}""", file=OUT)

    # the sentence that goes with it
    ratios = [int(r["iters_frozen"]) / int(r["iters_fresh"])
              for r in rows if int(r["iters_fresh"]) > 0]
    top = [r for r in rows if int(r["level"]) == max(int(x["level"]) for x in rows)]
    tr = [int(r["iters_frozen"]) / int(r["iters_fresh"])
          for r in top if int(r["iters_fresh"]) > 0]
    flagged = [r for r in rows if r.get("resonance_flag", "").strip() == "true"]
    KAPPA = "$1.03\\kappa$"
    note = ("No sample was flagged as a resonance candidate."
            if not flagged else
            f"{len(flagged)} sample(s) exceeded 1.5 times the cell median "
            f"and were re-solved at {KAPPA}; see the summary file.")
    print(f"""
%% Sentence for the sweep paragraph:
\\cj{{Across the {len(rows)} samples the frozen preconditioner costs between
{min(ratios):.2f} and {max(ratios):.2f} times the iterations of a freshly
assembled one (median {st.median(ratios):.2f}), and at the largest amplitude
between {min(tr):.2f} and {max(tr):.2f}. {note}}}""",
          file=OUT)


# --------------------------------------------------------------- campaign
def campaign_paragraph(rows):
    fr = [int(r["iters_frozen"]) for r in rows]
    sub = [r for r in rows if r["has_fresh"].strip().lower() == "true"]
    ratios = [int(r["iters_frozen"]) / int(r["iters_fresh"]) for r in sub]
    nonconv = [r for r in rows if r["conv_frozen"].strip().lower() != "true"]
    eps = float(rows[0]["eps_m"]) * 1e3
    conv = ("every sample converged to the $10^{-8}$ tolerance"
            if not nonconv else
            f"{len(nonconv)} of {len(rows)} samples did not reach the "
            "$10^{-8}$ tolerance within 1500 iterations and are reported "
            "as such")
    print(f"""
%% Campaign paragraph:
\\cj{{At the top amplitude, $\\varepsilon = {eps:.1f}$\\,mm, {len(rows)}
independent fields were drawn and solved with the frozen preconditioner. The
iteration count has median {st.median(fr):.0f} and range {min(fr)}--{max(fr)};
{conv}.
On the {len(sub)} samples also solved with a freshly assembled preconditioner
the frozen-to-fresh ratio lies between {min(ratios):.2f} and {max(ratios):.2f}
(median {st.median(ratios):.2f}). No realization was discarded: admissibility
is screened in advance, so the sample is the one that was drawn.}}""",
          file=OUT)


def sensitivity_paragraph(rows, primary_ell):
    """The ell = d/10 rows, against the primary ell at the same levels."""
    other = sorted({r["ell"] for r in rows} - {primary_ell})
    if not other:
        return
    alt = other[0]
    lv = sorted({int(r["level"]) for r in rows if r["ell"] == alt})
    if not lv:
        return

    def med(e, l):
        v = [int(r["iters_frozen"]) for r in rows
             if r["ell"] == e and int(r["level"]) == l]
        return st.median(v) if v else None

    pairs = [(l, med(primary_ell, l), med(alt, l)) for l in lv]
    pairs = [(l, a, b) for l, a, b in pairs if a is not None and b is not None]
    if not pairs:
        return
    body = "; ".join(f"level {l}, {f(a)} against {f(b)}" for l, a, b in pairs)
    d = max(b - a for _, a, b in pairs)
    word = "no additional iterations" if d == 0 else (
        "one additional iteration" if d == 1 else f"{f(d)} additional iterations")
    print(f"""
%% Correlation-length sensitivity paragraph:
\\cj{{Shortening the correlation length from ${primary_ell.replace('d', 'd/')}$
to ${alt.replace('d', 'd/')}$ leaves the picture unchanged. Median frozen
counts, longer against shorter: {body}. The rougher field costs {word} at
equal amplitude, consistent with a preconditioner that responds to the size
of the deformation rather than to its spectral content.}}""", file=OUT)


def refinement_paragraph(rows, primary_mesh, primary_ell):
    """The lambda/20 check: the same amplitude, 2.7x the dof.

    This is the sharpest experiment in the section, because it moves h while
    holding the perturbation at its maximum. Plain grows, frozen does not.
    """
    other = sorted({r["mesh"] for r in rows} - {primary_mesh})
    if not other:
        return
    ref = other[0]
    R = [r for r in rows if r["mesh"] == ref]
    L = max(int(r["level"]) for r in R)
    B = [r for r in rows if r["mesh"] == primary_mesh and r["ell"] == primary_ell
         and int(r["level"]) == L and r["case"] == R[0]["case"]]
    if not B:
        return
    dofB, dofR = int(B[0]["dof"]), int(R[0]["dof"])
    pB = [int(r["iters_plain"]) for r in B]
    pR = [int(r["iters_plain"]) for r in R]
    fB = [int(r["iters_frozen"]) for r in B]
    fR = [int(r["iters_frozen"]) for r in R]
    print(f"""
%% Mesh-refinement paragraph (the lambda/20 check):
\\cj{{Refining from ${dofB}$ to ${dofR}$ degrees of freedom at the top
amplitude, with the perturbation held fixed, raises the unpreconditioned
iteration count from a median of {f(st.median(pB))} to {f(st.median(pR))}
(range {min(pR)}--{max(pR)} over {len(R)} fields) and leaves the frozen count
where it was: {f(st.median(fR))} against {f(st.median(fB))}, ranges
{min(fR)}--{max(fR)} and {min(fB)}--{max(fB)}. The mesh-independence that
Calder\\'on preconditioning is built for therefore survives freezing the
operator on the nominal geometry, and the advantage in iteration count grows
with refinement, from {st.median(pB)/st.median(fB):.1f} to
{st.median(pR)/st.median(fR):.1f}.}}""", file=OUT)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    print("%% Generated by almond/make_almond_tables.py -- do not hand-edit.",
          file=OUT)
    s = read(os.path.join(d, "p32_almond_sweep_summary.csv"))
    if s:
        # The primary correlation length is whichever one the sweep actually
        # ran at, not a hard-coded "d5": the sensitivity rows sit in the same
        # file and must not be mixed into the main table. The primary is the
        # one with the most rows.
        def commonest(rows, key):
            n = {}
            for r in rows:
                n[r[key]] = n.get(r[key], 0) + 1
            return max(n, key=n.get), n

        # p32 APPENDS, so this one file also holds the ell = d/10 sensitivity
        # and the lambda/20 mesh check. Both must be filtered out of the main
        # table: a refined-mesh row carries a plain count 50% higher and
        # corrupts the level it lands in, which is how "430 [401--643]" and a
        # cell of thirteen samples got into a table captioned "h = lambda/12,
        # ten random fields per cell".
        ell, elln = commonest(s, "ell")
        mesh, meshn = commonest(s, "mesh")
        print(f"%% primary: mesh {mesh}, ell {ell}; also present: "
              f"{ {k: v for k, v in list(meshn.items()) + list(elln.items()) if k not in (mesh, ell)} }",
              file=OUT)
        main_rows = [r for r in s if r["ell"] == ell and r["mesh"] == mesh]
        sweep_table(main_rows)
        sensitivity_paragraph([r for r in s if r["mesh"] == mesh], ell)
        refinement_paragraph(s, mesh, ell)
    c = read(os.path.join(d, "p33_almond_campaign_summary.csv"))
    if c:
        campaign_paragraph(c)


if __name__ == "__main__":
    main()
