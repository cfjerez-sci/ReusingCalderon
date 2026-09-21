#!/usr/bin/env python3
"""[CJ-12] Analysis of p25_perturbation_linearity.csv.

Tests the two falsifiable consequences of Corollary 7 / Theorem 8:

  (P1) LINEARITY   slope of log||A_eps - A_0||_F against log(eps) == 1,
                   at every mesh level.
  (P2) h-UNIFORMITY  kappa_S(P_0 A_eps) as a function of eps is the same
                   curve at every mesh level.

Produces fig_perturbation_linearity.pdf (two panels) and a printed report.

  python3 figures/analyze_p25_linearity.py data/sweep/p25_perturbation_linearity.csv
"""
import csv, sys, math
from collections import defaultdict

src = sys.argv[1] if len(sys.argv) > 1 else "data/sweep/p25_perturbation_linearity.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "fig_perturbation_linearity.pdf"

rows = [r for r in csv.DictReader(open(src))]
if not rows:
    sys.exit("no rows in " + src)

by_h = defaultdict(list)
for r in rows:
    by_h[float(r["h"])].append(r)
for h in by_h:
    by_h[h].sort(key=lambda r: float(r["amplitude"]))

def lsq_slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs)
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    b = sxy/sxx
    a = my - b*mx
    ss_res = sum((y-(a+b*x))**2 for x, y in zip(xs, ys))
    ss_tot = sum((y-my)**2 for y in ys)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    return b, a, r2

print("="*74)
print("(P1)  LINEARITY IN epsilon   -- theory predicts slope 1")
print("="*74)
print(f"{'h':>8} {'dof':>7} {'pts':>4} {'slope':>8} {'R^2':>8}   verdict")
slopes = {}
for h in sorted(by_h):
    rs = by_h[h]
    xs = [math.log(float(r["amplitude"])) for r in rs]
    ys = [math.log(float(r["normF_dA"])) for r in rs]
    b, a, r2 = lsq_slope(xs, ys)
    slopes[h] = b
    ok = "OK" if abs(b-1) < 0.05 else ("near" if abs(b-1) < 0.15 else "** OFF **")
    print(f"{h:>8.4g} {rs[0]['dof']:>7} {len(rs):>4} {b:>8.4f} {r2:>8.5f}   {ok}")
print("\n  (slope is norm-independent here: the sweep rescales one fixed")
print("   deformation direction, so A_eps - A_0 = eps*D + O(eps^2).)")

have_k = any(r["kappaS_P0Aeps"] not in ("", "NaN") for r in rows)
if have_k:
    print()
    print("="*74)
    print("(P2)  h-UNIFORMITY of kappa_S(P_0 A_eps)  -- theory predicts one curve")
    print("="*74)
    amps = sorted({float(r["amplitude"]) for r in rows})
    hs = sorted(by_h)
    hdr = f"{'eps':>7} " + " ".join(f"{('h='+format(h,'g')):>12}" for h in hs) + f" {'spread':>9}"
    print(hdr)
    max_spread = 0.0
    for a in amps:
        vals = []
        for h in hs:
            m = [r for r in by_h[h] if abs(float(r["amplitude"])-a) < 1e-12]
            vals.append(float(m[0]["kappaS_P0Aeps"]) if m and m[0]["kappaS_P0Aeps"] not in ("","NaN") else None)
        good = [v for v in vals if v is not None]
        spread = (max(good)-min(good))/min(good) if len(good) > 1 else float("nan")
        if good and len(good) > 1:
            max_spread = max(max_spread, spread)
        cells = " ".join(f"{(format(v,'.4f') if v is not None else '-'):>12}" for v in vals)
        print(f"{100*a:>6.0f}% {cells} {(format(100*spread,'.1f')+'%' if spread==spread else '-'):>9}")
    print(f"\n  worst relative spread across mesh levels: {100*max_spread:.1f}%")
    print("  (small spread => the bound of Theorem 8 is h-uniform in practice)")

    print()
    print("="*74)
    print("kappa_S growth vs the (1+nu)/(1-nu) shape")
    print("="*74)
    for h in hs:
        rs = [r for r in by_h[h] if r["kappaS_P0Aeps"] not in ("","NaN")]
        if not rs: continue
        k0 = float(rs[0]["kappaS_P0A0"])
        print(f"  h={h:g}  kappa_S(P_0 A_0) = {k0:.4f}")
        for r in rs:
            k = float(r["kappaS_P0Aeps"]); e = float(r["amplitude"])
            ratio = k/k0
            # (1+c e)/(1-c e) = ratio  =>  c = (ratio-1)/(e (ratio+1))
            c = (ratio-1)/(e*(ratio+1)) if e > 0 else float("nan")
            print(f"     eps={100*e:>5.1f}%  kappa_S={k:>9.4f}  ratio={ratio:>7.4f}  implied nu/eps={c:>8.4f}")
        print("     (implied nu/eps roughly constant => nu grows linearly in eps)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("\n[matplotlib missing -- report only, no figure]")
    sys.exit(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
colors = ["#c62828", "#1565c0", "#2e7d32", "#6a4fa3", "#b35806"]
for i, h in enumerate(sorted(by_h)):
    rs = by_h[h]
    c = colors[i % len(colors)]
    e = [float(r["amplitude"]) for r in rs]
    d = [float(r["normF_dA"]) for r in rs]
    ax1.loglog(e, d, "o-", color=c, label=f"$h={h:g}$ ({rs[0]['dof']} dof)")
ref_e = [float(r["amplitude"]) for r in by_h[sorted(by_h)[0]]]
ref_d = [float(by_h[sorted(by_h)[0]][0]["normF_dA"]) * (x/ref_e[0]) for x in ref_e]
ax1.loglog(ref_e, ref_d, "k--", lw=1, alpha=.6, label="slope 1")
ax1.set_xlabel(r"perturbation amplitude $\varepsilon$")
ax1.set_ylabel(r"$\|\mathbf{A}_{\varepsilon,h}-\mathbf{A}_{0,h}\|_F$")
ax1.set_title("(a) linearity in $\\varepsilon$", fontsize=10)
ax1.legend(fontsize=7.5); ax1.grid(alpha=.3, which="both")

if have_k:
    for i, h in enumerate(sorted(by_h)):
        rs = [r for r in by_h[h] if r["kappaS_P0Aeps"] not in ("","NaN")]
        if not rs: continue
        c = colors[i % len(colors)]
        ax2.plot([float(r["amplitude"]) for r in rs],
                 [float(r["kappaS_P0Aeps"]) for r in rs],
                 "s-", color=c, label=f"$h={h:g}$")
    ax2.set_xlabel(r"perturbation amplitude $\varepsilon$")
    ax2.set_ylabel(r"$\kappa_S(\mathbf{P}_0\mathbf{A}_{\varepsilon,h})$")
    ax2.set_title("(b) $h$-uniformity of the conditioning bound", fontsize=10)
    ax2.legend(fontsize=7.5); ax2.grid(alpha=.3)
else:
    ax2.text(.5, .5, "run without --no-kappa\nfor panel (b)", ha="center", va="center",
             transform=ax2.transAxes, fontsize=9, color="0.4")
    ax2.set_xticks([]); ax2.set_yticks([])

plt.tight_layout()
plt.savefig(out, dpi=200, bbox_inches="tight")
print("\nsaved", out)
