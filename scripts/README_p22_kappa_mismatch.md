# Brief for a Claude Code session in the working repo

Start Claude Code in **`/Users/carlosjerez/Dartagnan/DATA/CODES/Julia/ReusingCalderon.jl`** — the working copy with the 146 GB Makeitso cache, *not* the `ReusingCalderon` publication replica. Paste everything below the rule as the first message.

The driver is already written against the repo's own API — there is nothing to wire. It reuses `make(Sim2.discretization; ...)`, `solve_plain_gmres` and `solve_calderon_gmres` exactly as `scripts/p10_kappa_mismatch_precond_sweep.jl` does. The five files it depends on (`methods/EFIE.jl`, `methods/EFIE_manual_solves.jl`, `problems/p10_perturbed_sphere.jl`, `postproc/shape_perturbation.jl`, `src/…jl`) are byte-identical between the two repos, so it behaves the same in either.

---

You are extending the numerical experiments of *"One Calderón Operator to Rule Them All: Robust Preconditioning under Shape Uncertainty"*.

Three new files are already in place:

- `scripts/p22_kappa_mismatch_bidirectional.jl` — the experiment driver
- `figures/analyze_kappa_mismatch.py` — tables, figures, ratio-law test, QC report
- `postproc/resonance_check.py` — interior Maxwell eigenvalues of the unit ball

(`p14`–`p21` belong to the Feibelman/nanophotonics work, hence `p22`.)

Repo conventions from `CLAUDE.md` apply: run scripts from the repo root, wrap work in its own `module` when sharing a REPL (the driver already does), `data/` is git-ignored, and **do not commit or push without asking Carlos**.

Read the header comment of the driver before anything else. Also read `kappa_mismatch/README_kappa_mismatch.md` in the manuscript folder if it is to hand — it carries the design rationale — but do not re-derive any of it.

## Goal

Characterise, **numerically only**, how a frozen Calderón preconditioner degrades when the solve wavenumber differs from the wavenumber it was assembled at — in **both** directions. The existing Section 5.6 experiment (`scripts/p10_kappa_mismatch_precond_sweep.jl`, freezing at κ = 1 and sweeping κ_solve upward over {1,2,4,8}) only establishes behaviour for increasing κ.

- **Part A**: κ_precond = 2 fixed; κ_solve ∈ {0.25, 0.5, 1, 2, 4, 8}, i.e. d = log₂(κ_solve/κ_precond) ∈ {−3…+2}.
- **Part D**: κ_precond ∈ {1, 2, 4} × κ_solve ∈ {0.5, 1, 2, 4, 8}, giving R_ij = iters_frozen / iters_fresh.

## Hard constraints

1. **No theorem about wavenumber reuse.** Remark 11 explains why the shape-perturbation argument gives no uniform (h,ν) result when κ changes — the −iκ and 1/(iκ) weights rescale the EFIE's two parts relative to one another. This experiment characterises; it does not prove.
2. **Do not edit the manuscript**, and do not touch `data/sweep/p10_kappa_mismatch_precond_summary.csv`. The driver writes a new file.
3. **Never discard a failed run.** The driver records failures with `converged = false` and a `FAILED:` note; leave them in.
4. **Never invent a number.** If something cannot be measured, say so.
5. **Change no numerical setting.** The driver inherits everything from `methods/EFIE.jl` and `methods/EFIE_manual_solves.jl`; `frozen` and `fresh` differ only in which `(Tyy, Nxy)` pair is passed.

## Step 1 — smoke test first

```
julia --project scripts/p22_kappa_mismatch_bidirectional.jl --smoke
python3 figures/analyze_kappa_mismatch.py \
    data/sweep/p22_kappa_mismatch_bidirectional_SMOKE.csv --k0 2.0 --outdir /tmp/smoke
```

This runs the full Parts A+D matrix at h = 0.4 in a few minutes. Its iteration counts are physically meaningless (κ = 8 at h = 0.4 is ~2 elements per wavelength) and must never reach the manuscript — the point is to prove the plumbing.

`/tmp/smoke/kappa_mismatch_qc.txt` must show: every solve converged; true residuals below 1e-6; frozen and fresh solutions agreeing to solver tolerance; dof/h/tolerances constant; and **the matched diagonal (κ_solve = κ_precond) reproducing the fresh count to within one iteration**, in both parts.

If the matched diagonal is off by more than one iteration, stop and diagnose. Do not proceed.

## Step 2 — the real run

```
julia --project scripts/p22_kappa_mismatch_bidirectional.jl --detune 2>&1 | tee data/sweep/p22_log.txt
```

`--detune` adds resonance controls at κ = 3.52 and κ = 8.4661. **Not optional** — see step 4.

Cost is dominated by `make(Sim2.discretization; κ=…)` at h = 0.1: roughly 440 s per *uncached* wavenumber, 713 MB of Makeitso cache each.

The cache in this repo has already been checked: at `h=0.1, realization=0, amplitude=0.03, radius=1.0` — exactly the parameter set the driver uses — **κ = 1, 2, 4 and 8 are already cached** from the p10 runs. Only κ = 0.25 and 0.5 are cold, plus 3.52 and 8.4661 under `--detune`. So expect roughly **15 min without `--detune`, 30 min with it**, not 75. Disk was 95 GB free against ~2.9 GB needed. Run it in the background and poll the log rather than blocking.

## Step 3 — analyse

```
python3 figures/analyze_kappa_mismatch.py \
    data/sweep/p22_kappa_mismatch_bidirectional.csv --k0 2.0 --outdir data/sweep/p22_results
```

Read `kappa_mismatch_qc.txt` **before** anything else. If QC does not pass, report that first and do not present the numbers as results.

## Step 4 — interpret, carefully

The question is whether degradation depends only on |log₂(κ/κ₀)|, or on direction, or on absolute κ. Two confounds must be accounted for, and they pull in opposite directions.

**Interior resonance.** The interior Maxwell spectrum of the unit ball begins at **κ = 2.743707** (run `postproc/resonance_check.py` to regenerate). So:

- κ = 0.25, 0.5, 1, 2 lie below the entire spectrum — exactly resonance-free.
- κ = 4 sits **3.2 %** from the TM(ℓ=2) eigenvalue 3.870239.
- κ = 8 sits **2.3 %** from the TE(ℓ=4) eigenvalue 8.182561.

The downward arm is clean; the upward arm is not. A raw "upward is worse" reading is confounded. The detuned controls (κ = 3.52, gap 9.9 %; κ = 8.4661, gap 3.0 % — the best available, the spectrum is dense up there) are what separate mismatch from resonance proximity. Compare them explicitly against κ = 4 and κ = 8.

Note that the *existing* Section 5.6 data is suggestive here: plain GMRES takes 272 iterations at κ = 2, 256 at κ = 4, but 353 at κ = 8. Do not over-read it — plain counts grow with κ anyway — but it is consistent with κ = 8 being the contaminated point.

**Low-frequency breakdown.** At κ = 0.25 and 0.5 the 1/(iκ) scalar-potential term dominates and conditioning degrades for reasons unrelated to mismatch. Calderón preconditioning does not cure this. The penalty R normalises much of it, since frozen and fresh solve the same Zxx, but not all — the fresh preconditioner is itself built at a breakdown wavenumber. Do not read that as mismatch either.

**The Part D ratio-law test is the primary discriminator**, not Part B's reciprocal pairs. The grid realises each log-ratio at several absolute κ (d = +1 from 1→2, 2→4, 4→8), so spread *within* a d-group measures dependence on absolute κ, and the pooled ±d comparison measures direction while averaging over the resonance contamination. Single reciprocal pairs are the weaker instrument.

Classify as (a) depends on |log₂| alone, (b) directional asymmetry, (c) substantial dependence on absolute κ, or (d) insufficient evidence — **from the computed data only**. The script proposes a mechanical verdict; sanity-check it and overrule it if the resonance caveat makes it unsafe.

## Step 5 — report

Give me: the tables; the figures; where the raw CSV and log are; a concise diagnostic interpretation; an explicit downward-vs-upward comparison including the detuned controls; any anomalies or resonance concerns; and a recommendation on whether this is strong enough to replace Section 5.6.

Do not edit the manuscript. If you think specific edits are warranted, list them for me to approve.

One limitation to state rather than fix: with only h = 0.1 the experiment says nothing about whether the mismatch penalty is h-uniform. Do not run a second sweep for it.
