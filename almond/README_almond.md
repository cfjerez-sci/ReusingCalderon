# NASA almond shape-uncertainty benchmark

Everything for the almond experiment: the geometry and mesher, the admissible
perturbation field, the dry-run validity checks, the Julia drivers and the
figure scripts.

## What the deformation is

The paper's own admissible class, Section 3.1:

    T(x) = x + eps * V(x),        V : R^3 -> R^3 smooth,

with `V` a vector-valued random Fourier feature expansion of a
squared-exponential covariance (three i.i.d. components, isotropic,
correlation length `ell`). It is **not** a displacement along surface
normals. Two consequences matter:

* `T` is injective on any convex set as soon as `eps * sup|DV|_2 < 1`, and
  `sup|DV|` is computable in closed form. The amplitude ladder is chosen so
  that `eps * sup|DV| <= 0.5` for every sample drawn, i.e. with a factor-two
  margin. **Validity is screened in advance, so no sample is ever rejected
  and the Monte Carlo statistics carry no rejection bias.** The
  triangle-triangle test is kept as a safety check; a failure is a bug to
  report, not a sample to drop.
* Case **U** applies `V` as it is, so the sharp tip moves with its
  neighbourhood. Case **T** multiplies `V` by a C^1 cutoff that vanishes
  identically within `d/10` of the tip and rises to 1 over the following
  `d/4`. The `grad chi` term is included in the Jacobian bound, so case T is
  screened on the field that is actually applied. The two lengths do
  different jobs: `d/10` is the physics ("the tip does not move"), `d/4` is
  the Lipschitz budget, chosen so that the cutoff does not, by itself, cap
  the amplitude below what the field allows.

## Files

| file | what |
|---|---|
| `almond/almond.py` | geometry, mesher, vector field + analytic Jacobian, `sup_jacobian`, self-intersection test |
| `almond/legacy.py` | the abandoned normal-displacement model, kept for the record |
| `almond/audit_bvh_and_v1.py` | intersection-test audit + the clean v1 table |
| `almond/dry_run.py` | validity of every campaign geometry, no assembly |
| `almond/export_for_julia.py` | writes `data/almond/` |
| `almond/export_io.py` | reads it back, for the figures |
| `almond/make_fig_almond_*.py` | Figures A, B, C |
| `postproc/almond_geometry.jl` | mesh loading, field evaluation, perturbation, certificate |
| `problems/p11_perturbed_almond.jl` | Makeitso problem definition |
| `scripts/p31_almond_feasibility.jl` | calibration, resonance screening, timing |
| `scripts/p32_almond_sweep.jl` | 5 levels x 2 cases x 10 seeds |
| `scripts/p33_almond_campaign.jl` | 100-sample Monte Carlo |
| `scripts/p34_almond_monostatic.jl` | monostatic azimuth sweep |
| `scripts/p35_almond_preflight.jl` | seconds-long gate before a long run |
| `scripts/run_almond.sh` | the whole campaign, in order, resumable |
| `almond/make_almond_tables.py` | CSVs to the LaTeX of Section 5.9 |
| `almond/jlint.py` | Julia checks for the mistakes a Python habit produces |

`data/almond/` holds the two nominal meshes (CSV + Gmsh copies), the 140
exported field parameter files, `fields_index.csv` (with each field's
`sup|DV|`) and `levels.csv` (the amplitude ladder). Regenerate with

    cd almond && python export_for_julia.py ../data/almond

which is deterministic: the same seeds give the same numbers.

## Configuration

| | |
|---|---|
| geometry | NASA almond, d = 9.936 in = 0.252374 m (Woo et al. 1993) |
| frequency | 3.5642 GHz, lambda = 84.125 mm, **electrical size 3 lambda**, kappa*d = 18.85 |
| primary mesh | h = lambda/12 = 7.010 mm, 893 vertices, 1782 faces, **2673 RWG dof** |
| refined mesh | h = lambda/20 = 4.206 mm, 2425 vertices, 4846 faces, **7269 dof** |
| correlation length | ell = d/5 primary, d/10 sensitivity |
| ladder | 2.2, 4.4, 6.6, 8.8, **11.0 mm** = lambda/38.2 ... **lambda/7.6** |
| top level | 1.57 h, 67.6 % of the seam half-thickness, `eps*sup|DV| <= 0.498` |

## Running it

One command, from the project root:

```bash
./scripts/run_almond.sh --check     # preflight only, seconds
./scripts/run_almond.sh             # stages 2-7, ~11.9 h
NFRESH=3 ./scripts/run_almond.sh    # ~8.3 h, see below
nohup ./scripts/run_almond.sh > run_almond.out 2>&1 &   # and walk away
```

It runs the preflight first and refuses to start if that fails, then each
stage in order, timing it and teeing its output to `data/sweep/logs/`. A
failing stage stops the run rather than burning hours on a broken premise.
Everything is resumable: the drivers skip samples already in their CSVs, so
rerunning after an interruption continues where it stopped. `--only 4` runs a
single stage.

| stage | what | as written | `NFRESH=3` |
|---|---|---|---|
| 2 | amplitude sweep, 5 levels x 2 cases x 10 seeds | 6.1 h | 2.1 h |
| 3 | correlation-length sensitivity, levels 1-2 | 2.5 h | 0.8 h |
| 4 | Monte Carlo campaign, 100 frozen + 20 fresh | 1.5 h | 1.5 h |
| 5 | mesh check at lambda/20, 3 seeds | 1.4 h | 1.4 h |
| 6 | monostatic azimuth sweep | 0.4 h | 0.4 h |
| 7 | tables and figures from the CSVs | seconds | seconds |
| | **total** | **11.9 h** | **8.3 h** |

Timings are measured, not extrapolated: stage 1 (`p31`) gave full assembly
213.3 s per sample (Zxx 7.3, Tyy 204.0, Nxy 1.5, bx 0.5), frozen-only 7.8 s,
and solves of 0.9 / 4.6 / 1.6 s for plain / frozen / fresh on an Apple M3 Max.

`NFRESH=n` solves the freshly assembled reference on the first `n` seeds of
each 10-seed cell instead of all ten. The fresh solve is what forces a Tyy
assembly, 204 s of the 220 s a sweep sample costs, so this is where almost all
the time is. The frozen counts, which are what the paper claims, still come
from every seed.

Stage 7 is the rule that no almond number is ever typed by hand: it reads the
summary CSVs and writes `data/sweep/sec59_results.tex`, the exact LaTeX that
replaces the `\todo` blocks in Section 5.9, sentences and min/median/max
included, plus the three figures beside it. It does not write into the
manuscript; copy the figures across when you are happy with them.

### Changing the correlation length

The ladder is set by the primary correlation length, so switching it means
re-exporting, which takes about two minutes:

```bash
cd almond && python export_for_julia.py ../data/almond 4 10   # d/4 and d/10
ELL=d4 ./scripts/run_almond.sh
```

| primary | worst sup\|DV\| | top level | | | |
|---|---|---|---|---|---|
| d/5 (as exported) | 45.28 m^-1 | 11.00 mm | lambda/7.6 | 1.57 h | 67.6 % |
| d/4 | 37.47 m^-1 | 13.30 mm | lambda/6.3 | 1.90 h | 81.8 % |

A longer correlation length buys amplitude by making the field smoother,
which is also easier for the preconditioner. That trade belongs in the paper
rather than in a default, which is why the lengths are arguments.

## What the preflight checks

Seconds, no assembly, no solves -- it touches only the cached geometry:

* both meshes load, are consistently oriented, enclose a positive volume and
  give 2673 and 7269 dof;
* the field index exists and contains the correlation length asked for;
* the top level's screening value `eps*sup|DV|` is under 0.5 for **every** field
  that will be drawn at it, so a re-export cannot silently push a sample over;
* the perturbation applies, moves the mesh by what the ladder says, and leaves
  connectivity untouched.

## What to read in stage 1's output

* `dof = 2673` and a positive enclosed volume.
* `t_T / t_Z` — the cost model of Section 5 uses it, and the campaign's
  saving is `(t_T + t_N) / (t_Z + t_T + t_N + t_b)`.
* The resonance screen prints all three iteration counts. If the count at
  `kappa` exceeds 1.5x the median of the three, shift `kappa` (pass
  `--kappa`) and rerun everything, rather than reporting a resonant case.
* `permutation valid? X: true Y: true`. If either is false, stop: the
  DOF-alignment argument does not apply and nothing downstream is meaningful.
* The projected wall clock at the end.
