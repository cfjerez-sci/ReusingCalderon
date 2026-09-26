# ReusingCalderon

Code and data accompanying the manuscript

> P. Escapil-Inchauspé and C. Jerez-Hanckes,
> **Freezing Calderón: Robust Preconditioning for Maxwell Scattering under Shape Uncertainty**,
> submitted to *SIAM Journal on Scientific Computing* (2026).

A Calderón (operator) preconditioner assembled once on a nominal geometry is
reused, unchanged, across families of perturbed geometries — sampled
realizations, mesh refinements, and fully coupled stochastic Galerkin systems.
This repository contains the Julia scripts, supporting infrastructure, and
summary data behind every table and figure in the paper.

## Requirements

- Julia 1.11 (experiments were run with Julia 1.11.5 on an Apple M3 Max, 36 GB RAM)
- [BEAST.jl](https://github.com/krcools/BEAST.jl) v2.8.0
- [CompScienceMeshes.jl](https://github.com/krcools/CompScienceMeshes.jl) v0.10.0
- DrWatson v2.19.1, Makeitso v2.1.1 (result registration / caching)

Exact versions of all dependencies are pinned in `Manifest.toml`. To set up:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Every script activates the project environment itself, so they can be run
directly, e.g. `julia scripts/p12_h_refinement_study.jl`. Full recomputation
takes hours per experiment (dense BEM assembly at up to 343 quadrature nodes);
the `data/sweep` CSVs let you inspect all reported numbers without recomputing.
Makeitso caches intermediate operators under `data/` (git-ignored; the cache
grows large).

## Layout

| Path | Contents |
|---|---|
| `scripts/` | Experiment drivers (see mapping below) |
| `problems/` | Perturbed-geometry problem definitions (sphere, ellipsoid, Fichera corner) |
| `methods/` | EFIE assembly, Calderón preconditioning, GMRES drivers |
| `postproc/` | Spherical-harmonic / face-bump shape perturbations, Mie reference, far fields |
| `geos/` | Gmsh geometry for the Fichera corner |
| `almond/` | NASA-almond geometry, random-field construction, validity checks, table and figure generators (Python) |
| `data/almond/` | The two shipped almond meshes, the 140 drawn field coefficient files, `fields_index.csv` (with each field's measured sup\|DV\|) and the amplitude ladder `levels.csv` |
| `data/sweep/` | Summary CSVs backing every table in the paper |
| `data/figexport/` | Exported mesh/RCS data (from `p11_meshes_and_rcs.jl`) used by the figure scripts |
| `data/versioninfo.txt` | Exact hardware/software stack of the reported runs |
| `figures/` | Manuscript figures (PDF) and the Python scripts that render them from the CSVs |

## Script → result mapping

Numbering follows the manuscript as resubmitted to SISC (September 26, 2026), which
numbers tables and figures by section; items marked SM are in the Supplementary
Materials.

| Paper result | Script | Output |
|---|---|---|
| Table 4.1 — per-matrix assembly times (sphere rows) | `scripts/p10_time_matrix_assembly_components.jl`, `..._amp30.jl` | `p10_matrix_assembly_component_times*.csv` |
| Table 5.1 — sphere amplitude sweep (3–50 %) | `scripts/p10_PEC_sphere_perturbation_amplitude_sweep.jl` | `p10_amplitude_sweep_summary.csv` |
| Table 5.2 — N=20 random campaign | `scripts/p10_PEC_sphere_perturbation_robustness_v2.jl`, `..._amp30.jl` | `p10_perturbation_summary_v2*.csv` |
| Table 5.3 — mesh refinement at fixed perturbation | `scripts/p12_h_refinement_study.jl` | `p12_h_refinement_summary.csv` |
| Table 5.4 — stochastic Galerkin p-refinement | `scripts/p13_sg_prefinement.jl` | `p13_sg_prefinement_summary.csv` |
| Table 5.5 — bidirectional wavenumber mismatch | `scripts/p22_kappa_mismatch_bidirectional.jl --lowk --detune` | `p22_kappa_mismatch_bidirectional_LOWK.csv` |
| Table 5.6 — triaxial ellipsoid sweep | `scripts/p10_ellipsoid_amplitude_sweep.jl` | `p10_ellipsoid_amplitude_sweep_summary.csv` |
| Table 5.7 — Fichera corner, perturbation **at** the corner | `scripts/p26_fichera_corner_sweep.jl --h 0.15` | `p26_fichera_corner_sweep_summary.csv` |
| Table 5.7 — the θ_min / σ_max mesh-quality columns | `scripts/p30_fichera_mesh_quality.jl` | `p30_fichera_mesh_quality.csv` |
| Table 4.1 — per-matrix assembly times (NASA almond row) | `scripts/p31_almond_feasibility.jl` | `p31_almond_feasibility.csv` |
| Table 5.8 — NASA almond amplitude sweep | `scripts/p32_almond_sweep.jl` | `p32_almond_sweep_summary.csv` |
| Fig. 3.1 — transport of the preconditioner | drawn inline in the manuscript (TikZ) | — |
| Fig. 5.1 — perturbed meshes, four geometries | `scripts/p11_meshes_and_rcs.jl` + `figures/make_fig_meshes_row4.py` | — |
| Fig. 5.2 — direct test of the perturbation estimate | `scripts/p25_perturbation_linearity.jl` + `figures/make_fig_linearity.py` | `p25_perturbation_linearity.csv` |
| Fig. 5.3 — NASA almond amplitude sweep | `almond/make_fig_almond_sweep.py` | from `p32_almond_sweep_summary.csv` |
| Fig. 5.4 — NASA almond Monte Carlo campaign | `almond/make_fig_almond_campaign.py` | from `p33_almond_campaign_*.csv` |
| Fig. SM1 (Supplementary Materials) — bistatic RCS, three geometries | `scripts/p11_meshes_and_rcs.jl` + `figures/make_fig_rcs.py` | — |
| §5.2 — storage artifact, not conditioning | `scripts/p23_hybrid_precond_sweep.jl` | `p23_hybrid_precond_summary.csv` |
| §5.2, §3.6 — exactness of the transported RWG–BC pairing | `scripts/p24_nxy_invariance_check.jl` | `p24_log.txt` |
| §5.6 — higher wavenumber (κ=4) | `scripts/p10_PEC_sphere_perturbation_amplitude_sweep_kappa4.jl` | `p10_amplitude_sweep_summary_kappa4.csv` |
| §5.8 — matched flat-face control at h=0.15 | `scripts/p10_fichera_amplitude_sweep.jl --h 0.15` | `p10_fichera_amplitude_sweep_summary_h0.15.csv` |
| §5.9 — Monte Carlo campaign at the top amplitude | `scripts/p33_almond_campaign.jl` | `p33_almond_campaign_summary.csv`, `..._rcs.csv` |
| §5.9 — monostatic sweep and the multi-RHS timing | `scripts/p34_almond_monostatic.jl` | `p34_almond_monostatic.csv` |
| §5.9 — the LaTeX of every almond number | `almond/make_almond_tables.py data/sweep` | `data/sweep/sec59_results.tex` |
| Reproducibility info | `scripts/p12_versioninfo_dump.jl` | `data/versioninfo.txt` |

## The NASA almond campaign (§5.9)

The almond experiment is self-contained and scripted end to end. Shapes are
drawn from the paper's own admissible class, `T(x) = x + ε V(x)` with `V` a
vector-valued random field, and **admissibility is screened before a shape is
constructed**: the amplitude ladder is chosen so that `ε·sup|DV| ≤ 1/2` for every
field in `fields_index.csv`, which makes the deformation injective. No
realization is ever rejected, so the reported statistics carry no rejection
bias. A triangle–triangle intersection test is retained as a check on the
implementation, not as a filter.

```bash
./scripts/run_almond.sh --check   # preflight only, seconds
NFRESH=3 ./scripts/run_almond.sh  # the full campaign, ~8 h on an M3 Max
```

`p35_almond_preflight.jl` gates the run: it verifies that both meshes load
and are consistently oriented, that the requested correlation length exists in
the exported field index, that the top amplitude satisfies the injectivity
screening bound for **every** field that will be drawn at it, and that the
perturbation applies without disturbing connectivity. The driver then runs the
amplitude sweep, the correlation-length sensitivity, the 100-sample campaign,
the λ/20 refinement check, the monostatic sweep, and finally the table and
figure generators. It is resumable: each driver skips samples already present
in its CSV.

The drawn fields are shipped rather than re-drawn, so every geometry behind
Section 5.9 reproduces exactly. To regenerate them from the seeds instead
(deterministic — the same seeds give the same numbers):

```bash
cd almond && python export_for_julia.py ../data/almond 4 10   # ℓ = d/4 primary, d/10 sensitivity
```

`almond/README_almond.md` documents the deformation model, the injectivity
screening bound and the amplitude ladder in full.

`scripts/p23_fingerprints.jl` is the shared edge/vertex-fingerprint and
permutation module used by the p23–p26 drivers;
`figures/analyze_p25_linearity.py` prints the least-squares slopes and the
h-uniformity spread quoted in §5.3.

Supporting scripts (`p10_investigate_*`, `p10_compare_matrix_entries.jl`,
`p10_sample_vertices_table.jl`, Fichera smoke tests) document the
DOF-alignment construction described in the implementation section.

## Notes

- Perturbed meshes preserve the nominal connectivity; RWG/BC degrees of
  freedom are aligned across geometries via a combinatorial edge-fingerprint
  permutation (see `scripts/p10_investigate_dof_indexing.jl` and the per-node
  permutation in `scripts/p13_sg_prefinement.jl`).
- The experiments are built on [BEAST.jl](https://github.com/krcools/BEAST.jl)
  and [CompScienceMeshes.jl](https://github.com/krcools/CompScienceMeshes.jl)
  by Kristof Cools and contributors, which supply the boundary element
  assembly and the RWG/BC discretizations these scripts drive. They are used
  as ordinary MIT-licensed dependencies; no third-party code is redistributed
  in this repository. `Manifest.toml` is unchanged from the runs reported in
  the paper, so the pinned dependency versions are exactly those used.

## License

MIT — see `LICENSE`.
