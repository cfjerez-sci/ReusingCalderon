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
| `data/sweep/` | Summary CSVs backing every table in the paper |
| `data/figexport/` | Exported mesh/RCS data (from `p11_meshes_and_rcs.jl`) used by the figure scripts |
| `data/versioninfo.txt` | Exact hardware/software stack of the reported runs |
| `figures/` | Manuscript figures (PDF) and the Python scripts that render them from the CSVs |

## Script → result mapping

Numbering follows the submitted manuscript.

| Paper result | Script | Output |
|---|---|---|
| Table 1 — per-matrix assembly times | `scripts/p10_time_matrix_assembly_components.jl`, `..._amp30.jl` | `p10_matrix_assembly_component_times*.csv` |
| Table 2 — sphere amplitude sweep (3–50 %) | `scripts/p10_PEC_sphere_perturbation_amplitude_sweep.jl` | `p10_amplitude_sweep_summary.csv` |
| Table 3 — N=20 random campaign | `scripts/p10_PEC_sphere_perturbation_robustness_v2.jl`, `..._amp30.jl` | `p10_perturbation_summary_v2*.csv` |
| Table 4 — mesh refinement at fixed perturbation | `scripts/p12_h_refinement_study.jl` | `p12_h_refinement_summary.csv` |
| Table 5 — stochastic Galerkin p-refinement | `scripts/p13_sg_prefinement.jl` | `p13_sg_prefinement_summary.csv` |
| Table 6 — bidirectional wavenumber mismatch | `scripts/p22_kappa_mismatch_bidirectional.jl --lowk --detune` | `p22_kappa_mismatch_bidirectional_LOWK.csv` |
| Table 7 — triaxial ellipsoid sweep | `scripts/p10_ellipsoid_amplitude_sweep.jl` | `p10_ellipsoid_amplitude_sweep_summary.csv` |
| Table 8 — Fichera corner, two-face bump | `scripts/p10_fichera_amplitude_sweep.jl` | `p10_fichera_amplitude_sweep_summary.csv` |
| Table 9 — Fichera corner, perturbation **at** the corner | `scripts/p26_fichera_corner_sweep.jl --h 0.15` | `p26_fichera_corner_sweep_summary.csv` |
| Fig. 1 — nominal vs perturbed meshes | `scripts/p11_meshes_and_rcs.jl` + `figures/make_fig_meshes_2x3.py` | — |
| Fig. 2 — GMRES iterations vs amplitude | `figures/make_fig_amplitude_only.py` | — |
| Fig. 3 — direct test of the perturbation estimate | `scripts/p25_perturbation_linearity.jl` + `figures/make_fig_linearity.py` | `p25_perturbation_linearity.csv` |
| Fig. 4 — bistatic RCS, sphere | `scripts/p11_meshes_and_rcs.jl` + `figures/make_fig_rcs_sphere.py` | — |
| §5.2 — storage artifact, not conditioning | `scripts/p23_hybrid_precond_sweep.jl` | `p23_hybrid_precond_summary.csv` |
| §5.2, §3.6 — exactness of the transported RWG–BC pairing | `scripts/p24_nxy_invariance_check.jl` | `p24_log.txt` |
| §5.6 — higher wavenumber (κ=4) | `scripts/p10_PEC_sphere_perturbation_amplitude_sweep_kappa4.jl` | `p10_amplitude_sweep_summary_kappa4.csv` |
| §5.8 — matched flat-face control at h=0.15 | `scripts/p10_fichera_amplitude_sweep.jl --h 0.15` | `p10_fichera_amplitude_sweep_summary_h0.15.csv` |
| Reproducibility info | `scripts/p12_versioninfo_dump.jl` | `data/versioninfo.txt` |

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
- The project scaffold derives from Kristof Cools' MIT-licensed
  `Exp25_CJH_KC_LocalMultiTrace.jl`; the Julia package name is kept so that
  the pinned `Manifest.toml` resolves unchanged.

## License

MIT — see `LICENSE`.
