# ReusingCalderon

Code and data accompanying the manuscript

> P. Escapil-Inchauspé and C. Jerez-Hanckes,
> **Reusing Calderón Preconditioners for Electric Field Integral Equations on Parametric and Uncertain Geometries**,
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

| Paper result | Script | Output CSV |
|---|---|---|
| Table 1 — sphere amplitude sweep (3–50 %) | `scripts/p10_PEC_sphere_perturbation_amplitude_sweep.jl` | `p10_amplitude_sweep_summary.csv` |
| Table 2 — ellipsoid amplitude sweep | `scripts/p10_ellipsoid_amplitude_sweep.jl` | `p10_ellipsoid_amplitude_sweep_summary.csv` |
| Table 3 — N=20 random campaign + timings | `scripts/p10_PEC_sphere_perturbation_robustness_v2.jl`, `..._amp30.jl`, `scripts/p10_time_matrix_assembly_components.jl`, `..._amp30.jl` | `p10_perturbation_summary_v2*.csv`, `p10_matrix_assembly_component_times*.csv` |
| Table 4 — mesh refinement (frozen flat in h) | `scripts/p12_h_refinement_study.jl` | `p12_h_refinement_summary.csv` |
| Table 5 — stochastic Galerkin p-refinement | `scripts/p13_sg_prefinement.jl` | `p13_sg_prefinement_summary.csv` |
| SG quadrature-convergence check (343 nodes) | `scripts/p13b_sg_quadcheck.jl` | `p13b_sg_quadcheck_summary.csv` |
| Fichera-corner sweep (beyond smooth theory) | `scripts/p10_fichera_amplitude_sweep.jl` | `p10_fichera_amplitude_sweep_summary.csv` |
| §5.5 — higher wavenumber (κ=4) | `scripts/p10_PEC_sphere_perturbation_amplitude_sweep_kappa4.jl` | `p10_amplitude_sweep_summary_kappa4.csv` |
| §5.7 — wavenumber-mismatch counterexperiment | `scripts/p10_kappa_mismatch_precond_sweep.jl` | `p10_kappa_mismatch_precond_summary.csv` |
| Fig. 1 — amplitude & mismatch overview | `figures/make_fig1_v2.py` | — |
| Fig. 2 — nominal vs perturbed meshes | `scripts/p11_meshes_and_rcs.jl` + `figures/make_fig_meshes.py` | — |
| Fig. 3 — bistatic RCS comparison | `scripts/p11_meshes_and_rcs.jl` + `figures/make_fig_rcs.py` | — |
| Reproducibility info | `scripts/p12_versioninfo_dump.jl` | `data/versioninfo.txt` |

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
